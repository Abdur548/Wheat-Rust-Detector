"""Inference efficiency: parameters, FLOPs, latency, throughput and peak memory.

Every model is measured in the same process on the same device with random weights
(weights do not affect cost). `canet_b4_twopass` is CANet as first implemented, running
the encoder twice per forward, so the gain from the single-pass fix is measured too.
"""
import json
import platform
import time

import torch

from .models import MODELS, build_model


def count_flops(model, size, device):
    from torch.utils.flop_counter import FlopCounterMode
    x = torch.randn(1, 3, size, size, device=device)
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        model(x)
    return fc.get_total_flops()


def time_forward(model, x, device, amp, warmup, iters):
    """Median wall time per forward, in milliseconds."""
    times = []
    with torch.no_grad(), torch.autocast(device.type, enabled=amp):
        for i in range(warmup + iters):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(x)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            if i >= warmup:
                times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return times[len(times) // 2]


def benchmark_model(name, size=512, batch=8, device=None, quick=False):
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(name, pretrained=False).to(device).eval()
    params = sum(p.numel() for p in model.parameters())
    res = {'model': name, 'description': MODELS[name], 'params_M': params / 1e6,
           'GFLOPs': count_flops(model, size, device) / 1e9, 'input': f'{size}x{size}'}
    res['GMACs'] = res['GFLOPs'] / 2

    if device.type == 'cuda':
        warm, iters = (3, 5) if quick else (20, 100)
        x1 = torch.randn(1, 3, size, size, device=device)
        xb = torch.randn(batch, 3, size, size, device=device)
        res['gpu_latency_ms_bs1_fp32'] = time_forward(model, x1, device, False, warm, iters)
        res['gpu_latency_ms_bs1_fp16'] = time_forward(model, x1, device, True, warm, iters)
        torch.cuda.reset_peak_memory_stats()
        ms = time_forward(model, xb, device, True, warm, iters)
        res[f'gpu_throughput_img_s_bs{batch}_fp16'] = batch * 1000 / ms
        res[f'gpu_peak_mem_MB_bs{batch}_fp16'] = torch.cuda.max_memory_allocated() / 2**20

    cpu = torch.device('cpu')
    model = model.to(cpu)
    warm, iters = (1, 2) if quick else (3, 10)
    res['cpu_latency_ms_bs1_fp32'] = time_forward(
        model, torch.randn(1, 3, size, size), cpu, False, warm, iters)
    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return res


def run_benchmarks(models, out_path, size=512, quick=False, log=print):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env = {'device': torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu',
           'cpu': platform.processor() or platform.machine(),
           'cpu_threads': torch.get_num_threads(), 'torch': torch.__version__}
    rows = []
    for name in models:
        r = benchmark_model(name, size=size, device=device, quick=quick)
        rows.append(r)
        log(f'[bench] {name}: {r["params_M"]:.2f}M params, {r["GFLOPs"]:.1f} GFLOPs, '
            + ', '.join(f'{k} {v:.1f}' for k, v in r.items() if k.startswith(('gpu_', 'cpu_'))))
    with open(out_path, 'w') as f:
        json.dump({'env': env, 'results': rows}, f, indent=2)
    return rows
