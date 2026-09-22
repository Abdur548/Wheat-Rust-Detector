"""CPU smoke tests: the full pipeline on a tiny synthetic dataset, plus metric checks.

    python -m pytest tests -q
"""
import json
import os

import cv2
import numpy as np
import pytest
import torch

from nwrd.metrics import ProbHistogram, per_image_counts, scores
from nwrd.models import build_model


def make_dataset(root, size=256):
    rng = np.random.default_rng(0)
    layout = {'train': (4, 2), 'val': (3, 1), 'test': (3, 1)}   # (rust, background) patches
    for split, (n_rust, n_bg) in layout.items():
        for sub in ('images', 'masks'):
            os.makedirs(os.path.join(root, split, sub))
        for i in range(n_rust + n_bg):
            img = rng.integers(0, 255, (size, size, 3), dtype=np.uint8)
            mask = np.zeros((size, size), np.uint8)
            if i < n_rust:
                y, x = rng.integers(0, size // 2, 2)
                mask[y:y + size // 4, x:x + size // 4] = 255
                img[mask > 0] = (200, 80, 30)
            cv2.imwrite(os.path.join(root, split, 'images', f'p{i}.png'), img)
            cv2.imwrite(os.path.join(root, split, 'masks', f'p{i}.png'), mask)


def test_metrics_pooled_and_histogram_agree():
    torch.manual_seed(0)
    probs, target = torch.rand(4, 1, 32, 32), (torch.rand(4, 1, 32, 32) > 0.8).float()
    target[0] = 0                                     # an all-background image
    hist = ProbHistogram()
    hist.update(probs, target)
    for t in (0.3, 0.5, 0.71):
        direct = per_image_counts(probs, target, t).sum(0)
        assert tuple(direct) == tuple(int(v) for v in hist.counts(t))
    s = scores(*hist.counts(0.5))
    assert 0 < s['IoU'] < s['Dice'] < 1
    assert abs(s['IoU'] - s['Dice'] / (2 - s['Dice'])) < 1e-9


@pytest.mark.parametrize('name', ['canet_b4', 'deeplabv3plus_r50', 'deeplabv3plus_b4', 'unet', 'unet_tuned'])
def test_models_output_logits_at_input_size(name):
    m = build_model(name, pretrained=False).eval()
    with torch.no_grad():
        assert m(torch.randn(1, 3, 256, 256)).shape == (1, 1, 256, 256)


def test_canet_single_pass_matches_two_pass():
    a = build_model('canet_b4', pretrained=False).eval()
    b = build_model('canet_b4_twopass', pretrained=False).eval()
    b.load_state_dict(a.state_dict())
    x = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        assert torch.equal(a(x), b(x))


def test_full_pipeline(tmp_path):
    from nwrd.run import main
    data, out = tmp_path / 'wheat_rust_patches', tmp_path / 'runs'
    make_dataset(str(data))
    args = ['--data-root', str(data), '--out', str(out), '--img-size', '256',
            '--models', 'canet_b4', 'unet', 'unet_tuned', '--seeds', '1', '2',
            '--epochs', '2', '--freeze-encoder-epochs', '1', '--batch-size', '2',
            '--num-workers', '0', '--bg-keep-ratio', '0.5', '--pretrained', 'false',
            '--bench-quick', '--bench-models', 'canet_b4', 'canet_b4_twopass']
    assert main(args) == 0
    m = json.load(open(out / 'canet_b4' / 'seed1' / 'metrics.json'))
    assert 0.05 <= m['threshold'] <= 0.95 and m['test_images'] == 4
    assert not (out / 'canet_b4' / 'seed1' / 'last.pt').exists()
    comp = json.load(open(out / 'comparison.json'))
    assert comp['significance']['unet']['seeds'] == [1, 2]
    assert 'test_rust_patches@thr' in m
    tuned = json.load(open(out / 'unet_tuned' / 'seed1' / 'history.json'))['history']
    shared = json.load(open(out / 'unet' / 'seed1' / 'history.json'))['history']
    assert tuned['lr'][0] == pytest.approx(1e-3) and shared['lr'][0] == pytest.approx(5e-5)
    assert 'paper-architecture baseline' in open(out / 'comparison.md', encoding='utf-8').read()
    assert 'Against the NWRD paper' in open(out / 'comparison.md', encoding='utf-8').read()
    exp = json.load(open(out / 'export' / 'results.json'))
    assert set(exp['final_metrics']) >= {'IoU', 'F1 / Dice'} and exp['split'] == 'test'
    for f in ('best_model.pth', 'confusion_matrix.png', 'training_curves.png',
              'qualitative_results.png'):
        assert (out / 'export' / f).exists(), f
    bench = json.load(open(out / 'benchmark.json'))['results']
    flops = {r['model']: r['GFLOPs'] for r in bench}
    assert flops['canet_b4_twopass'] > 1.3 * flops['canet_b4']   # encoder counted twice
    # Re-running is a no-op that re-aggregates.
    assert main(args) == 0


def test_time_budget_stops_and_resumes(tmp_path):
    from nwrd.run import main
    data, out = tmp_path / 'wheat_rust_patches', tmp_path / 'runs'
    make_dataset(str(data))
    base = ['--data-root', str(data), '--out', str(out), '--img-size', '256',
            '--models', 'canet_b4', '--seeds', '1', '--epochs', '3', '--batch-size', '2',
            '--num-workers', '0', '--pretrained', 'false', '--stage', 'train']
    assert main(base + ['--time-budget-hours', '1e-9']) == 3     # stops after epoch 1
    run = out / 'canet_b4' / 'seed1'
    assert (run / 'last.pt').exists() and not (run / 'TRAINED').exists()
    assert main(base) == 0
    hist = json.load(open(run / 'history.json'))['history']
    assert len(hist['train_loss']) == 3 and (run / 'TRAINED').exists()
