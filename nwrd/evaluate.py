"""Evaluation of a trained run.

Protocol: the binarisation threshold is chosen on the validation split, frozen, and
applied once to the held-out test split. Test metrics are also reported at the fixed 0.5
threshold, which needs no tuning at all.
"""
import json
import os

import numpy as np
import torch

from . import data as D
from .metrics import ProbHistogram, per_image_counts, scores
from .models import build_model, load_checkpoint


def predict_split(model, cfg, split, device, thresholds=()):
    """Returns the probability histogram, per-image counts at each threshold, and files."""
    files = sorted(os.listdir(D.split_dirs(cfg.data_root, split)[0]))
    files = files[:cfg.limit] if cfg.limit else files
    loader = torch.utils.data.DataLoader(
        D.RustDataset(cfg.data_root, split, files, D.eval_transform(cfg.img_size)),
        batch_size=cfg.batch_size, num_workers=cfg.num_workers, shuffle=False)
    amp = cfg.amp and device.type == 'cuda'
    hist = ProbHistogram()
    counts = {t: [] for t in thresholds}
    model.eval()
    with torch.no_grad():
        for imgs, masks, _ in loader:
            imgs, masks = imgs.to(device), masks.to(device)
            with torch.autocast(device.type, enabled=amp):
                probs = torch.sigmoid(model(imgs).float())
            hist.update(probs, masks)
            for t in thresholds:
                counts[t].append(per_image_counts(probs, masks, t))
    return hist, {t: np.concatenate(v) for t, v in counts.items()}, files


def evaluate_run(cfg, model_name, run_dir, log=print):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(model_name, pretrained=False, dropout=cfg.dropout).to(device)
    ckpt = load_checkpoint(model, os.path.join(run_dir, 'best.pt'), map_location=device)

    val_hist, _, _ = predict_split(model, cfg, 'val', device)
    thr, val_sweep = val_hist.best_threshold(lo=cfg.thr_lo, hi=cfg.thr_hi, step=cfg.thr_step)
    at_edge = thr in (cfg.thr_lo, cfg.thr_hi)

    test_hist, test_counts, test_files = predict_split(model, cfg, 'test', device,
                                                       thresholds=tuple(dict.fromkeys((thr, 0.5))))
    rust = np.array([c[0] + c[2] > 0 for c in test_counts[thr]])
    res = {
        'model': model_name,
        'best_epoch': ckpt.get('epoch'),
        'threshold': thr,
        'threshold_at_grid_edge': bool(at_edge),
        'val@thr': scores(*val_hist.counts(thr)),
        'val@0.5': scores(*val_hist.counts(0.5)),
        'test@thr': scores(*test_hist.counts(thr)),
        'test@0.5': scores(*test_hist.counts(0.5)),
        # NWRD paper Table 4 protocol: background-only patches removed from the test set.
        'test_rust_patches@thr': scores(*test_counts[thr][rust].sum(0)),
        'test_images': len(test_files),
        'test_rust_images': int(rust.sum()),
        'test_rust_pixel_fraction': float(test_hist.pos.sum() / (test_hist.pos.sum() + test_hist.neg.sum())),
    }
    with open(os.path.join(run_dir, 'metrics.json'), 'w') as f:
        json.dump(res, f, indent=2)
    with open(os.path.join(run_dir, 'val_threshold_sweep.json'), 'w') as f:
        json.dump(val_sweep, f)
    # Per-image confusion counts drive the paired bootstrap in compare.py.
    np.savez_compressed(os.path.join(run_dir, 'test_counts.npz'), files=np.array(test_files),
                        at_thr=test_counts[thr], at_05=test_counts[0.5])
    t = res['test@thr']
    log(f'[{model_name} {os.path.basename(run_dir)}] thr {thr:.2f}{" (GRID EDGE)" if at_edge else ""} | '
        f'test IoU {t["IoU"]:.4f} Dice {t["Dice"]:.4f} P {t["Precision"]:.4f} R {t["Recall"]:.4f}')
    return res


def plot_qualitative(cfg, run_dirs, path, n=6, seed=0):
    """Test patches with rust: image, ground truth, then each model's prediction at its own
    val-selected threshold. `run_dirs` maps model name -> run directory."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    counts = np.load(os.path.join(next(iter(run_dirs.values())), 'test_counts.npz'))
    rust = [i for i, c in enumerate(counts['at_thr']) if c[0] + c[2] > 0]
    idx = sorted(np.random.default_rng(seed).choice(rust, min(n, len(rust)), replace=False))
    files = [str(counts['files'][i]) for i in idx]
    ds = D.RustDataset(cfg.data_root, 'test', files, D.eval_transform(cfg.img_size))
    imgs = torch.stack([ds[i][0] for i in range(len(ds))])
    masks = torch.stack([ds[i][1] for i in range(len(ds))])

    preds = {}
    for name, run_dir in run_dirs.items():
        model = build_model(name, pretrained=False, dropout=cfg.dropout).to(device).eval()
        load_checkpoint(model, os.path.join(run_dir, 'best.pt'), map_location=device)
        with open(os.path.join(run_dir, 'metrics.json')) as f:
            thr = json.load(f)['threshold']
        with torch.no_grad():
            preds[name] = (torch.sigmoid(model(imgs.to(device))).cpu() >= thr, thr)
        del model

    mean, std = torch.tensor(D.MEAN)[:, None, None], torch.tensor(D.STD)[:, None, None]
    cols = ['Image', 'Ground truth'] + [f'{k} (t={t:.2f})' for k, (_, t) in preds.items()]
    fig, axes = plt.subplots(len(files), len(cols), figsize=(3.2 * len(cols), 3.2 * len(files)),
                             squeeze=False)
    for r in range(len(files)):
        axes[r][0].imshow((imgs[r] * std + mean).clamp(0, 1).permute(1, 2, 0).numpy())
        axes[r][1].imshow(masks[r, 0].numpy(), cmap='gray')
        for c, (p, _) in enumerate(preds.values(), start=2):
            axes[r][c].imshow(p[r, 0].numpy(), cmap='gray')
        for c, ax in enumerate(axes[r]):
            ax.axis('off')
            if r == 0:
                ax.set_title(cols[c], fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)
