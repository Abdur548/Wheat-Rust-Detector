"""Pixel metrics pooled over a whole split.

The notebooks averaged IoU per batch, which scores a correctly predicted all-background
batch as 0 and depresses the curve (~0.58 vs ~0.81 pooled). Everything here pools the
confusion counts over every pixel of the split, then computes the ratios once.
"""
import numpy as np
import torch

EPS = 1e-12


def scores(tp, fp, fn, tn):
    tp, fp, fn, tn = (float(v) for v in (tp, fp, fn, tn))
    return {
        'IoU': tp / (tp + fp + fn + EPS),
        'Dice': 2 * tp / (2 * tp + fp + fn + EPS),
        'Precision': tp / (tp + fp + EPS),
        'Recall': tp / (tp + fn + EPS),
        'Specificity': tn / (tn + fp + EPS),
        'Accuracy': (tp + tn) / (tp + tn + fp + fn + EPS),
    }


def per_image_counts(probs, target, threshold):
    """(B, 4) int64 array of tp, fp, fn, tn per image. Decision rule: prob >= threshold."""
    pred = probs >= threshold
    tgt = target > 0.5
    dims = tuple(range(1, probs.dim()))
    tp = (pred & tgt).sum(dims)
    fp = (pred & ~tgt).sum(dims)
    fn = (~pred & tgt).sum(dims)
    tn = (~pred & ~tgt).sum(dims)
    return torch.stack([tp, fp, fn, tn], 1).cpu().numpy().astype(np.int64)


class ProbHistogram:
    """Histograms of predicted probability for rust and background pixels.

    Gives exact pooled confusion counts at any threshold on a 1/bins grid without keeping
    every prediction in memory."""
    def __init__(self, bins=1000):
        self.bins = bins
        self.pos = np.zeros(bins, np.int64)
        self.neg = np.zeros(bins, np.int64)

    def update(self, probs, target):
        q = (probs.float() * self.bins).long().clamp_(0, self.bins - 1).flatten()
        t = (target > 0.5).flatten()
        self.pos += torch.bincount(q[t], minlength=self.bins).cpu().numpy()
        self.neg += torch.bincount(q[~t], minlength=self.bins).cpu().numpy()

    def counts(self, threshold):
        k = int(round(threshold * self.bins))
        tp, fp = self.pos[k:].sum(), self.neg[k:].sum()
        return tp, fp, self.pos.sum() - tp, self.neg.sum() - fp

    def sweep(self, lo=0.05, hi=0.95, step=0.01):
        rows = []
        for t in np.round(np.arange(lo, hi + step / 2, step), 4):
            s = scores(*self.counts(t))
            rows.append({'threshold': float(t), **s})
        return rows

    def best_threshold(self, **kw):
        """IoU = Dice / (2 - Dice), so maximising either picks the same threshold."""
        rows = self.sweep(**kw)
        return max(rows, key=lambda r: r['IoU'])['threshold'], rows
