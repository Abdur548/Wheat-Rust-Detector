"""NWRD patch dataset: split discovery, background subsampling, transforms."""
import glob
import json
import os
import random

import cv2
import numpy as np
import torch

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
DATASET_DIRNAME = 'wheat_rust_patches'


def find_data_root(hint=None):
    """Returns the directory containing train/ val/ test/.

    Searches `hint`, then Kaggle's input mount, then the kagglehub cache."""
    candidates = [hint] if hint else []
    candidates += ['/kaggle/input', os.path.expanduser('~/.cache/kagglehub')]
    for base in filter(None, candidates):
        if os.path.isdir(os.path.join(base, 'train', 'images')):
            return base
        hits = glob.glob(os.path.join(base, '**', DATASET_DIRNAME), recursive=True)
        if hits:
            return hits[0]
    raise FileNotFoundError(
        f'no {DATASET_DIRNAME}/ found under {candidates}; attach the Kaggle dataset '
        f'abdur548/nwrd-patched or pass --data-root')


def split_dirs(root, split):
    return os.path.join(root, split, 'images'), os.path.join(root, split, 'masks')


def read_mask(path):
    return (cv2.imread(path, cv2.IMREAD_GRAYSCALE) > 127).astype(np.float32)


def mask_stats(root, split, cache_dir=None):
    """Per-file rust pixel counts for a split, cached as JSON (reading 8k masks is slow)."""
    cache = os.path.join(cache_dir, f'mask_stats_{split}.json') if cache_dir else None
    if cache and os.path.exists(cache):
        with open(cache) as f:
            return json.load(f)
    _, mask_dir = split_dirs(root, split)
    stats = {}
    for f in sorted(os.listdir(mask_dir)):
        m = read_mask(os.path.join(mask_dir, f))
        stats[f] = {'pos': int(m.sum()), 'total': int(m.size)}
    if cache:
        os.makedirs(cache_dir, exist_ok=True)
        with open(cache, 'w') as f:
            json.dump(stats, f)
    return stats


def select_train_files(stats, bg_keep_ratio, seed):
    """All rust-positive patches plus a seeded random fraction of background-only ones.

    The seed is the *data* seed, fixed across models so every model sees the same patches."""
    disease = sorted(f for f, s in stats.items() if s['pos'] > 0)
    background = sorted(f for f, s in stats.items() if s['pos'] == 0)
    kept = random.Random(seed).sample(background, int(len(background) * bg_keep_ratio))
    return disease + sorted(kept), {'disease': len(disease), 'bg_kept': len(kept),
                                    'bg_total': len(background)}


def pos_weight_from_stats(stats, files):
    pos = sum(stats[f]['pos'] for f in files)
    total = sum(stats[f]['total'] for f in files)
    return (total - pos) / max(pos, 1)


def train_transform(size):
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    return A.Compose([
        A.Resize(size, size),
        A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(p=0.5), A.ElasticTransform(p=0.5),
        A.Normalize(mean=MEAN, std=STD), ToTensorV2()])


def eval_transform(size):
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    return A.Compose([A.Resize(size, size), A.Normalize(mean=MEAN, std=STD), ToTensorV2()])


class RustDataset(torch.utils.data.Dataset):
    def __init__(self, root, split, files, transform):
        self.img_dir, self.mask_dir = split_dirs(root, split)
        self.files, self.transform = list(files), transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        f = self.files[idx]
        img = cv2.cvtColor(cv2.imread(os.path.join(self.img_dir, f)), cv2.COLOR_BGR2RGB)
        out = self.transform(image=img, mask=read_mask(os.path.join(self.mask_dir, f)))
        return out['image'], out['mask'].unsqueeze(0), idx
