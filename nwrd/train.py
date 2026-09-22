"""Training for one (model, seed) run. Resumable from last.pt."""
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn

from . import data as D
from .metrics import ProbHistogram, scores
from .models import build_model


class CombinedLoss(nn.Module):
    def __init__(self, dice_w, bce_w, pos_weight):
        super().__init__()
        import segmentation_models_pytorch as smp
        self.dice_w, self.bce_w = dice_w, bce_w
        self.dice = smp.losses.DiceLoss(mode='binary', from_logits=True)
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, logits, target):
        return self.dice_w * self.dice(logits, target) + self.bce_w * self.bce(logits, target)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _limit(files, n):
    return files[:n] if n else files


def make_loaders(cfg, cache_dir):
    """Returns loaders, data info and the train generator (reseed it every epoch)."""
    root = cfg.data_root
    train_stats = D.mask_stats(root, 'train', cache_dir)
    train_files, sel = D.select_train_files(train_stats, cfg.bg_keep_ratio, cfg.data_seed)
    train_files = _limit(train_files, cfg.limit)
    val_files = _limit(sorted(D.mask_stats(root, 'val', cache_dir)), cfg.limit)
    pos_weight = D.pos_weight_from_stats(train_stats, train_files)

    kw = dict(batch_size=cfg.batch_size, num_workers=cfg.num_workers,
              pin_memory=torch.cuda.is_available())
    g = torch.Generator()
    train_loader = torch.utils.data.DataLoader(
        D.RustDataset(root, 'train', train_files, D.train_transform(cfg.img_size)),
        shuffle=True, drop_last=True, generator=g, **kw)
    val_loader = torch.utils.data.DataLoader(
        D.RustDataset(root, 'val', val_files, D.eval_transform(cfg.img_size)), shuffle=False, **kw)
    info = {**sel, 'train_used': len(train_files), 'val': len(val_files), 'pos_weight': pos_weight}
    return train_loader, val_loader, info, g


def run_validation(model, loader, criterion, device, amp):
    model.eval()
    hist, loss_sum, n = ProbHistogram(), 0.0, 0
    with torch.no_grad():
        for imgs, masks, _ in loader:
            imgs, masks = imgs.to(device, non_blocking=True), masks.to(device, non_blocking=True)
            with torch.autocast(device.type, enabled=amp):
                logits = model(imgs)
            loss_sum += criterion(logits.float(), masks).item() * imgs.size(0)
            n += imgs.size(0)
            hist.update(torch.sigmoid(logits.float()), masks)
    return loss_sum / n, hist


def train_run(cfg, model_name, seed, run_dir, log=print):
    os.makedirs(run_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    amp = cfg.amp and device.type == 'cuda'
    torch.backends.cudnn.benchmark = True
    set_seed(seed)

    train_loader, val_loader, info, gen = make_loaders(cfg, os.path.join(cfg.out_dir, '_cache'))
    log(f'[{model_name} s{seed}] data: {info}')
    model = build_model(model_name, pretrained=cfg.pretrained, dropout=cfg.dropout).to(device)
    criterion = CombinedLoss(cfg.dice_weight, cfg.bce_weight,
                             torch.tensor([info['pos_weight']], device=device))
    lr = cfg.unet_tuned_lr if model_name == 'unet_tuned' else cfg.lr
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=cfg.sched_t0, T_mult=cfg.sched_tmult, eta_min=cfg.sched_eta_min)
    scaler = torch.amp.GradScaler(device.type, enabled=amp)

    history = {k: [] for k in ('train_loss', 'val_loss', 'val_iou', 'val_dice',
                               'val_iou_best_thr', 'lr', 'epoch_seconds')}
    start_epoch, best_iou = 0, -1.0
    last_path, best_path = os.path.join(run_dir, 'last.pt'), os.path.join(run_dir, 'best.pt')
    if os.path.exists(last_path):
        ck = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ck['model'])
        optimizer.load_state_dict(ck['optimizer'])
        scheduler.load_state_dict(ck['scheduler'])
        scaler.load_state_dict(ck['scaler'])
        torch.set_rng_state(ck['rng_cpu'])
        if device.type == 'cuda' and ck.get('rng_cuda') is not None:
            torch.cuda.set_rng_state_all(ck['rng_cuda'])
        start_epoch, best_iou, history = ck['epoch'], ck['best_iou'], ck['history']
        log(f'[{model_name} s{seed}] resumed at epoch {start_epoch}')

    t_start = time.time()
    for epoch in range(start_epoch, cfg.epochs):
        if cfg.time_budget_hours and epoch > start_epoch:
            elapsed = time.time() - t_start
            per_epoch = elapsed / (epoch - start_epoch)
            if elapsed + 1.5 * per_epoch > cfg.time_budget_hours * 3600:
                log(f'[{model_name} s{seed}] time budget reached at epoch {epoch}; resume later')
                return False

        # Freezing only makes sense for an ImageNet-pretrained encoder.
        frozen = epoch < cfg.freeze_encoder_epochs and getattr(model, 'has_pretrained_encoder', True)
        for p in model.encoder.parameters():
            p.requires_grad = not frozen

        # Shuffle order and augmentation depend on (seed, epoch) only, so a resumed run
        # sees the same batches it would have seen uninterrupted.
        gen.manual_seed(seed * 10_000 + epoch)
        t0 = time.time()
        model.train()
        loss_sum, n = 0.0, 0
        for imgs, masks, _ in train_loader:
            imgs, masks = imgs.to(device, non_blocking=True), masks.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=amp):
                logits = model(imgs)
            loss = criterion(logits.float(), masks)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            loss_sum += loss.item() * imgs.size(0)
            n += imgs.size(0)
        lr = optimizer.param_groups[0]['lr']
        scheduler.step()

        val_loss, hist = run_validation(model, val_loader, criterion, device, amp)
        val = scores(*hist.counts(0.5))
        _, sweep = hist.best_threshold(lo=cfg.thr_lo, hi=cfg.thr_hi, step=cfg.thr_step)
        for k, v in (('train_loss', loss_sum / n), ('val_loss', val_loss),
                     ('val_iou', val['IoU']), ('val_dice', val['Dice']),
                     ('val_iou_best_thr', max(r['IoU'] for r in sweep)), ('lr', lr),
                     ('epoch_seconds', time.time() - t0)):
            history[k].append(v)
        log(f'[{model_name} s{seed}] ep {epoch + 1:02d}/{cfg.epochs} '
            f'{"(enc frozen) " if frozen else ""}train {loss_sum / n:.4f} val {val_loss:.4f} '
            f'IoU@0.5 {val["IoU"]:.4f} Dice@0.5 {val["Dice"]:.4f} ({history["epoch_seconds"][-1]:.0f}s)')

        # Model selection: pooled val IoU at the fixed 0.5 threshold.
        if val['IoU'] > best_iou:
            best_iou = val['IoU']
            torch.save({'model': model.state_dict(), 'epoch': epoch + 1, 'val_iou@0.5': best_iou,
                        'model_name': model_name, 'seed': seed}, best_path)
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(), 'scaler': scaler.state_dict(),
                    'rng_cpu': torch.get_rng_state(),
                    'rng_cuda': torch.cuda.get_rng_state_all() if device.type == 'cuda' else None,
                    'epoch': epoch + 1, 'best_iou': best_iou, 'history': history}, last_path)
        with open(os.path.join(run_dir, 'history.json'), 'w') as f:
            json.dump({'history': history, 'data': info, 'best_val_iou@0.5': best_iou}, f, indent=1)

    os.remove(last_path)   # optimizer state is ~2/3 of the file; best.pt is what we keep
    with open(os.path.join(run_dir, 'TRAINED'), 'w') as f:
        f.write(f'best val IoU@0.5 {best_iou:.6f}\n')
    return True
