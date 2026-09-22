"""Experiment protocol. Every model is trained with exactly these settings."""
from dataclasses import asdict, dataclass, field


@dataclass
class Config:
    data_root: str = ''
    out_dir: str = 'runs'
    models: list = field(default_factory=lambda: ['canet_b4', 'unet', 'deeplabv3plus_r50'])
    seeds: list = field(default_factory=lambda: [42, 43, 44])
    data_seed: int = 42          # background subsample; fixed so all runs see the same patches

    img_size: int = 512
    bg_keep_ratio: float = 0.20
    epochs: int = 30
    freeze_encoder_epochs: int = 3
    batch_size: int = 8
    num_workers: int = 2
    lr: float = 5e-5
    # 'unet_tuned' only: the shared lr suits fine-tuning pretrained encoders and may
    # under-train a from-scratch UNet, so this variant gets its own.
    unet_tuned_lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    dice_weight: float = 0.7
    bce_weight: float = 0.3
    dropout: float = 0.5         # CANet head dropout; smp baselines have none
    sched_t0: int = 5
    sched_tmult: int = 2
    sched_eta_min: float = 1e-7
    amp: bool = True
    pretrained: bool = True

    # threshold chosen on val over this grid, then frozen and applied to test
    thr_lo: float = 0.05
    thr_hi: float = 0.95
    thr_step: float = 0.01

    time_budget_hours: float = 0.0   # >0: stop cleanly before a platform session limit
    limit: int = 0                   # >0: use only N patches per split (smoke tests)

    def to_dict(self):
        return asdict(self)
