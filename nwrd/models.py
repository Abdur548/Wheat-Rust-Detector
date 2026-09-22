"""Model definitions: CANet (EfficientNet-B4) and the DeepLabV3+ baselines.

Module names inside CANet match the checkpoints trained by the sprint-2 notebook, so
existing `best_model.pth` files load unchanged.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

MODELS = {
    'canet_b4': 'CANet, EfficientNet-B4 encoder (ours)',
    'deeplabv3plus_r50': 'DeepLabV3+, ResNet-50 encoder (baseline)',
    'deeplabv3plus_b4': 'DeepLabV3+, EfficientNet-B4 encoder (same-encoder ablation)',
    'unet': 'UNet, 4 encoder / 4 decoder blocks, from scratch (NWRD paper baseline)',
    'unet_tuned': 'UNet (paper architecture) with a from-scratch learning rate (config.unet_tuned_lr)',
    # Benchmark only: CANet as first implemented, running the encoder twice per forward.
    'canet_b4_twopass': 'CANet, original two-pass forward',
}


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, p=1, d=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, padding=p, dilation=d, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.block(x)


class ContextFlow(nn.Module):
    """One CAM information flow: a shallow encoder-decoder at a given downsample scale."""
    def __init__(self, in_ch, out_ch, scale):
        super().__init__()
        self.scale = scale
        self.encode = ConvBNReLU(in_ch, out_ch)
        self.decode = ConvBNReLU(out_ch, out_ch)

    def forward(self, x):
        h, w = x.shape[-2:]
        xd = F.avg_pool2d(x, self.scale) if self.scale > 1 else x
        f = self.decode(self.encode(xd))
        return F.interpolate(f, (h, w), mode='bilinear', align_corners=False) if self.scale > 1 else f


class AttentionFusion(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.att = nn.Sequential(
            nn.Conv2d(in_ch, in_ch // 4, 1), nn.ReLU(inplace=True),
            nn.Conv2d(in_ch // 4, in_ch, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.att(x)


class CAM(nn.Module):
    """Chained context aggregation: a serial dilated global flow plus parallel context
    flows at scales 2/4/8, pre-fused by concatenation and re-fused by channel attention."""
    def __init__(self, in_ch, out_ch=256):
        super().__init__()
        self.global_flow = nn.Sequential(
            ConvBNReLU(in_ch, out_ch, k=3, p=2, d=2),
            ConvBNReLU(out_ch, out_ch, k=3, p=4, d=4),
            ConvBNReLU(out_ch, out_ch, k=3, p=8, d=8))
        self.context_flows = nn.ModuleList([
            ContextFlow(in_ch, out_ch, 2),
            ContextFlow(in_ch, out_ch, 4),
            ContextFlow(in_ch, out_ch, 8)])
        self.pre_fusion = ConvBNReLU(out_ch * 4, out_ch, k=1, p=0)
        self.re_fusion = AttentionFusion(out_ch)
        self.out_conv = ConvBNReLU(out_ch, out_ch)

    def forward(self, x):
        gf = self.global_flow(x)
        cfs = [cf(x) for cf in self.context_flows]
        fused = self.pre_fusion(torch.cat([gf] + cfs, dim=1))
        return self.out_conv(self.re_fusion(fused))


class AsymmetricDecoder(nn.Module):
    """Fuses upsampled 1/32 context with reduced 1/4 low-level features."""
    def __init__(self, high_ch, low_ch, out_ch=128):
        super().__init__()
        self.low_reduce = ConvBNReLU(low_ch, 48, k=1, p=0)
        self.fuse = nn.Sequential(
            ConvBNReLU(high_ch + 48, out_ch), ConvBNReLU(out_ch, out_ch))

    def forward(self, high, low):
        high_up = F.interpolate(high, low.shape[-2:], mode='bilinear', align_corners=False)
        return self.fuse(torch.cat([high_up, self.low_reduce(low)], dim=1))


class CANet(nn.Module):
    def __init__(self, dropout=0.3, pretrained=True, single_pass=True):
        super().__init__()
        from efficientnet_pytorch import EfficientNet
        self.encoder = (EfficientNet.from_pretrained('efficientnet-b4') if pretrained
                        else EfficientNet.from_name('efficientnet-b4'))
        self.single_pass = single_pass
        self.reduce = ConvBNReLU(1792, 512, k=1, p=0)
        self.cam = CAM(512, 256)
        self.decoder = AsymmetricDecoder(256, 32, 128)   # reduction_2: 32ch at 1/4
        self.dropout = nn.Dropout2d(dropout)
        self.head = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1))

    def forward(self, x):
        if self.single_pass:
            # reduction_6 is exactly extract_features(x); one encoder pass yields both.
            ep = self.encoder.extract_endpoints(x)
            enc, low = ep['reduction_6'], ep['reduction_2']
        else:
            enc = self.encoder.extract_features(x)
            low = self.encoder.extract_endpoints(x)['reduction_2']
        x_cam = self.cam(self.reduce(enc))
        x_dec = self.dropout(self.decoder(x_cam, low))
        return F.interpolate(self.head(x_dec), scale_factor=4, mode='bilinear', align_corners=False)


class _DoubleConv(nn.Sequential):
    def __init__(self, in_ch, out_ch):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))


class UNet(nn.Module):
    """Plain UNet as used by the NWRD paper (Anwar et al., Sensors 2023): 4 encoder and 4
    decoder blocks, 64-1024 channels, transposed-conv upsampling, no pretraining."""
    has_pretrained_encoder = False

    def __init__(self, base=64):
        super().__init__()
        ch = [base * 2 ** i for i in range(5)]
        self.inc = _DoubleConv(3, ch[0])
        self.encoder = nn.ModuleList(_DoubleConv(ch[i], ch[i + 1]) for i in range(4))
        self.up = nn.ModuleList(nn.ConvTranspose2d(ch[i + 1], ch[i], 2, stride=2) for i in reversed(range(4)))
        self.decoder = nn.ModuleList(_DoubleConv(ch[i] * 2, ch[i]) for i in reversed(range(4)))
        self.head = nn.Conv2d(ch[0], 1, 1)

    def forward(self, x):
        skips = [self.inc(x)]
        for down in self.encoder:
            skips.append(down(F.max_pool2d(skips[-1], 2)))
        x = skips.pop()
        for up, dec in zip(self.up, self.decoder):
            x = dec(torch.cat([skips.pop(), up(x)], 1))
        return self.head(x)


def build_model(name, pretrained=True, dropout=0.3):
    """All models return raw logits of shape (B, 1, H, W) and expose `.encoder`."""
    if name == 'canet_b4':
        return CANet(dropout=dropout, pretrained=pretrained)
    if name == 'canet_b4_twopass':
        return CANet(dropout=dropout, pretrained=pretrained, single_pass=False)
    if name in ('unet', 'unet_tuned'):
        return UNet()
    if name in ('deeplabv3plus_r50', 'deeplabv3plus_b4'):
        import segmentation_models_pytorch as smp
        encoder = 'resnet50' if name.endswith('r50') else 'efficientnet-b4'
        return smp.DeepLabV3Plus(encoder_name=encoder,
                                 encoder_weights='imagenet' if pretrained else None,
                                 classes=1)
    raise ValueError(f'unknown model {name!r}; choose from {sorted(MODELS)}')


def load_checkpoint(model, path, map_location='cpu'):
    """Loads either a pipeline checkpoint or a sprint-2 notebook checkpoint ({'model': sd})."""
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt)
    return ckpt
