# Code Templates (PyTorch + TensorFlow)

Starter patterns for the most common architectures. These are templates to adapt, not finished products. Each one is annotated with shape comments because RS shape bugs are silent and expensive.

The user works in both PyTorch and TensorFlow — default to PyTorch and offer TF on request, or follow whatever framework the user's existing code uses.

## Multispectral input adapter (PyTorch)

Adapt an ImageNet-pretrained encoder to N-channel multispectral input.

```python
import torch
import torch.nn as nn
import torchvision.models as tvm

def adapt_first_conv(model: nn.Module, in_channels: int, conv_attr: str = "conv1"):
    """Replace the first conv to accept `in_channels` while preserving pretrained weights.

    Strategy: average the RGB-trained weights across the channel dim, replicate to N channels.
    This preserves the spatial filters learned on ImageNet.
    """
    old_conv = getattr(model, conv_attr)
    new_conv = nn.Conv2d(
        in_channels=in_channels,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=old_conv.bias is not None,
    )
    with torch.no_grad():
        # (out, 3, k, k) -> (out, 1, k, k) mean -> repeat to (out, in_channels, k, k)
        avg_weight = old_conv.weight.mean(dim=1, keepdim=True)
        new_conv.weight.copy_(avg_weight.repeat(1, in_channels, 1, 1) / in_channels)
        if old_conv.bias is not None:
            new_conv.bias.copy_(old_conv.bias)
    setattr(model, conv_attr, new_conv)
    return model

# Usage: ResNet-50 for 13-band Sentinel-2
backbone = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2)
backbone = adapt_first_conv(backbone, in_channels=13)
```

## Minimal U-Net (PyTorch)

A clean U-Net implementation; copy and modify.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
            nn.GroupNorm(min(32, out_c), out_c),  # GN > BN for small batches
            nn.GELU(),
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            nn.GroupNorm(min(32, out_c), out_c),
            nn.GELU(),
        )
    def forward(self, x):
        return self.block(x)

class UNet(nn.Module):
    def __init__(self, in_channels=13, num_classes=10, base=64):
        super().__init__()
        self.enc1 = DoubleConv(in_channels, base)              # (B, 64, H,   W)
        self.enc2 = DoubleConv(base, base * 2)                 # (B,128, H/2, W/2)
        self.enc3 = DoubleConv(base * 2, base * 4)             # (B,256, H/4, W/4)
        self.enc4 = DoubleConv(base * 4, base * 8)             # (B,512, H/8, W/8)
        self.bottleneck = DoubleConv(base * 8, base * 16)      # (B,1024,H/16,W/16)

        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2)
        self.dec4 = DoubleConv(base * 16, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = DoubleConv(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = DoubleConv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = DoubleConv(base * 2, base)

        self.head = nn.Conv2d(base, num_classes, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        # x: (B, in_channels, H, W); H and W must be divisible by 16
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b  = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)  # (B, num_classes, H, W)
```

## Siamese change-detection (PyTorch)

```python
class FCSiamDiff(nn.Module):
    """FC-Siam-diff: shared encoder, absolute-difference fusion at each skip level."""
    def __init__(self, in_channels=4, num_classes=2, base=32):
        super().__init__()
        # Shared encoder (weights tied across the two date streams)
        self.enc1 = DoubleConv(in_channels, base)
        self.enc2 = DoubleConv(base, base * 2)
        self.enc3 = DoubleConv(base * 2, base * 4)
        self.enc4 = DoubleConv(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2)
        # Decoder takes |t1 - t2| at each level
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = DoubleConv(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = DoubleConv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = DoubleConv(base * 2, base)
        self.head = nn.Conv2d(base, num_classes, 1)

    def encode(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        return e1, e2, e3, e4

    def forward(self, x1, x2):
        a1, a2, a3, a4 = self.encode(x1)
        b1, b2, b3, b4 = self.encode(x2)
        d1 = (a1 - b1).abs()
        d2 = (a2 - b2).abs()
        d3 = (a3 - b3).abs()
        d4 = (a4 - b4).abs()
        u3 = self.dec3(torch.cat([self.up3(d4), d3], dim=1))
        u2 = self.dec2(torch.cat([self.up2(u3), d2], dim=1))
        u1 = self.dec1(torch.cat([self.up1(u2), d1], dim=1))
        return self.head(u1)
```

## Combined Dice + CE loss (PyTorch)

```python
import torch.nn.functional as F

class DiceCELoss(nn.Module):
    """Robust default for imbalanced semantic segmentation."""
    def __init__(self, num_classes, class_weights=None, ignore_index=-100, dice_weight=0.5):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.dice_weight = dice_weight
        self.register_buffer(
            "class_weights",
            torch.tensor(class_weights) if class_weights is not None else None
        )

    def forward(self, logits, targets):
        # logits: (B, C, H, W); targets: (B, H, W)
        ce = F.cross_entropy(logits, targets, weight=self.class_weights,
                             ignore_index=self.ignore_index)
        # Dice computed only on valid pixels
        valid = (targets != self.ignore_index)
        targets_clipped = targets.clone()
        targets_clipped[~valid] = 0
        onehot = F.one_hot(targets_clipped, self.num_classes).permute(0, 3, 1, 2).float()
        probs = F.softmax(logits, dim=1)
        valid_f = valid.unsqueeze(1).float()
        intersection = (probs * onehot * valid_f).sum(dim=(0, 2, 3))
        cardinality = (probs + onehot).mul(valid_f).sum(dim=(0, 2, 3))
        dice = 1 - (2 * intersection + 1e-6) / (cardinality + 1e-6)
        dice_loss = dice.mean()
        return (1 - self.dice_weight) * ce + self.dice_weight * dice_loss
```

## Training loop pattern (PyTorch)

A spine that handles the common landmines: mixed precision, gradient accumulation, proper scheduler stepping, EMA hook point.

```python
import torch
from torch.cuda.amp import autocast, GradScaler

def train_one_epoch(model, loader, optimizer, scheduler, loss_fn, device,
                    accum_steps=1, scaler=None, ema=None):
    model.train()
    optimizer.zero_grad()
    running = 0.0
    for step, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        with autocast(enabled=scaler is not None):
            logits = model(x)
            loss = loss_fn(logits, y) / accum_steps
        if scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        if (step + 1) % accum_steps == 0:
            if scaler:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            optimizer.zero_grad()
            if ema:
                ema.update(model)
        running += loss.item() * accum_steps
    # Step the scheduler at epoch boundary by default (most schedulers expect this)
    scheduler.step()
    return running / len(loader)
```

**Common LR scheduler bug**: `CosineAnnealingLR(T_max=N)` where `N` accidentally equals batch count instead of epoch count, or vice versa. Verify by printing the LR at the start of each epoch.

## Tile-based inference with overlap blending (PyTorch)

```python
def tiled_inference(model, image, tile_size=512, overlap=128, num_classes=10, device="cuda"):
    """
    image: (C, H, W) tensor
    Returns: (num_classes, H, W) logits, accumulated with cosine-weighted overlap.
    """
    C, H, W = image.shape
    stride = tile_size - overlap
    # Cosine window for blending (avoids edge artifacts)
    win_1d = torch.cos(torch.linspace(-torch.pi / 2, torch.pi / 2, tile_size)).clamp(min=0)
    weight = (win_1d[:, None] * win_1d[None, :]).to(device)  # (tile_size, tile_size)

    out = torch.zeros(num_classes, H, W, device=device)
    norm = torch.zeros(1, H, W, device=device)

    model.eval()
    with torch.no_grad():
        for y0 in range(0, max(1, H - tile_size + 1), stride):
            for x0 in range(0, max(1, W - tile_size + 1), stride):
                y1 = min(y0 + tile_size, H)
                x1 = min(x0 + tile_size, W)
                tile = image[:, y0:y1, x0:x1].unsqueeze(0).to(device)
                # Pad if at the edge
                pad_h = tile_size - (y1 - y0)
                pad_w = tile_size - (x1 - x0)
                if pad_h or pad_w:
                    tile = F.pad(tile, (0, pad_w, 0, pad_h), mode="reflect")
                logits = model(tile)[0]  # (num_classes, tile_size, tile_size)
                logits = logits[:, : y1 - y0, : x1 - x0]
                w = weight[: y1 - y0, : x1 - x0]
                out[:, y0:y1, x0:x1] += logits * w
                norm[:, y0:y1, x0:x1] += w
    return out / norm.clamp(min=1e-6)
```

## TensorFlow / Keras: U-Net skeleton

```python
import tensorflow as tf
from tensorflow.keras import layers, models

def double_conv(x, c):
    x = layers.Conv2D(c, 3, padding="same", use_bias=False)(x)
    x = layers.GroupNormalization(groups=min(32, c))(x)
    x = layers.Activation("gelu")(x)
    x = layers.Conv2D(c, 3, padding="same", use_bias=False)(x)
    x = layers.GroupNormalization(groups=min(32, c))(x)
    x = layers.Activation("gelu")(x)
    return x

def build_unet(input_shape=(256, 256, 13), num_classes=10, base=64):
    inputs = layers.Input(shape=input_shape)
    e1 = double_conv(inputs, base)
    e2 = double_conv(layers.MaxPool2D()(e1), base * 2)
    e3 = double_conv(layers.MaxPool2D()(e2), base * 4)
    e4 = double_conv(layers.MaxPool2D()(e3), base * 8)
    b  = double_conv(layers.MaxPool2D()(e4), base * 16)

    u4 = layers.Conv2DTranspose(base * 8, 2, strides=2)(b)
    d4 = double_conv(layers.Concatenate()([u4, e4]), base * 8)
    u3 = layers.Conv2DTranspose(base * 4, 2, strides=2)(d4)
    d3 = double_conv(layers.Concatenate()([u3, e3]), base * 4)
    u2 = layers.Conv2DTranspose(base * 2, 2, strides=2)(d3)
    d2 = double_conv(layers.Concatenate()([u2, e2]), base * 2)
    u1 = layers.Conv2DTranspose(base, 2, strides=2)(d2)
    d1 = double_conv(layers.Concatenate()([u1, e1]), base)

    outputs = layers.Conv2D(num_classes, 1)(d1)
    return models.Model(inputs, outputs)
```

## Deep ensemble training pattern

```python
# Train M models with different seeds, save each.
for seed in [0, 1, 2, 3, 4]:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = build_model()
    train(model, ...)
    torch.save(model.state_dict(), f"ckpt_seed{seed}.pt")

# Inference: average softmax probabilities (NOT logits — calibration matters).
def ensemble_predict(models, x):
    probs = [F.softmax(m(x), dim=1) for m in models]
    return torch.stack(probs).mean(0)
```

## When to bundle a script vs paste inline

- If the user is iterating, paste the relevant snippet inline so they can edit it.
- If the user wants a full project skeleton, structure as `model.py`, `losses.py`, `data.py`, `train.py`, `infer.py` and present them as files.
- Always include shape annotations on tensor operations. Always.
