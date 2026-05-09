# Segmentation

Covers semantic segmentation, instance segmentation, and panoptic segmentation for remote sensing. Read this when the task is per-pixel labeling.

## Decision tree: which family fits

The right family depends on three task properties: **GSD**, **class granularity**, and **whether boundaries matter more than regions**.

```
Is there a strong global-context need (e.g. land-use classes that depend on neighborhood)?
├── YES → Transformer-based: SegFormer, UNetFormer, Mask2Former
└── NO  → Is boundary precision critical (cadastral, building footprints)?
        ├── YES → U-Net++, HRNet, or U-Net with boundary-aware loss
        └── NO  → Plain U-Net or DeepLabV3+ is usually enough
```

Then layer on remote-sensing-specific concerns:

- Multi-temporal? → see `temporal-analysis.md` (U-TAE, etc.)
- Foundation-model fine-tuning available? → see `foundation-models.md` first
- Classes very imbalanced? → architecture matters less than loss/sampling

## Architecture families

### U-Net and descendants

The default for remote sensing because of its label efficiency. Symmetric encoder-decoder with skip connections. Variants worth knowing:

- **U-Net++** (Zhou et al. 2018): nested skip pathways, better gradient flow, more parameters. Use when you have enough data to justify the extra capacity.
- **Attention U-Net**: gated skip connections that suppress irrelevant encoder features. Helps when the foreground is small and surrounded by distractors (e.g., small ponds in farmland).
- **ResU-Net / ResU-Net-a**: residual blocks in the encoder. Standard upgrade.
- **HRNet**: maintains high-resolution representations throughout, instead of recovering from low-res. Strong on tasks needing fine detail (roads, building edges).

### Transformer-based segmentation

- **SegFormer** (Xie et al. 2021): hierarchical transformer encoder + lightweight all-MLP decoder. Excellent accuracy/efficiency tradeoff. Strong default in 2024+ for tasks where U-Net plateaus.
- **UNetFormer** (Wang et al. 2022): explicitly designed for remote sensing. CNN encoder + transformer decoder with global-local attention. Strong on aerial imagery (LoveDA, UAVid).
- **Mask2Former**: unified semantic/instance/panoptic via masked attention. Heavy but state-of-the-art when compute allows.
- **DC-Swin**: Swin Transformer encoder with dense connection decoder, tuned for remote sensing.

### State-space / Mamba-based

- **VMamba, Vim**: linear-complexity alternatives to attention. Promising for very high-resolution tiles where transformers OOM.
- **RS-Mamba** and **Samba**: remote-sensing specific. Worth experimenting with on large-scene segmentation.

### Foundation-model adapted

For most tasks today with <50k labeled tiles, this beats from-scratch training. See `foundation-models.md`.

## Design patterns specific to remote sensing

**Tile size and overlap**: Pick tile size from receptive field, not from "what fits in memory". For 10 m GSD, a 512×512 tile covers 5.12 km — usually enough for context. At 0.3 m GSD aerial, 1024×1024 only covers 307 m, which may be too little for context-dependent classes. Use 25–50% overlap at inference to avoid edge artifacts; blend with cosine or gaussian weights, not arithmetic mean.

**Loss for imbalanced semantic classes**:

- **Weighted cross-entropy**: weights = inverse class frequency (or its square root). Simple and works.
- **Focal loss** (γ=2 default): better when imbalance is extreme.
- **Dice loss**: directly optimizes overlap; pair with CE for stability.
- **Lovász-Softmax**: optimizes IoU directly; helpful when IoU is the eval metric and classes have small support.
- **Boundary loss**: when boundaries dominate the metric (cadastral mapping).

A common robust default: `0.5 * CE_weighted + 0.5 * Dice`.

**Multi-spectral input adapter**: the standard ImageNet 3-channel weights need adapting. Three options:

1. Replicate first-conv weights across N channels with `1/N` scaling, then fine-tune.
2. Train a small N→3 conv and freeze it initially.
3. Initialize first conv with channel-wise mean of RGB weights, then unfreeze.

Don't drop bands silently. SWIR and red-edge are often the most discriminative for vegetation tasks.

## Instance and panoptic segmentation

For remote sensing, instance segmentation matters most for: building extraction (each building separate), tree-crown delineation, and ship detection.

- **Mask R-CNN**: still the workhorse. Stable, well-supported.
- **Mask2Former**: SOTA on benchmarks; one model handles semantic/instance/panoptic.
- **DETR/Mask DINO**: query-based; better for sparse instances and scales well to many classes.
- **Cascade Mask R-CNN**: better for small dense instances (urban buildings).

Special case for remote sensing: **panoptic segmentation** is rarely what you actually want. Most tasks are either pure semantic ("what is this pixel") or pure instance ("how many objects, where"). Asking for panoptic doubles the labeling cost; verify the task really needs it.

## Diagnostics for segmentation

When reviewing a segmentation model:

1. **Confusion matrix per class**, not just mIoU. mIoU hides class collapse.
2. **Boundary IoU** vs **mask IoU**: large gap means the model gets the rough region right but the boundaries wrong.
3. **Region-level test**: train and test on different geographic regions, not random splits. The drop is the real generalization number.
4. **Per-tile IoU distribution**: a few catastrophic tiles can dominate; look at the tail.
5. **Spectral confusion**: which bands does the model rely on? Drop a band at inference; the IoU drop reveals dependence.

## Datasets to know

- **EuroSAT / EuroSAT-MS** — small (27k tiles), 10 classes, Sentinel-2. Classification, but used as pretraining benchmark.
- **LoveDA** — 5,987 high-res aerial images with urban/rural domain shift built in. Great for testing geographic generalization.
- **iSAID** — large-scale instance segmentation in aerial.
- **Potsdam / Vaihingen** — classic high-res aerial with DSM, but small. Treat results with caution.
- **DynamicEarthNet** — daily PlanetScope, 7 classes, 75 cubes. Good for temporal segmentation.
- **OpenEarthMap** — 5,000 high-res tiles, 8 classes, global coverage.

## Common review findings

When reviewing someone else's segmentation model, these come up most:

- Random tile split instead of region split → optimistic numbers.
- ImageNet 3-channel weights with multi-spectral input dropped to RGB → wasted signal.
- No class weighting on imbalanced data → mIoU dominated by majority class.
- Tile size too small for the GSD → not enough context.
- Skipping inference overlap blending → visible seams in mosaics.
