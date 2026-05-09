# Object Detection (Including Instance-Level Understanding)

Covers object detection in aerial and satellite imagery, including oriented (rotated) bounding boxes, small-object detection, and DETR-family methods. Read this when the task is sparse object localization rather than dense per-pixel labeling.

## What's different about RS object detection

Detection in remote sensing differs from natural images in three ways that drive architectural choice:

1. **Objects are small and dense.** Ships, vehicles, and small buildings often span 10–30 pixels. Anchor-based methods designed for ImageNet-scale objects underperform.
2. **Objects have arbitrary orientation.** A ship at 30° vs a ship at 90° is the same class but the axis-aligned bounding box differs wildly. Oriented bboxes (OBBs) are the right output.
3. **Backgrounds are vast and varied.** A 10k×10k aerial image has overwhelming negative-to-positive ratio. Hard negative mining matters more than in COCO.

## Detector families

### Two-stage (R-CNN family)

- **Faster R-CNN**: still a strong baseline; well-supported, slow.
- **Cascade R-CNN**: progressively refines proposals, better at high IoU thresholds. Worth using when localization precision matters.
- **Mask R-CNN, Cascade Mask R-CNN**: when instance masks are needed.

### One-stage anchor-based

- **RetinaNet**: focal loss handles class imbalance well. Strong default for RS, especially with small objects.
- **YOLOv5 / v8 / v9 / v10**: fast, well-tuned. RS forks (YOLO-OBB) add oriented-bbox support.

### Anchor-free

- **FCOS**: per-pixel detector, simpler than anchor-based, strong baseline.
- **CenterNet**: detects object centers, regresses size. Very fast, slightly worse on small dense objects.

### Query-based (DETR family)

- **DETR, Deformable DETR**: transformer-based, set prediction with bipartite matching. Slow to converge, but strong on cluttered scenes.
- **DINO** (the detection one, not the SSL one): convergence-fixed DETR with denoising training. SOTA on COCO, transfers well to RS.
- **Co-DETR**: collaborative training for faster convergence.
- **RT-DETR, RT-DETRv2**: real-time DETR variants. Use when latency matters.

### Oriented-bbox specific

- **R3Det, S2A-Net**: refined feature alignment for OBB.
- **Oriented R-CNN**: two-stage OBB detector.
- **ReDet**: rotation-equivariant feature extraction.
- **YOLOv8-OBB / YOLOv9-OBB**: practical, well-supported.
- **DOTA-style benchmarks**: standard testbed.

## Small-object detection patterns

When most objects are <30 pixels:

- **Use higher-resolution feature maps**: keep skip connections from early layers; use FPN with P2/P3 prominent.
- **Smaller anchors / smaller stride**: default COCO anchors miss tiny objects.
- **Tile inference with overlap**: don't downsample large aerial images; tile them with 200–400 px overlap and stitch detections via NMS.
- **Copy-paste augmentation**: paste small objects into other tiles to balance the sample distribution.
- **High-resolution backbone**: HRNet or transformer with smaller patch sizes (8 vs 16).
- **Avoid heavy downsampling**: 32× stride loses small-object signal; cap at 16×.

## Oriented bbox: representations

Multiple parameterizations exist, with different gradient behavior:

| Representation                | Form                     | Notes                                               |
| ----------------------------- | ------------------------ | --------------------------------------------------- |
| 5-parameter (cx, cy, w, h, θ) | angle in radians         | Standard; suffers angle discontinuity at boundaries |
| 8-parameter (4 corners)       | (x1,y1,...,x4,y4)        | No angle ambiguity; harder to regress               |
| Gaussian (GWD, KLD)           | 2D Gaussian distribution | Smooth loss, no angle discontinuity, SOTA on DOTA   |
| Mid-line / sliding vertex     | various                  | Used in CSL, R3Det                                  |

**Practical advice**: Gaussian-based losses (GWD, KLD, KFIoU) avoid the angle-discontinuity problem and are the modern default for OBB regression.

## DETR-family for remote sensing

DETR works well for:

- Cluttered scenes with many overlapping objects.
- Multi-class scenarios with long-tail distributions.
- When you want to skip NMS.

DETR struggles with:

- Very small objects (DETR's slow convergence makes small-object features hard to learn). Deformable DETR mitigates this.
- Real-time inference (use RT-DETR variants).
- Limited data (DETR is data-hungry; pretrain or use Co-DETR).

For aerial imagery, **Deformable DETR + DINO denoising + multi-scale features** is currently the strongest non-RS-specific recipe.

## Foundation models for detection

- **GroundingDINO** (Liu et al. 2023): open-vocabulary detection, zero-shot via text. Adapt to RS for "find all storage tanks" style queries.
- **OWL-ViT, OWLv2**: open-vocabulary object detection.
- **SAM as a detection helper**: segment-everything output → bounding boxes. Useful for class-agnostic detection.

For RS-specific detection foundation models, see `foundation-models.md`.

## Class imbalance in detection

Background dominates. Handle it via:

- **Focal loss** (RetinaNet's contribution): standard.
- **OHEM (Online Hard Example Mining)**: focus training on misclassified hard negatives.
- **Sampling**: ensure each batch has positive examples (especially for rare classes).
- **Long-tail loss functions** (Equalized Focal Loss, Seesaw Loss): for many-class RS detection.

## Datasets

- **DOTA / DOTAv2** — oriented-bbox aerial detection, 18 classes. Standard OBB benchmark.
- **DIOR / DIOR-R** — large-scale aerial detection, 20 classes.
- **HRSC2016** — ship detection with oriented bboxes.
- **xView** — 60-class aerial with very small objects (some 10 px).
- **VisDrone** — drone-captured, dense small objects.
- **iSAID** — instance segmentation in aerial; can be reduced to detection.
- **FAIR1M** — fine-grained aerial detection.

## Common review findings

- Axis-aligned bbox for inherently oriented objects (ships, vehicles) → 30%+ wasted recall on densely packed scenes.
- COCO-tuned anchors used for aerial → small objects missed entirely.
- Image downsampled to 1024×1024 instead of tiling → small objects lost.
- No NMS strategy across tile boundaries → duplicate detections.
- mAP at COCO IoU thresholds (0.5:0.95) reported without IoU=0.5 specifically → small-object recall hidden.
- DETR used with <5k training images and no pretraining → undertrained.
- Reported mAP on DOTA-v1 only when DOTA-v2 exists → benchmarks may be saturated.
