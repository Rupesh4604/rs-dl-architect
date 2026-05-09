# Change Detection

Covers bi-temporal and multi-temporal change detection. Read this when the task involves comparing observations of the same area across time to identify what has changed.

## Task taxonomy first

"Change detection" is overloaded. Identify which sub-task before picking architecture:

| Sub-task                            | Output                           | Typical use                                 |
| ----------------------------------- | -------------------------------- | ------------------------------------------- |
| **Binary change detection (BCD)**   | Per-pixel change/no-change mask  | Damage assessment, urbanization             |
| **Semantic change detection (SCD)** | "From-class" + "to-class" labels | Land-use transition mapping                 |
| **Multi-class change detection**    | Type-of-change label             | Disaster typing (flood vs fire vs collapse) |
| **Object-level change detection**   | Per-object change events         | Building demolition/construction            |
| **Continuous monitoring**           | Per-date label, trend detection  | Crop phenology, deforestation alerts        |

The first three are bi-temporal (two dates). The last two are multi-temporal. Architecture choice diverges sharply.

## Bi-temporal architecture families

### Siamese networks

The default. Two encoders share weights, process the two dates, then a fusion module produces the change map.

- **FC-Siam-conc / FC-Siam-diff** (Daudt et al. 2018): concat or absolute-difference fusion at each skip level. Simple, strong baseline.
- **STANet** (Chen & Shi 2020): siamese + attention modules at multiple scales. Standard upgrade.
- **DTCDSCN**: dual-task siamese (change + semantic), helps when SCD labels exist.
- **SNUNet**: nested siamese + ECA attention, strong on building change.

Decision: when in doubt, start with FC-Siam-diff and see if you actually need more.

### Transformer-based

- **BIT (Bitemporal Image Transformer)** (Chen et al. 2021): tokenizes feature maps, applies transformer to model long-range temporal dependencies. Strong baseline, modest compute.
- **ChangeFormer** (Bandara & Patel 2022): hierarchical transformer in a siamese setup with MLP fusion. Generally better than BIT on standard benchmarks.
- **TransUNetCD, MSCANet**: mix CNN encoders with transformer fusion.

### Mamba-based (newer)

- **ChangeMamba** (Chen et al. 2024): replaces attention with state-space modeling, scales to large scenes more cheaply. Worth considering for high-resolution wide-area tasks.

### Foundation-model-based

- Use a frozen RS foundation model (Prithvi, SatMAE, Clay) as encoder, train only the change-detection head. Often beats from-scratch siamese on small labeled sets.
- **GeoSAM-CD** style: prompt a segmentation foundation model with the difference embedding.

See `foundation-models.md` for adaptation patterns.

## Multi-temporal / continuous

Not a siamese problem. Use temporal modeling:

- **U-TAE** (Garnot & Landrieu 2021): temporal attention encoder, designed for time series of satellite images. Strong default.
- **TempCNN, ConvLSTM**: see `temporal-analysis.md`.
- **Transformer-based temporal stacks**: SITS-Former, etc.
- **CCDC and BFAST** (statistical): not deep learning, but valid baselines for trend-based change. Worth mentioning when the user is over-engineering.

## Design patterns specific to change detection

**Geometric registration is upstream of architecture**: if the two dates aren't sub-pixel registered, the model spends capacity learning to align them, and false positives spike along edges. Verify registration before recommending more complex models. Often the right answer is "fix your alignment, not your model".

**Radiometric / illumination differences**: different sun angle, atmosphere, sensor calibration. Architectures alone can't fix this — pair with histogram matching, relative radiometric normalization, or domain adaptation. BIT and ChangeFormer are more robust to this than CNN siamese, but no architecture is immune.

**Class imbalance is extreme**: change pixels are typically <5% of the image. Required:

- Online hard example mining or focal loss.
- Sampling that ensures each batch has change pixels (not random tiles).
- Eval metrics: F1, IoU on the change class — not pixel accuracy. Pixel accuracy is meaningless here.

**Difference-vs-concat fusion**: for natural changes (vegetation, water), absolute-difference fusion at skip levels works well. For categorical changes (building added vs removed), concat fusion preserves more information. SCD generally needs concat or attention fusion.

**Pseudo-change is the enemy**: registration error, illumination, phenology. A good change-detection model is one that ignores these. Evaluate by including known-no-change pairs from different seasons in your test set.

## Datasets to know

- **LEVIR-CD** — 637 pairs, 1024×1024, building change. The most-cited benchmark; saturated.
- **WHU-CD** — building change, single pair tiled.
- **S2Looking** — Sentinel-2 building change at 0.5 m–0.8 m, harder than LEVIR.
- **xBD** — disaster damage assessment, 4 damage levels.
- **DSIFN-CD** — multi-class change.
- **HRSCD** — semantic change detection.
- **DynamicEarthNet** — daily PlanetScope, multi-temporal continuous.

## Common review findings

When reviewing a change-detection paper or implementation:

- LEVIR-CD-only evaluation → numbers say nothing about generalization. Ask for at least one cross-dataset test.
- No ablation showing the model handles registration error → fragile.
- Reported pixel accuracy → automatic red flag.
- "Difference image + classifier" baseline missing from comparison → claims of beating CNN baseline may be unfair.
- Single seed, no variance reported → in change detection where IoU swings 2–5 points across seeds, this is critical.
- Training and test pairs from the same scene/area → leakage.
