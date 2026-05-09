# Temporal Analysis (Time Series of Satellite Images)

Covers models that process a stack of dates: ConvLSTM, U-TAE, TempCNN, transformer-based temporal models. Read this when the input is a sequence of observations of the same area.

## Task taxonomy

| Sub-task                        | Output                             | Examples                                |
| ------------------------------- | ---------------------------------- | --------------------------------------- |
| **Per-date classification**     | Label per timestep                 | Crop type, flood/no-flood per date      |
| **Sequence classification**     | Single label for whole sequence    | Crop type from full season              |
| **Sequence segmentation**       | Per-pixel label, possibly per-date | Crop maps, deforestation alerts         |
| **Forecasting**                 | Future timesteps                   | Yield prediction, phenology forecasting |
| **Continuous change detection** | Change events along the sequence   | Disturbance detection, BFAST-style      |

Architecture choice depends most on whether the temporal axis is regular (daily, every-5-days), irregular (cloud-induced gaps), and whether positional encoding of dates is informative.

## Architecture families

### Recurrent (ConvLSTM, ConvGRU)

The original deep approach. Process the sequence one date at a time, maintain a spatial-recurrent hidden state.

- Strength: handles arbitrary sequence lengths, maintains spatial structure.
- Weakness: slow training, no parallelism over time, struggles with long sequences.
- Modern role: still a reasonable baseline, but rarely SOTA. Worth using when the sequence is short (≤10) and the task is conceptually sequential (e.g., flood evolution).

### 3D convolutions

Treat time as a third spatial axis.

- **TempCNN** (Pelletier et al. 2019): 1D conv over time, applied per-pixel. Fast, surprisingly strong on crop classification.
- **Spatio-temporal 3D CNN**: 3D kernels over (T, H, W). Heavy, but useful when local spatio-temporal patterns matter (e.g., short-window phenology).

### Transformer-based temporal

- **U-TAE** (Garnot & Landrieu 2021): U-Net spatial encoder + temporal attention over the date axis at each spatial location. Strong default for crop classification on Sentinel-2 stacks.
- **TSViT** (Tarasiou et al. 2023): tokenize spatio-temporal cubes, factorized attention. SOTA on PASTIS.
- **Presto** (Tseng et al. 2023): pixel time series + transformer; pretrained, designed for label-efficient transfer.
- **SITS-Former, ALISE**: full transformer encoders over irregular satellite image time series, with date positional encoding.

### Hybrid / state-space

- **PSE+L-TAE**: pixel-set encoder + lightweight temporal attention (efficient, designed for parcel-level classification).
- **Mamba-based temporal**: emerging; ChangeMamba and related extensions.

### Foundation-model temporal

- **Prithvi-EO** has temporal variants pretrained on multi-temporal Sentinel-2 / HLS.
- **Presto** is itself a foundation model for pixel time series.
- For most temporal tasks with limited labels, fine-tune a temporal foundation model first; train from scratch only when nothing fits.

## Design patterns

**Date positional encoding**: when sequences are irregular (clouds skip dates), feed the **actual date** (day-of-year or absolute) as positional encoding, not the index. The model needs to know that a 3-week gap is different from a 1-day gap.

```python
# Day-of-year encoding (one common pattern)
doy = torch.tensor([d.timetuple().tm_yday for d in dates])  # (T,)
pe = sinusoidal(doy / 365.0)  # (T, D)
```

**Cloud masking and gaps**: two strategies:

1. **Mask-aware attention**: pass a binary mask of valid pixels per date; attention ignores masked positions. Cleanest.
2. **Interpolation pre-processing**: linear or harmonic interpolation to a regular grid. Loses information but simplifies the model.

For short sequences (<20 dates), masking is preferable. For very long irregular sequences, interpolation may be necessary for compute reasons.

**Temporal length tradeoffs**: more dates = more compute but typically better accuracy until diminishing returns. For crop classification, full-season ~30 Sentinel-2 dates is the sweet spot. For change detection, 2 well-chosen dates often beat 10 noisy dates.

**Per-pixel vs per-tile temporal modeling**:

- **Per-pixel** (TempCNN, Presto, PSE+L-TAE): treat each pixel as an independent time series. Fast, label-efficient, ignores spatial context. Strong for parcel-level crop ID.
- **Per-tile** (U-TAE, TSViT): preserve spatial structure across time. Needed for spatial outputs (segmentation), or when the task depends on spatial context evolving (urban growth).

## Specific tasks — what works

### Crop type classification

- Default: U-TAE for spatial output, PSE+L-TAE for parcel-level.
- Foundation model: try Presto first if you have <10k labeled parcels.
- Bands: red-edge (Sentinel-2 B5, B6, B7) and SWIR (B11, B12) are most discriminative. Don't drop them.
- Sequence: full growing season, 5–10 day cadence after cloud removal.

### Deforestation / disturbance alerts

- Default: continuous change detection on dense time series. CCDC-style baselines are competitive.
- Deep alternative: temporal transformer trained to predict per-date disturbance probability.
- Watch for: false positives from clouds, shadows, and seasonal vegetation. Class imbalance is extreme (>99% no-disturbance).

### Burned area mapping

- Often bi-temporal (pre/post fire) → see `change-detection.md`.
- Multi-temporal helps for fire progression.

### Flood detection

- SAR is preferred (cloud penetration). Time series on SAR can use simpler models (TempCNN suffices).
- Often combined with SAR+optical fusion → see `data-fusion.md`.

## Datasets

- **PASTIS / PASTIS-R** — French parcel-level crop classification, multi-temporal Sentinel-2 (and SAR for -R). Standard benchmark for U-TAE and successors.
- **Sen12MS / SEN12MS-CR** — multi-modal multi-temporal, 100k+ patches.
- **TimeSen2Crop** — time-series crop classification, 1M parcels.
- **DynamicEarthNet** — daily PlanetScope, 75 cubes, 24 months.
- **MultiEarth** — irregular multi-modal multi-temporal.

## Common review findings

- Sequence model trained on a fixed-length stack, broken when test sequences vary in length → no length-invariance design.
- Date information ignored (just treats sequence as ordered indices) → fragile to gaps.
- No comparison against a strong simple baseline (TempCNN, Random Forest on hand-crafted temporal features) → may not beat them.
- Per-pixel temporal model deployed for spatial-output task (or vice versa) → architecture mismatch.
- Cloud masks ignored at training → model learns to interpret clouds as features.
- Single-region evaluation → temporal models often fail to transfer across climate zones; always test cross-region.
