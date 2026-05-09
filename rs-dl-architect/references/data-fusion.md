# Data Fusion (Multi-Modal / Multi-Source)

Covers fusion of heterogeneous data sources: SAR + optical, optical + DEM, hyperspectral + LiDAR, RGB + thermal, and similar combinations. Read this when the input contains physically distinct modalities that can't simply be channel-stacked.

## Why fusion is its own problem

Fusion isn't "just concatenate channels". The modalities have different:

- **Statistics**: SAR is multiplicative speckle noise, optical is additive. Mixing them at input level breaks normalization assumptions.
- **Resolutions**: Sentinel-1 (10 m) vs Sentinel-2 (10/20/60 m) vs PlanetScope (3 m). Need resampling.
- **Geometric viewing**: SAR is side-looking, optical is near-nadir. Same pixel sees different physics.
- **Information content**: SAR penetrates clouds, optical doesn't. They're complementary, not redundant.

Treating fusion as a channel-stack works on small benchmarks and breaks on real-world data. The architecture's job is to respect these differences.

## Fusion stages

There are three classical stages. Most modern architectures use a hybrid.

### Early fusion (input-level)

Stack all bands as channels into a single encoder. Simplest. Works only when modalities are similar enough (e.g., RGB + NIR, or different optical sensors). Fails when statistics differ wildly.

When to use: optical + optical, or optical + thermal at same resolution.

### Late fusion (decision-level)

Independent encoders per modality, fuse only at the prediction head (concat, sum, or learned weights of logits/embeddings). Most robust to modality dropout (one stream missing at inference). Loses cross-modal feature interactions in the encoder.

When to use: when modalities are radically different (SAR + hyperspectral) or when you need to handle missing modalities at inference.

### Mid / feature-level fusion

Independent encoders, fuse at intermediate feature maps (multiple skip levels in U-Net-style, or cross-attention between encoder branches).

When to use: most cases. This is the modern default. Variants:

- **Concatenation fusion**: simplest, more parameters. Strong baseline.
- **Summation / weighted sum**: fewer parameters. Works when both streams are pre-aligned in feature space.
- **Cross-attention fusion**: one modality attends to the other. Captures asymmetric relationships (e.g., SAR provides structure, optical refines it).
- **Gated fusion**: learnable gate controls how much each modality contributes per spatial location. Helps when one modality is unreliable in some regions (e.g., optical under clouds).
- **MMTM (Multimodal Transfer Module)**: bidirectional gating between streams.
- **TokenFusion**: in transformers, replace less informative tokens of one modality with the other's tokens.

## Architectures worth knowing

### Two-stream encoders

- **DeepLabV3+ dual-encoder**: standard segmentation backbone with two parallel encoders, fuse at the ASPP output.
- **MS-RNN, FuseNet** (older): canonical RGB+depth and RGB+thermal references, principles still apply.

### Transformer fusion

- **CMX** (Liu et al. 2023): cross-modal feature rectification + cross-attention fusion. Strong on RGB+thermal and RGB+depth, transfers cleanly to remote sensing.
- **MultiMAE**: masked autoencoder pretrained jointly on multiple modalities. The pretraining strategy is more interesting than the fusion module — see `foundation-models.md`.

### Remote-sensing specific

- **DOFA** (Dynamic One-For-All): sensor-agnostic foundation-model encoder, handles arbitrary band combinations. Worth considering when the modality mix changes per task.
- **DeCUR**: decoupled common-unique representation learning for SAR+optical. Useful when optical is sometimes missing (cloud cover).
- **CrossViT-style**: dual-branch transformer with cross-attention; clean architecture for SAR+optical.

## Modality dropout and robustness

Real deployments often miss modalities (cloud-covered optical, SAR sensor outage, missing DEM tile). The architecture should not silently fail.

Patterns:

- **Train with random modality dropout**: zero out one stream with probability ~0.3 during training. Forces each branch to be useful alone.
- **Modality tokens**: include a learnable token per modality; missing modality drops its token instead of feeding zeros.
- **Foundation-model approach**: pretrain on aligned modalities, then the model has learned to interpolate missing ones from context.

If the user is fusing modalities without modality dropout in training, that's a robustness issue worth flagging.

## Resolution alignment

Different sensors → different GSD. Three options:

1. **Upsample low-res to high-res** before encoder. Simple, but introduces interpolation artifacts that the model may learn as features.
2. **Independent encoders at native resolution, align in feature space** (e.g., adaptive pooling or learned upsampling at fusion point). More principled.
3. **Use foundation model that handles native multi-resolution** (Prithvi, DOFA).

For SAR (10 m) + Sentinel-2 (10/20/60 m), most groups upsample 20 m and 60 m bands to 10 m. This is fine but worth being explicit about.

## Specific modality pairs — what works

### SAR + Optical

- Best fusion strategy: mid-level cross-attention or gated fusion. Late fusion underperforms because complementarity is in the features.
- Pretraining: SAR-specific or joint-modality MAE (DeCUR, DOFA).
- Common task: cloud-resilient land cover, flood mapping, biomass estimation.
- Watch for: SAR speckle filtering before encoder is debated — recent work suggests letting the model see raw speckle works fine.

### Optical + DEM (or DSM)

- DEM provides height/slope; usually fused as additional channels (early fusion) or as a single-channel stream.
- For building extraction, DSM is enormously informative — late or mid fusion both work.
- For land use, DEM is often weakly informative — early fusion is fine.

### Hyperspectral + LiDAR

- Hyperspectral provides material composition, LiDAR provides 3D structure. Highly complementary.
- Spectral attention for hyperspectral (handle 100+ bands) + spatial encoder for LiDAR-derived features.
- Common architectures: dual-branch CNN with attention fusion (e.g., FusAtNet, EndNet).

### RGB + Thermal

- Thermal is a registration challenge (resolution, parallax). Mid-level fusion with cross-attention (CMX) is the modern default.
- Common task: wildlife detection, search and rescue, thermal anomaly mapping.

## Common review findings

- Concatenating SAR linear-scale and optical reflectance directly without normalization → models learn the scale difference, not features.
- No modality-dropout training → model fails when a modality is missing at inference, but this isn't tested.
- Comparing fusion model to single-modality baseline using only the average case → fusion may help on hard cases (clouds) but hurt on easy ones; report stratified results.
- Ignoring spatial misalignment between modalities → especially for SAR+optical, sub-pixel registration matters.
- Fusion at every level when only one or two levels add value → unnecessary parameters; ablate fusion levels.
