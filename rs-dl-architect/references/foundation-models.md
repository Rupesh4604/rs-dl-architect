# Foundation Models for Remote Sensing

Covers self-supervised and large-scale pretrained models for remote sensing, plus adaptation patterns for downstream tasks. Read this when the user is fine-tuning or adapting a pretrained model — and reflexively when the user has <50k labeled samples for any task. Foundation models are usually the right starting point now, not the last resort.

## The model landscape

Models worth knowing, organized by what they were pretrained on.

### Pretrained on remote-sensing imagery

- **SatMAE** (Cong et al. 2022): MAE on multispectral and multi-temporal Sentinel-2. Variants: SatMAE++, SatMAE-Time. Strong for multispectral fine-tuning.
- **Prithvi-EO** (NASA-IBM 2023; Prithvi-EO 2.0 in 2024): MAE on HLS (Harmonized Landsat-Sentinel) with temporal stacks. 100M and 600M variants. Strong for temporal tasks.
- **Clay** (Clay Foundation 2024): MAE-style foundation model spanning multiple sensors and resolutions. Sensor-aware embeddings.
- **DOFA** (Dynamic One-For-All): handles arbitrary band combinations via dynamic input projection. Sensor-agnostic.
- **CROMA**: contrastive multi-modal MAE on SAR+optical.
- **SoftCon**: contrastive RS pretraining with semantic-aware soft positives.
- **RemoteCLIP** (Liu et al. 2024): CLIP-style image-text alignment on remote sensing. Strong zero-shot classification, good for retrieval.
- **GeoSAM / SAM-RS variants**: SAM adapted to remote sensing for promptable segmentation.

### Pretrained on natural images, transferable to RS

- **DINOv2 / DINOv3** (Meta): self-supervised ViT, very strong dense-feature representations. With proper input adaptation, often competitive with RS-specific models on RGB tasks.
- **SAM / SAM 2** (Meta): promptable segmentation, weak zero-shot on RS but excellent base for prompt-tuned RS-SAM variants.
- **CLIP / SigLIP / OpenCLIP**: image-text foundation models. Useful for retrieval and few-shot classification with text queries.

### Vision-language for RS

- **RemoteCLIP**, **GeoChat**, **EarthGPT**, **RSGPT**, **GeoChat**: VL models trained on RS captions/QA. Good for visual question answering and grounded retrieval.

## Choosing a foundation model

| Task property                      | Pick                                                         |
| ---------------------------------- | ------------------------------------------------------------ |
| Multispectral input, single date   | SatMAE++, Prithvi-EO 2.0 (single-frame), Clay                |
| Multispectral input, time series   | Prithvi-EO 2.0 (temporal), SatMAE-Time, Presto (pixel-level) |
| SAR + optical                      | CROMA, DOFA                                                  |
| RGB only, high-res aerial          | DINOv2 (with first-conv adapt) often beats RS-specific       |
| Promptable segmentation            | SAM 2 → fine-tune to RS, or use GeoSAM                       |
| Zero-shot classification with text | RemoteCLIP, OpenCLIP                                         |
| Sensor-agnostic / mixed inputs     | DOFA, Clay                                                   |

When in doubt, run two quickly: a RS-specific foundation model and DINOv2. The result is often surprising.

## Adaptation patterns

### Linear probing

Freeze the encoder, train only a linear head on encoder features. Use as a baseline to understand what the foundation model already knows. If linear probing gets within 5 points of the SOTA, you don't need fancy fine-tuning.

### Full fine-tuning

Unfreeze everything. Best accuracy when labeled data is sufficient (>10k samples). Risk: catastrophic forgetting of useful features when labeled data is small or noisy.

### Parameter-efficient fine-tuning (PEFT)

The right default for limited data:

- **LoRA** (low-rank adaptation): inject low-rank update matrices into attention layers. ~1% of full-fine-tuning parameters. Default for ViT-based foundation models.
- **Adapter modules**: small bottleneck MLPs after attention/FFN. Slightly more parameters than LoRA, similar accuracy.
- **Prefix / prompt tuning**: prepend learnable tokens. Lightest, can be brittle.
- **BitFit**: train only biases. Surprisingly effective baseline.

For ViT-based RS foundation models, LoRA on attention with rank 8–16 is the typical first try.

### Decoder fine-tuning for dense tasks

For segmentation/detection, freeze encoder, train a task-specific decoder. Two patterns:

1. **UPerNet / SegFormer head**: standard, works for ViT encoders.
2. **Linear probing at multiple depths**: pick features from layers 4, 8, 12, 16, fuse via simple decoder. SegMAE-style.
3. **DPT (Dense Prediction Transformer)** decoder: when you need high-res output from a low-res ViT.

### Prompt-based segmentation (SAM family)

SAM-style models accept point/box/mask prompts. For RS:

- **Auto-prompting**: train a small network to generate prompts from the image, then feed to SAM. RSPrompter does this.
- **Prompt-tuning**: learnable prompt tokens optimized for a downstream task while keeping SAM frozen.
- **LoRA on SAM**: fine-tune SAM with LoRA + task head; SAM-Adapter, GeoSAM.

## Multi-spectral input adaptation

Foundation models pretrained on RGB need adapters for multispectral input. Patterns:

1. **Replicate first-conv weights**: average pretrained RGB conv weights, replicate across N input channels. Simple, surprisingly effective for ViT patch embeddings.
2. **Channel-wise input adapter**: train a small N→3 conv that projects multispectral → RGB-like, freeze the rest of the encoder.
3. **Per-band tokenization** (in ViT): each band gets its own patch embedding, concat in the token sequence. More parameters but preserves spectral information.
4. **Use an RS-pretrained model**: SatMAE, Prithvi, Clay accept multispectral natively.

For Sentinel-2: never drop bands silently. Red-edge (B5, B6, B7) and SWIR (B11, B12) carry critical signal for vegetation and surface composition.

## Evaluation patterns

When evaluating an adapted foundation model:

1. **Compare against from-scratch on the same architecture and budget**. The foundation model should win, especially with limited labels. If it doesn't, the adaptation is broken.
2. **Linear probe baseline** before full fine-tuning. Know what the frozen model already gives you.
3. **Few-shot scaling**: report accuracy at 100, 1k, 10k labels. Foundation models should win the most at 100, less at 10k.
4. **Cross-region transfer**: foundation models often generalize better geographically. Test it.

## Common pitfalls

- **Using DINOv2/CLIP weights with multispectral input via "just pad to 3 channels"**: silently throws away SWIR and red-edge signal.
- **Full fine-tuning on a small labeled set**: catastrophic forgetting. Use PEFT.
- **Comparing fine-tuned foundation model vs from-scratch U-Net at same epoch count**: foundation models may need fewer epochs but different learning rates. Compare at convergence.
- **Linear probing only the last layer**: ViT foundation models often have richer features in middle layers. Probe multi-depth.
- **Forgetting the GSD mismatch**: SatMAE was pretrained at certain GSD; fine-tuning at very different GSD may underperform. Check pretraining GSD.
- **Treating GeoSAM-style models as drop-in replacements for U-Net**: they need prompts or auto-prompting infrastructure; they're not plug-and-play for arbitrary segmentation.

## Reading the user's checkpoint

When asked to analyze someone else's foundation-model adaptation:

1. Identify the base model and pretraining (encoder name + checkpoint).
2. Check input adapter: how are bands handled?
3. Identify which parts are frozen vs trained.
4. Check the head: linear, decoder, prompt-based?
5. Look for LoRA / adapter modules in the state dict (key patterns: `lora_A`, `lora_B`, `adapter`, `prompt_embed`).
6. Map this to one of the adaptation patterns above and explain the design choice.
