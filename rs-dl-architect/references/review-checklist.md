# Architecture Review Checklist

A rigorous checklist for reviewing deep-learning architectures — whether the user hands you their `model.py`, a paper, or a half-finished design. Read this in **review mode**.

The goal of a review isn't to be exhaustive — it's to surface the issues that, if unaddressed, will most likely cause the model to fail. Prioritize ruthlessly.

## Review philosophy

A useful review:

- Identifies issues with **specific evidence** (line, equation, claim).
- Distinguishes **blocking issues** (likely to cause failure or invalid claims) from **suggestions** (preferences or minor improvements).
- Proposes **concrete fixes**, not vague directions ("consider regularization" is not a fix).
- Acknowledges what the model **does well** — full-negative reviews are usually missing context.

Aim for: 2–4 blocking issues, 3–6 suggestions, 1–3 strengths. If you can't find blocking issues, say so explicitly.

## The checklist

Walk through these in order. Skip a section if it's obviously not applicable, but state that you're skipping it.

### 1. Task–architecture match

- Does the architecture's output structure match the task? (dense prediction → segmentation backbone; sparse → detection; etc.)
- Is the receptive field large enough for the task's required context? (compute it, don't guess)
- For RS: does the model handle the actual band count, or is it silently dropping bands?
- For RS: is the GSD compatible with the architecture's pretraining (if pretrained)?

### 2. Data flow and tensor shapes

- Annotate every layer with input/output shapes. Inconsistencies here are the #1 source of silent bugs.
- Skip connections: are spatial dimensions actually compatible? (off-by-one from odd input sizes is common)
- Multi-branch fusion: are channel counts and spatial resolutions aligned at the fusion point?
- Output: does the final layer produce the right shape and value range?

### 3. Loss and supervision

- Is the loss appropriate for the task? (Dice/CE for segmentation, focal for imbalance, etc.)
- Are class weights / sampling addressing the imbalance?
- Are auxiliary losses (deep supervision, boundary loss, contrastive) actually helping, or just adding noise? Look for ablations.
- For probabilistic models: is the loss properly calibrated to the output parameterization (softmax vs sigmoid, log-prob vs prob)?

### 4. Optimization and training

- Optimizer, learning rate, scheduler — are they sensible for the architecture? (Transformer-based usually wants AdamW + warmup + cosine; CNNs are more forgiving)
- Batch size: is it large enough for stable BatchNorm? If small, is BN replaced with GroupNorm/LayerNorm?
- LR scheduler coupling: is the scheduler's `T_max` actually tied to the training length? (a frequent bug — e.g., `CosineAnnealingLR(T_max=N)` with `N` being a constant rather than the real epoch count)
- Gradient clipping: present where needed (transformers, RNNs)?
- Mixed precision: used where it helps, disabled where it causes instability (e.g., loss with `log` or `exp` operations)?

### 5. Data pipeline

- Augmentations: appropriate for the modality? (rotations OK for RS; horizontal flip OK for nadir, questionable for oblique aerial; color jitter on multispectral can destroy spectral signal)
- Normalization: per-band statistics computed from training set (not ImageNet stats for multispectral)?
- Train/val/test split: is it spatially or temporally separated, or random tiles? Random tile splits in RS leak information through spatial autocorrelation.
- Class balance in batches: ensured for rare classes?
- Cloud / no-data masks: handled?

### 6. Evaluation

- Metrics: appropriate? (mIoU not pixel accuracy for segmentation; F1 for imbalanced classification; AP at IoU=0.5 for small-object detection)
- Test set: held out from training? Held out by **region/scene**, not random tiles?
- Multiple seeds: variance reported? Single-seed numbers in RS are unreliable.
- Strong baselines: compared against simple alternatives (Random Forest, plain U-Net, etc.)? Beating ResNet-18 is not impressive if RF wasn't tried.
- Stratified results: reported by region, by class, by season?

### 7. Generalization claims

- Cross-region test included?
- Cross-sensor test if the paper claims sensor invariance?
- Cross-temporal test if the paper claims temporal robustness?
- Are limitations honestly stated, or hedged into footnotes?

### 8. Reproducibility

- Code released?
- Random seeds fixed (and the model is actually deterministic given the seed)?
- Pretrained weights released or accessible?
- Dependency versions specified?
- Training logs or W&B/TensorBoard runs available?

### 9. Compute and efficiency

- Parameter count and FLOPs reported?
- Training time / hardware specified?
- For real-time claims: inference latency on stated hardware?
- Memory footprint at training and inference batch sizes?

### 10. Novelty and contribution (for paper review)

- What is actually new? Architecture, training method, data, application?
- Is the novelty incremental or substantive? Both can be valuable, but should be honestly framed.
- Are baselines from comparable years and budgets?
- Are claims supported by ablations?

## Output format for a review

Structure the review as:

```
## Strengths (1–3)
- [specific, with evidence]

## Blocking issues (2–4)
- [issue + evidence + concrete fix]

## Suggestions (3–6)
- [non-blocking improvements]

## Open questions (0–3)
- [things that need clarification before judging]
```

Don't pad. If you only have 2 blocking issues, list 2.

## Reviewing code specifically

When the user hands you a `model.py` or a checkpoint:

1. **Trace forward()** end-to-end with concrete shape annotations.
2. **Check the loss function** in the training loop — common bugs: forgetting to ignore the no-data class, wrong reduction, double-counting auxiliary losses.
3. **Check the LR scheduler** wiring — coupling bugs are silent killers.
4. **Check the dataloader** — augmentations applied to labels too where appropriate, no-data handling, normalization.
5. **Look for the small things**: `model.train()` vs `model.eval()` switches around validation, gradient accumulation correctness, EMA update timing.
6. **Run a forward pass mentally** with a fake input — does it work?

## Reviewing a paper specifically

When the user wants you to review a paper:

1. **State the central claim** in one sentence. If you can't, the paper has a clarity problem.
2. **Identify the contribution** (architecture, training method, dataset, application).
3. **Map the experiments** to the claim. Does each experiment test something the claim asserts?
4. **Check ablations**: does each architectural component have an ablation? If a component is justified by intuition only, that's a weakness.
5. **Check the baselines**: are they current, fair, and well-tuned?
6. **Check related work**: are key papers cited? Missing comparisons can indicate weak claims.

## Example of a good review note

Bad:

> "The model could benefit from more regularization."

Good:

> "BatchNorm is used throughout but training batch size is 4 per GPU. With this batch size, BN statistics are very noisy and likely contributing to the training instability shown in Figure 5. Replace BN with GroupNorm (e.g., 32 groups) or LayerNorm in the encoder; this is a 1-line change and typically resolves small-batch training issues."

The good version is specific, evidence-based, and actionable.
