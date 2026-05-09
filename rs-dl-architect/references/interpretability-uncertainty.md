# Interpretability and Uncertainty

Covers explanation methods (what does the model attend to, why this prediction) and uncertainty quantification (how confident is the model, when is it likely wrong). Read this when the user is past pure accuracy chasing — debugging predictions, deploying to high-stakes decisions, or writing a paper that needs to defend its claims.

## Why these matter for remote sensing

In RS deployment, a wrong prediction can mean a missed flood alert, a misidentified illegal mine, a misclassified crop affecting subsidy decisions. Pure accuracy on a held-out set under-specifies trust. Interpretability tells you _what_ the model uses; uncertainty tells you _when_ to trust it. Both matter.

## Interpretability methods

### Gradient-based

- **Grad-CAM, Grad-CAM++, Score-CAM**: class-activation maps from gradients flowing through a chosen layer. Fast, easy, the standard first thing to try.
- **Integrated gradients**: integrate gradients along a path from a baseline. Less noisy than vanilla gradient, satisfies axiomatic properties.
- **SmoothGrad**: average gradients over noisy inputs. Cleaner than vanilla saliency.

For multispectral input, run Grad-CAM per band group and inspect which bands the model relies on. A model that ignores SWIR for a vegetation task is suspicious.

### Attention-based (transformers)

- **Attention rollout**: multiply attention matrices across layers to get image-level attribution. Standard for ViT.
- **Attention flow**: graph-based variant.
- **Grad-CAM for transformers**: works on any model with a forward hook; often more reliable than raw attention.

Be cautious: attention is not explanation. Many studies show attention maps and saliency disagree. Use multiple methods and look for consistency.

### Perturbation-based

- **LIME**: local linear approximation around a sample. Slow but model-agnostic.
- **SHAP**: game-theoretic attribution. More principled than LIME, also slow.
- **Occlusion sensitivity**: slide an occluding patch over the image, watch the prediction change. Brutally simple, often illuminating.

For RS: occlusion sensitivity over **bands** (not just spatial regions) is underused and very informative.

### Concept-based

- **TCAV (Testing with Concept Activation Vectors)**: how much does a concept (e.g., "bright SAR backscatter") drive this prediction?
- **Concept bottleneck models**: model is forced to predict via human-interpretable concepts. Useful when you can curate concepts.

### Counterfactual / contrastive

- "What would change in the input to flip the prediction?" Counterfactual saliency (e.g., CEM, DiCE). Underused in RS; powerful for understanding decision boundaries.

## Uncertainty quantification

Two flavors that are often conflated:

- **Aleatoric**: data uncertainty. Inherent ambiguity in the input (e.g., a mixed pixel that's genuinely 50% forest, 50% grassland). Cannot be reduced by more data.
- **Epistemic**: model uncertainty. Lack of training data in this region of input space. Reduced by more / better data.

A good UQ method separates these.

### Methods

#### Monte Carlo dropout (Gal & Ghahramani 2016)

Train with dropout, keep dropout active at inference, sample N forward passes. Variance ≈ epistemic uncertainty.

- Pros: trivial to add to existing models.
- Cons: theoretical justification is contested; tends to be overconfident; sample-count tradeoff (N=20–50 typical).
- Practical: standard first try, but rarely the best.

#### Deep ensembles (Lakshminarayanan et al. 2017)

Train M independent models with different seeds; ensemble predictions. Variance across ensemble members ≈ epistemic uncertainty.

- Pros: state-of-the-art calibration in many studies; simple.
- Cons: M× training and inference cost. M=5–10 is typical.
- Practical: the strongest baseline; if you can afford it, do it.

#### Variational and Bayesian neural networks

Place priors on weights, learn posterior. Theoretically clean, practically heavy and often unstable for large vision models. Use sparingly; ensembles usually win.

#### Evidential deep learning

Predict parameters of a higher-order distribution (e.g., Dirichlet for classification, Normal-Inverse-Gamma for regression). One forward pass gives both prediction and uncertainty.

- Pros: cheap at inference; principled separation of aleatoric/epistemic.
- Cons: harder to train, sensitive to loss formulation. Recent work (e.g., posterior network) improves this.
- Practical: consider when inference cost matters.

#### Test-time augmentation (TTA) variance

Augment input N times (rotations, flips, brightness shifts), measure prediction variance. Cheap epistemic-like estimate.

- Pros: free, applies to any model.
- Cons: not principled, but empirically useful as a sanity check.

#### Conformal prediction

Distribution-free: guarantees that the prediction set contains the true label with probability ≥ 1−α. Wraps any model; provides coverage guarantee.

- Pros: rigorous, model-agnostic, doesn't need retraining.
- Cons: produces sets, not single confidence scores; requires calibration set.
- Practical: increasingly used when you need formal guarantees (regulatory or scientific contexts).

### Calibration

Separate from raw uncertainty: a model is calibrated if predicted probability matches empirical frequency.

- **Reliability diagrams** and **Expected Calibration Error (ECE)**: standard diagnostics.
- **Temperature scaling**: divide logits by a learned scalar. Cheap post-hoc fix; usually the first thing to try.
- **Platt scaling, isotonic regression**: alternatives.

A common failure: model has high accuracy but ECE is terrible (overconfident on errors). Always report calibration alongside accuracy in deployment-oriented work.

## Combining interpretability and uncertainty

The most informative pattern: **flag low-confidence predictions and explain them differently than high-confidence ones**.

- Low-confidence pixel/object → show interpretability map; use to debug the model or escalate to manual review.
- High-confidence misprediction → most dangerous case; interpretability can reveal what spurious signal the model latched onto.

## Evaluation

For interpretability:

- **Sanity checks** (Adebayo et al. 2018): saliency map should change when the model is randomized. Methods that pass this are more trustworthy.
- **Pointing game**: does the saliency hit the ground-truth object?
- **Insertion / deletion tests**: insert pixels in saliency-order, watch prediction climb; delete in saliency-order, watch it fall.

For uncertainty:

- **NLL on test set**: rewards good calibration.
- **Brier score**: for classification.
- **Selective prediction / abstention curves**: at threshold τ, predict only when confidence > τ; plot accuracy vs coverage.
- **OOD detection benchmarks**: does the model produce high uncertainty on out-of-distribution inputs (different sensor, different region)?

For RS specifically: cross-region OOD evaluation is the gold standard. A UQ method that produces high uncertainty on out-of-region tiles is doing its job.

## Common pitfalls

- Reporting Grad-CAM without sanity checks → may be visualizing input edges, not model decisions.
- MC dropout at inference but not at training (or with very low p) → samples are nearly identical; "uncertainty" is meaningless.
- Single ensemble member used at inference for "speed" → you've thrown away the uncertainty.
- Conformal prediction calibration set drawn from the same scene as test → coverage guarantee violated by spatial autocorrelation.
- Conflating softmax confidence with calibrated probability → most networks are overconfident; raw softmax is not calibrated.
- Reporting interpretability on cherry-picked examples → show failure cases too.

## When the user is just starting

A reasonable first uncertainty pipeline for RS:

1. Train your model normally.
2. Train 5 seeds → deep ensemble.
3. Apply temperature scaling on a held-out calibration set.
4. Report ECE and selective prediction curves alongside accuracy.
5. Generate Grad-CAM for both correct and incorrect high-confidence predictions; look for spurious patterns.

This catches most issues without exotic methods.
