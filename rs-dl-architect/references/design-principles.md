# Design Principles

Cross-cutting heuristics that apply across all the subdomains. Read this when starting a new design from scratch, or when a task crosses multiple subdomains and the per-subdomain guides don't dominate.

## Principle 1: Match output structure to task

The output's spatial structure is the first design decision, and most others follow from it.

| Output structure    | Decoder shape                                | Examples                          |
| ------------------- | -------------------------------------------- | --------------------------------- |
| Per-pixel           | Encoder–decoder, recover full resolution     | Segmentation, regression maps     |
| Per-object (sparse) | Encoder + detection head, no full upsampling | Detection, instance counting      |
| Per-image (global)  | Encoder + global pool + MLP                  | Classification, regression scalar |
| Per-pixel-per-time  | Spatio-temporal encoder–decoder              | Continuous monitoring             |

Don't fight the structure. Using a segmentation backbone for image classification wastes 80% of the parameters; using a classification backbone for segmentation throws away spatial information.

## Principle 2: Receptive field must match context need

Compute the receptive field of the proposed model (effective, not theoretical). Compare it to the spatial scale of the predictive context.

- Land-cover classification: the relevant context is often 50–200 m (1–4 Sentinel-2 pixels). Small RF works.
- Building extraction: the relevant context is the building footprint plus a margin (~100 m at 0.3 m GSD ≈ 300 px). Need much larger RF.
- Crop classification: temporal context (full season) often matters more than spatial context. Use temporal models.
- Disaster damage: the building is the unit, but neighborhood context (collapsed buildings cluster) helps. Medium-large RF.

A receptive field too small fails silently — the model just plateaus at suboptimal accuracy.

## Principle 3: Label cost is the binding constraint

In RS, labels are usually expensive. Architecture choice should be informed by label budget, not just accuracy.

- 100 labels: foundation-model fine-tuning, parameter-efficient tuning, heavy augmentation. Don't train a 100M-param ViT from scratch.
- 1k labels: small CNN or PEFT'd foundation model. From-scratch transformers usually fail.
- 10k labels: medium models work. Foundation-model fine-tuning still usually wins.
- 100k+ labels: train from scratch is feasible. Foundation models still help with domain shift.
- 1M+ labels: from-scratch architecture matters; foundation models help less.

If the user asks for the "best" architecture without stating label budget, ask. The answer changes by 2 orders of magnitude.

## Principle 4: Inductive biases earn their keep

A good inductive bias is one whose corresponding training data would be expensive or unavailable. Examples:

- **Translation equivariance** (CNNs): correct for most natural images. Strong bias, basically free.
- **Multi-scale via pyramids/FPN**: correct for scenes with objects at varied scales. Worth the parameters.
- **Rotation invariance** (group-equivariant convs, oriented filters): correct for aerial/satellite (no canonical "up"). Often underused; worth trying.
- **Local + global** (transformer + CNN hybrids): correct when context spans both small and large scales. Worth the complexity.
- **Permutation invariance over time** (set transformer for irregular time series): correct when temporal ordering is provided via positional encoding, not implicit ordering.

Inductive biases that aren't earned: skip them. Adding "channel attention" everywhere because a paper did it is not an inductive bias, it's a parameter splurge.

## Principle 5: Make components ablation-friendly

Build models such that components can be turned off cleanly. This pays off during development and when reviewers ask "did the new module actually help?". Patterns:

- **Residual additions**: `out = base(x) + module(x)` lets you ablate `module` by zeroing it.
- **Gated additions**: `out = base(x) + α * module(x)` with learnable α; trace what α converges to.
- **Optional skip connections**: pass them through a `nn.Identity()` when ablating.

Models that are tightly coupled (every component depends on every other) are hard to debug and hard to publish.

## Principle 6: Diagnose before you scale

When a model underperforms, the first reflex is often to scale it up (more layers, more channels, more attention). This is usually wrong. Scale only after diagnosis:

- Is the model **overfitting**? Train loss low, val loss high → regularize, augment, add data, not scale up.
- Is the model **underfitting**? Train loss high → scale up may help.
- Is the model **mis-specified**? Train loss decreases but val loss is meaningless → the loss or labels may be wrong.
- Is the **data the bottleneck**? Scaling the model won't help; scale the data or label more.

Scaling the model is the last move, not the first.

## Principle 7: Match the loss to the metric

The loss function should reward the model for the thing being measured. When they diverge:

- mIoU as the metric, cross-entropy as the loss → the model optimizes pixel accuracy, IoU is what's evaluated. Add Dice or Lovász loss.
- mAP as the metric, naive cross-entropy on positives → the model is rewarded for recall, mAP cares about ranking. Use focal loss or RankNet-style.
- Boundary F1 as the metric, region-based loss → the model gets boundaries wrong. Add boundary loss.

Don't expect the optimizer to magically reconcile mismatched loss/metric.

## Principle 8: Validate the data before validating the model

Many "model issues" are data issues. Check before tuning:

- **Label noise**: random sample 50 labels and re-label them yourself. Inter-annotator agreement on RS labels is rarely above 90%.
- **Spatial leakage**: are train and val tiles from the same scene? Same orbit?
- **Class boundaries**: are the labels for the same class consistent across regions?
- **No-data masking**: is no-data labeled as a class, or excluded?

Fixing the data usually beats fixing the model.

## Principle 9: Honest evaluation beats optimistic evaluation

It's tempting to choose the eval setup that makes the numbers look best. Resist:

- **Random tile splits** in RS leak through spatial autocorrelation. Use region splits.
- **Single seed** numbers are noise. Report mean ± std over ≥3 seeds.
- **Best-checkpoint** selection inflates results. Use early-stopping on val, then evaluate on test.
- **Cherry-picked metrics**: report all the standard ones for the task, not just the favorable ones.

Optimistic numbers always come back to bite you in deployment, in review, or in follow-up papers.

## Principle 10: Simplicity is a feature

For most RS tasks, a well-tuned U-Net or simple CNN beats an elaborate custom architecture. Reasons:

- Simple models are easier to debug, deploy, and reproduce.
- Simple models leave more compute for hyperparameter search and seeds.
- Simple models are easier to explain to stakeholders who must trust the predictions.
- Most "improvements" in the literature don't replicate; simple baselines do.

Reach for complexity only when simplicity demonstrably fails. The default direction is the boring one.

## A useful design dialogue

When approaching a new task, work through this internally before recommending:

1. What's predicted? (output structure)
2. What's available? (input modality, resolution, temporal stack, label budget)
3. What's the binding constraint? (compute, labels, latency, accuracy)
4. What's the simplest model that could plausibly work?
5. What inductive biases would buy real labeled data here?
6. What can be borrowed from a foundation model?
7. What's the honest evaluation that mirrors deployment?

Then propose. Defending a recommendation is easier when the dialogue is explicit.
