# PoolFormer-S24 Reduced-GELU / Width-Redistribution Experiment Plan

## Objective

Modify PoolFormer-S24 to study whether face-recognition or image-classification accuracy can be preserved while reducing the number of explicit nonlinear activation layers.

The central hypothesis is:

> PoolFormer-S24 may not need a GELU in every MetaFormer block. If GELU layers are selectively retained and the hidden width of those active MLP blocks is increased, some or most of the accuracy lost from reducing GELU depth may be recovered while keeping total parameters and FLOPs approximately constant.

The experiment should be implemented in two main phases:

1. **GELU-depth ablation**
   - progressively replace selected GELU layers with `Identity`
   - keep all MLP widths unchanged

2. **Width redistribution**
   - keep fewer GELU blocks
   - widen the MLP hidden dimension in blocks that retain GELU
   - narrow the MLP hidden dimension in blocks where GELU is removed
   - approximately preserve total Params/FLOPs

Do **not** introduce polynomial activations yet.

Use only:

```text
GELU
Identity
```

so the effect of explicit activation reduction can be isolated.

---

# 1. Important Terminology

PoolFormer uses GroupNorm in addition to GELU.

GroupNorm is input-dependent:

```text
mean and variance are computed from the current feature tensor
```

so GroupNorm is also nonlinear.

Therefore, this experiment must **not** claim that the network's total nonlinear depth is reduced from 24 to 12.

Use the terminology:

```text
GELU depth
explicit activation depth
number of GELU layers
number of GELU scalar evaluations
```

Avoid claiming:

```text
total nonlinear depth
fully linear block
```

when GroupNorm remains present.

The purpose of this experiment is specifically:

> reduce explicit GELU activations while leaving the rest of PoolFormer unchanged.

---

# 2. Baseline PoolFormer-S24 Architecture

Inspect the current repository implementation first.

The expected PoolFormer-S24 stage configuration is:

```text
Stage 1: 4 blocks
Stage 2: 4 blocks
Stage 3: 12 blocks
Stage 4: 4 blocks
```

Expected channel dimensions:

```text
Stage 1: 64
Stage 2: 128
Stage 3: 320
Stage 4: 512
```

Expected default MLP expansion ratio:

```text
mlp_ratio = 4
```

Therefore a normal MLP sub-block is conceptually:

```text
C
↓
1×1 Conv / Linear
C → 4C
↓
GELU
↓
1×1 Conv / Linear
4C → C
```

A simplified PoolFormer block is:

```text
input
 │
 ├───────────────────────────────┐
 │                               │
GroupNorm                        │
 ↓                               │
Pooling token mixer              │
 ↓                               │
LayerScale                       │
 │                               │
 +───────────────────────────────┘
 ↓
x'
 │
 ├───────────────────────────────┐
 │                               │
GroupNorm                        │
 ↓                               │
MLP: C → 4C                      │
 ↓                               │
GELU                             │
 ↓                               │
MLP: 4C → C                      │
 ↓                               │
LayerScale                       │
 │                               │
 +───────────────────────────────┘
```

The baseline has:

```text
4 + 4 + 12 + 4 = 24 GELU layers
```

Before modifying the model, verify:

```text
block counts
channel widths
GELU locations
normalization locations
stage transitions
default mlp_ratio
```

against the actual repository implementation.

---

# 3. Required Refactoring

Create a configurable MLP module.

The modified MLP should support:

```python
Mlp(
    in_features,
    hidden_features=None,
    out_features=None,
    use_activation=True,
)
```

or equivalent naming consistent with the repository.

The important variables are:

```text
hidden_features
use_activation
```

Conceptual behavior:

```text
input C
  ↓
Linear / Conv1x1: C → hidden_features
  ↓
GELU or Identity
  ↓
Linear / Conv1x1: hidden_features → C
  ↓
output
```

Requirements:

- `use_activation=True` → original GELU
- `use_activation=False` → `nn.Identity()`
- `hidden_features` can vary per block
- external stage width must remain unchanged
- token-mixing branch must remain unchanged
- GroupNorm must remain unchanged
- LayerScale must remain unchanged
- DropPath behavior must remain unchanged
- preserve baseline checkpoint compatibility whenever the hidden width equals the original width

---

# 4. Configuration Interface

Do not hard-code every experimental architecture separately.

Create a clean per-block configuration.

Example activation mask:

```python
gelu_mask = {
    "stage1": [True, False, True, False],
    "stage2": [True, False, True, False],
    "stage3": [
        True, False,
        True, False,
        True, False,
        True, False,
        True, False,
        True, False,
    ],
    "stage4": [True, False, True, False],
}
```

Example per-block MLP ratios:

```python
mlp_ratios = {
    "stage1": [4, 4, 4, 4],
    "stage2": [4, 4, 4, 4],
    "stage3": [4] * 12,
    "stage4": [4, 4, 4, 4],
}
```

Prefer named presets:

```text
baseline
gelu18
gelu12
gelu8
gelu4

gelu12_w5
gelu12_w6
gelu12_w7
```

The model/training script should support an option such as:

```bash
--arch-config gelu12
```

or:

```bash
--gelu-config gelu12
```

Use the repository's existing style.

---

# 5. Experiment A — GELU-Depth Ablation

Purpose:

> isolate the effect of reducing explicit GELU depth.

For all Experiment A models:

```text
mlp_ratio = 4 for every block
```

Do not redistribute width yet.

Therefore:

- MLP structure stays unchanged
- Params stay essentially unchanged
- FLOPs stay essentially unchanged
- GroupNorm stays unchanged
- Pooling token mixer stays unchanged
- only selected GELUs become `Identity`

Test:

```text
24 → 18 → 12 → 8 → 4
```

GELU layers.

---

# 6. A0 — Baseline

```text
Stage 1:
G G G G

Stage 2:
G G G G

Stage 3:
G G G G G G G G G G G G

Stage 4:
G G G G
```

Where:

```text
G = GELU
- = Identity
```

Expected GELU depth:

```text
24
```

---

# 7. Mandatory First 50%-GELU Architecture

This configuration is mandatory.

Name it:

```text
gelu12
```

Use the following mask:

```text
Stage 1 [4]:
G - G -

Stage 2 [4]:
G - G -

Stage 3 [12]:
G - G - G - G - G - G -

Stage 4 [4]:
G - G -
```

Explicit boolean mask:

```python
gelu_mask = {
    "stage1": [
        True,
        False,
        True,
        False,
    ],

    "stage2": [
        True,
        False,
        True,
        False,
    ],

    "stage3": [
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
    ],

    "stage4": [
        True,
        False,
        True,
        False,
    ],
}
```

Total GELU depth:

```text
Stage 1 = 2
Stage 2 = 2
Stage 3 = 6
Stage 4 = 2
----------------
Total   = 12
```

This is exactly:

```text
12 / 24 = 50%
```

of the baseline GELU depth.

### Rationale

This is not assumed to be the optimal placement.

It is the first controlled reduced-GELU architecture because:

- GELUs remain distributed across every stage
- no stage loses explicit activation completely
- Stage 3 retains regularly spaced GELUs
- no width changes are introduced
- the architecture is easy to interpret

---

# 8. Other GELU-Depth Configurations

Implement:

```text
gelu18
gelu8
gelu4
```

Target GELU depths:

| Config | GELU depth |
|---|---:|
| baseline | 24 |
| gelu18 | 18 |
| gelu12 | 12 |
| gelu8 | 8 |
| gelu4 | 4 |

Use deterministic masks.

Initial rules:

1. keep at least one GELU in every stage
2. distribute retained GELUs across long stages
3. avoid placing all retained GELUs next to each other
4. do not modify GroupNorm
5. do not modify token mixer
6. do not modify stage transitions

Print:

```text
final GELU mask
GELU depth
per-stage GELU count
```

during model construction.

---

# 9. Experiment B — Width Redistribution

Only begin this after Experiment A is working correctly.

Purpose:

> At fixed GELU depth, test whether concentrating hidden width around the remaining GELU blocks can recover lost accuracy.

Use:

```text
gelu12
```

as the first target.

The original MLP is:

```text
C → 4C → GELU → C
```

Define:

```text
r = MLP expansion ratio
```

For an active GELU block:

```text
C → r_G*C → GELU → C
```

For an inactive block:

```text
C → r_I*C → Identity → C
```

---

# 10. FLOP / Parameter Matching Principle

For one MLP block:

```text
fc1: C → rC
fc2: rC → C
```

Approximate parameter cost:

```text
C*(rC) + (rC)*C
= 2rC^2
```

Approximate MAC cost:

```text
2 * H * W * r * C^2
```

Therefore, within a same-resolution stage:

```text
sum(mlp_ratio_i)
```

approximately determines MLP Params/FLOPs.

Baseline:

```text
every block: r = 4
```

For a stage with `N` blocks:

```text
budget = 4N
```

---

# 11. Mandatory Width-Redistribution Experiment

For the 50% GELU model:

```text
half blocks = GELU
half blocks = Identity
```

Test:

```text
active GELU block: r_G = 6
inactive block:    r_I = 2
```

because:

```text
(6 + 2) / 2 = 4
```

which matches the baseline average MLP ratio.

Name this model:

```text
gelu12_w6
```

Conceptually:

```text
Baseline:
G4 G4 G4 G4 ...

Reduced model:
G6 -2 G6 -2 ...
```

Where:

```text
G6 = GELU block with mlp_ratio=6
-2 = Identity block with mlp_ratio=2
```

This is the most important first width-redistribution experiment.

---

# 12. Additional Width-Redistribution Models

At fixed GELU depth = 12, test:

| Model | Active GELU ratio | Identity ratio | Average ratio |
|---|---:|---:|---:|
| gelu12_w4 | 4 | 4 | 4 |
| gelu12_w5 | 5 | 3 | 4 |
| gelu12_w6 | 6 | 2 | 4 |
| gelu12_w7 | 7 | 1 | 4 |

Therefore:

```text
gelu12_w4
gelu12_w5
gelu12_w6
gelu12_w7
```

should all have approximately the same total MLP compute.

Verify actual total Params/FLOPs using a profiler.

Do not rely only on the analytical estimate.

---

# 13. Optional Aggressive 25%-GELU Experiment

After the 50% experiment is complete, implement an optional 25% GELU architecture.

Target:

```text
6 GELU blocks total
```

For Stage 3 specifically:

```text
12 baseline blocks
```

retain approximately:

```text
3 GELU blocks
```

Example Stage 3 pattern:

```text
G - - - G - - - G - - -
```

The baseline Stage-3 ratio budget is:

```text
12 * 4 = 48
```

A possible fixed-budget design is:

```text
3 active blocks with r_G = 10
9 inactive blocks with r_I = 2
```

because:

```text
3*10 + 9*2
= 30 + 18
= 48
```

This experiment tests a more extreme hypothesis:

> can a small number of very wide nonlinear MLPs replace many moderately wide nonlinear MLPs?

Do not run this before the 50% experiment is validated.

---

# 14. Stage-Sensitivity Experiment

Add a stage-level ablation mode.

Configurations:

```text
remove_stage1_gelu
remove_stage2_gelu
remove_stage3_gelu
remove_stage4_gelu
```

For example:

```text
remove_stage3_gelu
```

means:

```text
Stage 1: all GELU
Stage 2: all GELU
Stage 3: all Identity
Stage 4: all GELU
```

Keep:

```text
mlp_ratio = 4
```

for all blocks.

Report:

```text
Δ accuracy relative to baseline
```

This gives a simple stage criticality ranking.

Use the result later to design better non-uniform GELU masks.

---

# 15. Optional Block-Sensitivity Experiment

If training cost allows, implement single-block removal analysis.

For each GELU block `i`:

```text
replace only GELU_i with Identity
```

fine-tune briefly or evaluate directly.

Record:

```text
Δ accuracy_i
```

This produces a per-block sensitivity score.

Possible later use:

```text
retain the most important GELUs
remove the least important GELUs
```

instead of using uniformly spaced masks.

---

# 16. Model Statistics

Every experiment must automatically report:

```text
Total parameters
Total MACs/FLOPs
GELU depth
Number of GELU layers
Number of GELU scalar evaluations
Per-stage GELU count
Per-stage MLP ratio distribution
```

Define GELU scalar evaluations:

```text
N_GELU = Σ(H_l × W_l × hidden_channels_l)
```

over every active GELU tensor.

This metric is essential.

A model can have:

```text
12 GELU layers
```

but if each GELU block is widened from `4C` to `6C`, the scalar nonlinear-operation count does not decrease by exactly 50%.

Therefore report both:

```text
GELU depth
N_GELU
```

---

# 17. GroupNorm Statistics

Because GroupNorm remains nonlinear, also report:

```text
number of GroupNorm layers
```

This should stay constant across the main experiments.

The experiment should explicitly document:

```text
GELU is reduced
GroupNorm is unchanged
```

Do not count GroupNorm as part of the GELU-depth metric.

Later work may investigate GroupNorm removal/replacement separately.

---

# 18. Correctness Tests

Before training, implement tests.

## Baseline equivalence

When:

```text
use_activation=True
mlp_ratio=4
```

the configurable implementation must reproduce the original PoolFormer-S24 architecture.

If using identical weights:

```text
outputs should match within floating-point tolerance
```

---

## Identity test

When:

```text
use_activation=False
```

verify that:

```python
activation is nn.Identity()
```

and that the output shape is unchanged.

---

## Shape tests

Check all stages:

```text
input shape
hidden shape
output shape
stage-transition shape
residual-add shape
```

---

## Config validation

Validate:

```text
len(stage1_mask) == 4
len(stage2_mask) == 4
len(stage3_mask) == 12
len(stage4_mask) == 4
```

and corresponding `mlp_ratio` list lengths.

Fail early on invalid configuration.

---

# 19. Training Strategy

## Phase 1 — Fast screening

Use a pretrained baseline checkpoint if available.

For Experiment A:

```text
load baseline checkpoint
replace selected GELUs with Identity
fine-tune
```

Because the MLP widths remain unchanged, most checkpoint weights should load directly.

Use identical:

```text
dataset
augmentation
optimizer
learning-rate schedule
batch size
training epochs
random seed
```

for all compared models.

Do not use knowledge distillation in the first controlled comparison.

---

# 20. Width-Redistribution Initialization

When changing:

```text
mlp_ratio = 4
```

to:

```text
r_G = 5, 6, 7
```

or:

```text
r_I = 3, 2, 1
```

checkpoint shapes will no longer match directly.

Implement a clean initialization strategy.

Preferred options:

1. initialize the new hidden dimensions normally using the repository's standard initialization
2. optionally copy overlapping channels from the baseline checkpoint
3. clearly log which weights were loaded and which were reinitialized

Do not silently ignore incompatible checkpoint weights.

For final controlled comparison, train promising width-redistributed models from scratch.

---

# 21. Full Training Phase

After screening, fully train at least:

```text
baseline
gelu12_w4
gelu12_w6
gelu8
best gelu8 widened model, if implemented
```

If compute permits also train:

```text
gelu18
gelu12_w5
gelu12_w7
gelu4
```

Use identical training conditions.

Prefer multiple seeds for final results.

---

# 22. Evaluation Metrics

Use the evaluation setup appropriate for the repository/task.

If using PoolFormer for ImageNet-style classification, report:

```text
Top-1 accuracy
Top-5 accuracy
validation loss
```

If adapting PoolFormer-S24 as a face-recognition backbone, report at minimum:

```text
LFW
CFP-FP
AgeDB-30
CPLFW
CALFW
```

If available:

```text
IJB-B
IJB-C
```

and:

```text
TAR @ FAR = 1e-4
TAR @ FAR = 1e-5
```

Also log:

```text
training loss
validation loss
gradient norm
```

---

# 23. Required Experiment Table

Generate CSV and/or JSON with one row per experiment.

Suggested fields:

```text
experiment_name
gelu_mask
gelu_depth
gelu_scalar_count
groupnorm_count
params
flops
stage1_ratios
stage2_ratios
stage3_ratios
stage4_ratios
training_epoch
training_loss
validation_loss
top1
top5
LFW
CFP_FP
AgeDB30
CPLFW
CALFW
IJB_B_TAR_FAR_1e4
IJB_C_TAR_FAR_1e4
checkpoint_path
```

Unsupported task-specific metrics can remain blank.

---

# 24. Required Plots

Generate scripts for the following plots.

## Accuracy vs GELU depth

```text
x-axis: GELU depth
y-axis: accuracy
```

---

## Accuracy vs GELU scalar count

```text
x-axis: N_GELU
y-axis: accuracy
```

---

## Accuracy vs active MLP ratio

For fixed GELU depth = 12:

```text
x-axis: active-block mlp_ratio
y-axis: accuracy
```

---

## Accuracy vs compute

```text
x-axis: FLOPs or Params
y-axis: accuracy
```

---

## Summary Pareto plot

Show:

```text
accuracy
vs
GELU scalar evaluations
```

and highlight:

```text
baseline
gelu12_w4
gelu12_w6
best reduced model
```

---

# 25. Experimental Questions

The implementation should allow us to answer:

1. How much accuracy is lost when GELU depth decreases:

```text
24 → 18 → 12 → 8 → 4
```

while keeping MLP width fixed?

2. Are some stages more sensitive to GELU removal than others?

3. At 50% GELU depth, can widening the remaining GELU blocks recover accuracy?

4. Under approximately fixed Params/FLOPs, which is better:

```text
many moderate-width GELU blocks
```

or:

```text
fewer wide GELU blocks
```

5. Is accuracy more strongly related to:

```text
GELU depth
```

or:

```text
GELU scalar evaluation count
```

6. Is there a minimum GELU depth below which width redistribution can no longer recover accuracy?

7. Does concentrating MLP capacity around nonlinear blocks consistently improve the accuracy/nonlinear-cost tradeoff?

---

# 26. Main Controlled Comparison

The most important comparison is:

## Baseline

```text
24 GELU blocks
all mlp_ratio = 4
```

versus:

## 50%-GELU, no redistribution

```text
12 GELU blocks
12 Identity blocks

active ratio   = 4
inactive ratio = 4
```

versus:

## 50%-GELU, redistributed width

```text
12 GELU blocks
12 Identity blocks

active ratio   = 6
inactive ratio = 2
```

These three models should form the central experiment:

```text
baseline
gelu12_w4
gelu12_w6
```

They test:

```text
effect of reducing GELU depth
```

and then:

```text
whether width concentration recovers accuracy
```

while keeping average MLP compute approximately fixed.

---

# 27. Do Not Change Yet

For this experiment, do NOT yet:

```text
replace GELU with x^2
replace GELU with polynomial approximations
remove GroupNorm
replace GroupNorm
change pooling token mixer
change LayerScale
change stage depth
change stage output channels
replace MLP with depthwise convolution
add attention modules
change classifier / embedding head
change input resolution
introduce FHE-specific constraints
```

The controlled variables should be only:

```text
GELU placement
GELU count
MLP hidden width
```

---

# 28. Implementation Order

Follow this order.

## Step 1

Inspect the current PoolFormer-S24 code and confirm:

```text
stage depths
stage channels
GELU locations
GroupNorm locations
mlp_ratio
stage-transition implementation
```

## Step 2

Refactor MLP to support:

```text
use_activation
per-block hidden width / mlp_ratio
```

without changing baseline behavior.

## Step 3

Verify baseline numerical equivalence.

## Step 4

Implement architecture presets:

```text
baseline
gelu12
```

## Step 5

Add:

```text
gelu18
gelu8
gelu4
```

## Step 6

Add profiling:

```text
Params
FLOPs
GELU depth
GELU scalar count
GroupNorm count
```

## Step 7

Run Experiment A.

## Step 8

Implement width redistribution:

```text
gelu12_w5
gelu12_w6
gelu12_w7
```

## Step 9

Profile actual Params/FLOPs.

Target:

```text
Params difference < 2%
FLOPs difference < 2%
```

relative to baseline when possible.

If exact matching is not possible, report the actual differences.

## Step 10

Run full controlled training for the best candidates.

---

# 29. Expected Deliverables

The Codex CLI agent should produce:

1. configurable PoolFormer-S24 implementation
2. baseline preset
3. mandatory 50%-GELU `gelu12` preset
4. `gelu18`, `gelu8`, and `gelu4` presets
5. `gelu12_w4`, `gelu12_w5`, `gelu12_w6`, `gelu12_w7`
6. stage-sensitivity presets
7. Params/FLOPs/GELU profiler
8. GELU scalar-count profiler
9. training commands/scripts
10. evaluation commands/scripts
11. experiment CSV/JSON summary
12. plotting utility
13. README describing how to reproduce experiments

The architecture/config system should be modular enough that later experiments can replace only the remaining GELU layers with polynomial activations.

---

# 30. Long-Term Extension

Do not implement this now, but design the code so later experiments can replace:

```text
GELU
```

with:

```text
PolynomialActivation
```

only in the retained active blocks.

A later research pipeline may therefore become:

```text
PoolFormer-S24
↓
reduce GELU depth
↓
redistribute MLP width
↓
identify best low-GELU architecture
↓
replace only remaining GELUs with low-degree polynomial
↓
investigate GroupNorm separately
```

This separation is important because it prevents polynomial-instability effects from being mixed with architecture-level nonlinearity-reduction effects.
