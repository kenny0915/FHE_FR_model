# iResNet50 Reduced-Nonlinearity Experiment Plan

## Objective

Modify the InsightFace-style `iResNet50` backbone to study whether face-recognition accuracy can be preserved while reducing nonlinear depth.

The main hypothesis is:

> iResNet50 may not need a PReLU in every residual block. If nonlinear activations are selectively retained and their surrounding hidden representations are widened, some of the accuracy lost by reducing nonlinear depth may be recovered while keeping total parameters and FLOPs approximately constant.

This experiment should be implemented in two phases:

1. **Nonlinearity ablation:** progressively remove PReLU layers without changing convolution width.
2. **Width redistribution:** keep fewer PReLU layers, widen the blocks containing PReLU, and narrow linear blocks so that total Params/FLOPs remain approximately equal to the baseline.

Do **not** introduce polynomial activations in this experiment yet. Use only `PReLU` and `Identity` so that the role of nonlinear depth can be isolated.

---

# 1. Baseline Architecture

Use the current InsightFace-style iResNet50 implementation as the baseline.

Expected stage configuration:

```text
Stem
Stage 1: 3 IBasicBlocks
Stage 2: 4 IBasicBlocks
Stage 3: 14 IBasicBlocks
Stage 4: 3 IBasicBlocks
```

The channel configuration is expected to be approximately:

```text
Stage 1: 64
Stage 2: 128
Stage 3: 256
Stage 4: 512
```

Each normal `IBasicBlock` should be conceptually similar to:

```text
input
  │
  ├──────────────────────────── shortcut
  │
  BN
  ↓
Conv 3×3
  ↓
 BN
  ↓
PReLU
  ↓
Conv 3×3
  ↓
 BN
  │
  + shortcut
  ↓
output
```

The standard model contains approximately:

```text
Stem PReLU: 1

Stage 1: 3
Stage 2: 4
Stage 3: 14
Stage 4: 3

Total nonlinear depth ≈ 25
```

Before changing anything, verify this against the actual repository implementation.

---

# 2. Required Refactoring

Create a configurable version of `IBasicBlock`.

The modified block must support:

```python
IBasicBlock(
    in_channels,
    out_channels,
    stride,
    mid_channels=None,
    use_activation=True,
)
```

Default behavior:

```python
mid_channels = out_channels
use_activation = True
```

Conceptual structure:

```text
input
  │
  ├──────────────────────────────────────── shortcut
  │
  BN
  ↓
Conv3×3(in_channels → mid_channels)
  ↓
 BN
  ↓
PReLU(mid_channels) OR Identity()
  ↓
Conv3×3(mid_channels → out_channels, stride=stride)
  ↓
 BN
  │
  + shortcut
  ↓
output
```

Requirements:

- `use_activation=True` → use the original PReLU.
- `use_activation=False` → replace PReLU with `nn.Identity()`.
- `mid_channels` controls the internal block width.
- Do not change the external stage channel dimensions.
- Preserve the original shortcut/downsampling behavior.
- Preserve checkpoint compatibility where possible when `mid_channels == out_channels` and `use_activation=True`.

---

# 3. Architecture Configuration Interface

Do not hard-code activation removal separately for every experiment.

Create a clean configuration mechanism.

For example:

```python
activation_mask = {
    "stem": True,
    "stage1": [True, False, True],
    "stage2": [True, False, False, True],
    "stage3": [...],
    "stage4": [True, True, True],
}
```

Also support per-block internal width:

```python
mid_widths = {
    "stage1": [64, 64, 64],
    "stage2": [128, 128, 128, 128],
    "stage3": [256] * 14,
    "stage4": [512, 512, 512],
}
```

Prefer implementing experiment presets by name:

```text
baseline
nl17
nl13
nl9
nl5
nl13_wide125
nl13_wide150
nl13_wide200
```

The training script should accept something similar to:

```bash
--arch-config nl13
```

or

```bash
--nonlinear-config nl13
```

Use the coding style already present in the repository.

---

# 4. Experiment A — Nonlinear Depth Ablation

The purpose of Experiment A is to isolate the effect of nonlinear depth.

For all Experiment A models:

```text
mid_channels = original channel count
```

Do not redistribute width.

Therefore:

- convolution Params remain essentially unchanged;
- convolution FLOPs remain essentially unchanged;
- network depth remains unchanged;
- only selected PReLUs become `Identity`.

Run the following configurations.

---

## A0 — Baseline

```text
Stem:
P

Stage 1:
P P P

Stage 2:
P P P P

Stage 3:
P P P P P P P P P P P P P P

Stage 4:
P P P
```

Where:

```text
P = PReLU
- = Identity
```

Expected nonlinear depth:

```text
25
```

---

# 5. Required First 50%-Nonlinearity Architecture

This architecture is mandatory.

Name it:

```text
nl13
```

or another clear equivalent.

Use:

```text
Stem:
P

Stage 1 [3]:
P - P

Stage 2 [4]:
P - - P

Stage 3 [14]:
P - - P - - P - - P - - - P

Stage 4 [3]:
P P P
```

Explicit boolean masks:

```python
activation_mask = {
    "stem": True,

    "stage1": [
        True,
        False,
        True,
    ],

    "stage2": [
        True,
        False,
        False,
        True,
    ],

    "stage3": [
        True,
        False,
        False,
        True,
        False,
        False,
        True,
        False,
        False,
        True,
        False,
        False,
        False,
        True,
    ],

    "stage4": [
        True,
        True,
        True,
    ],
}
```

Nonlinear depth:

```text
Stem   = 1
Stage1 = 2
Stage2 = 2
Stage3 = 5
Stage4 = 3
----------------
Total  = 13
```

This is approximately 50% of the baseline nonlinear depth:

```text
13 / 25 ≈ 52%
```

### Rationale

This is not assumed to be the optimal placement.

It is the first controlled reduced-nonlinearity architecture because:

- every stage still contains nonlinear processing;
- the first/last parts of stages retain nonlinear transformations;
- Stage 3 nonlinearities are distributed through the long 14-block stage;
- Stage 4 is initially preserved;
- no width change is introduced yet.

---

# 6. Other Nonlinear-Depth Configurations

Implement at least these additional ablation points:

```text
nl17
nl9
nl5
```

Target nonlinear depths:

| Config | Approx. nonlinear depth |
|---|---:|
| baseline | 25 |
| nl17 | 17 |
| nl13 | 13 |
| nl9 | 9 |
| nl5 | 5 |

The exact masks for `nl17`, `nl9`, and `nl5` may be generated using a deterministic rule, but:

1. keep the stem activation initially;
2. avoid removing every activation from an entire stage;
3. initially keep stage-transition blocks nonlinear;
4. distribute retained nonlinearities reasonably across long stages.

Print the final mask and nonlinear depth at model construction time.

---

# 7. Experiment B — Width Redistribution

Only start this after Experiment A works.

The objective is:

> At a fixed nonlinear depth, test whether widening the blocks that retain PReLU can recover accuracy.

Use `nl13` as the first target.

For an active block:

```text
C → rC → PReLU → C
```

For a linear block:

```text
C → qC → Identity → C
```

Test active-block width ratios:

```text
r = 1.25
r = 1.50
r = 2.00
```

Create configurations such as:

```text
nl13_wide125
nl13_wide150
nl13_wide200
```

---

# 8. FLOP/Parameter Matching

The models in Experiment B should have approximately the same total Params and FLOPs as the baseline.

For a normal non-downsampling block:

```text
Conv 1: 3×3, C → M
Conv 2: 3×3, M → C
```

Approximate parameter cost:

```text
9*C*M + 9*M*C
= 18*C*M
```

Therefore, within one same-resolution stage, compute can be approximately redistributed using:

```text
sum(mid_channels_i) ≈ number_of_blocks * baseline_channel
```

Example: Stage 3.

Baseline:

```text
14 blocks
C = 256
```

Suppose `nl13` retains 5 active blocks in Stage 3.

For:

```text
active width = 1.5C = 384
```

there are:

```text
5 active blocks
9 linear blocks
```

To approximately preserve stage compute:

```text
5*(1.5C) + 9*(qC) = 14C
```

Therefore:

```text
q ≈ 0.722
```

For `C=256`:

```text
qC ≈ 185
```

Use a hardware-friendly aligned value such as:

```text
192
```

Then verify actual Params/FLOPs with a profiler.

Do not rely only on the analytical approximation.

---

# 9. Important Constraint for the First Width Experiments

Do not aggressively modify stage-transition/downsampling blocks in the first implementation.

For blocks that:

- change spatial resolution;
- change channel count;
- use shortcut downsampling;

keep the original width and keep their PReLU initially.

Reason:

Their FLOP calculation differs from normal blocks and modifying them would introduce extra experimental variables.

First redistribute width only among normal blocks at the same spatial resolution.

---

# 10. Model Statistics

Every experiment must automatically report:

```text
Total parameters
Total MACs/FLOPs
Nonlinear depth
Number of active PReLU layers
Total nonlinear scalar evaluations
```

Define nonlinear scalar evaluations as:

```text
N_NL = Σ(H_l × W_l × C_l)
```

over all active PReLU tensors.

This metric is important because two networks can have the same nonlinear depth but very different nonlinear operation counts.

Example:

```text
112×112×64 PReLU
```

is much more expensive in scalar nonlinear evaluations than:

```text
7×7×512 PReLU
```

Save these statistics with every experiment result.

---

# 11. Correctness Tests

Before training, implement tests that verify:

### Baseline equivalence

When:

```text
use_activation=True
mid_channels=original_channels
```

the new configurable block must reproduce the original architecture.

If loading the same weights, outputs should match numerically within floating-point tolerance.

### Identity configuration

When:

```text
use_activation=False
```

verify that the activation is exactly `nn.Identity()`.

### Shape tests

For every stage:

```text
input/output tensor dimensions
shortcut dimensions
stride behavior
```

must remain correct.

### Config validation

Check:

```text
len(stage1_mask) == 3
len(stage2_mask) == 4
len(stage3_mask) == 14
len(stage4_mask) == 3
```

and equivalent checks for width arrays.

Fail early on invalid configuration.

---

# 12. Training Strategy

## Phase 1 — Fast architecture screening

Use the baseline checkpoint when compatible.

For reduced-nonlinearity models:

1. initialize from the pretrained iResNet50 checkpoint;
2. PReLU → Identity blocks simply ignore/remove the PReLU parameters;
3. fine-tune using the same face-recognition training objective;
4. use the same dataset and augmentation;
5. use the same evaluation protocol.

The purpose is rapid sensitivity analysis.

Do not use knowledge distillation in the first comparison.

Do not introduce extra regularization only for reduced-nonlinearity models unless explicitly required for numerical stability.

---

## Phase 2 — Full controlled training

After identifying promising configurations, train the best candidates from scratch.

At minimum compare:

```text
baseline
nl13
best nl13 widened model
nl9
best nl9 widened model, if implemented
```

Use identical:

```text
training dataset
batch size
optimizer
learning-rate schedule
number of epochs
augmentation
ArcFace/margin-loss configuration
random-seed protocol
```

Prefer multiple random seeds for final comparisons if compute permits.

---

# 13. Evaluation Metrics

Do not evaluate only on LFW.

At minimum, if already supported by the repository, report:

```text
LFW
CFP-FP
AgeDB-30
CPLFW
CALFW
```

If available, also evaluate:

```text
IJB-B
IJB-C
```

and report verification metrics such as:

```text
TAR @ FAR = 1e-4
TAR @ FAR = 1e-5
```

Also save training metrics:

```text
training loss
validation loss
gradient norm
```

If convenient, additionally log:

```text
embedding L2 norm
intra-class cosine similarity
inter-class cosine similarity
```

---

# 14. Required Experiment Tables

Generate a CSV/JSON summary with one row per experiment.

Suggested fields:

```text
experiment_name
activation_mask
nonlinear_depth
nonlinear_element_count
params
flops
training_epoch
training_loss
LFW
CFP_FP
AgeDB30
CPLFW
CALFW
IJB_B_TAR_FAR_1e4
IJB_C_TAR_FAR_1e4
checkpoint_path
```

Missing unsupported metrics can be left blank.

---

# 15. Required Plots

Generate scripts to plot:

### Accuracy vs nonlinear depth

```text
x-axis: nonlinear depth
y-axis: face verification accuracy
```

### Accuracy vs nonlinear element count

```text
x-axis: nonlinear scalar evaluations
y-axis: face verification accuracy
```

### Accuracy vs active-block width

For fixed nonlinear depth, e.g. `nl13`:

```text
x-axis: active-block width ratio
y-axis: accuracy
```

### FLOPs/Params validation

Plot or tabulate:

```text
model
Params
FLOPs
nonlinear depth
nonlinear element count
accuracy
```

---

# 16. Experimental Questions

The implementation should allow us to answer:

1. How much accuracy is lost when nonlinear depth decreases from:

```text
25 → 17 → 13 → 9 → 5
```

while keeping convolution architecture unchanged?

2. Are some stages much more sensitive to PReLU removal than others?

3. At approximately 50% nonlinear depth (`nl13`), can widening the remaining nonlinear blocks recover accuracy?

4. Under fixed Params/FLOPs, which is better:

```text
many narrow nonlinear blocks
```

or:

```text
few wide nonlinear blocks
```

5. Is accuracy more strongly correlated with:

```text
nonlinear depth
```

or:

```text
nonlinear scalar evaluation count
```

6. Is there a minimum nonlinear depth below which width redistribution cannot recover performance?

---

# 17. Stage-Sensitivity Experiment

Add an optional experiment mode that removes PReLU from one stage at a time.

Configurations:

```text
remove_stage1_activation
remove_stage2_activation
remove_stage3_activation
remove_stage4_activation
```

All other stages retain PReLU.

This gives a simple stage sensitivity ranking.

Report:

```text
Δ accuracy relative to baseline
```

This result can later be used to design better activation masks than the manually chosen `nl13` pattern.

---

# 18. Implementation Order

Follow this order.

## Step 1

Inspect the existing iResNet50 implementation and confirm:

```text
stage block counts
channel widths
PReLU locations
stride/downsampling locations
```

## Step 2

Implement configurable:

```text
use_activation
mid_channels
```

without changing baseline behavior.

## Step 3

Verify baseline numerical equivalence.

## Step 4

Implement activation-mask presets.

Mandatory first presets:

```text
baseline
nl13
```

## Step 5

Add:

```text
nl17
nl9
nl5
```

## Step 6

Add model complexity reporting:

```text
Params
FLOPs
nonlinear depth
nonlinear element count
```

## Step 7

Run short fine-tuning experiments for Experiment A.

## Step 8

Implement width redistribution for `nl13`.

## Step 9

Profile and adjust widths until total Params/FLOPs are close to baseline.

Target tolerance:

```text
Params difference: preferably < 2%
FLOPs difference: preferably < 2%
```

If exact matching is difficult, report the actual difference instead of hiding it.

## Step 10

Run full controlled training only for the most promising configurations.

---

# 19. Do Not Do Yet

For this experiment, do NOT yet:

```text
replace PReLU with x^2
replace PReLU with polynomial approximations
introduce FHE-specific constraints
change loss functions
replace standard convolution with depthwise convolution
convert blocks to MobileNetV2 blocks
add attention modules
change embedding dimensions
change input resolution
```

Those changes would introduce additional variables.

The current experiment is specifically about:

```text
nonlinear depth
×
nonlinear placement
×
hidden width
```

under approximately fixed model compute.

---

# 20. Expected Deliverables

The Codex agent should produce:

1. Configurable iResNet50 implementation.
2. `baseline` architecture preset.
3. Mandatory `nl13` 50%-nonlinearity preset.
4. `nl17`, `nl9`, and `nl5` presets.
5. Width-redistributed `nl13` variants.
6. Params/FLOPs/nonlinearity profiler.
7. Training commands or scripts.
8. Evaluation commands or scripts.
9. CSV/JSON experiment summary.
10. Plotting utility.
11. Short README describing how to reproduce each experiment.

The implementation should be modular enough that later experiments can replace the remaining PReLU layers with polynomial activations without redesigning the backbone configuration system.
