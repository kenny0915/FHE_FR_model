# NL13 PReLU-to-Polynomial Replacement Results

## Objective and acceptance gate

This experiment replaces the 13 retained PReLUs in the `nl13` iResNet-50
backbone in forward order. A checkpoint qualifies only when strict inference
over all 469,375 IJB-C images reports zero non-finite augmented embedding rows
and TAR at FAR `1e-4` is at least 90%.

The polynomial target at activation `i` is its frozen channel-wise PReLU on a
public interval `[-S_i, S_i]`. Inference evaluates
`S_i * HerPN(x / S_i)`, which folds to one degree-2 polynomial. Each replaced
activation therefore adds one sequential ciphertext-ciphertext multiplication
for the square; calibration, range conditioning, and guards are plaintext
training procedures and do not increase encrypted multiplicative depth.

## Validated frontier

| Polynomial prefix | Last replaced activation | Non-finite IJB-C rows | TAR at FAR `1e-4` | Gate |
|---:|---|---:|---:|---|
| 3 | `layer1.2.prelu` | 0 | 95.20% | pass |
| 4 | `layer2.0.prelu` | 0 | 92.85% | pass |
| 5 | `layer2.3.prelu` | 0 | 91.79% | pass |
| 6, fast schedule | `layer3.0.prelu` | 0 | 84.72% | accuracy fail |
| 6, slow blend + fixed-graph recovery | `layer3.0.prelu` | 0 | **92.69%** | **pass** |
| 7, fast schedule | `layer3.3.prelu` | 0 | 77.52% | accuracy fail |
| 7, slow blend + fixed-graph recovery, epoch 11 | `layer3.3.prelu` | 0 | 84.24% | accuracy fail |
| 7, slow blend + fixed-graph recovery, epoch 14 | `layer3.3.prelu` | 0 | **87.36%** | accuracy fail |

The maximum validated prefix is therefore **six polynomial activations**,
leaving seven PReLUs. The qualifying checkpoint is preserved independently of
the later seven-layer experiment at:

```text
work_dirs/ms1mv3_r50_nl13_prelu_herpn_scaled_group06_accuracy_recovery/
model_group06_ijbc_92p69.pt
```

Its complete IJB-C TAR vector for FAR
`[1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]` is
`[80.50, 88.09, 92.69, 95.59, 97.45, 98.64]` percent. The exact checkpoint
state has its first six `.blend` buffers equal to one and its remaining seven
equal to zero.

## Method that reached six replacements

1. Start from the trained NL13 PReLU backbone, not the 25-activation baseline.
2. Convert singleton activations in forward order so every interval is
   measured on the graph containing the already converted prefix.
3. Estimate the central input magnitude with the maximum sampled 99.95th
   percentile over 512 batches per rank and use a 2x interval margin.
4. Hold out 10% of calibration batches and record the global maximum with its
   rank, batch, sample, channel, and spatial coordinate.
5. Reject excessive tail ratios, same-stage scale growth, absolute input
   scales, non-finite gradients, non-finite validation outputs, and validation
   embedding magnitudes over `1e6`.
6. Before a difficult blend, condition only the pending activation's upstream
   graph at 1% backbone LR with range-loss weight 1.0. Recalibrate strictly at
   the blend boundary and floor `S_i` from the observed tail when the required
   expansion remains within its explicit cap.
7. Blend gradually, recalibrate downstream BatchNorm statistics, save one
   checkpoint per completed activation, and run the full strict IJB-C evaluator
   rather than accepting LFW/CFP-FP/AgeDB alone.
8. For the sixth activation, use one local-fit epoch, a strict tail-floor scale
   of `29.69291`, a two-epoch blend, then fine-tune the fixed-six graph with a
   frozen baseline-NL13 embedding teacher and 10% backbone LR. This recovered
   TAR at FAR `1e-4` from 84.72% to 92.69%.

The primary configurations are:

```text
configs/ms1mv3_r50_nl13_prelu_herpn_scaled.py
configs/ms1mv3_r50_nl13_prelu_herpn_scaled_recover_group03.py
configs/ms1mv3_r50_nl13_prelu_herpn_scaled_recover_group03_resume.py
configs/ms1mv3_r50_nl13_prelu_herpn_scaled_group06_accuracy_recovery.py
configs/ms1mv3_r50_nl13_prelu_herpn_scaled_group06_accuracy_resume.py
configs/ms1mv3_r50_nl13_prelu_herpn_scaled_group06_accuracy_finetune.py
```

## Why earlier replacement attempts failed

The large dataset is part of the explanation, but dataset size alone is not
the root cause.

- A quadratic has unbounded `x^2` growth outside its fitted interval. A rare
  activation that is harmless under PReLU can become much larger under the
  polynomial, and successive quadratic sites compound the error.
- Mean losses and ordinary validation sets under-sample extreme tails. The
  unscaled run looked normal on LFW while CFP-FP/AgeDB magnitudes rose through
  tens, hundreds, and thousands; it finally hit a validation embedding of
  `9.9331e6` after the second grouped conversion.
- The 5.18-million-image training set and 469,375-image IJB-C evaluation make
  rare inputs much more likely to appear than in a small calibration sample.
  This supports the unexpected-input hypothesis, but the failure requires the
  polynomial's unsafe extrapolation and its composition through later layers.
- Progressive conversion changes downstream BatchNorm input distributions.
  Without causal interval measurement and post-conversion BN recalibration, an
  interval fitted on the original PReLU graph becomes stale.
- Increasing local-fit duration is not always a cure. For `layer3.6.prelu`,
  extending conditioning changed the strict observed maximum from `655.35` to
  `22564.06` while the robust magnitude was only `39.04`. The required tail
  interval expansion became `39.74x`, so the guard correctly stopped before
  blending group 8.
- Numerical finiteness is weaker than recognition quality. The fast group-6
  and group-7 runs produced zero non-finite IJB-C rows but only 84.72% and
  77.52% TAR, respectively. Slow blending and embedding recovery rescued group
  6, but the same stronger treatment only raised group 7 from 84.24% after one
  recovery epoch to 87.36% after four.
- Extending `num_epoch` while restoring an already exhausted polynomial LR
  scheduler silently leaves optimizer groups at exactly zero LR. The first
  fixed-seven recovery attempt was therefore flat at loss 44.7 and 73.33%
  LFW. The opt-in `resume_rebase_lr_scheduler` fix rebased the extended run at
  step 50,580 (base LR `3.0769e-4`, effective backbone LR `3.0769e-5`), rapidly
  restored LFW, and produced the meaningful final 87.36% IJB-C result.
- The seventh activation is an accuracy boundary rather than a non-finite
  boundary. Its slow blend stayed finite, but loss rose to about 45 and the
  pre-clip gradient norm exceeded 9,000. Four real-LR recovery epochs reduced
  loss to about 14.4, yet did not recover the required low-FAR separation.

## Strict result artifacts

```text
work_dirs/ms1mv3_r50_nl13_prelu_herpn_scaled/
  ijbc_scaled_group03/nl13_herpn_scaled_group03/ijbc_tar_at_far.csv

work_dirs/ms1mv3_r50_nl13_prelu_herpn_scaled_recover_group03/
  ijbc_scaled_recover_group04/nl13_herpn_scaled_recover_group04/ijbc_tar_at_far.csv
  ijbc_scaled_recover_group05/nl13_herpn_scaled_recover_group05/ijbc_tar_at_far.csv
  ijbc_scaled_recover_group06/nl13_herpn_scaled_recover_group06/ijbc_tar_at_far.csv
  ijbc_scaled_recover_group07/nl13_herpn_scaled_recover_group07/ijbc_tar_at_far.csv

work_dirs/ms1mv3_r50_nl13_prelu_herpn_scaled_group06_accuracy_recovery/
  ijbc_group06_accuracy_final/nl13_herpn_group06_accuracy_final/ijbc_tar_at_far.csv
  ijbc_group07_accuracy_epoch11/nl13_herpn_group07_accuracy_epoch11/ijbc_tar_at_far.csv
  ijbc_group07_accuracy_final/nl13_herpn_group07_accuracy_final/ijbc_tar_at_far.csv
```

## Why the frontier stops at six

The seventh replacement completed the strongest practical recovery attempted:
strict tail-aware scale `135.2361`, two-epoch blend, BatchNorm recalibration,
and four fixed-graph epochs with baseline embedding distillation and a rebased
non-zero LR. Its final full IJB-C vector was
`[71.79, 80.35, 87.36, 92.23, 95.71, 97.90]` percent with zero non-finite rows.
It therefore misses the requested low-FAR gate by 2.64 percentage points.

Attempting group 8 is less promising and was stopped by the safety guard before
blend. One strict pass measured observed input `655.35`, robust input `15.74`,
and required `2.861x` interval expansion beyond its then-allowed `2x` cap.
Extending conditioning to two epochs made the tail dramatically worse:
observed input `22564.06`, robust input `39.04`, tail ratio `288.99`, and
required expansion `39.74x` beyond a `3x` cap. This is direct evidence that
more fitting can move rare tails outward rather than cure them. Blending group
8 would violate the explicit numerical-safety policy and is not justified
after group 7 already fails accuracy.
