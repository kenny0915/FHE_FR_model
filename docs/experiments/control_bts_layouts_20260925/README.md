# Control epoch3: five BTS rescaling layouts

Source: `work_dirs/accuracy_recovery_ijbc_20260925/inputs/control.pt`, the control epoch3 checkpoint.
SHA-256: `5e2b6e237475150e7767a8362bbae43a648d8a9bca0420020ec12db97f223657`.
Completed Slurm job 438201 (exit 0; 1m05s). No retraining or IJB-C calibration.

All layouts use the identical 1,000 MS1MV3 original-orientation images, sampled without replacement with seed 42, and target absolute maximum 0.8. FP32 with TF32 disabled. Positions are zero-based residual block **outputs** (the designated BTS inputs).

Each stage shares one coordinate scale to preserve identity shortcuts without new inference operators. Stage 1 remains scale 1 in the 5-BTS layout because that stage has no selected boundary. No change to the 25 degree-2 activations or their one-multiplication-level-per-activation cost.

The initial approximation target remains per-channel PReLU, with per-channel interval transformed from `[-lam_fit, lam_fit]` to `[-s*lam_fit, s*lam_fit]`; `q_new(s*x)=s*q_old(x)`. This is frozen-BN inference equivalence, not a training-resume transformation.

## Requested layouts

| Version | Checkpoint directory | Boundaries (prefix `layer` omitted) |
| --- | --- | --- |
| Original (6) | `bts6/student_scaled.pt` | 1.2, 2.3, 3.3, 3.7, 3.11, 4.1 |
| #1 (14) | `bts14/student_scaled.pt` | 1.0, 1.2, 2.1, 2.3, 3.0, 3.2, 3.4, 3.6, 3.8, 3.10, 3.12, 3.13, 4.0, 4.2 |
| #2 (9) | `bts9/student_scaled.pt` | 1.1, 2.1, 2.3, 3.1, 3.4, 3.7, 3.10, 3.13, 4.1 |
| #3 (7) | `bts7/student_scaled.pt` | 1.2, 2.3, 3.2, 3.6, 3.10, 3.13, 4.2 |
| #4 (5) | `bts5/student_scaled.pt` | 2.0, 3.0, 3.5, 3.10, 4.0 |

All checkpoint directories are below `work_dirs/control_bts_layouts_20260925/exports/`.

## Verified exported checkpoints

| Layout | Stage 1 scale | Stage 2 scale | Stage 3 scale | Stage 4 scale | Max embedding absolute error | Max relative L2 error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| bts6 | 0.1891087095 | 0.2562613212 | 0.07410368251 | 0.4499699445 | 4.02927e-05 | 1.15639e-05 |
| bts14 | 0.1891087095 | 0.2562613212 | 0.06356061552 | 0.8646066237 | 4.76837e-05 | 1.14174e-05 |
| bts9 | 0.1653409373 | 0.2562613212 | 0.06356061552 | 0.4499699445 | 4.21405e-05 | 1.11413e-05 |
| bts7 | 0.1891087095 | 0.2562613212 | 0.06356061552 | 0.8646066237 | 4.76837e-05 | 1.14174e-05 |
| bts5 | 1 | 0.3246474002 | 0.0789593483 | 1 | 3.46303e-05 | 9.47277e-06 |

All five exported checkpoints were reloaded and compared with the unscaled source over all 1,000 calibration images. Both final embeddings and each selected boundary passed `assert_close(atol=1e-4, rtol=1e-4)`. All measured embeddings and boundaries were finite; all selected rescaled boundaries were inside [-1,1] on calibration images.

The 14- and 7-BTS layouts select the same four stage scales on this calibration set. Their BTS observation schedules remain different.

Each checkpoint records its boundary names in `residual_graph_scaling.boundaries`. The updated `eval_ijbc.py --bts-failure-report` reads this metadata automatically; legacy checkpoints without it keep the six-boundary default. Use the updated evaluator, not an old frozen snapshot, for these exports.

Validation covers these calibration images only: no held-out range guarantee, new full IJB-C TAR, or encrypted bootstrapping result is claimed. The unscaled control had 4 nonfinite IJB-C source images; parameter rescaling does not establish their removal. BTS-range filtering must be evaluated separately for each layout.

Reproduction: submit `controlled_degree2/control_bts_layouts.slurm` with `CHECKPOINT`, `DATASET_ROOT`, and a fresh `BTS_OUTPUT_ROOT`. Full inputs and code revision are in [submission.json](submission.json); per-boundary ranges, equivalence metrics, and output hashes are in [calibration_reports.json](calibration_reports.json).

Tests: 32 passed (`test_bts_failure_audit`, `test_rescale_residual_graph`, `test_layer_statistics`, `test_ordered_prefetch`), covering all layouts, absent-stage scaling, invalid boundaries, nonfinite ranges, dynamic hooks, whole-source zeroing, residual graph equivalence, and serialization.
