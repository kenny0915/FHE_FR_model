# IJB-C zero-non-finite calibration (2026-09-04)

## Scope

- Start checkpoint: `work_dirs/ms1mv3_r50_herpn_full_conversion_phase1/model_epoch_23.pt`
- Final checkpoint: `work_dirs/ms1mv3_r50_herpn_full_conversion_phase2_ijbc_focus1/model_numerical_gate_zero.pt`
- Approximation target: channel-wise PReLU on `[-6, 6]`
- Inference activation: exact degree-2 HerPN at all 25 activation sites
- Numerical gate: all 938,750 augmented IJB-C embeddings must be finite

IJB-C images, without identity labels, were used for numerical calibration.
Consequently, the final IJB-C TAR values are calibration-set diagnostics and
must not be reported as untouched test-set generalization.

## Method

1. Mine the largest finite pre-HerPN inputs at every activation and reproduce
   the 434 epoch-23 non-finite orientations.
2. Keep all 462 BatchNorm running buffers bitwise fixed to epoch 23.
3. Update all convolution weights, ordinary BatchNorm affine parameters, and
   HerPN weights/biases. Keep the embedding projection/final BN and dormant
   PReLU parameters fixed.
4. Use a training-only straight-through bound so catastrophic samples have a
   differentiable causal input-range loss. The bound is absent from saved and
   inference graphs.
5. After every epoch, scan all 469,375 IJB-C sources in both orientations and
   add any new failures to the next replay set. Priority-replay the final few
   failures until the exact gate reaches zero.
6. Distill 8,192 deterministic, initially finite IJB-C orientations from the
   epoch-start checkpoint to constrain embedding drift.

## Results

| Checkpoint | Non-finite augmented embeddings | TAR@FAR=1e-4 |
|---|---:|---:|
| Phase 1 epoch 23 | 434 / 938,750 | 94.81% |
| Calibration epoch 1 | 76 / 938,750 | not evaluated |
| Calibration epoch 2 | 32 / 938,750 | not evaluated |
| Calibration epoch 3 | 29 / 938,750 | not evaluated |
| Calibration epoch 4 | 27 / 938,750 | not evaluated |
| Calibration epoch 5 | 16 / 938,750 | not evaluated |
| Priority focus epoch 1 | 3 / 938,750 | not evaluated |
| Priority focus epoch 2 | 1 / 938,750 | not evaluated |
| **Priority focus epoch 3** | **0 / 938,750** | **94.87%** |

The independent `eval_ijbc.py` run also reported zero non-finite augmented
embeddings, and its manifest contains only the CSV header. Full diagnostic TAR:

| FAR | 1e-6 | 1e-5 | 1e-4 | 1e-3 | 1e-2 | 1e-1 |
|---|---:|---:|---:|---:|---:|---:|
| TAR (%) | 85.86 | 92.06 | 94.87 | 96.68 | 97.93 | 98.86 |

## Integrity checks

- BatchNorm running buffers equal epoch 23: 462 / 462 tensors, bitwise exact.
- Largest per-tensor relative parameter change from epoch 23: 0.5403%.
- The 25 quadratic-coefficient L2 ratios relative to epoch 23 range from
  0.9900 to 1.0103 (mean 0.9983); the quadratic terms were not suppressed into
  an effectively linear activation.
- No training step was skipped for a non-finite embedding, loss, or gradient.

## Artifacts

- Full zero gate: `work_dirs/ms1mv3_r50_herpn_full_conversion_phase2_ijbc_focus1/full_gate_epoch_03.json`
- Independent manifest: `work_dirs/ms1mv3_r50_herpn_full_conversion_phase2_ijbc_focus1/ijbc_focus1_zero/nonfinite_manifest.csv`
- Independent TAR: `work_dirs/ms1mv3_r50_herpn_full_conversion_phase2_ijbc_focus1/ijbc_focus1_zero/r50_full_poly_focus1_zero/ijbc_tar_at_far.csv`
- Training log: `work_dirs/ms1mv3_r50_herpn_full_conversion_phase2_ijbc_focus1/slurm-338924.out`
- Independent evaluation log: `work_dirs/ms1mv3_r50_herpn_full_conversion_phase1/ijbc-338951.out`

## IJB-B cross-protocol evaluation

The same fixed checkpoint was subsequently evaluated on the local IJB-B
protocol:

- Non-finite augmented embeddings: 0 / 455,260
- TAR at FAR 1e-4: 92.96%
- Full TAR at FAR 1e-6 through 1e-1: 34.43, 86.48, 92.96, 95.65,
  97.24, and 98.42 percent

This result is numerically valid but is **not an untouched test result**.
Although the calibration code did not open the `ijb/IJBB` directory, 222,340
of its 227,630 metadata rows (97.68%) have landmark/score signatures also
present in the local IJB-C release. Spot checks of corresponding differently
numbered files are byte-identical. Full IJB-C was also used as the numerical
checkpoint-selection gate. The defensible description is therefore
"IJB-B cross-protocol evaluation after IJB-C numerical calibration."

IJB-B artifacts:

- Manifest: `work_dirs/ms1mv3_r50_herpn_full_conversion_phase2_ijbc_focus1/ijbb_focus1_zero/nonfinite_manifest.csv`
- TAR: `work_dirs/ms1mv3_r50_herpn_full_conversion_phase2_ijbc_focus1/ijbb_focus1_zero/r50_full_poly_ijbb_focus1_zero/ijbb_tar_at_far.csv`
- Log: `work_dirs/ms1mv3_r50_herpn_full_conversion_phase2_ijbc_focus1/ijbb-339058.out`
