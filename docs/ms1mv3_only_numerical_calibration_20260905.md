# MS1Mv3-only numerical calibration (2026-09-05)

## Scope

- Start checkpoint: `work_dirs/ms1mv3_r50_herpn_full_conversion_phase1/model_epoch_23.pt`
- Final checkpoint: `work_dirs/ms1mv3_r50_herpn_full_conversion_phase2_ms1mv3_focus1/model_numerical_gate_zero.pt`
- Approximation target: channel-wise PReLU on `[-6, 6]`
- Inference activation: exact degree-2 HerPN at all 25 activation sites
- Training and checkpoint selection data: MS1Mv3 only
- MS1Mv3 numerical gate: all 5,179,510 sources in both deterministic
  original/flip orientations must produce finite embeddings

No IJB-C image, label, tail manifest, or metric was read before the final
checkpoint was fixed. The IJB-C run reported below is therefore the first
evaluation of this MS1Mv3-selected checkpoint. Its TAR is nevertheless only a
diagnostic score because the checkpoint did not pass the strict IJB-C
non-finite gate.

## Method

1. Scan the complete MS1Mv3 RecordIO dataset in both deterministic
   orientations. Mine the top 64 finite pre-HerPN tails at each of the 25
   activations and reproduce all epoch-23 output failures.
2. Keep all 462 BatchNorm running buffers bitwise fixed to phase1 epoch 23.
3. Update all convolution weights, ordinary BatchNorm affine parameters, and
   HerPN weights/biases. Keep the embedding projection, final BatchNorm, and
   dormant PReLU parameters fixed.
4. Apply a training-only straight-through clamp at 6 so catastrophic examples
   retain a differentiable causal range loss. This clamp is absent from the
   saved inference/FHE graph.
5. Penalize the earliest activation whose sample maximum exceeds the
   `0.75 * 6 = 4.5` guard, while preserving normalized teacher embeddings on
   8,192 deterministic, initially normal MS1Mv3 orientations.
6. Combine preservation and tail gradients with conflict-aware projection,
   per-tensor update caps, and a 1% relative trust region.
7. Run an exact full-MS1Mv3 gate after every epoch and priority-replay the
   remaining failures. Select the lowest-failure MS1Mv3 checkpoint without
   consulting IJB-C.

The activation remains an exact channel-wise quadratic, so this calibration
does not add polynomial degree or FHE multiplicative depth.

## MS1Mv3 results

The initial mining pass found 7,283 unique hard-tail source indices and exactly
244 non-finite augmented embeddings. Training used the top-64 activation tails
plus failures as replay data.

| Checkpoint | Non-finite augmented embeddings |
|---|---:|
| Phase1 epoch 23 | 244 / 10,359,020 |
| Calibration epoch 1 | 24 / 10,359,020 |
| Calibration epoch 2 | 11 / 10,359,020 |
| Calibration epoch 3 | 6 / 10,359,020 |
| Calibration epoch 4 | 2 / 10,359,020 |
| Calibration epoch 5 | 3 / 10,359,020 |
| **Priority focus epoch 1, starting from epoch 4** | **0 / 10,359,020** |

Epoch 5 was not selected because its full MS1Mv3 gate regressed from 2 to 3.
The priority continuation started from epoch 4, repeated its two remaining
failures 64 times, and reached zero after 500 update steps. No training step
was skipped for a non-finite embedding, loss, or gradient.

## First IJB-C evaluation

Slurm job 341790 evaluated the fixed MS1Mv3-only checkpoint on all 469,375
IJB-C sources in both orientations:

- Non-finite augmented embeddings: **116 / 938,750 (0.01236%)**
- Affected IJB-C sources: **77 / 469,375 (0.01640%)**
- Epoch-23 failures resolved: 330 / 434
- Epoch-23 failures still present: 104 / 434
- Newly failing orientations: 12

All 116 failures first became non-finite in the quadratic `bn2` branch of a
stage-3 HerPN activation. The first-failure distribution was:

| First non-finite module | Rows |
|---|---:|
| `layer3.2.prelu.herpn.bn2` | 54 |
| `layer3.4.prelu.herpn.bn2` | 16 |
| `layer3.3.prelu.herpn.bn2` | 15 |
| `layer3.1.prelu.herpn.bn2` | 12 |
| `layer3.6.prelu.herpn.bn2` | 8 |
| `layer3.7.prelu.herpn.bn2` | 7 |
| `layer3.9.prelu.herpn.bn2` | 4 |

The finite tensor immediately before the first failing `bn2` already had an
absolute maximum between `6.82e19` and `2.96e36` (median `3.04e26`). Thus the
remaining issue is quadratic tail amplification under IJB-C distribution
shift, not a changed BN running-statistic buffer.

The evaluator replaces non-finite values with zero only to finish its
diagnostic ROC calculation. Therefore the following numbers are **not strict
IJB-C accuracy**:

| FAR | 1e-6 | 1e-5 | 1e-4 | 1e-3 | 1e-2 | 1e-1 |
|---|---:|---:|---:|---:|---:|---:|
| Diagnostic TAR (%) | 85.57 | 92.03 | 94.85 | 96.63 | 97.90 | 98.83 |

Compared with phase1 epoch 23, the IJB-C failure count fell from 434 to 116
(73.3% reduction), but an MS1Mv3 zero gate alone does not guarantee an IJB-C
zero gate. This is direct evidence that the rare numerical tail is partly
dataset-specific.

## Integrity checks

- BatchNorm running buffers equal epoch 23: 462 / 462 tensors, bitwise exact.
- Frozen projection/final-BN/dormant-PReLU parameters equal epoch 23: 32 / 32
  tensors, bitwise exact.
- Largest per-tensor change using the training trust-region scale
  `max(||parameter||_2, 1)` relative to epoch 23: 0.4097%.
- The 25 folded quadratic-coefficient L2 ratios relative to epoch 23 range
  from 0.9923 to 1.0063 (mean 0.9981). The quadratic terms were not collapsed
  into effectively linear activations.
- Every tensor in the saved checkpoint is finite.

## Artifacts

- Tail manifest: `work_dirs/ms1mv3_r50_herpn_full_conversion_phase2_ms1mv3_tail_mining_all25/epoch23_ms1mv3_tails_all25.json`
- Final MS1Mv3 zero gate: `work_dirs/ms1mv3_r50_herpn_full_conversion_phase2_ms1mv3_focus1/full_gate_epoch_01.json`
- Final checkpoint: `work_dirs/ms1mv3_r50_herpn_full_conversion_phase2_ms1mv3_focus1/model_numerical_gate_zero.pt`
- IJB-C non-finite manifest: `work_dirs/ms1mv3_r50_herpn_full_conversion_phase2_ms1mv3_focus1/ijbc_ms1mv3_only_zero/nonfinite_manifest.csv`
- IJB-C diagnostic TAR: `work_dirs/ms1mv3_r50_herpn_full_conversion_phase2_ms1mv3_focus1/ijbc_ms1mv3_only_zero/r50_full_poly_ms1mv3_only_zero/ijbc_tar_at_far.csv`
- Mining log (Slurm 341537): `work_dirs/ms1mv3_r50_herpn_full_conversion_phase2_ms1mv3_tail_mining_all25/slurm-341537.out`
- Calibration log (Slurm 341538): `work_dirs/ms1mv3_r50_herpn_full_conversion_phase2_ms1mv3_calibration/slurm-341538.out`
- Priority-focus log (Slurm 341776): `work_dirs/ms1mv3_r50_herpn_full_conversion_phase2_ms1mv3_focus1/slurm-341776.out`
- IJB-C evaluation log (Slurm 341790): `work_dirs/ms1mv3_r50_herpn_full_conversion_phase1/ijbc-341790.out`
