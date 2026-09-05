# WIDER + MS1Mv3 IJB-free numerical calibration (2026-09-06)

## Objective and reporting boundary

The experiment starts from the exact 25/25 degree-2 HerPN phase1 checkpoint
and attempts to remove IJB-C non-finite embeddings without using IJB images
for gradients or checkpoint selection.

- Start: `work_dirs/ms1mv3_r50_herpn_full_conversion_phase1/model_epoch_23.pt`
- Approximation target: channel-wise PReLU on `[-6, 6]`
- Accepted numerical range: every pre-HerPN input must satisfy `|x| <= 4`
- Inference graph: 25 degree-2 HerPN activations; no activation was reduced to
  degree one
- Training/selection data: WIDER FACE train and MS1Mv3 only
- IJB-C: evaluated once after the non-IJB checkpoints and gates were fixed

This is accurately described as **IJB-C-free fine-tuning**. It is not a claim
that the whole research program never observed IJB-C: aggregate results from
earlier models were already known before this experiment.

## WIDER protocol

WIDER FACE supplies bounding boxes but not five-point landmarks. The protocol
therefore uses deterministic bbox stress crops rather than claiming ArcFace
landmark alignment:

1. Parse `wider_face_train_bbx_gt.txt`, including its four official zero-face
   sentinel records.
2. Reject invalid boxes and faces whose minimum side is below 20 pixels.
3. Make a square crop centered on the box, with side
   `1.35 * max(width, height)`, constant padding value 127.5, and resize to
   112 by 112.
4. Convert BGR to RGB and normalize exactly as `uint8 / 127.5 - 1`.
5. Use SHA-256 of the source scene path for a source-image-level 10-fold
   split. Fold 0 is held-out validation; folds 1--9 are calibration. Faces
   from the same scene cannot leak across the split.

The resulting sets are:

| Split | Face crops | Source scenes | Evaluated orientations |
|---|---:|---:|---:|
| Calibration | 57,361 | 11,208 | original + flip during mining/training |
| Held-out validation | 5,742 | 1,202 | 11,484 |

Tail mining on phase1 epoch 23 retained the top 2,048 inputs at each of the
25 activations. Their union contained 20,120 hard crops. It found 37
non-finite orientations from 22 crops; visual inspection showed mostly clear
but high-contrast or edited faces, masks/helmets/occlusions, and poster/text
overlays rather than tiny-box interpolation artifacts.

## Calibration method

All runs keep the exact inference graph and use the following training-only
recovery procedure:

- Freeze all 154 BatchNorm modules' 462 running-statistic buffers.
- Update all convolution weights, ordinary BN affine parameters, and HerPN
  weights/biases.
- Freeze the embedding projection, final BN tensors, and dormant PReLU
  weights.
- Use conflict-aware clean-distillation and causal earliest-layer range
  gradients, bounded local pixel adversaries, per-tensor update clipping, and
  a 1% relative trust region.
- Select checkpoints only with exact full-dataset original-plus-flip gates.

The WIDER-only calibration reached a held-out WIDER zero gate in one epoch:

| Model | WIDER non-finite | WIDER range failures | Max pre-HerPN |
|---|---:|---:|---:|
| WIDER calibration epoch 1 | 0 / 11,484 | 0 / 11,484 | 2.1794 |

However, its exact full MS1Mv3 gate still had 188 non-finite embeddings and
199 numerical failures among 10,359,020 orientations. WIDER alone therefore
did not cover the MS1Mv3 tail.

The next pass replayed those MS1Mv3 failures 128 times plus a fixed top-64
per-activation MS1Mv3 background:

| Focus1 checkpoint | Non-finite | Range failures | Numerical union |
|---|---:|---:|---:|
| WIDER start | 188 | 199 | 199 |
| Epoch 1 | 14 | 15 | 15 |
| Epoch 2 | 6 | 7 | 7 |
| Epoch 3 | 8 | 8 | 8 |
| Epoch 4 | 2 | 2 | **2** |
| Epoch 5 | 2 | 4 | 4 |

The non-monotonic counts came from boundary samples trading places, not from
skipped gradients: all training runs reported zero skipped steps. All five
focus1 checkpoints retained a zero held-out WIDER gate, with maximum
pre-HerPN values between 1.936 and 2.067.

The final cumulative focus starts from the best focus1 checkpoint (epoch 4),
replays the union of all 26 orientations that failed in focus1, and halves
the per-tensor step limit from `1e-5` to `5e-6`. With 256 repeats plus the
fixed background, its 7,273-row replay set is covered in one 300-step epoch.

| Final checkpoint | Dataset | Non-finite | Range failures | Max pre-HerPN |
|---|---|---:|---:|---:|
| Focus2 epoch 1 | Full MS1Mv3 | **0 / 10,359,020** | **0** | 2.7612 |
| Focus2 epoch 1 | Held-out WIDER | **0 / 11,484** | **0** | 1.9728 |

## Checkpoint integrity

The reproducible report in `checkpoint_integrity.json` verifies:

- all 800 / 800 checkpoint tensors are finite;
- `model_epoch_01.pt` and `model_numerical_gate_zero.pt` are bitwise equal for
  all 800 state tensors;
- all 462 / 462 BN running buffers are bitwise equal to phase1 epoch 23;
- all 32 / 32 frozen projection/final-BN/dormant-PReLU tensors are bitwise
  equal to phase1 epoch 23;
- the largest per-tensor relative parameter change from phase1 epoch 23 is
  0.4326%, at `layer2.0.bn1.weight`, below the 1% trust region;
- the 25 folded quadratic-coefficient L2 ratios relative to phase1 range from
  0.9911 to 1.0081 (mean 0.9981), so the numerical result was not obtained by
  collapsing the activations into effectively linear functions.

## Single final IJB-C result

Slurm job 345283 completed normally after processing all 469,375 sources in
both orientations.

| Checkpoint | Non-finite augmented embeddings | Affected source rows | Diagnostic TAR@FAR=1e-4 |
|---|---:|---:|---:|
| Phase1 epoch 23 | 434 / 938,750 | 308 | 94.81% |
| Previous MS1Mv3 robust zero | 108 / 938,750 | 69 | 94.70% |
| **WIDER + MS1Mv3 focus2** | **97 / 938,750** | **62** | **94.73%** |

Full final diagnostic TAR values are:

| FAR | 1e-6 | 1e-5 | 1e-4 | 1e-3 | 1e-2 | 1e-1 |
|---|---:|---:|---:|---:|---:|---:|
| TAR (%) | 85.38 | 91.70 | 94.73 | 96.53 | 97.92 | 98.85 |

These TAR values are diagnostic: the evaluator replaced 49,664 non-finite
feature values across 62 aggregated image rows with zero. They must not be
reported as a strict IJB-C accuracy result.

All 97 final failed orientations are subsets of both the phase1 set of 434
and the previous MS1Mv3-robust set of 108. The method removed 337 / 434
(77.6%) relative to phase1 and another 11 / 108 (10.2%) relative to the
previous non-IJB calibration, while introducing no new IJB-C failures.

The remaining first-overflow sites are all stage-3 quadratic branches:

| First non-finite module | Orientations |
|---|---:|
| `layer3.2.prelu.herpn.bn2` | 43 |
| `layer3.3.prelu.herpn.bn2` | 19 |
| `layer3.4.prelu.herpn.bn2` | 14 |
| `layer3.5.prelu.herpn.bn2` | 12 |
| `layer3.1.prelu.herpn.bn2` | 8 |
| `layer3.6.prelu.herpn.bn2` | 1 |

## Conclusion

WIDER calibration is useful: it reduced the residual IJB-C failures from 108
to 97 without introducing a new failed orientation, and cumulative small-step
focus produced strict zero gates on both non-IJB datasets. It is nevertheless
**not sufficient to achieve the target IJB-C zero gate**. Passing natural
MS1Mv3 and held-out WIDER gates does not certify the unseen IJB alignment and
capture tail; a non-constant quadratic remains globally unbounded.

Any later experiment may still be trained without IJB samples, but after this
measurement it should not be marketed as an untouched IJB-C development
decision. Defensible next directions are predeclared, non-IJB stress families
(alignment jitter, padding, contrast/gamma, blur/compression and bounded pixel
attacks) or an architecture-level range guarantee using only FHE-compatible
fixed affine scaling/operator-norm constraints. The 97 IJB-C rows should not
be replayed if the goal is IJB-free training.

## Artifacts and jobs

- WIDER tail manifest: `work_dirs/ms1mv3_r50_herpn_full_conversion_phase2_wider_tail_mining/epoch23_wider_tails_top2048.json`
- WIDER-only zero checkpoint: `work_dirs/ms1mv3_r50_herpn_full_conversion_phase2_wider_robust_margin/model_numerical_gate_zero.pt`
- WIDER-only full-MS gate: `work_dirs/ms1mv3_r50_herpn_full_conversion_phase2_wider_robust_margin/ms1mv3_gate/model_numerical_gate_zero_ms1mv3_gate.json`
- Focus1 gates: `work_dirs/ms1mv3_r50_herpn_full_conversion_phase2_wider_ms1mv3_focus1/full_gate_epoch_*.json`
- Final full-MS zero gate: `work_dirs/ms1mv3_r50_herpn_full_conversion_phase2_wider_ms1mv3_focus2/full_gate_epoch_01.json`
- Final checkpoint: `work_dirs/ms1mv3_r50_herpn_full_conversion_phase2_wider_ms1mv3_focus2/model_numerical_gate_zero.pt`
- Integrity report: `work_dirs/ms1mv3_r50_herpn_full_conversion_phase2_wider_ms1mv3_focus2/checkpoint_integrity.json`
- IJB-C manifest: `work_dirs/ms1mv3_r50_herpn_full_conversion_phase2_wider_ms1mv3_focus2/ijbc_wider_ms1mv3_ijbfree_zero/nonfinite_manifest.csv`
- IJB-C TAR: `work_dirs/ms1mv3_r50_herpn_full_conversion_phase2_wider_ms1mv3_focus2/ijbc_wider_ms1mv3_ijbfree_zero/r50_full_poly_wider_ms1mv3_ijbfree_zero/ijbc_tar_at_far.csv`
- Jobs: WIDER mining 344845; WIDER training 344906; first full-MS gate
  344921; focus1 training 344969; focus1 WIDER gates 345094 and 345183;
  focus2 training 345224; final focus2 WIDER gate 345239; IJB-C 345283.
