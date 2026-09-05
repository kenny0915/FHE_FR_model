# MS1Mv3-only adversarial range-margin recovery (2026-09-05)

## Outcome

Starting from the 25/25-poly phase1 epoch-23 checkpoint, the final model passes
a full MS1Mv3 original-plus-horizontal-flip scan with both:

- 0 / 10,359,020 non-finite output embeddings; and
- 0 / 10,359,020 samples whose input to any of the 25 HerPN activations
  exceeds the predeclared safety interval `[-4, 4]`.

The largest observed absolute pre-HerPN input is 2.2546024.  The inference
activation remains an exact channel-wise degree-2 polynomial approximating
PReLU on `[-6, 6]`; neither a clamp nor an adversarial branch is saved in the
inference graph.

## Motivation and protocol

The preceding MS1Mv3-only calibration reached zero non-finite MS1Mv3 outputs,
but its first IJB-C evaluation still found 116 / 938,750 non-finite augmented
embeddings.  Merely remaining finite on the training distribution therefore
left too little range margin for a shifted evaluation tail.

This experiment did not use an IJB image, label, embedding, or score for
gradient updates or checkpoint selection.  It used the following predeclared
MS1Mv3-only protocol:

1. Mine the top 4,096 original/flip inputs independently at each of all 25
   HerPN activation sites from phase1 epoch 23.
2. Replay these natural hard tails and valid-pixel PGD variants.  The PGD
   radius is `16/255` on the model's normalized `[-1, 1]` input, equivalent to
   `8/255` in ordinary `[0, 1]` pixels; it uses three `4/255` steps and a
   random start.
3. Update all convolution weights, ordinary BN affine parameters, and HerPN
   coefficients together.  Freeze all BN running buffers, the final
   projection/final BN, and dormant PReLU parameters.
4. Preserve ordinary behavior with a frozen phase-start teacher, conflict-aware
   gradients, per-group clipping, per-step update limits, and a 1% relative
   parameter trust region.
5. After each epoch, scan all 5,179,510 MS1Mv3 sources in both orientations.
   A checkpoint passes only when the union of output non-finite rows and any
   pre-HerPN input outside `[-4, 4]` is empty.

The approximation interval is still `[-6, 6]`; `[-4, 4]` is a deliberately
stricter numerical guard interval rather than a new activation approximation.
The straight-through training clamp and PGD construction are training-only.

## Full-dataset gates

The broad run started directly from phase1 epoch 23:

| Epoch | Output non-finite | Outside `[-4,4]` | Numerical union |
|---:|---:|---:|---:|
| 1 | 71 | 75 | 75 |
| 2 | 48 | 59 | 59 |
| 3 | 43 | 46 | 46 |
| 4 | 38 | 51 | 51 |
| 5 | 37 | 47 | 47 |

Epoch 3 was selected by the predeclared numerical-union gate.  A priority
focus run then oversampled those 46 failures while retaining 64 high-tail
background examples per activation:

| Focus epoch | Output non-finite | Outside `[-4,4]` | Numerical union |
|---:|---:|---:|---:|
| 1 | 10 | 15 | 15 |
| 2 | 12 | 15 | 15 |
| 3 | 2 | 3 | 3 |
| 4 | 4 | 6 | 6 |
| 5 | 0 | 0 | 0 |

The non-monotonic middle epochs show that different boundary samples can trade
places; full-dataset selection is necessary and output-only counts are not a
sufficient gate.

## Checkpoint integrity

- All 800 saved tensors are finite.
- `model_epoch_05.pt` and `model_numerical_gate_zero.pt` are tensor-wise
  identical (800 / 800 tensors).
- BN running buffers are bitwise equal to phase1 epoch 23 (462 / 462 tensors).
- Frozen projection/final-BN/dormant-PReLU parameters are bitwise equal to
  phase1 epoch 23 (32 / 32 tensors).
- The largest per-tensor change using `max(||parameter||_2, 1)` as denominator
  is 0.7007%, within the 1% trust region.
- Across the 25 folded quadratic terms, the final-to-phase1 L2 ratios range
  from 0.9902 to 1.0307 (mean 0.9993).  The numerical gate was not passed by
  collapsing the model to effectively linear activations.

## Final IJB-C evaluation

The final IJB-C run is a single held-out evaluation after MS1Mv3 checkpoint
selection.  No further model change is permitted based on this result.

It found 108 / 938,750 non-finite augmented embeddings (0.01150%), affecting
69 / 469,375 image rows after original/flip aggregation.  Consequently, the
TAR values below use the evaluator's diagnostic zero replacement and are not
a strict IJB-C accuracy result:

| FAR | 1e-6 | 1e-5 | 1e-4 | 1e-3 | 1e-2 | 1e-1 |
|---|---:|---:|---:|---:|---:|---:|
| Diagnostic TAR (%) | 84.06 | 91.64 | 94.70 | 96.53 | 97.91 | 98.79 |

The preceding MS1Mv3-only checkpoint had 116 non-finite augmented embeddings.
All 108 failures here are a subset of those 116: the stronger margin removed
eight failures and introduced none relative to that model.  However, it did
not resolve any of the 104 residual rows that were already non-finite in the
original phase1 epoch-23 IJB-C scan.  The remaining failures first overflow in
the stage-3 quadratic `bn2` branch: 35 at `layer3.2`, 26 at `layer3.3`, 15 at
`layer3.4`, 12 at `layer3.1`, and the remaining 20 across `layer3.5`,
`layer3.8`, `layer3.12`, and `layer3.13`.

This rejects the hypothesis that a strict natural-plus-local-adversarial
MS1Mv3 range margin alone is sufficient to guarantee finite IJB-C inference.
The IJB-C tail is still outside the support represented by these MS1Mv3
examples and bounded local perturbations.

## Artifacts

- Phase1 start: `work_dirs/ms1mv3_r50_herpn_full_conversion_phase1/model_epoch_23.pt`
- Top-4,096 tail manifest: `work_dirs/ms1mv3_r50_herpn_full_conversion_phase2_ms1mv3_tail_mining_top4096/epoch23_ms1mv3_tails_top4096.json`
- Broad gates: `work_dirs/ms1mv3_r50_herpn_full_conversion_phase2_ms1mv3_robust_margin/full_gate_epoch_*.json`
- Final zero gate: `work_dirs/ms1mv3_r50_herpn_full_conversion_phase2_ms1mv3_robust_margin_focus1/full_gate_epoch_05.json`
- Final checkpoint: `work_dirs/ms1mv3_r50_herpn_full_conversion_phase2_ms1mv3_robust_margin_focus1/model_numerical_gate_zero.pt`
- Final IJB-C non-finite manifest: `work_dirs/ms1mv3_r50_herpn_full_conversion_phase2_ms1mv3_robust_margin_focus1/ijbc_ms1mv3_robust_margin_zero/nonfinite_manifest.csv`
- Final diagnostic TAR: `work_dirs/ms1mv3_r50_herpn_full_conversion_phase2_ms1mv3_robust_margin_focus1/ijbc_ms1mv3_robust_margin_zero/r50_full_poly_ms1mv3_robust_margin_zero/ijbc_tar_at_far.csv`
- Mining log (Slurm 341948): `work_dirs/ms1mv3_r50_herpn_full_conversion_phase2_ms1mv3_tail_mining_top4096/slurm-341948.out`
- Broad-run log (Slurm 342010): `work_dirs/ms1mv3_r50_herpn_full_conversion_phase2_ms1mv3_robust_margin/slurm-342010.out`
- Focus-run log (Slurm 342455): `work_dirs/ms1mv3_r50_herpn_full_conversion_phase2_ms1mv3_robust_margin_focus1/slurm-342455.out`
- Final IJB-C log (Slurm 342894): `work_dirs/ms1mv3_r50_herpn_full_conversion_phase1/ijbc-342894.out`
