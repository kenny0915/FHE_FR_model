# Progressive-final polish A/B/C/D (2026-09-15)

User requested four parallel short experiments, verified runnable before submission,
then no ongoing job monitoring. This campaign does not use IJB images/labels in
training, routing, hyperparameter selection, or checkpoint selection.

## Fixed starting point and common settings

- `work_dirs/controlled_degree2_tail_ms1mv3_20260907/progressive/student_final.pt`
- Teacher: `work_dirs/ms1mv3_r50/model.pt`.
- 2 epochs per arm; 4 H200s per arm, up to 16 concurrent training GPUs.
- Global batch 2048 = 4 GPUs x 128 x accumulation 4. FP32 in **every** arm.
  A is a controlled rerun of the old objective, not a bit-exact BF16 replay.
- SGD/Nesterov .9; base LR .002, .5-epoch warmup, cosine floor .01;
  weight decay .0005; gradient norm limit 5. Loss/gradient nonfinites abort
  before an update. No automatic retries, sanitization, or strategy changes.
- Same seed 20260915, source order and excluded audit rows. Augmentation and
  model-dependent replay can diverge across arms, so training is not bit-paired.
- Polynomial coefficients, fitting intervals and regularization radii fixed.
  Original target is each channel's PReLU on saved `[-lam_fit,lam_fit]`;
  inference never clips. No new inference operators or multiplicative depth.
- Original polish augmentation: lowres .2, photo .2, crop .1, stress .4,
  pathology .05; online training-tail replay capacity 1024, fraction .25,
  mix begins at step 100. Pathology excluded from KD, retained for range loss.
- Original objective: cosine KD + hint (1 -> .3) + range (ramp ->1)
  + causal-tail (1) + .0001 operator penalty, margin .1.

## Arms

| Arm | Causal scope | BN running moments | Student training path |
|---|---|---|---|
| A | layer3 only | adaptive | original clipped polish |
| B | all 25 sites | adaptive | same clipped polish |
| C | all 25 sites | fixed | same clipped polish |
| D | all 25 sites | fixed | safe-row unclipped KD + finite-prefix repair |

D first does a no-gradient eval-mode unclipped probe. Each source row is
assigned to its first input above 4x its channel fitting radius. Safe rows
receive the same ordinary loss through an unclipped graph. Unsafe rows are
grouped by that boundary; a fresh differentiable forward stops **before**
the selected square, and penalizes finite inputs relative to 1x fitting radius:
mean squared excess + .01 times per-row maximum squared excess. Losses are
weighted by the fraction of original microbatch rows in each route. No graph
with an invalid probe output is differentiated. A nonfinite prefix or an
unroutable failure aborts rather than inventing a valid gradient. The 4x/1x
thresholds are experiment choices, not proven safe bounds or inference clamps.
The safe-row proportion and repair-row proportion are saved for each epoch.
Unexpected nonfinite gradients still abort; the routing rule is not a theorem
of backward finiteness.

## Evaluation and provenance

1. One shared baseline job scans all MS1MV3 original/flip inputs and evaluates
   full IJB-C. Training waits for this job to finish, but consumes only its
   **MS1MV3** manifest. IJB-C scores are report-only, not a gate or fitting input.
2. Each training arm evaluates before training and after each epoch:
   - full LFW canary (a failed finite gate is recorded, not silently scored);
   - the same 8192 deterministic MS1MV3 sources / 16384 original+flip rows,
     excluded from this fine-tuning. The initialization/teacher may have seen
     them previously; this is not a newly unseen generalization dataset;
   - fixed baseline MS1MV3 failures/high-tail rows, observed only as diagnostics.
     These can overlap training and are explicitly not held-out accuracy.
   - all module-boundary finite counts and per-site maximum input/radius.
3. Every arm saves `epoch1.pt` and `epoch2.pt`. No mutable best/final alias.
   Checkpoints remain diagnostic until full evaluation.
4. Dependent final jobs scan **full MS1MV3 and full IJB-C**, only for fixed
   epoch2. Export uses 25 channelwise pure quadratics, BN affine folding,
   original/flip and remainder finite audit. Sanitized TAR is diagnostic;
   `valid_tar_at_far_1e4` is null if any IJB-C module/embedding is nonfinite.

Code is frozen in a git-archive snapshot for all production jobs, with shared
dataset symlinks. Configs record source and teacher hashes and audit-row hashes.
Jobs have 4-hour hard caps and no continuation daemon. Evaluation dependencies
use `aftercorr` for the matching training array task, so an unsuccessful arm
cannot evaluate an absent or stale checkpoint. No running training is launched
on the checkout host.

CPU tests cover scope/BN switches, mixed safe/repair rows, stopping before an
unsafe square, fixed coefficients and pathological-row repair gradients.
A bounded four-GPU smoke exercises each arm before production submission.
Actual GPU smoke/submission IDs and limitations are recorded separately in
`polish_abcd_submission.json`.
