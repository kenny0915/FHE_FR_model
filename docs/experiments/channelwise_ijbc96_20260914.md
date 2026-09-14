# Reproducible PReLU-to-quadratic IJB-C 96% goal

Status: CPU validation and 16-H200 smoke passed; accuracy target not achieved.

User-authorized objective: start from original PReLU iResNet50, use an exact
degree-two polynomial at all 25 activation sites, no inference clipping,
and reach unrounded IJB-C TAR >=96% at requested FAR=1e-4 with zero non-finite
intermediate values and augmented embeddings. Layer/channel-specific
coefficients and IJB-C calibration are explicitly allowed. Report the final
metric as **IJB-C calibration-set performance**, not untouched test accuracy.
No IJB-C identity/pair labels are used for gradient training in the initial
policy; numerical calibration, teacher distillation and metric-based selection
may use IJB-C. Any later change must be recorded explicitly.

## Source and initial experiment

Use only `work_dirs/ms1mv3_r50/model.pt` as model initialization. Do not load
old polynomial student weights, run10 fits or their interval buffers.
Fresh teacher calibration uses 100,000 MS1MV3 training images, an identity
split seeded 20260911, and the existing per-channel histograms. Approximate
each channel's original PReLU on [-R_c,R_c], where R_c is its absolute-input
99.95th percentile upper histogram edge times 1.5 (minimum .05). Use the
existing weighted least-squares fit with 5% uniform edge weight and initial
regularization radius .6 R_c. Save source hashes, intervals, fit errors,
teacher maxima, split and complete configuration.

Initial revised recipe: frozen teacher and BN running moments; train Conv,
BN affines, classification head and per-channel polynomial coefficients.
Warm the head for one epoch, convert one activation at a time over 12.5
epochs (half an epoch per site), then finish 24 epochs in total. This is a
fixed-duration initial curriculum, **not an adaptive numerical gate** and
not an established convergent recipe. The opened prefix is never clipped;
the current site blends PReLU/quadratic and the unopened suffix uses PReLU.
Final inference ignores all blends and uses only quadratics. An unopened
site avoids evaluating its unused square, preventing a 0*Inf blend.

FP32, global batch 2048 on 16 H200s; SGD/Nesterov with base backbone LR .001,
head LR .005, coefficient LR .00002 and range weight 1. Existing ArcFace,
teacher embedding/stage hints, training-only augmentation and tail replay
are reused. These are proposed initial settings, not proven optimal values.
Non-finite embeddings/loss/gradients abort before an optimizer update.
Any conversion failure must be investigated before retrying or extending;
do not silently continue training with sanitized features.

## Calibration and acceptance

After a reproducible converted candidate exists, use IJB-C orientations to
mine finite tails and numerical failures, and distill original teacher
embeddings while repairing the earliest unsafe polynomial inputs. The
existing HerPN-specific calibrator cannot directly consume this model.
`calibrate_ijbc_channelwise.py` provides a separate one-GPU adapter: teacher
cosine/block distillation on a temporarily bounded auxiliary graph and an
unclipped finite-prefix loss, with optional original/flip failure replay.
BN moments and range buffers stay fixed; spatial Conv/BN affine and quadratic
coefficients train. No labels are read. Saved checkpoints are diagnostic until
full evaluation; this adapter has CPU gradient/restoration tests but still
requires a real-candidate GPU smoke before production calibration.
Track accuracy and numerical stability separately. Reaching a finite gate
on MS1MV3 alone is not evidence of a finite IJB-C gate.

The final evaluator must run the exported graph itself, using
`--network r50_controlled_d2 --polynomial-coefficient-mode channelwise`,
`--polynomial-export CERTIFICATE.json --finite-audit AUDIT.json`.
The export allows only affine convolution/linear, additions, multiplications
and structural operations, rejects inference clamps and zero curvature in
any channel, and replaces BN with fixed affine constants. L2 normalization,
template/media aggregation and scoring remain in the existing plaintext
evaluation protocol. Each activation contributes one square; the deepest
path has 25 nonlinear square levels, excluding affine costs.

Acceptance requires all 469,375 source images and both orientations
(938,750 embeddings), zero audited intermediate and embedding non-finites,
the actual selected ROC FAR, the unrounded TAR sidecar, and checkpoint hash.
Compare with the original PReLU baseline using the identical evaluator.

## Execution

`channelwise_goal.slurm` defaults to a 20-minute isolated 16-H200 smoke,
using a separate small calibration and two distributed updates. It cannot
establish full-conversion convergence. Production requires `CHANNEL_SMOKE=0`,
an explicitly supplied `CHANNEL_DEADLINE` and appropriate Slurm time limit.
No full training runs on the login host. Resource instructions were updated
in AGENTS.md during preparation to specify 16 H200s. The initial production
attempt has a six-hour cap; further allocations require checking actual
progress and the user's resource preferences, not automatic blind restarts.

66 focused CPU tests passed for channelwise policy/export/calibration/acceptance
and legacy recipe/controlled/shared behavior. Shell syntax, Python compilation
and git whitespace checks passed. `channelwise_acceptance.py` rejects rounded
threshold hits, partial image coverage, non-finite boundaries/embeddings and
checkpoint hash mismatches. Existing unrelated worktree files and the
concurrent AGENTS.md edit are not included in implementation commits.

GPU smoke job 383290 completed in 98 seconds with Slurm exit 0. Fresh isolated
calibration succeeded; 16 ranks x 128 images completed two updates with
unclipped stem conversion. First loss 3.4841; maximum stem weight change
.000553. This proves execution only, not full conversion or accuracy.
Initial implementation commit 0324a53 was pushed to origin/main.

First production job **383299** started on nodes 25a-hgpn019–020, 16 H200s,
using commit bd701c4 (also pushed). Output:
`work_dirs/channelwise_ijbc96_20260914`. The fresh preparation, source hashes,
training checkpoints and per-epoch metrics stay under this directory.
Its initial absolute deadline is Unix 1789385375; Slurm limit is six hours.
No candidate IJB-C score is available yet. This goal remains active.

Preparation completed and recorded teacher SHA-256
`ac658cc7cdbce5de90b8cd36de19b29f22ee283e016884f85252aa0a50a1841a`.
The fresh teacher-development check reproduced clean TAR 99.01615% and
lowres20 TAR 34.57615% at sampled FAR=1e-4, with zero non-finite observations.
These are MS1MV3 development metrics, not IJB-C accuracy. Head warmup started
with finite losses and all conversion alphas zero.

## Immutable final evaluation

The existing `job_ijbc.slurm` supports `CHANNELWISE_ACCEPTANCE=1`, together
with `POLYNOMIAL_EXPORT=1`, `FINITE_AUDIT=1` and
`POLYNOMIAL_COEFFICIENT_MODE=channelwise`. This mode requires a fresh result
directory and snapshots the selected checkpoint there before evaluating any
batch. The exporter and acceptance report identify the snapshot by SHA-256.
Actual root input/output hook counts must both equal 938,750; metadata row
counts alone do not certify complete inference. The final script writes
`acceptance.json` and exits nonzero when the 96%/finite/coverage gate is unmet;
that report distinguishes a completed below-target evaluation from a runtime
failure. Additional focused audit/export tests: 35 passed.

IJB-C evaluation job **383314** is queued with `afterok:383299` and
`--kill-on-invalid-dep=yes`. It requests one GPU only after the training
allocation succeeds. It snapshots `student_best.pt` into
`work_dirs/channelwise_ijbc96_20260914/first_ijbc/evaluated_checkpoint.pt`,
then evaluates the exported graph and writes strict acceptance. A missing
selected checkpoint fails before image inference. This is the first
uncalibrated candidate assessment; any subsequent IJB-C-guided adjustments
and selection will be recorded as calibration-set work. Current training
is still in head warmup; no final student verification metric exists yet.

## First unbounded conversion milestone

Job 383299 completed head-warmup epoch 0 and saved `last.pt`. A CPU state
audit found all backbone tensors finite, 25 coefficient tensors and 17,664
coefficient values; provenance matches the original PReLU teacher. The
checkpoint correctly records `pure_quadratic=false` during conversion.

Fresh per-channel fit radii span .05–2.76513457. Many stem channels hit the
.05 floor because their teacher BN weights/activation magnitudes are nearly
zero (median observed stem maximum approximately 1.42e-14). Their large
recorded relative fit error is not a network-wide relative error: the
histogram has a 1e-5 lower edge and the target energy is nearly zero. Preserve
this initial policy and monitor actual distillation/range behavior instead
of changing radii based on that ratio alone. Details are saved in
`fresh_calibration_summary.json` and `epoch0_state_audit.json` under the run.

At epoch 1, step 1375/2477, the stem is fully quadratic and the first
residual-block activation has blend .110214. Training remains unclipped;
loss 8.4183, KD .0724, range .0133, and completed updates remain finite.
This verifies the first transition, not a full-network finite gate or final
accuracy. Training is still RUNNING; evaluation 383314 remains dependent.

## Stem and Layer1 completed

At epoch 3, step 425/2477, the first four activation sites have alpha=1
and `layer2.0.prelu` has alpha=.343157. Training remains unclipped and
finite (loss 8.1790, KD .1162, range .0104).

Preserved the preceding atomic epoch-2 checkpoint as
`epoch2_conversion.pt` before `last.pt` can be replaced. Its integrity audit
(`epoch2_state_audit.json`) confirms all backbone tensors finite and all
237 BN running buffers bitwise identical to the original teacher. Exactly
the four opened sites have changed quadratic coefficients; every unopened
site's coefficients still equal its fresh fit. Maximum coefficient changes
are .001754 (stem), .000389, .000302 and .000634 (Layer1 sites).
The snapshot is a conversion checkpoint, explicitly `pure_quadratic=false`,
and is not an accepted full-poly or IJB-C candidate. Job 383299 continues;
383314 is still waiting on its successful completion.

## Layer2 completed; Layer3 conversion underway

Job 383299 reached epoch 5 with the first eight activation sites fully
quadratic and no training clipping. At step 775/2477, `layer3.0.prelu` has
alpha=.625757; loss 7.8935, KD .1293 and range .0106 remain finite. Accounting
confirms RUNNING; dependent IJB-C job 383314 remains PENDING.

Preserved `epoch4_conversion.pt` as an immutable link to the completed
epoch-4 checkpoint. `epoch4_state_audit.json` confirms all backbone tensors
finite and all 237 BN running buffers unchanged. Exactly the stem, three
Layer1 sites and four Layer2 sites have updated coefficients; Layer3/Layer4
coefficients still match their fresh fits in this snapshot. The largest
coefficient change is .002835 at `layer2.2.prelu`. This is still a conversion
snapshot with `pure_quadratic=false`, not a full-network finite/accuracy gate.
No student IJB-C score has been produced and no test metric changed this run.
