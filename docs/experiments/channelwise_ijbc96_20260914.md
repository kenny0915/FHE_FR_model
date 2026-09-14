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
