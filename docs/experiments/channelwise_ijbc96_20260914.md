# Reproducible PReLU-to-quadratic IJB-C 96% goal

Status: best fully finite fresh-campaign IJB-C calibration-set TAR is
95.7815615892008%, with zero nonfinite intermediate values and embeddings.
The higher diagnostic TAR of 95.9145% has 104 nonfinite embeddings and is
ineligible. The 96% target is not achieved. The final head-adaptation evaluation
completed at 95.76622181316152%, also fully finite. This recipe has been stopped;
main job 384727 and pending evaluation 384733 were cancelled after saving results.

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

## First numerical failure and diagnostic continuation

Job 383299 aborted in epoch 5 after the last logged step 850/2477 (Layer3
first-site alpha .686314). All ranks passed the embedding finite check but
the synchronized **training loss** finite check failed before backward or an
optimizer update. This does not identify which loss component overflowed;
no failed batch was captured by the original trainer. Do not infer an
embedding overflow or a particular layer from the last finite log alone.

The Slurm allocation lingered after the worker failure. Cancellation of it
and dependent evaluation 383314 was requested, but the scheduler RPC timed
out; cancellation has not been confirmed. No IJB-C result exists.
`--capture-numerical-failure` now saves the exact augmented per-rank batches,
component/stage/activation diagnostics and rank-0 pre-update backbone/head/
optimizer state before numerical abort, with no sanitization or skipped update.
The hook is observational and leaves the successful update path unchanged.
Fourteen focused tests passed, including preservation of the failing batch
and unchanged weights. Shell syntax and whitespace checks passed.

Next action is a 20-minute diagnostic continuation from the immutable
epoch-4 checkpoint in a fresh output directory, copying the original
preparation/split and preserving the original optimizer/training policy.
Only failure capture is added. The transient training-tail replay cache is
rebuilt on resume, so this is not a bit-exact replay of the failed epoch.
The purpose is to capture a concrete numerical failure, or establish that
the resumed trajectory progresses, before selecting a numerical repair.

The diagnostic output directory is
`work_dirs/channelwise_ijbc96_diag_20260914`; original preparation and split
are preserved there. Source checkpoint SHA-256:
`fee1837a8d8934741dd256f67e235eeeff91ca97902906a889ce501e6872f5bb`.
The first `sbatch` request used `afterany:383299` so no diagnostic allocation
could overlap the lingering original allocation. Submission also returned a
socket timeout. Repeated accounting searches for the unique diagnostic job
name initially found no job. A later accounting query identified **383449**,
submitted at 14:29:37 Taipei time and PENDING: the timed-out mutation was
accepted. No duplicate was submitted. The original allocation/cancellation
state was subsequently reconciled: 383299 is FAILED (exit 1, 59m56s) and
383314 is CANCELLED. Diagnostic 383449 started at 14:32:00 Taipei time on
25a-hgpn153–154, using 16 H200s and the declared 20-minute cap.

`replay_numerical_failure.py` will replay a captured state/batch on one GPU
without updates. It reports forward-boundary finiteness, embedding norm
overflow, per-stage FP32 versus FP64 errors, active versus masked errors,
and range-loss mean versus sum-then-scale arithmetic. A CPU test demonstrates
the masked `0*Inf` NaN distinction and active FP32 square overflow; it does
not establish the cause of the real failed run. The completed replay and resulting policy revision are recorded below.


### Captured failure and pathological-augmentation ablation

Diagnostic 383449 failed at epoch 5, step 84. All 16 exact augmented batches
and the pre-update checkpoint are saved under
`work_dirs/channelwise_ijbc96_diag_20260914/numerical_failure_e5_s84`.
Only rank 15 had a nonfinite loss. Single-H200 replay **383511** completed
and reproduced classification 8.23009, KD NaN and boundary Inf. Batch row
125 was an artificial pathological image excluded from identity losses.
It dominated the activation tail (stage 3 maximum about 3.41e21); all
forward boundaries remained finite but its FP32 embedding norm overflowed.
Masked squared errors produced `0*Inf`, while range penalties also overflowed.
Active rows had finite squared errors, although their stage 3 hint error
was still large (~892); this does not prove ordinary inputs are stable.

The loss now selects active rows before cosine/squared-error computation.
Historical replay retains the legacy formula unless a checkpoint explicitly
records `loss_masking=active_rows`. Range penalties continue to cover every
training input. An explicit resume-policy revision is required to change
pathological augmentation frequency. The proposed ablation sets its fraction
to zero while retaining ordinary crop, low-resolution, photographic and
stress augmentation. It resumes the original epoch-4 checkpoint into a fresh
output directory, with the original preparation, optimizer and other policy.
A read-only active-row backward check precedes this run. No IJB-C row may be
excluded from final evaluation. Validation: 60 related CPU tests passed;
after adding the backward diagnostic, all 11 campaign tests passed again.


Active-row GPU backward diagnostic **383543** completed in 9 seconds.
All 125 retained identity rows produced finite loss 113.984436, KD 56.256958
and range penalty 49.497391. All 238 gradient tensors were finite; there
were zero optimizer updates. The historical full batch still reproduced
its original NaN/Inf losses in the same diagnostic.

Ablation job **383545** is submitted on 16 H200s, with a 4h35m allocation
cap and the original absolute deadline 1789385375. Output:
`work_dirs/channelwise_ijbc96_nopath_20260914`. It resumes the original
preserved epoch-4 state, sets pathological fraction to zero, uses active-row
loss masking, and preserves the remaining optimization/conversion policy.
`ablation_provenance.json` records the source hash, policy revision,
submission and diagnostic. Code commit: `f266f49`. No new student IJB-C
score or successful fully converted checkpoint is available yet.


### FAR acceptance precision

Final acceptance now uses the best empirical ROC TAR at measured FAR
**at or below 1e-4**, without rounding. The previous guard allowed an overly
broad FAR interval for a nearest-ROC-point result. Evaluation preserves the
historical nearest-point table and adds `at_or_below_requested_far` to each
raw point; acceptance requires this explicit constrained point. A regression
case where nearest-point TAR is 96.1% above the requested FAR but the valid
TAR is only 95.5% now fails acceptance. All 12 campaign tests pass. This
changes evaluation/acceptance only; active training 383545 is unaffected.


### No-pathology run still fails on an identity row

383545 FAILED after 4m29s, epoch 5 step 845. The exact state and all rank
batches are in `work_dirs/channelwise_ijbc96_nopath_20260914/numerical_failure_e5_s845`.
Rank 15 row 25 is an identity-loss-enabled augmented image (all 128 masks
true); input values remain in [-1,1]. Replay **383564** completed and
confirmed the same row dominates ratios from stem 3.22 through layer1.0
6.90, layer1.2 9.70, layer2.0 38.72, layer2.3 6.17e10 and layer3.0 5.74e20.
There are 512 nonfinite embedding values. Removing pathological augmentation
alone is insufficient; no fully quadratic student or IJB-C metric resulted.

A bounded diagnostic now tests detached, finite-prefix BN-affine repair
on the captured row, preserving the actual partial PReLU/quadratic phase
and adding no clipping. It stops before an escaping quadratic and trains
only its existing upstream BN affine. This is numerical evidence only:
no resumable checkpoint or accuracy improvement is claimed. Initial guard
4 and target 2 are multiples of each saved PReLU fit interval radius;
the original approximation target and interval themselves are unchanged.


Partial repair probe **383585** completed in 9 seconds, code `e63a681`.
At 200 SGD steps the original 128-row batch changed from one nonfinite
embedding/norm row to zero. All tensors outside the permitted BN affines
were unchanged; no inference clipping was introduced. The guard was NOT
fully satisfied: last escaping site layer2.1 had ratio 11.83 versus guard 4.
Source hash `1c6d1822ad3589d9e215c63e3479e0a490eee87188b9b80ef2e92724afeaf5e9`.
Report: `work_dirs/channelwise_ijbc96_nopath_20260914/numerical_failure_e5_s845/repair_probe/report.json`.
This supports investigating guarded prefix repair during conversion, but
provides no full-conversion, IJB-C, or accuracy evidence. All 35 related
recovery/campaign tests passed before GPU submission.


### Fidelity tradeoff in partial BN repair

Probe **383600** (2,000 range-only steps) kept the full batch finite but
reduced teacher cosine on the original 127 finite rows from 0.93377 to
0.81646; the guard still failed. This policy is not adopted for training.
Probe **383604** adds a separate 32-row finite teacher-distillation anchor,
with weight 100 and `log1p(prefix_loss / guard**2)`. The failure row is always
included in the prefix objective. After 2,000 steps the 127-row teacher
cosine improved to 0.97502, but the failed row still had an overflowing norm.
Thus neither result meets even the numerical repair gate, much less IJB-C
acceptance. Lower teacher weights 10 and 1 are being tested for 3,000 steps
each on one H200 each. These experiments do not save production checkpoints.


### Lower-weight probes and source-image reconstruction

383613 (teacher weight 1) passed the partial-phase interval guard after
about 1,850 updates; all 128 embeddings/norms were finite. Teacher cosine
on the original 127 finite rows was 0.91918 (source 0.93377). 383612 (weight
10, 3,000 steps) kept all embeddings/norms finite and raised the corresponding
cosine to 0.95627, but did not satisfy every interval guard. These are only
same-batch diagnostics, with no IJB-C or held-out accuracy evidence.

DistributedSampler reconstruction located rank 15 batch row 25 at MS1MV3
source row **4510751**, identity label **81726**. It lies outside the eight-row
replay-injection prefix. Original and augmented images are preserved as
`row25_original.png` and `row25.png` alongside `row25_source.json`. Visual
inspection shows a normal-color original and a strongly saturated red/blue
augmented image. The exact randomly selected augmentation operation was not
captured, so it cannot be attributed to a particular transform with certainty.

The next controlled ablation disables the extra stress family (probability
0.1 to 0), alongside the already disabled artificial pathological rows.
It retains crop 0.1, lowres 0.2 and photo 0.2. This avoids adopting the
costly, only same-batch-validated repair policy prematurely. It resumes the
same preserved original epoch-4 checkpoint in a fresh directory and retains
full IJB-C coverage as the final criterion. An explicit resume-policy
revision is required and recorded for this change.


No-stress ablation **383627** is RUNNING on 16 H200s, output
`work_dirs/channelwise_ijbc96_nostress_20260914`, code `86258c5`.
Allocation cap is 4h15m; original absolute deadline 1789385375 is retained.
Initial resumed epoch-5 step-0 loss is 7.7510, KD 0.1225, range 0.0088,
with clipping false. All 19 recipe/campaign tests passed. This initial
finite update does not establish stability through later conversion.


### No-stress failure and finite-prefix training routing

383627 FAILED at epoch 5 step 702 after 3m57s. Dependent full IJB-C job
383634 was CANCELLED without running. Rank 2 had finite embeddings but
stage 3 magnitude 2.03e19; KD and range arithmetic overflowed. Captures:
`work_dirs/channelwise_ijbc96_nostress_20260914/numerical_failure_e5_s702`.
Disabling the stress family is therefore insufficient for stable conversion.

`GuardedConversion` is a new training-only component. It probes each input
before evaluating an escaping quadratic, routes escaping rows to detached
finite-prefix BN repair, and computes full features for the remaining rows.
No training row is discarded: the caller must retain repair rows for replay
and must mask the all-repair dummy feature from identity losses. This is
not an inference replacement; full inference and IJB-C never use this router.
It is not yet integrated into multi-GPU training.

CPU validation: 17 routing/recovery tests passed, then all three routing
tests passed with explicit trainable-coefficient gradient coverage. GPU
check **383641** completed in 6 seconds using the exact captured rank-2
batch: 127 full rows, one prefix-repair row, finite loss 0.610792, repair
0.008856, and all 237 gradient tensors finite. Zero optimizer updates.
Code `ce610d9`; `guarded_backward.json` records the outcome. This establishes
single-GPU backward viability only; distributed integration and full target
accuracy remain unproven.


### Distributed training integration of prefix routing

Optional `--training-prefix-guard 4 --training-prefix-target 2` now routes
training rows before unsafe squaring. Safe rows retain ArcFace, stage KD and
range losses, scaled by their batch fraction; repair rows contribute a
weighted log-prefix penalty. Entirely repaired ranks call the head with a
zero-weight dummy row solely for DDP synchronization. The backbone wrapper
uses unused-parameter detection because repair-only ranks touch BN affines
but not all convolutions. All actual repair rows remain in the training-tail
replay cache with their measured escape scores. Validation/checkpoint export
use the original backbone and never the router. Enabling the policy on resume
requires an explicit revision record. Frozen BN running statistics and the
current per-site conversion schedule are preserved.

A two-process Gloo test alternates which rank receives only overflowing
inputs, performs three optimizer updates, and verifies exact synchronized
backbone/head weights after each. Additional coverage checks repaired-row
retention in replay, all-repair gradients, and unchanged ordinary outputs.


Integrated validation passed all 24 routing/campaign/recipe tests.
16-H200 smoke **383671** completed in 65 seconds with two optimizer updates,
backbone delta 0.000576 and no nonfinite values. All 2,048 smoke rows used
full features; GPU repair-path evidence remains the captured-batch single-GPU
test and the mixed-rank Gloo test described above.

Production continuation **383686** is submitted from the same original
epoch-4 checkpoint, output `work_dirs/channelwise_ijbc96_guarded_20260914`.
Code `c9a57c1`, guard 4, target 2, pathology/stress probabilities zero;
other training policy and source preparation are retained. Allocation cap
3h55m; absolute deadline 1789385375 unchanged. Every 25 steps, global
`GUARDED_ROWS full=... repair=...` counts expose the repair fraction.
No accuracy or full-inference stability result is available yet.


383686 has reached epoch 5 step 750 with finite loss, crossing the preceding
no-stress failure at step 702. Logged full-row fractions are approximately
94–98%; at step 750, 2,004 rows used full features and 44 prefix repair.
This is training routing evidence, not a full-inference finite audit.

Full IJB-C evaluation **383719** is PENDING with `afterany:383686` and one
H200. It requires the newly trained `student_best.pt`, snapshots its weights,
and runs channelwise polynomial export, all-image/orientation finite audit
and conservative FAR acceptance. Using afterany allows a valid full-conversion
checkpoint to be evaluated even if later training reaches its deadline.
If no fully converted finite development-selected checkpoint was produced,
the initial file check aborts evaluation. Result directory: `ijbc_full` under
the guarded run. Submission response was delayed but confirmed job 383719;
no duplicate was submitted.


### First completed guarded epoch

383686 completed epoch 5 and entered epoch 6. Immutable hardlink
`work_dirs/channelwise_ijbc96_guarded_20260914/epoch5_conversion.pt` is
preserved with SHA-256
`491ef3355fe879bd1ea76e0645ae8abc7d572a300df89ecf148c6acc0ea21084`.
`epoch5_state_audit.json` confirms all backbone tensors finite, all 237 BN
running-statistic/count buffers unchanged from original epoch 4, and matching
original PReLU teacher hash. Ten activation sites have updated coefficients,
through layer3.1. The checkpoint correctly remains `pure_quadratic=false`:
later activations are still in the PReLU conversion curriculum. No full-model
finite audit or IJB-C accuracy claim follows from this partial checkpoint.


### Guarded routing abort after epoch 6

383686 completed epoch 6 (preserved `epoch6_conversion.pt`) but stopped
advancing after epoch 7 step 400. A worker raised the router's ambiguous
`prefix routing changed between probe and repair` error. It could mean
that a subgroup no longer crossed the guard, or that its repair loss was
nonfinite; the original message did not distinguish them and no failure
batch was captured. The allocation remained RUNNING while peers waited,
so 383686 and dependent 383719 were explicitly cancelled; scancel succeeded.

The router now retains the first probed escape site when recomputing the
subgroup gradient. This handles small batch-shape-dependent threshold changes
without dropping the sample's repair gradient. Earlier genuine escapes still
stop computation first. The revised error includes probe site, repair site
and loss, and training now coordinates routing errors across ranks and saves
all exact batches before aborting. A regression test exercises a row that
rounds below the guard but remains above the repair target. All 33 routing,
recovery and campaign tests passed. This addresses a plausible routing cause,
not a proven diagnosis of the uncaptured real batch; resumed evidence is needed.


### GPU credit rejection on corrected continuation

Epoch-6 source SHA-256 is
`fe522be8f8b7908ef4cdd6927214131b1e2aaa50c5be262a8f0532cf37232601`;
all source backbone tensors are finite. Corrected continuation is prepared
in `work_dirs/channelwise_ijbc96_guarded_resume_20260914`, code `e23ce2c`,
with unchanged guard/target, augmentation, optimizer and conversion policy.
The replay cache is rebuilt, so the resumed epoch is not bit-exact.

Slurm rejected the submission (return code 1, no new job ID):
`You do not have enough credits in your iService wallet -397.5509`.
383686 and 383719 are confirmed cancelled. No GPU training or evaluation
for this campaign remains running. The user has been asked to replenish
MST114196 or identify an authorized alternative account. Read-only association
lookup also lists mst113064, but no work was charged to that other project.
The saved absolute deadline must be refreshed if it expires before credits
are restored. This is an external resource dependency; no full-conversion
IJB-C result or 96% achievement is claimed.


### Credits restored and continuation launched

The user reported replenishing credits. On September 14 at 20:48 Taipei,
`sbatch --test-only` passed for MST114196. Source epoch-6 SHA-256 was checked
again and matched the preserved value. The expired deadline was replaced by
1789406337 (a new 4.5-hour training window). Corrected continuation **384727**
is RUNNING on 16 H200s, output
`work_dirs/channelwise_ijbc96_guarded_resume_20260914`. It entered epoch 7;
step 25 loss was finite at 8.1956, with 1,996 full rows and 52 repair rows.
This verifies that the resumed training and repair path execute, not that the
previous uncaptured routing failure is conclusively fixed.

Full IJB-C evaluation **384733** is PENDING with `afterany:384727`, requiring
this run's newly produced `student_best.pt`. It snapshots the checkpoint,
exports the channelwise polynomial graph and audits all original/flip rows
before conservative FAR acceptance. No fresh IJB-C score is available yet.
The prior credit-rejected submission is retained in `resume_provenance.json`.


### First all-quadratic checkpoint and early full IJB-C diagnosis

384727 reached epoch 14 with all 25 alphas at one and clipping disabled.
The first unrestricted development validation failed: `nonfinite=712020353`
is the sum of nonfinite boundary values and failed embedding rows, NOT a
count of failed images. No TAR is reported by that development gate.
Training continues; the immutable `epoch14_quadratic.pt` snapshot is retained.

An early full IJB-C diagnostic runs as step **384727.2**, sharing one of the
already allocated H200s on 25a-hgpn026 (`srun --overlap`), with a one-hour cap.
It does not allocate a seventeenth GPU, but can slow the synchronized training.
Command and logs are saved as `epoch14_eval_step.json/.out/.err`; result
folder `ijbc_epoch14`. The complete 469,375-image evaluation has started.
The exported certificate confirms pure polynomial inference, no clipping,
25 channelwise quadratic sites and 17,664 coefficients. Snapshot SHA-256:
`69cf56b3938118d7b8dc698089c8a60d2fe09a6c83682867a904759a3b3b5675`.
This proves the structural requirement for this snapshot, not finite
inference or target accuracy. Full results remain pending and will guide
whether/how the permitted IJB-C calibration is needed.


### Calibration adapter checked on a real all-quadratic candidate

Epoch 15 unrestricted development validation still fails, with aggregate
nonfinite count 849160683 versus epoch 14's 712020353. These counts combine
boundary values and bad embedding rows; they are not image counts or TAR.
The epoch-14 full IJB-C evaluation continues toward a source-level manifest.

In parallel, calibration smoke step **384727.3** completed 25 updates in
27 seconds on another already allocated H200. It used an isolated epoch-14
copy, 800 sampled IJB-C orientations, original teacher KD, the bounded
auxiliary training branch and unbounded finite-prefix repair. Saved inference
has no clipping; all fixed buffers were checked unchanged and gradients/
updated parameters finite. Step-25 KD 0.101471, prefix 0.180486, total loss
0.606596. Output `ijbc_calibration_smoke` under the resumed run; command/logs
`calibration_smoke_step.json/.out/.err`. This validates the real calibration
pipeline, not full-inference finiteness or accuracy. Production calibration
will use the completed IJB-C failure manifest and requires a new full audit.


### Epoch-14 full IJB-C result and audit accounting correction

The complete IJB-C diagnostic finished: conservative TAR **95.53612517%**
at actual FAR **9.9495285e-5**. The manifest contains **9,712** original/flip
embedding rows with nonfinite values. Therefore this result fails both
accuracy and finite-inference requirements. The graph certificate and
checkpoint identity pass. The score is diagnostic, using the evaluator's
reported nonfinite handling, not a zero-nonfinite result.

Diagnostic tracing replayed those 9,712 rows through the same audited model,
inflating audited input/output rows from 938,750 to 948,462 and duplicating
some boundary failure counts. The new audit pause context excludes diagnostic
replays from subsequent full-evaluation accounting and restores collection
even if tracing raises. All 14 campaign tests pass. This accounting fix does
not change model weights or the existing TAR, and does not remove real failures.

Next calibration uses the immutable epoch-14 checkpoint and the complete
failure manifest: 1,000 steps, batch 32 (half random IJB-C orientations,
half manifest replay), LR 1e-5, original teacher KD and finite-prefix loss.
Saved inference remains unclipped; a new full IJB-C audit is required.

Production calibration is now running as step **384727.4**, sharing one
already allocated H200 on node 026 with a 40-minute deadline. Output
`ijbc_calibration_epoch14_1000`; command/logs `calibration1000_step.json/.out/.err`.
The first 75 updates completed with finite recorded losses; acceptance still
requires the saved unclipped model and a new full evaluation.

### Completed calibration and fixed-subset comparison

384727.4 completed all 1,000 updates in 5m03s. Candidate SHA-256:
`55e6133f9d72248626a2634f098702c58bf30a2e6083b9ba2418ae164b495b61`.
Full exported-graph IJB-C evaluation **384727.5** is running under
`ijbc_calibration_epoch14_1000_full`; logs `calibration1000_eval_step.*`.

Diagnostic step **384727.6** compared source and candidate on the same seeded
512 manifest orientations plus 512 random orientations, using exported graphs.
Manifest embedding nonfinite rows decreased **512 -> 408**; random failures
remained **9 -> 9**. Valid random-row teacher cosine was .89672 -> .89780.
One additional calibrated manifest row has finite coordinates but an overflowing
FP32 embedding norm (409 invalid norms versus 408 nonfinite embeddings).
The production evaluator aggregates templates in float64, so this FP32 probe
norm count is a stricter diagnostic, not its reported nonfinite count.
Records, hashes and executable command are in `calibration1000_probe.json`
and `calibration1000_probe_step.json`. This subset cannot establish acceptance.

Based on partial numerical improvement, continuation **384727.7** starts from
this calibrated candidate for another 1,000 steps, LR **1e-4**, seed 20260915,
with the same original failure manifest and half-random batches. Output
`ijbc_calibration_epoch14_lr1e4`; command/logs `calibration_lr1e4_step.*`.
It shares a second already allocated H200 while the first candidate's full
evaluation runs; the primary 16-GPU training remains active.

### Stronger calibration and exact-inference distillation option

The first LR-1e-4 continuation completed 1,000 steps, final prefix .20958.
Candidate SHA `0c71925fdfacd1f8a1356008fc45e195698265798854238ad36e4702adc973de`.
The identical exported-graph subset now has **64/512** manifest and **4/512**
random nonfinite embeddings (versus 408 and 9 after the first calibration).
Valid random-row teacher cosine decreased to .88436; compared sets differ by
newly repaired rows, so this is not a paired-only accuracy measurement.
No full TAR is available for this candidate. Probe logs `calibration_lr1e4_probe_step.*`.
A further 2,000-step LR-1e-4 continuation, seed 20260916, is running in
`ijbc_calibration_epoch14_lr1e4_3000`, starting from this immutable completed
candidate. Main epoch 16 still fails development finiteness; its snapshot
is retained as `epoch16_quadratic.pt`.

The calibration adapter now optionally adds teacher distillation on its actual
unclipped graph (`--exact-kd-weight`, default zero preserves previous runs).
A no-grad probe selects only rows with all polynomial inputs within four
saved fit radii and finite nonzero FP32 embedding norms, then reruns only those
images with autograd. Thus failed rows cannot poison gradients through masked
NaNs. The loss is weighted by the selected fraction of the original batch.
This training gate does not exist in exported inference. CPU tests verify
correct loss, finite nonzero coefficient gradients with an overflowing excluded
row, all-excluded zero gradients and rejection of clipped evaluation. Real-GPU
validation and any benefit to full IJB-C accuracy remain to be established.

The additional 2,000-step continuation is Slurm step **384727.9**. The exact-KD
GPU smoke **384727.10** subsequently completed 25 updates in 33 seconds using
a separate copy of the 1,000-step LR-1e-4 candidate. Final exact KD .090612,
31/32 safe rows, prefix .258782, total .695178; gradients, updated parameters
and fixed-buffer checks passed. Saved inference remains unclipped. Output
`ijbc_calibration_exactkd_smoke`, command/logs `calibration_exactkd_smoke_step.*`.
It is a pipeline validation, not a measured accuracy improvement. All 15
campaign tests passed before the smoke; implementation commit `afecffd`.

### First calibration: complete IJB-C result

Full evaluation **384727.5** completed with conservative TAR **95.63839035%**
at actual FAR **9.6170250e-5**, versus uncalibrated epoch 14's 95.53612517%.
Nonfinite augmented embeddings decreased **9,712 -> 7,931** (18.3% reduction).
This is still a diagnostic score with nonfinite handling, not a passing model.
The corrected audit observes exactly **938,750 input and output rows** across
all 469,375 source images; full coverage, graph and checkpoint identity pass.
Accuracy and finiteness fail; `acceptance.json` records `target_met=false`.
Artifacts are under `ijbc_calibration_epoch14_1000_full`; evaluated hash is
`55e6133f9d72248626a2634f098702c58bf30a2e6083b9ba2418ae164b495b61`.

### Three-thousand LR-1e-4 updates and measured feature drift

Step **384727.9** completed its 2,000 additional updates, giving 1,000 updates
at LR 1e-5 followed by 3,000 at LR 1e-4. Final candidate hash:
`15dcf54332304311be385f2a337a7bd0710b3144ec4492d324bd15a0aa23d280`.
Full IJB-C evaluation **384727.12** is running in
`ijbc_calibration_epoch14_lr1e4_3000_full`, logs `calibration_lr1e4_3000_eval_step.*`.
Fixed-subset probe **384727.13** reports **3/512** manifest and **1/512** random
nonfinite embeddings. On the SAME 503 initially valid random rows, teacher
cosine decreased **.89672 -> .87757**; this confirms drift beyond changed row
membership. It does not directly measure TAR. Complete record/command:
`calibration_lr1e4_3000_probe.json` / `calibration_lr1e4_3000_probe_step.json`.

Exact-inference KD continuation **384727.14** starts from this candidate for
1,000 steps, LR 1e-4, exact KD weight **5**, range weight 1, seed 20260918,
with unchanged original failure replay and half-random calibration batches.
Output `ijbc_calibration_epoch14_exactkd5`; logs `calibration_exactkd5_step.*`.
This explicitly recorded policy revision aims to recover actual-inference
teacher agreement while retaining range repair; success is not assumed.

Probe **384727.11** also checked the preserved epoch-16 main-training checkpoint,
hash `57299d130e28e82d3d1d640f9644c007a1f737360646d92b67844cc778881934`.
It has 351/512 old-manifest and 9/512 random failures, random teacher cosine
.89974. Main training continues; these subset results alone select no winner.

### Output-head adaptation option and historical PReLU reference

The adapter adds `--parameter-scope spatial|all|head`; default `spatial`
preserves preceding runs. `all` also updates final FC/BatchNorm1d affines,
while `head` freezes the spatial backbone and all polynomial coefficients.
BN moments remain fixed in every mode. This permits testing whether adapting
the final linear map recovers agreement after numerical repair changed its
input distribution. CPU tests verify an actual head-only SGD update changes
only output affine parameters and leaves body/buffers unchanged; all 16 campaign
tests pass. No GPU/head-only accuracy result exists yet.

Historical PReLU scores `work_dirs/ms1mv3_r50/ijbc_result/ms1mv3_r50/ijbc.npy`
were reanalyzed over 15,658,489 pairs using the same conservative FAR rule:
TAR **96.55877691%**, actual FAR **9.8919798e-5**. Score SHA:
`f1ee38cd9e59dedb9a49ccf9b2404a0317e77acb80fd4e5941d3cb2ecc266524`.
This existing score artifact lacks original checkpoint/evaluation metadata;
it is a historical reference, not a newly provenance-verified teacher run.
Command and report: `historical_prelu_reanalysis_step.json` and
`historical_prelu_reanalysis.json` under the current run.

### Exact-KD result and output-head calibration

Exact-KD step **384727.14** completed 1,000 updates in 7m55s. Candidate hash
`227b6e557deda394458bd2abf19ad3d6d74e381391e0dbfaae8a0c73a8ec496e`.
Probe **384727.16** finds **2/512** old-manifest and **1/512** random nonfinite
embeddings. On the same 503 initially valid random rows, teacher cosine
recovers from the pre-exact-KD .87757 to **.89309** (original .89672).
This supports the exact-inference loss, but is not a full TAR/finite result.
Reports `calibration_exactkd5_probe.json` / `calibration_exactkd5_probe_step.*`.

Output-head-only GPU smoke **384727.17** completed 25 updates in 19 seconds:
LR 1e-3, exact KD weight 5, range weight 0. All finite-gradient/parameter and
fixed-buffer checks passed. The spatial backbone and polynomial coefficients
are frozen. Production **384727.18** starts from the completed exact-KD
candidate (not the smoke) for 2,000 head-only steps with these settings and
seed 20260919, output `ijbc_calibration_epoch14_head2000`; command/logs
`calibration_head2000_step.*`. It can adapt output geometry, but cannot repair
an overflowing frozen spatial backbone. The pending full-scan failure manifest
will guide any subsequent joint numerical repair.

### Complete longer-calibration result and residual-repair plan

Full evaluation **384727.12** completed: conservative TAR **95.70486271%**,
actual FAR **9.9495285e-5**, **110** nonfinite augmented embeddings. All 938,750
input/output rows are audited; graph and checkpoint identity pass. Accuracy
and finite requirements still fail. The manifest is under
`ijbc_calibration_epoch14_lr1e4_3000_full/nonfinite_manifest.csv`.

Head-only step **384727.18** completed 2,000 updates. Candidate hash
`777e6fb12a0fe149f060131012dd022b754046dd23fb7b624dd5f31204086d82`.
The fixed probe retains 2/512 manifest and 1/512 random failures, with paired
random teacher cosine **.89537** (up from .89309 before head adaptation).
Valid embedding norms have median 15.25 and maximum 36.91; the corresponding
teacher median is 21.99. This is a subset diagnostic, not TAR. Reports
`calibration_head2000_probe.json` / `calibration_head2000_probe_step.*`.

Next: fully evaluate this head-adapted candidate and run joint calibration
from a separate copy using the latest 110-row failure manifest, LR 1e-4,
exact KD weight 5, range weight 1, all spatial/quadratic/output affine parameters.
The evaluation launcher now preserves an explicitly supplied CUDA device mask,
allowing overlapping single-GPU evaluations on distinct allocated GPUs. It
still defaults to device 0; finite audit requires exactly one visible GPU.
Shell syntax and whitespace checks pass.

### Adaptive replay of the remaining numerical failures

Head-only full evaluation is **384727.20**, output
`ijbc_calibration_epoch14_head2000_full`. Joint residual calibration
**384727.21** completed 1,000 updates in 7m53s from the head-only candidate,
using the latest 110-row manifest, all parameter scopes, LR 1e-4, exact KD
weight 5 and range weight 1. Candidate hash
`7d2f4940394905be3b1611d5b592f6853c1996e9f1110365c08176113cd46a0a`.

Probe **384727.22** covers the original fixed 1,024 orientations AND all 110
latest failures. The original manifest/random groups now both have **0**
nonfinite embeddings. The 110-row group improves **93 -> 21** failures
relative to the head-only source. Paired random teacher cosine .89529 ->
.89364 (511 shared-valid rows). This is still a failing candidate, not a full
evaluation or zero-nonfinite claim. Reports `calibration_residual110_probe.*`.

Step **384727.23** reproduced/extracted those exact 21 remaining orientations
into `calibration_residual110_remaining.json`, tagged with candidate hash.
Calibration **384727.24** now replays this smaller set for **500** steps,
with the same LR/weights/all-parameter scope and seed 20260921. Output
`ijbc_calibration_epoch14_residual21`; logs `calibration_residual21_step.*`.
The next probe must still cover all 110 previous failures and the original
1,024 orientations, so narrowing replay cannot silently hide regressions.

### First candidate finite on all known failures; full audit pending

Head-adapted full evaluation **384727.20** completed: conservative TAR
**95.91450631%**, actual FAR **9.9878943e-5**, **104** nonfinite augmented
embeddings. Eleven failures were absent from the preceding 110-row manifest;
all subsequent probes include BOTH complete manifests to detect regressions.
Full coverage, graph and checkpoint identity pass; accuracy and finiteness fail.

The 500-update 21-row repair **384727.24** produced SHA
`fa926f14f85ebd1334e049ca479e725c112b7a20a9c5b4da1eafeee84ef14e08`.
Probe **384727.25** found four failures in the previous-110 group and six in the
new head-evaluation group, six unique failures overall. The fixed original
1,024 orientations remained finite. Exact remaining rows were recorded in
`calibration_residual21_remaining.json`.

Focused six-row repair **384727.26** completed another 500 updates with the
same all-parameter LR-1e-4 / exact-KD-5 / range-1 policy, seed 20260922.
Candidate SHA **8298aa8c0addaf8432648207a8ff9d4f3d14ddfc60c8dce07eee7527bcf2b69b**.
Probe **384727.27** finds zero nonfinite embeddings and zero invalid FP32 norms
in every group: original 512 manifest + 512 random, previous 110 failures and
latest 104 failures. These are 1,238 evaluated records / **1,139 unique
orientations**, NOT full IJB-C coverage. Paired random teacher cosine declined
.89235 -> .88816, motivating further output-head adaptation. Reports
`calibration_residual6_probe.json` and `calibration_residual6_probe_step.*`.

Full exported IJB-C audit **384727.28** is now running in
`ijbc_calibration_epoch14_residual6_full`, with a 40-minute step cap. No passing
full finite/accuracy result is claimed yet. In parallel, head-only continuation
**384727.29** freezes this spatial/polynomial backbone and uses **uniform random
IJB-C orientations without failure replay**, 2,000 steps, batch 64, LR .003,
exact KD weight 5, range weight 0, seed 20260923. Output
`ijbc_calibration_epoch14_head_random`; logs `calibration_head_random_step.*`.
It tests recovery of ordinary-input feature agreement after focused repairs.
Neither calibration phase uses IJB-C identity/pair labels for gradients.
The separate MS1MV3 main training uses ArcFace (margin .5, scale 64), KD and
range control; no AdaFace loss is used in this campaign.

### Full numerical repair confirmed; continuation reassessment (September 15)

Full exported evaluation **384727.28** completed in 18m27s. The residual-six
candidate achieves **95.7815615892008% TAR**, actual FAR
**9.911162731572719e-5**, with **zero** nonfinite intermediate values and
embeddings across all **469,375** source images / **938,750** original and
flip orientations, including the remainder batch. All acceptance checks
except TAR pass. The step reports FAILED because the accuracy gate returns
nonzero; extraction and numerical auditing completed successfully.
The gap to the goal is **0.2184384107992 percentage points**.

The compact machine-readable report is
[channelwise_ijbc96_residual6_result.json](channelwise_ijbc96_residual6_result.json),
including the original report hashes and graph certificate. Full reports and
the immutable evaluated checkpoint remain under
`work_dirs/channelwise_ijbc96_guarded_resume_20260914/ijbc_calibration_epoch14_residual6_full`.
This is **IJB-C calibration-set performance**, not untouched test accuracy.

Random-IJB head-only calibration **384727.29** completed 2,000 updates in
7m35s. Checkpoint SHA
`bf07956815e06230f5a36d5f36752ceabe23614695610373d59baa15bb8eba44`.
Probe **384727.30** preserves zero failures on all 1,139 unique known probe
orientations and improves paired random teacher cosine .88816 -> .89569.
Full exported evaluation **384727.31** is running in
`ijbc_calibration_epoch14_head_random_full`; probe improvement does not prove
full TAR improvement or full finiteness.

The user authorized at most 16 H200 GPUs and 48 hours, and confirmed account
MST114196 was replenished. A Slurm test-only submission succeeded, but no new
continuation job was submitted. Following the user's question about whether
continued training is worthwhile, further main-training submission is on
hold: recent MS-only validation still overflows, while the completed IJB
repair now demonstrably fixes numerical failures at some accuracy cost.
The pending head-only full evaluation will inform the next decision. Existing
main training and evaluation remain separate; no IJB-calibrated weights have
been fed back into MS1MV3 main training.

A verified machine-readable source chain is preserved in
[channelwise_ijbc96_calibration_chain.json](channelwise_ijbc96_calibration_chain.json).
It contains the nine actual calibration stages from the immutable main epoch-14
checkpoint to the random-head candidate, their full configurations, completed
step counts, replay-manifest hashes, and checkpoint hashes. Each predecessor
hash was recomputed from disk and matched its successor's recorded source;
all stages use the original PReLU teacher hash and preparation provenance.
This records the successful lineage, excluding discarded smoke/diagnostic runs.

### Final head result and stopping this recipe (September 15, 01:03 Taipei)

Full exported evaluation **384727.31** completed in 18m12s:
**95.76622181316152% TAR**, actual FAR **9.936739925718713e-5**,
**zero** nonfinite intermediate values and embeddings over all 938,750
orientations. Graph, source hash, full coverage and numerical checks pass;
the accuracy gate fails. See
[channelwise_ijbc96_head_random_result.json](channelwise_ijbc96_head_random_result.json).
The improved small-probe teacher cosine did **not** improve full TAR: it is
0.01533977603928 percentage points below the residual-six candidate.
No passing 96% claim is warranted.

The main run's just-completed epoch 19 validation still reports
`activation_max_ratio=Infinity`, `nonfinite=568045742` (aggregate boundary
scalar/embedding counter, not an image count). Its checkpoint `last.pt` is
preserved; training was partway through epoch 20. These observations do not
support extending the same main-training or head-only calibration recipe.
After the completed full evaluation, pending evaluation **384733** and main
allocation **384727** were cancelled to avoid further resource use. No new
continuation was submitted. The best eligible result of this recipe remains
**95.7815615892008% / zero nonfinite**, an IJB-C calibration-set result.
The broader 96% objective remains unmet; stopping this recipe does not
redefine or satisfy that objective.

### New bounded hypothesis: exact-path head calibration and feature magnitude

The original recipe remains stopped. Inspection of the unchanged IJB-C
protocol confirms that original/flip embeddings are summed without per-image
L2 normalization (`use_norm_score`), then weighted by detector confidence
before media/template aggregation. Earlier head calibration used cosine KD,
which does not constrain embedding magnitude, and combined a temporarily
clipped auxiliary graph with the exact unclipped graph. Better small-probe
cosine did not predict better full TAR.

A distinct one-H200, one-hour maximum ablation is prepared in
`controlled_degree2/channelwise_exact_head.slurm`. Both arms start the fully
finite residual-six source, freeze spatial layers and polynomial coefficients,
and optimize only final Linear/BN1d affines on the exact unclipped graph.
Arm 0 uses cosine KD; arm 1 adds embedding squared error divided by the batch
mean teacher squared norm. Same seed 20260924, 2,000 steps, batch 64, LR .003,
exact KD weight 1, guard 4, range and auxiliary weights zero. A separate
25-update smoke runs first. Each arm gets a full exported IJB-C audit; numerical
or extraction failures abort, and a verified passing arm stops further work.
No labels are used for gradients, no inference operations are added, BN/range
buffers remain fixed, and approximation targets/intervals remain unchanged.
This is a hypothesis test, not evidence that 96% is attainable.

The adapter defaults preserve all earlier runs. New tests establish a nonzero
magnitude-correcting gradient when cosine already matches, frozen polynomial
parameters, no auxiliary clipping in exact-only mode, and finite connected
zero gradients when all rows are excluded for a frozen-body head. All 18
campaign tests pass; the new Slurm script also passes `bash -n`.
