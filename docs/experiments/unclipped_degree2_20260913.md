# Unclipped, layer-shared degree-two campaign

Target: unrounded IJBC TAR >= 96.56% at requested FAR=1e-4, zero non-finite
intermediate/output values, and a polynomial image-to-embedding backbone.
The earlier FAR=.1 gate and bounded model do not qualify for this campaign.

Budget begins conservatively at 2026-09-13 06:40 UTC, including preparation;
hard deadline 2026-09-15 06:40 UTC. At most 16 H200 GPUs concurrently. Each
training job uses two nodes x eight H200 via nano4 Slurm conventions. No
full training on the login host. Reserve <=6 hours training and <=4 hours
final evaluation per attempt, always capped by the common deadline. Actual
allocation and job IDs are recorded in work_dirs/unclipped_round_20260913/
campaign.json and Slurm accounting. Unused allocations are not reported as
consumed time. Queue/preparation time counts against the wall-clock budget.

Source: shared_d2_D_20260913/continuation_best.pt, the internally selected
bounded epoch-7 backbone plus matching classifier. Its preparation, teacher,
MS1MV3 identity split and calibration provenance must match exactly. All
runs use a fresh optimizer, frozen BN running statistics, trainable Conv/BN
affines and exactly 75 polynomial coefficients, with scalar radius per site.
No per-channel polynomial coefficients are introduced.

Approximation target: the original pooled PReLU responses over each site's
MS1MV3-derived [-radius, radius], fitted with 95% empirical symmetric
histogram weight and 5% uniform interval weight. Those original ranges are
retained as training regularization references, not inference bounds.
Every site's inference expression is c+b*x+a*x*x. Training-only coefficient
projection bounds dimensionless (c/R,b,a*R) by (.5,1.5,curvature_cap), with
nonzero |a*R| >=1e-6. It is absent from the exported forward pass.

Predeclared independent strategies (all start again from the same source):

| Strategy | Training clamp removal | Curvature cap | Epochs | Backbone LR | Coefficient LR | Range weight |
|---|---|---:|---:|---:|---:|---:|
| direct | all removed immediately | none | 12 | .0002 | .00002 | 1 |
| gradual | one site at a time, forward order, over 6 epochs | none | 18 | .0005 | .00005 | 5 |
| projected | same 6-epoch curriculum | .15 | 18 | .0005 | .00005 | 5 |
| slow_projected | 12-epoch curriculum | .3 | 28 | .0002 | .00002 | 10 |
| small_curvature | 6-epoch curriculum | .05 | 24 | .001 | .0001 | 5 |
| slow_free | 12-epoch curriculum | none | 28 | .0002 | .00001 | 10 |

Classifier LR .0025, global batch 2048, FP32, existing ArcFace + teacher
embedding/stage hints, MS1MV3 augmentation and training-tail replay. Each
strategy includes epochs with no training clamps at all; evaluation never
uses a clamp, even during the curriculum. Previous internal loss discontinuity
on simultaneous clamp removal motivates the gradual alternatives. All six
policies are fixed before evaluating any new candidate on IJBC. Finishing
this comparison early does not imply budget exhaustion; any further design
must be justified only from MS1MV3 evidence and separately logged.

Only full-curriculum epochs enter model selection. Select maximum
min(clean+flip,lowres20) MS1MV3 development TAR at sampled FAR=1e-4, gated
by zero non-finite values in clean/flip/lowres/shift/dark inference. The
existing seed-20260911 2% identity holdout, six images per identity and two
million fixed impostor draws are reused. The teacher may have seen holdout
identities in pretraining; student training/calibration excludes them.

Final IJBC is run once per independently selected candidate; test results
only trigger the requested success stop. They cannot determine coefficient,
range, architecture or training changes. A failed training job with no
checkpoint has unavailable final metrics, not invented accuracy or finiteness.
A diagnostic last checkpoint is explicitly marked if no candidate passed
internal selection; a non-finite final audit cannot qualify regardless of TAR.

Final evaluation uses the existing alignment, flip fusion, media/template
aggregation, detector scores and ROC conventions. Export folds BN statistics
into fixed affine constants and removes dropout/training machinery. A strict
FX graph allowlist accepts only add/multiply, Conv/Linear (affine sums of
products), and structural operations. The evaluator saves and runs this
export itself, with a finite audit and unrounded FAR=.0001 metric. Existing
plaintext embedding normalization and scoring remain outside this backbone.
Every quadratic must have nonzero curvature and one shared triplet. The
certificate is structural; finiteness is empirical on evaluated samples,
not a proof over all real inputs. No multiplicative-depth optimization here.

Implementation checks: 25 focused CPU tests, including coefficient projection,
clamp curriculum, hidden nonlinearity rejection, full-backbone export,
BatchNorm affine equivalence, warm-start provenance, and the strict FAR gate.
Shell syntax and git whitespace checks also pass. GPU execution follows.

## Execution log

07:05 UTC: direct job 380601 failed at its first MS1MV3 training batch with
non-finite embeddings, before any optimizer update or completed checkpoint.
Elapsed allocation 148 seconds on 16 H200s. Internal student validation and
final IJBC TAR are unavailable because it produced no trained candidate.
The prior source is not silently substituted as a new trained model.

Gradual job 380603 started at 06:55:36 UTC on nodes 018–019, 16 H200s. The
teacher holdout reproduced clean TAR .9901615381 and lowres TAR .3457614779,
with zero non-finite outputs. More than 2000 student updates have passed the
training finite guards; several prefix clamps are removed, but the suffix
still uses training-only clamps. This is not a full unclipped inference gate.
No new IJBC result has been consumed. The fixed policies remain unchanged.

Accounting snapshot at 07:05:31 UTC: .4253 elapsed wall hours, 3.3022 allocated
GPU-hours including the failed direct job and elapsed gradual allocation.
The live budget.json is generated from sacct ElapsedRaw and AllocTRES rather
than requested time limits. Its Slurm start/end timestamps are cluster-local
Asia/Taipei; collection/deadline timestamps use UTC. Code/test commit a1f1a29
is pushed to origin/main. Six additional focused tests pass after adding the
read-only accounting utility (26 tests total across the relevant suites).

## Conditional training-data-gated repair, prepared before new IJBC results

The direct run's first-batch non-finites motivate an adaptive alternative if
the fixed curricula cannot safely remove all clamps. This reuses the existing
sitewise finite-prefix repair machinery, now admitting the same shared D
source with strict provenance and exactly 75 trainable polynomial coefficients.
It is prepared, not yet submitted, and does not replace a running policy.
Reserve remaining time (up to 24 hours including accuracy continuation, plus
final evaluation), always within the same absolute campaign deadline.

Repair Conv weights, spatial BN affines and shared coefficients jointly;
keep BN running moments, embedding projection and classifier fixed. Training
uses a bounded auxiliary teacher embedding/all-block hint/ArcFace objective
and an all-site maximum range penalty targeting .9 of each original MS1MV3
radius. A separate candidate-prefix pass stops *before* an escaping square,
so finite prefix gradients can repair the failure without backpropagating
through Inf/NaN. These training guards are absent from candidate deployment.
Use the existing globally averaged conflict-aware gradient combination, SGD
LR 1e-4, per-tensor actual-update cap 1e-4, and exact training-row/variant replay.
Unlike the old per-channel repair, all 75 shared coefficients may adapt.

Open sites in forward order only after the existing 8192-training-image
clean/flip/lowres/shift/dark gate reports exactly 40960 rows, no non-finite
values, and maximum normalized input ratio <=2. Recheck every 250 updates and
save every 25. Once all 25 sites open, require a complete clean/flip MS1MV3
training-corpus finite gate. Then run 24 epochs of ordinary *fully unclipped*
accuracy training with fresh momentum, frozen BN, source backbone/head LRs
scaled by .1 and coefficient LR .0001, selecting solely by the existing full
holdout criterion. No IJBC image, failure manifest, range or score enters
repair or handoff. The Slurm script has no automatic requeue past the deadline.
Shared resume verifies the source, fixed repair policy, accuracy epochs and
absolute deadline. Repaired interim checkpoints are diagnostic only until
accuracy/finite validation and final exported IJBC checks actually pass.

44 focused CPU tests passed after adding shared repair support; the final
resume-deadline regression is checked separately. No H200 smoke or successful
repair is claimed yet.

Small export probe of gradual epoch 0 (SHA-256
74e263232660ba2bcda70f9ff32afc7f1870537b39a135dcfbc7850d8643647a): MS1MV3
holdout source rows 3053 and 3056, original and flip, were finite in both the
source graph and serialized/reloaded polynomial export. Maximum embedding
difference 2.3841857910e-6. Source rows and boundary audits are in
work_dirs/unclipped_round_20260913/gradual/early_export_probe.json. This is
four-forward export validation, not a model-selection or dataset-wide gate.

07:19 UTC decision: gradual training job 380603 failed after 22m03s, in epoch
2 shortly after opening site nine. The finite-embedding guard passed, but the
subsequent training-loss guard failed. This does not identify a particular
layer as the cause; the unclipped prefix can yield very large finite values
whose training range penalties overflow even while the bounded suffix keeps
embeddings finite. No update with the bad loss was taken.

The predeclared diagnostic last-checkpoint evaluation is job 380686, using
saved epoch 1 (SHA-256
89e752090600d235dd1bf6f6b9b46a8bccb3a37034dc7b9bd7419d63a93fb818). It never
reached ordinary full-curriculum holdout selection. Final test results remain
unavailable at this decision point; no IJBC metric or failing image is used.

Prioritize the prepared adaptive-prefix strategy after the unchanged projected
arm, ahead of the remaining slower fixed curricula. This allocation revision
is supported by the two MS1MV3 failures, not IJBC. Its maximum training and
accuracy allocation is 24 hours within the common deadline; the remaining
fixed alternatives retain their original policies and six-hour caps. Stop
immediately if a final candidate meets the target instead of launching more.

The controller is being resumed against the existing ledger/evaluation job;
recorded jobs are monitored, never blindly resubmitted. A process lock prevents
duplicate controllers. The old/new policy lists and reason are preserved in
campaign.json. Fixed diagnostic checkpoints additionally receive the full
MS1MV3 holdout report after their IJBC allocation finishes but before the
controller consumes the IJBC metric. This supplies missing non-IJBC reporting;
it does not reselect or alter a checkpoint. All allocations remain sequential.
The standalone holdout job rechecks the saved split hash and checkpoint hash.
Ten focused tests pass, including restart without duplicate submission.

Gated repair additionally requires an isolated two-update H200 x16 smoke job
(maximum 20 minutes) before its full allocation. It exercises the actual shared
source/head, gradients and checkpoint invariants with a 32-image training gate;
full repair is not submitted unless Slurm completes and smoke.pt exists.
This smoke is accounted separately and does not certify inference or accuracy.

Success also requires a zero-nonfinite internal holdout report for the fixed
checkpoint. An IJBC-only pass cannot override an observed internal failure or
an unavailable internal audit. Twelve focused controller/export/recovery tests
pass, including this combined success gate, before any new final metric is
consumed. No training parameters change from these orchestration checks.

## Gradual final diagnostic outcome; projected arm starts unchanged

Final IJBC 380686 and fixed MS1MV3 holdout 380728 completed. The epoch-1
checkpoint's MS1MV3 audit observed 2,379,654,743 non-finite values (repeated
module boundaries plus invalid embeddings/norms) across 55,460 forwards;
verification TAR was therefore withheld. Its IJBC export was structurally
polynomial but produced 25,541 non-finite augmented embedding rows among
938,750 original/flip forwards, and 30,625,358,238 non-finite boundary values.
The reported TAR was 2.7560464284% at requested FAR=1e-4 (actual nearest FAR
.00010000682911083698). This is a **diagnostic, invalid** accuracy result:
the existing evaluator replaces bad features with zero. The candidate fails
numerical validity regardless of that number. The corresponding compact
report is unclipped_degree2_gradual_result.json.

Projector job 380734 started on 16 H200s, nodes 018–019, with the original
predeclared cap .15, six-epoch clamp curriculum, 18 total epochs, backbone LR
.0005, coefficient LR .00005, range weight 5, and a fresh start from D. No
parameter was changed in response to the gradual candidate's IJBC result.
The training-data-gated fallback was already prepared and prioritized before
that result existed. Its GPU smoke remains required before full repair.

## Projected training outcome (before its final metric)

Projected training job 380734 failed after 24m08s in epoch 2, shortly after
opening site ten. As in the free-coefficient arm, the loss finite guard failed
before an optimizer update. It had progressed beyond the free arm's ninth-site
failure, but neither completed the curriculum. Its fixed epoch-1 diagnostic
checkpoint has SHA-256
f55ea7cc2b510e2a7d21911859c51f829a83d01d1055d41959530cdb0bd35c07;
final IJBC job 380792 is still running. No metric from that job has been used.
The prepared gated repair and its smoke prerequisite retain their parameters.

Future unclipped accuracy training now also writes continuation_best.pt as an
independent atomic copy of the complete selected training state. It contains
its own matching backbone, head and optimizer; later last.pt replacement or
in-place writes cannot alter that snapshot. This supports an internal-only
continuation without mismatching checkpoint epochs. Legacy recovery namespaces
that predate the new optional flags remain accepted by the resume-policy
check. Twenty-nine focused recipe/recovery/campaign tests passed after these
changes. The two already-failed training runs are not changed retroactively.

## Projected final diagnostic result and gated repair execution

Projected final IJBC 380792 and fixed holdout 380893 completed. The holdout
observed 223,296,064 non-finite values across 55,460 forwards, so TAR was
withheld. IJBC reported 18.9139438564% TAR at requested FAR=1e-4 (actual
.00009994288612547199), but 4,287 of 938,750 augmented embedding rows were
non-finite, with 1,290,206,600 non-finite boundary observations. This remains
an **invalid diagnostic** score despite the structurally polynomial export.
See unclipped_degree2_projected_result.json. It does not establish a valid
accuracy improvement and is not used to tune subsequent parameters.

Gated smoke 380897 completed on H200 x16 with two joint updates and
JOINT_SMOKE_OK. It trained 234 tensors (Conv/spatial-BN tensors plus 25 shared
coefficient triplets), changed the stem convolution by up to .00010278821,
and passed frozen-tensor checks on save. Its 160-forward first-site gate was
finite but had maximum ratio 6.4301, above the advancement threshold. This
is execution validation of the training machinery, not deployment safety.

Full gated repair 380900 started on nodes 173–174 after smoke completion,
using the unchanged declared policy and the original D source. Main batch is
2048 across 16 H200s, plus up to 512 exact replay/prefix probes per update.
The production first-site gate covers 40,960 training forwards: initial max
ratio 6.8518133, after 250 updates 6.8157592. Both prefix gates have zero
non-finite rows, but neither passes the <=2 advancement criterion. The suffix
remains a training-only bounded auxiliary computation. No all-25 unclipped
inference claim is made. Further allocation decisions will use these MS1MV3
measurements only; no new IJBC evaluation is used during repair.
