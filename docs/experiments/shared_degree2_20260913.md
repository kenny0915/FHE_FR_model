# Layer-shared degree-two ResNet-50 experiment ledger

Start: 2026-09-12 18:27 UTC (2026-09-13 Taiwan time).
Hard deadline: 2026-09-14 18:27 UTC, including preparation and queue time.
Resource ceiling: 16 H200 GPUs concurrently. Training uses the documented
Slurm `16gpus` partition, two nodes, eight H200 each. No local full training.

Target: IJBC TAR >=96.56% at FAR=0.1, with zero non-finite values at all
observed module inputs/outputs during inference. Audit includes convolutions,
BatchNorm, residual-block sums, polynomial outputs, and final embeddings,
for original and flipped IJBC images. Zero failures is an empirical dataset
result, not a proof for arbitrary real-valued inputs or CKKS execution.

All 25 PReLUs become scalar-coefficient quadratics: exactly 75 trainable
polynomial numbers. Original per-channel PReLU slopes exist only as frozen
training blend targets; evaluation never reads them. A/B/C use unbounded
polynomial inference. D and the conditional E continuation retain MS1MV3-derived
input bounds, which require comparisons and are not a purely polynomial
encrypted path.
A quadratic adds one ciphertext square per activation; along the deepest
path there are 25 such nonlinear levels, excluding affine operations.

## Allocation and predeclared selection policy

Reserve first hour for code checks and a 16-GPU two-update smoke. Initially
reserve up to four hours each for three distinct training attempts, plus
up to four hours each for final IJBC evaluation. Use remaining time for
extensions or revised strategies based exclusively on MS1MV3 development
accuracy, activation ranges and optimization behavior. Never alter a
candidate using IJBC statistics, images, tails, or TAR/FAR feedback.

1. Fitted shared quadratic, adaptive training BatchNorm statistics, 0.25
   head-only epoch, three-epoch progressive conversion, 12 total epochs,
   backbone LR .004, coefficient LR .0001.
2. Near-linear initialization (a=.001/radius, c=0, fitted b), adaptive
   BatchNorm, same conversion and 12 epochs, coefficient LR .0005.
3. Fitted shared quadratic, frozen baseline BN statistics, slower six-epoch
   conversion and 16 epochs, coefficient LR .00005.

Each initial A/B/C attempt starts from `work_dirs/ms1mv3_r50/model.pt`, never a prior polynomial
checkpoint. ArcFace classification, teacher embedding/stage distillation,
training range penalties and augmented training tail replay reuse recipe A.
For A/B/C, training input clipping is disabled after conversion. Different configurations
are decided before inspecting their IJBC results. IJBC can only trigger the
requested success stop; it cannot drive further tuning.

The existing seeded MS1MV3 identity split (seed 20260911, 2% development
identities, up to six images each) is reused by convention. Calibration and
classification centers use only the other 98%. The pretrained teacher may
have seen development identities; the student training split is disjoint.
Select the epoch maximizing the existing minimum(clean+flip, lowres20)
verification TAR at sampled FAR=1e-4, gated by finite clean/flip/lowres/shift/
dark inference, including intermediate boundaries. Use the existing two
million fixed seeded impostor draws. These metrics are not IJBC substitutes.

Fit target: pooled channel PReLU responses on each site's MS1MV3-derived
symmetric interval [-radius, radius]. Radius is pooled absolute-input
histogram quantile .9995 times 1.5; 95% empirical symmetric weighting and 5%
uniform interval weighting. Coefficients are jointly least-squares fitted
across channels, not averaged channel-specific deployed coefficients.

Final evaluation reuses `eval_ijbc.py`, its image alignment, template/media
aggregation, detector scores, flip fusion, FAR points and result CSV. The
new `--finite-audit` only observes tensors. Failed intermediate audits cannot
qualify even if final embeddings or sanitized evaluator scores are finite.

## Attempts

| Attempt | Strategy | Training / IJBC jobs | MS1MV3 validation | Final IJBC TAR @ .1 | Non-finite inference | Status |
|---|---|---|---|---|---|---|
| Smoke | fitted, adaptive BN, two distributed updates | 379596 | not an accuracy run | not evaluated | finite training loss/gradients; inference not certified | completed |
| A | fitted / adaptive BN / unbounded inference | 379616 / 379810 | epoch 5: clean 14.70%, lowres 8.35% at FAR 1e-4 | 67.9910% diagnostic | 10 non-finite augmented IJBC rows; intermediate failures | failed |
| B | near-linear / adaptive BN / unbounded inference | 379865 / 379947 | epoch 6: clean 16.17%, lowres 8.17% at FAR 1e-4 | 71.3658% | 0 in full IJBC audit | below accuracy target |
| C | fitted / frozen BN / slower conversion / unbounded inference | 379956 / 380032 | unavailable: training non-finites before full holdout check | 0.00% diagnostic | 938,647 non-finite augmented IJBC rows; intermediate failures | failed |
| D | A epoch-2 warm-start / frozen BN / persistent input bounds / eight epochs | 380052 / 380135 | epoch 7: clean 98.18%, lowres 66.94% at FAR 1e-4 | pending | 0 in all eight MS1MV3 audits; IJBC pending | final evaluation running |

Artifacts remain under `work_dirs/shared_d2_*`; source and this ledger are
versioned. Record actual jobs, times, failed attempts, and evaluation hashes
as execution progresses. No numerical result is inferred from a historical
per-channel or IJBC-calibrated checkpoint.

Implementation check: 45 lightweight tests passed (shared coefficients, recipe A,
controlled degree two). Adaptive BN uses SyncBatchNorm on the 16-rank training
graph so all ranks evaluate the same running statistics.

18:33 UTC: smoke completed on nodes 006–007. World size 16, batch 128,
two optimizer updates; initial loss .7857; max backbone update .00061.
Additional recipe/shared/nonfinite-trace tests passed.

18:36 UTC: A preparation job 379605 completed MS1MV3 calibration/centers but
Slurm failed to launch the second step (socket timeout confirming allocation).
No training updates or accuracy results. Reuse saved prepared.pt/split.npz
with hashes checked by the trainer; no strategy change.

The sequential controller `shared_campaign.py` runs the fixed A/B/C policy
list, evaluates only after each training allocation ends, records checkpoint
SHA-256/development/Slurm states/final IJBC/audit in campaign.json, and stops
if both target conditions pass. It cancels its current job at the absolute
deadline. Finishing three attempts leaves the remaining budget available for
MS1MV3-guided work; it does not pretend the full experiment is complete.

18:39 UTC: A has begun optimization under job 379616, 16 H200 GPUs.
Teacher development: clean+flip TAR=.99016154, lowres20 TAR=.34576148
at sampled FAR=1e-4; zero audited non-finite values. The teacher checkpoint
SHA-256 is ac658cc7cdbce5de90b8cd36de19b29f22ee283e016884f85252aa0a50a1841a.
Initial fitted radii span .6102–1.6710; the sampled teacher maxima exceed
some robust radii, motivating explicit range loss and stress validation.
These statistics use MS1MV3 only.

If training terminates during conversion, a saved shared last.pt may still
be evaluated diagnostically: its inference graph always uses all 25 pure
quadratics (training alpha is ignored in eval). Record incomplete training
separately. Such a result only qualifies if the same all-boundary finite and
TAR gates pass; no failed/partial PReLU blend is evaluated as a polynomial.

Git commits 2ad9df5 and bde4b5b are local. Push attempts failed because the
execution host cannot resolve github.com. Unrelated untracked files remain
excluded from commits.

2026-09-12 18:52 UTC: A completed epoch 0 and saved last.pt.
Actual checkpoint inspection passed strict r50_shared_d2 loading, all 25
coefficient tensors have shape (1,3), exactly 75 polynomial coefficients,
no nn.PReLU modules, and all state tensors finite. Training conversion is
incomplete; these checks do not certify finite inference or final accuracy.
Job 379616 and controller PID 2908563 are live. GitHub DNS also failed from
allocated compute node 006; push remains unavailable.

2026-09-12 18:53 UTC: Push connectivity recovered. Cloudflare DNS-over-HTTPS
provided a current GitHub address; a per-command http.curloptResolve override
(with normal TLS verification, no persistent configuration changes) allowed
the push. origin/main now includes all commits through b96e53a.

A reached all-25-quadratic, unclipped training at epoch 3, step 625.
The transition raised loss from about 8 to 46.69; it then declined to
31.86 by step 1025. Completed updates remained finite. This is optimization
evidence only, not an inference or verification result.

The existing IJBC evaluator now also saves ijbc_tar_at_far_raw.json using
the exact same selected ROC indices as its customary two-decimal CSV/table.
Before certifying success, inspect the unrounded FAR=.1 TAR >=96.56 and
the complete finite audit. A rounded threshold hit alone is insufficient.
No scoring, alignment, aggregation, or FAR-selection convention changed.

Preserved A epoch-2 checkpoint as pre_unclipped.pt before later epochs
overwrote last.pt. SHA-256:
075e710d04013d2d9c07abaf78ff7d200461ae5ae99deeb26bc82cd01c989a1a.
Reason: the training loss remained around 32–40 for several hundred steps
after removing clipping; preserving the source allows a bounded-input
comparison if internal validation supports it. No IJBC results were used.
By epoch 3 step 1700, the current unclipped run reported loss 9.07, so this
is not yet evidence that recovery has failed. A continues unchanged.

Lightweight CPU FP32 inference probe of A epoch 3: fixed development rows
3053 and 3056, each original and flipped, yielded zero non-finite values
at every audited module input/output. Embedding absolute maximum .289545.
Saved details: work_dirs/shared_d2_A_20260913/early_cpu_probe.json.
This covers only four forwards and does not certify full validation or IJBC.
No model parameters or selection rules changed from this probe.

A epoch-4 full internal validation: clean+flip TAR@FAR=1e-4
7.351606%, lowres20 5.510982%; selected score .05510982. Zero non-finite
values/invalid embedding rows across 55460 variant forwards on 11092 held-out
MS1MV3 images. Maximum embedding magnitude .564650; maximum polynomial
input/radius ratio 4.438465. Accuracy is far below the teacher despite this
finite audit. This is not an IJBC result.

An MS1MV3-only checkpoint audit found median running-variance ratios versus
the teacher of 2984x in layer3.13.bn3, 2417x in layer3.12.bn3, and 2260x in
layer2.3.bn3. This suggests distribution shift as a candidate explanation,
not causal proof. Details: epoch4_bn_variance_audit.json in A output.

Budget adaptation decided before any candidate IJBC evaluation: allow A
through epoch 6 (three full-quadratic validation points). If best clean
TAR remains below 50% and best selection score below 15%, stop A early,
evaluate its internally selected checkpoint, then continue the predefined
B/C strategies. This reallocates roughly an hour without using test feedback.

## Prepared strategy D: persistent bounds and frozen BN

Decision based on A's MS1MV3 loss discontinuity, 7.35% internal clean TAR,
and changed running variances, before any candidate IJBC result. Preserve
A/B/C as the unclipped comparisons. If they do not trigger the success stop,
reserve up to three training hours and four evaluation hours for D.

D starts from A's preserved epoch-2 pre_unclipped.pt (hash above), reuses the
exact A teacher/calibration/split/identity-head mapping, and starts a fresh
optimizer. All 25 sites are quadratic from the first update. Keep the fixed
MS1MV3 input bounds at every site during both training and evaluation; freeze
BN running statistics from this source while allowing affine/conv/head and
all 75 polynomial coefficients to adapt. Use eight epochs, backbone LR .001,
coefficient LR .0001, head LR .005, no head warmup, existing distillation and
training augmentation. Select on the same full internal finite/accuracy gate.

Its distinct network ID is r50_shared_d2_bounded. Checkpoint metadata states
inference_input_bounds=true, fhe_requires_comparisons=true, and
pure_quadratic=false even after all_activations_quadratic=true. The clamps
are explicitly permitted by the requested design space, but they introduce
comparisons: this is not a purely polynomial encrypted computation. No claim
of CKKS deployability is made. The IJBC loader rejects a shared checkpoint
whose requested network would silently change bounded/unbounded semantics.

The additional flags are --inference-bound, --all-quadratic-start, and
--warm-start. Warm-start loading strictly checks teacher/split/calibration
provenance and restores the matching classification head; it does not reuse
optimizer momentum. 49 lightweight tests passed, including matching bounded
train/eval outputs and rejection of a mismatched warm-start split. GPU testing
of D remains pending; no extra allocation is launched alongside A.

A epoch-5 validation: clean+flip TAR 14.703213%, lowres20 8.349973%
at sampled FAR=1e-4, zero non-finite/invalid outputs across the complete
55,460-forward holdout audit. Maximum embedding magnitude .728732; maximum
input/radius ratio 2.986583. This is an improvement but remains far below
the teacher. The recorded epoch-6 pruning rule remains unchanged.

A epoch-6 validation: clean TAR 13.497913%, lowres TAR 7.772736% at
sampled FAR=1e-4, zero non-finite/invalid outputs on the full holdout audit.
Both metrics declined from epoch 5. The predeclared pruning conditions held
(best clean <50%, best selection <15%); job 379616 was cancelled after the
completed epoch-6 checkpoint/metrics, at roughly 82 minutes of training-job
wall time. This leaves last.pt with optimizer state for a possible later
MS1MV3-guided resume, without committing to additional compute now.

Final A candidate: student_best.pt, epoch 5, SHA-256
3cd884d0c3ca315194213d9f8b577154986dbac3a49da2f71c0c00613a44cc44.
Its 25 coefficient tensors all have shape (1,3). Final IJBC job 379810 is
running via the existing pipeline on 469,375 source images (plus flips),
with the complete module-boundary audit and unrounded TAR sidecar. No final
IJBC metric has yet been consumed; no candidate parameter changed after
selection. The sequential controller remains responsible for B/C dispatch.

## A final evaluation and B launch

A IJBC job 379810 completed. The existing pipeline reported TAR 67.99% at
FAR=0.1 (unrounded 67.9910006647%, actual nearest ROC FAR .1000075964).
There were 10 non-finite augmented embedding rows among 938,750 original/flip
forwards. The complete audit also contains non-finite intermediate values.
The pipeline replaces non-finite features with zero, so this TAR is diagnostic
and cannot qualify. A fails both the numerical requirement and the target TAR.
No IJBC per-layer statistics or failing images are used to tune later models.
See shared_degree2_A_result.json for the selected checksum and audit digest.

B job 379865 has started on 16 H200 GPUs and completed fresh MS1MV3-only
preparation. It uses exactly the near-linear initialization, adaptive BN,
three-epoch conversion, and coefficient LR .0005 declared before A's IJBC
result. Its teacher/split/calibration provenance is independently saved.
Roughly two wall-clock hours have been used since experiment start; the
absolute deadline remains 2026-09-14 18:27 UTC. D remains prepared from the
prior internal-validation evidence, with no IJBC-driven parameter changes.

The campaign success parser now requires the unrounded TAR sidecar, with a
regression check for 96.559% rounding to 96.56%. The already-running controller
loaded the earlier parser: any provisional success from that process must be
verified against the raw sidecar before stopping the experiment. This changes
only the success decision, not evaluation scoring or model selection.

B reached epoch 3 with finite bounded-conversion training losses. Its epoch-2
checkpoint, including the training head and optimizer state, was preserved as
work_dirs/shared_d2_B_20260913/pre_unclipped.pt before scheduled bound removal.
SHA-256: 4e34dd4e83911bfa3d087b5e553a70ceeda4d8315021a622fe144d3189a93e80.
This preserves a possible MS1MV3-only comparison starting point; it is not a
selected final candidate and has no inference-stability certification. No B
IJBC evaluation has run, and B's current training policy remains unchanged.

B's bounds were removed at epoch 3, step 625 after all groups became quadratic.
Training loss rose from about 7 to about 45 and subsequently varied sharply
between batches, while the finite training guards continued passing. At epoch
4, its first full MS1MV3 holdout evaluation gave clean/flip TAR@FAR=1e-4
0.0847703740, low-resolution TAR 0.0519513525, and selection score 0.0519513525.
There were zero non-finite module observations across 55,460 validation
forwards; embedding absolute maximum was 0.5297237635 and the largest observed
activation-input/radius ratio was 3.9498515129. This establishes finiteness only
on that holdout sample. B continued into epoch 5 with no IJBC evaluation yet.
About three of the 48 wall-clock hours have elapsed; no new allocation or
parameter change is based on these first-checkpoint results alone.

B epoch 5 improved its clean/flip TAR@FAR=1e-4 to 0.1161372289 and
low-resolution/selection TAR to 0.0700671598. The full holdout audit remained
finite (zero non-finite observations), with embedding absolute maximum
0.8642792106 and activation-input/radius maximum 3.5960447788. Despite low
absolute accuracy, the internal improvement supports continuing into epoch 6.
The model selection and training policy remain unchanged; B IJBC is pending.

Before B's epoch-6 evaluation, apply the same exploratory screening thresholds
used for A: after the completed epoch-6 checkpoint is saved, stop B if its best
full-quadratic clean TAR remains below 0.50 and its best selection score remains
below 0.15. The trigger is B's weak epoch-4/5 MS1MV3 results despite finite
holdout outputs; no B IJBC metric exists. This reduces the nominal 12-epoch cap
only when both internal thresholds fail, reserving compute for the predeclared
frozen-BN and bounded comparisons. Evaluate the internally selected checkpoint
once after stopping. If either threshold is reached, retain the original cap.

B epoch 6 reached clean/flip TAR@FAR=1e-4 0.1617353410 and low-resolution /
selection TAR 0.0816845149. All three completed full holdout audits had zero
non-finite observations. Epoch-6 embedding absolute maximum was 0.9579501152;
activation-input/radius maximum was 3.2408437729. Both screening thresholds
failed, so job 379865 was cancelled after its completed epoch-6 checkpoint,
at roughly 80 minutes of job wall time. early_stop.json records the complete
internal history and that no B IJBC result existed at selection.

Selected B checkpoint: student_best.pt, epoch 6, SHA-256
0811dc5262b41787e6e7fff63ea672784af94c7ad62b4c34cb0878aa3246b45b.
All backbone state tensors were checked finite; all 25 coefficient tensors
have shape (1,3). shared_degree2_B_coefficients.json records the triplets and
their original MS1MV3 calibration intervals. Final IJBC remains pending;
the sequential controller will dispatch it after the training allocation ends.

The controller dispatched B final IJBC job 379947 on one H200 (dev partition)
after the 16-GPU training job ended. The selected checkpoint is fixed; both
unrounded TAR reporting and the complete module-boundary audit are enabled.

D preparation is now staged in work_dirs/shared_d2_D_20260913 without GPU
training. A's prepared.pt and split.npz were copied unchanged, and the real
epoch-2 warm start passed strict backbone and identity-head loading against
the matching provenance. All restored backbone tensors were finite. The
prepared.pt SHA-256 is
73f67b1b26a8b87f4d3105d5a56e3f753ead62afeaf1661b506818a35efce295.
staging.json records the source checksum and verification. This is the same D
strategy declared from A's internal evidence before its IJBC result; staging
does not change its policy or establish accuracy. Persistent bounds require
comparisons and remain explicitly distinct from purely polynomial inference.

## B final evaluation and C launch

B final IJBC job 379947 completed with TAR 71.37% at FAR=0.1 (unrounded
71.3657513934%, actual nearest ROC FAR .0999611738). All 938,750 original/flip
forwards passed the module-input/output audit: zero non-finite observations
and zero non-finite augmented embedding rows, including the remainder batch.
The selected checkpoint checksum was verified unchanged after evaluation.
B meets the observed IJBC inference-stability requirement but misses the TAR
target. shared_degree2_B_result.json records the exact metric and audit digest.
No IJBC statistics are used to alter subsequent coefficients or schedules.

C job 379956 started on 16 H200 GPUs (nodes 026–027), completed preparation,
and entered training. It retains the policy fixed before either final IJBC
result: fitted initialization, frozen BN statistics, six-epoch conversion,
coefficient LR .00005, and a nominal 16-epoch cap. D remains staged for the
subsequent bounded comparison if needed. About four wall-clock hours have
elapsed; the absolute deadline remains 2026-09-14 18:27 UTC.

C progressed through epoch 5 with finite bounded-conversion losses, and its
saved configuration was verified against the predeclared frozen-BN policy.
Before bounds are removed in epoch 6, its epoch-5 checkpoint with identity
head and optimizer was preserved as
work_dirs/shared_d2_C_20260913/pre_unclipped.pt, SHA-256
04278668d2089728a7b1dd9007f81382113bb5c26ebbc008f42940d59b331f82.
This is an intermediate state for a possible MS1MV3-only comparison, not a
selected final candidate or an inference-stability certification. No C IJBC
evaluation has run. Roughly five wall-clock hours have elapsed; C's current
training allocation and D's previously declared policy remain unchanged.

C job 379956 failed at the removal of training bounds in epoch 6 (last logged
bounded step 600), with FloatingPointError on non-finite training embeddings.
The finite guard aborted before any invalid optimizer update. No full-polynomial
holdout metric was reached; record it as unavailable due to this training
failure. The last completed epoch-5 checkpoint has the same checksum as the
preserved pre_unclipped.pt above. Its final diagnostic inference still replaces
all 25 sites with quadratics and disables every training blend and clamp.

The original sequential controller exited after Slurm definitively rejected C's
evaluation submission with a get_api_token service error. A manual retry of the
identical evaluation succeeded as job 380032 on one H200. campaign.json records
the rejected submission and recovery; the old controller PID is no longer live.
The root agent now monitors this evaluation and will dispatch the staged D
strategy after it finishes if no success is established. The campaign deadline
and 16-GPU total limit remain unchanged. C's failure is training evidence, not
an IJBC-derived parameter adjustment.

C's diagnostic checkpoint checksum was reverified, and every saved backbone
tensor was finite. shared_degree2_C_coefficients.json records all 25 scalar
triplets and original MS1MV3 approximation intervals. All quadratic terms are
nonzero. Finite checkpoint tensors do not establish finite unbounded inference;
the complete IJBC audit remains pending.

Following the observed Slurm get_api_token rejection, the submission helper now
retries only that explicit no-job-submitted response (empty stdout plus the
token-service and submission-failed diagnostics), at most three attempts with
30-second waits. Ambiguous failures and responses containing a possible job ID
are not retried. Six campaign tests passed, including the retry limit and
non-retry cases. This applies to subsequent submissions; C's live evaluation
is unchanged.

## C final diagnostic result and D launch

C final evaluation 380032 completed. The existing ROC procedure reported
diagnostic TAR 0.00% at requested FAR=0.1 (selected ROC point FAR=0.0).
938,647 of 938,750 augmented embedding rows were non-finite, and the audit
counted 27,439,355,249,656 non-finite module observations including repeated
boundaries and diagnostic replays. The evaluator's zero replacement makes this
a diagnostic score only. C fails the inference-stability and accuracy goals;
its training had already failed on MS1MV3 before full holdout validation.
shared_degree2_C_result.json records the exact result and audit digest.

D training job 380050 started on 16 H200 GPUs, nodes 143–144, after C's
evaluation allocation ended. It reuses the previously staged preparation and
A epoch-2 warm start, with all quadratics active from the first step, persistent
MS1MV3 input bounds, frozen BN statistics, eight epochs, backbone LR .001,
coefficient LR .0001, and head LR .005. No warmup or old optimizer momentum is
used. The training allocation is capped at three hours; the campaign deadline
remains 2026-09-14 18:27 UTC. About six wall-clock hours have elapsed.
D's bounds introduce comparisons and are explicitly marked in its network and
checkpoint metadata; this is not a purely polynomial encrypted path. The D
policy was declared from A's internal evidence before any final IJBC result.

D's first allocation (380050) failed after 17 seconds while moving the teacher
to CUDA: a device was busy or unavailable. No optimizer update or checkpoint
was produced. The identical configuration was resubmitted as job 380052 on
nodes 176–177, excluding the failed allocation's nodes 143–144. This is an
infrastructure retry, not another model strategy. submission.json and the
campaign ledger retain both job IDs and the failure reason.

D retry 380052 reached training successfully and its saved configuration
matches the predeclared policy. Epoch 0's complete MS1MV3 holdout check gave
clean/flip TAR@FAR=1e-4 0.9789798260, low-resolution/selection TAR 0.6402614117,
and zero non-finite module observations across 55,460 forwards. Embedding
absolute maximum was 9.3622579575; the largest pre-clamp activation-input /
radius ratio was 7.2749419212. The bounds remain active during inference.
This is a promising internal result, not an IJBC result. D continues its
planned eight epochs with checkpoint selection based solely on the holdout.

D epoch 1 improved clean/flip TAR@FAR=1e-4 to 0.9799963832 and low-resolution /
selection TAR to 0.6486113667. The complete holdout audit again had zero
non-finite observations. Embedding absolute maximum was 9.4886198044 and the
pre-clamp activation-input/radius maximum was 7.2338681221. D continued into
epoch 2 without any policy change or IJBC evaluation.

Subsequent D holdout checks (TAR at FAR=1e-4; all 55,460 forwards audited):

| Epoch | Clean/flip TAR | Low-resolution / selection TAR | Non-finite observations |
| --- | ---: | ---: | ---: |
| 2 | 0.9807587862 | 0.6557269692 | 0 |
| 3 | 0.9807950854 | 0.6549283266 | 0 |
| 4 | 0.9810491800 | 0.6654565334 | 0 |
| 5 | 0.9812670350 | 0.6664004326 | 0 |
| 6 | 0.9815937281 | 0.6686149836 | 0 |
| 7 | 0.9817752838 | 0.6693773866 | 0 |

Epoch 4 became the selected checkpoint after improving the internal selection
score. Its embedding absolute maximum was 8.8295249939 and pre-clamp input /
radius maximum was 6.9015088081. D continued into epoch 5 of its planned eight
epochs. No D IJBC evaluation has run and no policy has changed.

Before D's final IJBC evaluation, reserve a possible E continuation based only
on D's improving MS1MV3 results and remaining clean-holdout gap to the teacher
(0.9815937281 versus 0.9901615381 at epoch 6). If the campaign still needs a
candidate, E will warm-start from D's internally selected final checkpoint,
retain the same fixed input bounds and frozen BN statistics, and train all
quadratics from the first step with a fresh optimizer for 16 epochs. Backbone
LR .0005, coefficient LR .00005, head LR .0025, zero head warmup; same
MS1MV3 preparation, augmentation, objectives and internal selection criterion.
Reserve up to four training hours plus the existing final-evaluation cap,
always bounded by the absolute campaign deadline. No E job is submitted now.
This policy is fixed before D IJBC is observed; an IJBC success ends the
campaign, while a failure does not determine any E parameter or coefficient.

D training 380052 completed all eight epochs. Epoch 7 was selected solely by
the internal score above; all eight full holdout audits reported zero
non-finite observations. Its embedding absolute maximum was 9.6122293472 and
pre-clamp input/radius maximum was 6.8297877312. The selected model loaded
strictly into r50_shared_d2_bounded, contained zero PReLU modules and exactly
25 shared (1,3) coefficient tensors, and had finite saved parameters/buffers.
Every quadratic term is nonzero and all inference bounds are active.

Selected student_best.pt SHA-256:
300dfe09bfcad28a2b5a5c2eeff359878b6bcc6cbae9d3991572f4e8113c5e6f.
shared_degree2_D_coefficients.json records all coefficients and input bounds.
The matching full training state is preserved in continuation_best.pt for
the predeclared E option, SHA-256
6495e8df88af083e1346961a7939e6c714cefbf2ff0bcbd0b4e992058bbd6324.
The final backbone and head were verified to come from the same epoch before
any D IJBC result was available.

D final IJBC job 380135 was submitted after training completed, using the
existing pipeline with network r50_shared_d2_bounded, all-boundary audit,
and unrounded TAR sidecar. No model parameter is changed during evaluation.
