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
