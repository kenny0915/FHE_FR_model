# Clipping mismatch × MS1MV3 identity supervision

Status: experiment design, not an implemented trainer or a submitted campaign.
This specifies the next controlled experiment after relational recovery. It
does not claim that 96% has been reached, or that the proposed runner exists.

## Questions and primary outcome

1. Does reducing the difference between clipped training and unclipped
   deployment recover recognition accuracy?
2. Does MS1MV3 identity supervision add value beyond matching PReLU teacher
   embeddings and intermediate features?
3. Do these two changes help each other, or does identity optimization amplify
   the quadratic tails when clipping is removed?

The external target remains **strict IJB-C TAR at measured FAR <= 1e-4 >=96%**.
Report TAR at 1e-5/1e-6 and numerical failure counts as secondary outcomes.
Small inference nonfinite rates are permitted; the user has not specified a
numerical acceptance ceiling. Report the accuracy/failure tradeoff without
silently declaring an arbitrary rate acceptable.

IJB-C images, identities, templates, range statistics and failure manifests
must not enter training, class-center preparation, routing, calibration,
early stopping or checkpoint selection. IJB-C is a final report only. Prior
benchmark results are already known, so this is an exploratory experiment,
not a claim that IJB-C is an untouched test set.

## Four arms: a 2 × 2 comparison

| Arm | Differentiable backbone path | Identity objective |
| --- | --- | --- |
| A: clipped KD | Clipped for every row | None |
| B: aligned KD | Unclipped eligible rows; clipped fallback otherwise | None |
| C: clipped KD + ID | Clipped for every row | ArcFace |
| D: aligned KD + ID | Unclipped eligible rows; clipped fallback otherwise | ArcFace |

The clipping factor compares **two training policies**, not universally
unclipped training: B/D deliberately retain a measured fallback. Do not label
them pure-unclipped or discard fallback rows from the denominator. All rows
receive the same applicable task losses; there is no extra repair loss.

At the fixed final epoch compare B-A and D-C for the path-policy effect,
C-A and D-B for the identity effect, and `(D-C)-(B-A)` for interaction.
These are percentage-point differences, not relative percentages. One seed
is an exploratory comparison, not evidence of statistical significance.

## Common source and data

- Use the original unscaled progressive best, SHA-256
  `3b3298217fcc54a368b045f50d85e8e0d9c845f305e7f1212640838c160f6b39`.
  Do not start from a BTS export or select relation01 because it scored best
  on IJB-C. Reuse the original PReLU teacher at
  `work_dirs/ms1mv3_r50/model.pt`, SHA-256
  `ac658cc7cdbce5de90b8cd36de19b29f22ee283e016884f85252aa0a50a1841a`.
- Seed 20260929. Split MS1MV3 identities 98% train / 2% development once,
  before creating centers or diagnostic manifests; all four arms use the
  identical saved source-row and identity maps. Verify RecordIO labels
  against the manifest. Development identities have no classifier columns.
- This is a holdout from the **new fine-tuning** only. The teacher and
  progressive initialization may already have seen these identities; do not
  claim an identity-unseen pretraining benchmark.
- Training augmentation is identical across arms: existing random horizontal
  flip, lowres/photo/crop probability .05 each. No stress/pathology examples,
  replay, adversarial samples, or class-balanced sampling in this first
  comparison. Sample by image as before. Use source-index/epoch-keyed
  augmentation or saved RNG replay so routing does not change augmentation
  draws. Audit actual source ordering rather than promising bitwise CUDA
  determinism.
- Disable relational KD in all arms. Freeze all 25 quadratic coefficient
  triplets, slopes, fitting intervals and BN running buffers. Train backbone
  Conv/BN affine/embedding projection parameters. All models have dropout 0.
- The initial approximation target remains channelwise PReLU on each saved
  `[-lam_fit[c], lam_fit[c]]`; `lam_reg=.6*lam_fit`. No interval widening,
  recalibration or BTS rescaling in this experiment.

Because this introduces a common fine-tuning holdout and routing diagnostics,
A must be rerun. Historical pointwise results are context, not the controlled
A measurement for this matrix.

## Phase 0: diagnose the mismatch without training

Use a frozen manifest of up to eight distinct images per development identity
and both orientations, with no training augmentation. Also retain a separate
8,192-row training-only diagnostic manifest. Neither manifest uses IJB-C.

On exactly the same tensors compare the source student in two eval modes:
all quadratic inputs clipped to `lam_fit`, and completely unclipped. Compare
both with the PReLU teacher using identical preprocessing. Record:

- Per-layer input/fit-radius quantiles (p50/p95/p99/p99.9), maximum and number
  of rows containing any exceedance; record nonfinite rows separately.
- Per-image embedding cosine difference between the two student graphs and
  their respective teacher KD losses. These differences are reported on the
  paired finite subset with its explicit coverage, not treated as an
  all-sample recognition score.
- Verification TAR on a fixed MS1MV3 development pair manifest, plus failure
  counts. Use distinct-image genuine pairs and two million seeded unique
  impostor pairs. Archive the pair IDs. FAR=1e-4 here is a development proxy,
  not the IJB-C template protocol or two million independent identities.
- For all-sample development scoring, either orientation failing zeros both
  entire embeddings before flip fusion. Keep every verification pair, including
  zero-vector pairs; do not drop failed source images.

A large clipped/unclipped embedding gap motivates alignment but does not prove
that it causes the full IJB-C gap. Clipped eval is diagnostic only and is never
presented as an FHE-deployable model.

## Shared routing and nonfinite policy

For every augmented training microbatch, **all four arms** run the same
no-gradient, fully unclipped probe using their current model and frozen BN
statistics. A row is eligible only if all polynomial inputs/outputs and its
embedding are finite, the embedding norm is nonzero, and every channel input
satisfies `abs(x)/lam_fit <=4`. The guard is fixed before training; it is a
screening heuristic, not a proof of finite backward or a deployment bound.
Check all 25 activation sites, not only BTS or stage outputs.

Discard the probe graph and clear captured tensors. For A/C, compute fresh
clipped training forwards for both eligible and fallback partitions. For
B/D, compute a fresh unclipped forward for the eligible partition and a fresh
clipped forward for the fallback partition. The same source row keeps its
same augmentation, label and teacher target through routing. Never reuse a
nonfinite probe output in the differentiable loss, even behind a zero mask.
The actual deployed graph is always unclipped with no router or fallback.

Compute sums of per-row losses across both partitions, divided by the original
microbatch count. For feature hints use the same detached teacher variance
normalizer from the full original microbatch for both partitions. Averaging
partition means equally or recomputing hint normalization per partition would
change the loss when the eligible fraction changes. The A/C partitioned result
must be tested against an unsplit clipped reference for loss and gradients.

All four arms therefore retain the same source rows; none removes rare tails
or adds a new prefix-repair objective. Routing masks can diverge as models
train, so log eligible/fallback rates and identity distributions per arm. If
B/D mostly fall back, report that the treatment had little unclipped exposure;
do not interpret a null result as evidence that clipping alignment cannot help.

If a differentiable forward, loss or gradient unexpectedly becomes nonfinite,
reject the entire in-progress optimizer update on every DDP rank, save source
IDs, augmented inputs, route masks and first offending site, and abort the
pilot. Do not silently skip batches, sanitize gradients, or dynamically change
guard thresholds. A revised policy would require a separately named experiment.
Accepting rare inference failures does not permit corrupt optimizer state.

## Identity supervision

Prepare one shared full-class ArcFace head for C/D from **training identities
only**: select up to four distinct source images per identity, compute teacher
embeddings in original and flipped orientations, normalize each embedding,
average them per identity, then normalize the class center. Save contiguous
identity remapping, selected rows, teacher hash and center hash. Reject missing,
zero-norm or nonfinite centers. C/D begin with bitwise identical centers.

Use the existing `recipe_a.IdentityHead` math: normalized embeddings and
class weights, scale 64, ArcFace margin ramp linearly from 0 to .5 over the
first epoch, then fixed .5. Train the full head, no PartialFC class sampling.
Initial head LR .01; weight decay 0. Backbone LR follows the common schedule.
These are predeclared pilot settings, not established optimum hyperparameters.
No backbone warm-up or additional training epochs are unique to C/D.

For all rows, including clipped fallback rows:

`L = L_cosine_KD + h(t)*L_hint + .01*L_range + id_weight*L_ArcFace`

`id_weight=0` for A/B and `.1` for C/D. A/B omit the classification term from
autograd entirely; do not compute a disabled nonfinite objective and multiply
it by zero. Log raw/weighted loss contributions and backbone gradient norms
from KD versus ID on fixed diagnostic steps. Large classifier loss alone does
not establish useful backbone supervision.

ArcFace normalization, square root, branching and softmax are plaintext
training-head operations. Export only the backbone. It retains the same
25 degree-2 polynomials, one square level per activation, and no new encrypted
inference operators.

## Common optimization and fixed reporting

- Six epochs, FP32/TF32 disabled, four H200s, microbatch 128/GPU, accumulation
  4, global batch 2048. SGD/Nesterov .9, backbone LR .0004, .5-epoch warmup,
  cosine floor .01, backbone weight decay .0005. Head LR uses the same
  warmup/cosine factors. Gradient norm cap 5 for backbone and, separately,
  5 for the classification head: a large head gradient must not silently
  reduce the backbone's step relative to A/B.
- Hint weight .1 -> .03 by the same cosine schedule. No causal-tail,
  operator-bound, relational, or extra repair objective.
- Repeat paired clipped/unclipped development diagnostics at initialization
  and after every epoch; LFW/CPLFW remain diagnostic. Finite-subset KD metrics
  and failure-inclusive verification metrics must be clearly distinguished.
  Compute the teacher's development baseline once.
- Save every epoch plus resumable backbone/head/optimizer state and complete
  provenance. Predeclare **epoch6 of all four arms** for final comparison.
  No zero-failure best-checkpoint gate, canary-based early stopping, or IJB-C
  checkpoint choice. Report an interrupted arm as incomplete.
- After all four arms finish, evaluate these four fixed checkpoints on full
  IJB-C, plus the original PReLU teacher under the same alignment, flip,
  faceness/template aggregation and strict-FAR protocol. Ignore BTS range;
  zero both views on a nonfinite source and retain all images/pairs. Record
  source/view failure counts and rates alongside TAR. This is final evaluation,
  not an input for another in-campaign sweep.

## Implementation and execution gates

The current `train.py` discards labels, hardcodes training clipping and lacks
this routing/classifier integration. Do not claim that toggling its existing
flags implements this design. Reuse components selectively:

- `recipe_a.identity_split`, label-checked `SourceRows`, center-building math
  and `IdentityHead`; **do not call recipe_a.prepare unchanged**, because it
  also recalibrates polynomial intervals and has different training policies.
- `polish_abcd` no-gradient probe pattern; **do not reuse arm D unchanged**:
  it repairs unsafe prefixes and has different causal/range objectives.
- Existing stable gradient checks, controlled checkpoint format and dynamic
  inference audits.

Before GPU execution, unit/integration tests must cover label-index mapping,
disjoint train/dev identity manifests, center source exclusion, C/D identical
head initialization, all-safe/all-fallback/mixed routes, mask and label
alignment, partition-invariant clipped loss/gradients, frozen BN/coefficients,
head gradients without teacher gradients, both student branches receiving
gradients, and unchanged unclipped export/reload output.

Use one DDP wrapper for the complete objective or an equivalently tested
synchronization strategy. Test ranks with different route occupancy, including
an empty partition; branch handling must not deadlock collectives or leave
unused-parameter reductions unresolved. Empty branches must not manufacture
a zero dependency through NaN/Inf tensors. Keep classification head state
separate from the exported inference state.

Then run each arm for two actual accumulated updates on four H200s using the
**full production class head** for C/D, not a 64-class substitute. Validate
finite updates, loss weights, BN/polynomial invariants, routing counts and
export/reload. Force mixed and all-fallback route cases in a bounded smoke,
without overriding production route decisions in its training checkpoint.

Only after those checks pass, run the four predeclared six-epoch arms. Start
with one four-GPU task at a time, following the successful serial NCCL setup;
exclude 25a-hgpn165 and 25a-hgpn167 until their previous initialization issue
is resolved. Use an eight-hour per-arm cap, record actual peak memory and
GPU-hours, and do not claim compute matching: probes cost extra versus older
training, and C/D add full-class-head computation. No full training on the
checkout host. No jobs have been submitted for this design.
