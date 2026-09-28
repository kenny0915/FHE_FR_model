# MS1MV3-only relational recovery, 2026-09-29

Target: strict IJB-C TAR at FAR <= 1e-4 of at least 96%. This is an experimental
target, not a promised outcome. A small inference nonfinite rate is allowed;
nonfinite optimization updates remain rejected. No numerical failure-rate
acceptance threshold has been invented on the user's behalf.

## Locked comparison

All three arms start from the same **unscaled original progressive best**,
SHA-256 `3b3298217fcc54a368b045f50d85e8e0d9c845f305e7f1212640838c160f6b39`.
Do not choose a previous recovery checkpoint based on IJB-C scores. Use the
original `work_dirs/ms1mv3_r50/model.pt` PReLU teacher. MS1MV3 supplies all
training images. LFW/CPLFW are diagnostics only, with no automatic schedule
changes, restarts, or early stopping. No IJB-C images, labels, templates,
ranges or failure manifests enter training or calibration.

| Arm | Cosine KD weight | Relational KL weight |
| --- | ---: | ---: |
| pointwise | 1 | 0 |
| relation01 | 1 | 0.1 |
| relation1 | 1 | 1 |

Common: 6 epochs, FP32, four H200 GPUs per arm, microbatch 128/GPU,
accumulation 4, global batch 2048, seed 20260929. Frozen BN running statistics
and quadratic coefficients. Train Conv/BN affine/embedding-head parameters.
SGD/Nesterov .9, global LR .0004, .5-epoch warmup and cosine schedule,
weight decay .0005, gradient norm cap 5. Reuse the prior fixed arm's
hint .1 -> .03, range weight .01, light crop/photo/lowres .05 each, and no
stress, pathology, replay, causal-tail or operator-bound objectives.

The pairwise objective forms a cosine similarity matrix within each **local
128-image microbatch**, excludes self-pairs, and minimizes
`KL(softmax(teacher_similarity / .05) || softmax(student_similarity / .05))`
averaged across anchor rows. There is no T-squared multiplier. Both student
endpoints have gradients; teacher endpoints are detached. It supplies soft
teacher relationship supervision, not an ArcFace identity classification
loss. Gradient accumulation does not enlarge the candidate set to 2048;
each anchor sees at most 127 other local images. Log raw relational loss
alongside the cosine, hint, and range terms.

This tests whether matching teacher similarity geometry adds value beyond
longer pointwise recovery. It does not isolate the effects of prior recipe
changes. Training inputs still clip at the saved fitting radius; deployment
is unclipped. That mismatch remains an explicit limitation, not silently
claimed to be solved by the new objective.

All 25 activations remain `c0+c1*x+c2*x^2`, originally approximating each
teacher channel's PReLU on its saved `[-lam_fit[c], lam_fit[c]]` interval.
Intervals are not recalibrated or widened; `lam_reg=.6*lam_fit`. One square
level per activation, unchanged inference graph. Softmax and normalization
in the new loss operate only during plaintext training, not encrypted inference.

## Output and evaluation policy

Save every epoch. Predeclare **epoch6.pt for every arm** as the primary final
comparison. The legacy finite-gated `student_best.pt` is diagnostic and is
not the selection rule. Record LFW/CPLFW failures even if their score is null.
Range logs and finite loss/gradient checks remain enabled. Exported checkpoints
do not guarantee zero nonfinite inference outputs.

Do not schedule IJB-C during training. After all arms finish, evaluate the
three fixed epoch-6 checkpoints under the same full IJB-C protocol, with
nonfinite-only source zeroing and all source images/pairs retained. Report
strict TAR and source/view nonfinite rates together; no finite-only scoring,
IJB-driven epoch selection or tuning loop. Previous IJB-C results have already
been observed, so this is exploratory benchmark reporting, not a newly
untouched test set. A matched original PReLU teacher evaluation is still
needed to quantify the remaining teacher-to-quadratic accuracy gap.

## Execution

`relational_recovery.slurm` can run up to three four-H200 tasks. It uses the
explicit verified Python interpreter through `accuracy_runtime.sh`.
Set `ACCURACY_CODE_ROOT`, `CHECKPOINT`, `TEACHER`, `DATASET_ROOT`, `CANARY_ROOT`,
and a fresh `OUTPUT_ROOT`. `ACCURACY_MODE=smoke` runs two accumulated updates
and full canaries through the identical production launcher. Submit full
training only after all smoke tasks complete and checkpoint invariants pass.
Production has a six-hour cap; no automatic extension or retry is configured.
No full training runs on the checkout host.

### Execution validation

61 CPU tests passed. Initial concurrent smoke arrays 450892 and 450896 each
failed at DDP/NCCL initialization for the two co-located tasks, on nodes
25a-hgpn165 and 25a-hgpn167 respectively, before the first training forward.
The pointwise task on a different node succeeded both times. This identifies
an execution problem but does not establish a hardware or NCCL root cause.

Serial smoke array 450899 completed all three tasks with exit 0 on nodes
25a-hgpn121 and 25a-hgpn012. Each used the production microbatch/global batch,
completed two optimizer updates and full LFW/CPLFW diagnostics. The relational
terms were active and gradients finite. Reloaded epoch checkpoints were
checked for finite tensors, changed trainable weights, and unchanged
polynomial coefficients, intervals, slopes and BN running buffers.

The actual production submission overrides concurrency to `--array=0-2%1`
and excludes both previously failing nodes. Thus the three arms run in
sequence on four H200s at a time. Training policy and loss settings are
unchanged. This workaround passed smoke; it is not a guarantee against
later infrastructure or optimization failure. All attempts, code/input
hashes, smoke metrics and the production ID are retained in
`relational_recovery_submission.json`. No IJB-C job is submitted with training.
