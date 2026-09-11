# A-v2: sitewise joint unclipped recovery

This is an explicit **revision of A's training protocol**, not an unchanged
A/B ablation. Keep the original A and BN-only recovery results separately.
Any later controlled A/B comparison must declare the recovery protocol and
additional optimization/data exposure for each arm.

## Invariants

- Start again from the preserved teacher-only A epoch-8 checkpoint in
  `work_dirs/recipe_a_376833/last.pt`, not the drifted BN-only recovery.
  Snapshot it into a new output directory, verifying its SHA-256. Neither
  original nor previous recovery directories are overwritten.
- Original PReLU iResNet50 teacher and original MS1MV3 identity-disjoint
  student training/development split. **Only training rows** supply repair,
  replay, stage gates and full-corpus gates. No IJB-C data, results, ranges,
  student_best weights or old run10 range buffers enter this procedure.
- All 25 activations remain fixed channelwise `c0+c1*x+c2*x^2`, fitting the
  original channelwise PReLU on `[-lam_fit[c], lam_fit[c]]`. Coefficients,
  intervals, slopes and all BN running buffers remain bitwise fixed. Each
  activation still requires one square; no inference operation is added.
- FP32 backbone, per-GPU batch 128, global main batch 2048 on 16 H200 GPUs.
  The extra repair probe has up to 32 examples per GPU; report that extra
  training exposure rather than treating it as the original compute budget.

## Changes from BN-only recovery

1. Open **one activation at a time in forward order**, from stem through all
   24 residual blocks. The candidate prefix is genuinely unclipped; the
   suffix remains temporarily clipped. The old 14-layer Layer3 jump is gone.
2. Train Conv2d weights and BatchNorm2d affines throughout the backbone.
   Keep the embedding projection, final BatchNorm1d, classifier and all
   polynomial state frozen during repair. This allows redistribution of
   activation tails, not merely local BN contraction. Save/resume validates
   every non-trainable tensor against the original snapshot.
3. On the clipped auxiliary graph, use teacher cosine KD, all 24 residual
   block normalized feature hints (weight 0.3), and 0.1 times the fixed
   classifier's ArcFace loss. Retain original bounded augmentation and the
   0.02 pathological fraction, excluded from identity/hint objectives.
4. Add an all-25-site normalized range objective on the auxiliary graph:
   per-image maximum squared excess above `0.9*lam_fit`, plus 0.01 times
   mean squared excess, averaged over sites. This supplies deep-layer
   correction even when the earliest escaping sample cannot reach them.
5. Separately probe exact replay examples (or a slice of the main batch if
   none are mined) on the unclipped candidate prefix. Stop before the first
   square whose input leaves `lam_fit`. Retry safe rows to reach deeper
   sites. Unlike v1, do **not** detach BN inputs: finite-prefix gradients
   update upstream Conv/BN weights too. No downstream non-finite tensor
   contributes to backward; intermediate graph flags/hooks are restored.
6. Average identity and range gradients independently over all ranks, then
   apply the existing conflict-aware projection to the global gradients.
   Tail/clean norm cap is 10000, with the helper's nonzero floor. SGD LR
   1e-4, no momentum/weight decay; each tensor's actual update is capped at
   `1e-4 * max(parameter_L2_norm, 1)`. There is no cumulative trust-region
   ball that could permanently pin a badly scaled source to an unsafe point.
   Identity preservation is a training objective, not an accuracy guarantee.
7. Scan 8192 fixed training rows in clean/flip/lowres20/shift4/dark variants
   on entry to **every site**, then every 250 updates. Require the exact
   expected row count, finite polynomial inputs/outputs, finite nonzero
   embedding norms, and maximum input/fit-interval ratio <= 2 before opening
   the next site. `2` is a predeclared bounded-extrapolation screening rule,
   not a proved safe bound, not a range-buffer change, and not chosen from
   IJB-C. It rejects the old finite-but-4.49e10-ratio prefix. The loss still
   targets 0.9 and stops unsafe training probes at 1.0.
8. Mine numerical failures **and** finite rows above that ratio at every
   site gate. Replay their exact source index AND deterministic variant on
   every repair update; never replace an observed shifted/dark failure with
   only its clean image. Each rank's diagnostic list is capped at 4096;
   counters are not capped. A bounded union (65536 records) keeps earlier
   counterexamples within the site. Replay membership is checked against
   the training split and saved in the checkpoint.
9. After site 25 passes, scan every original training row clean and flipped.
   Require exactly twice the training-row count and zero numerical failures.
   Full-scan interval escapes are diagnostic (not strict zero containment).
   If the scan fails, mine failures immediately and continue site-25 repair;
   do not declare success or start accuracy training.

There is no fixed repair-step exhaustion exit. Numerical losses/gradients,
corrupt provenance or system failures still fail visibly. Gate convergence,
accuracy recovery, IJB-C zero failures, and TAR 95.5% are **not guaranteed**.

## Resume and accuracy handoff

Atomic `recovery_last.pt` every 25 updates includes the policy identifier,
site/step, exact replay records, optimizer and latest gate report. New-output
`--resume` accepts only A-v2 checkpoints with matching fixed policy/source;
legacy BN-only checkpoints are rejected. The snapshot is the reference for
frozen-tensor checks even after many restarts. Sampling and replay iterators
restart deterministically, not bitwise at the old data cursor.

The Slurm script requests requeue ten minutes before the two-day wall limit.
Restart uses the same job output and its latest checkpoint. Cluster support
and queue availability still apply. A completed full gate bypasses repair on
restart; accuracy training resumes its own latest completed-epoch `last.pt`.
Arbitrary optimization/code failures are not put into a crash/requeue loop.

Only after all checks pass, write `recovered.pt` and resume A's original
accuracy training at epoch 9 with cleared conversion momentum and 0.1 times
the original base backbone/head LR. The original development-selection
protocol then resumes; it does not participate in recovery decisions.

```bash
sbatch --exclude=25a-hgpn144 --time=00:20:00 --signal=B:USR1@60 \
  --export=ALL,RECIPE_SMOKE=1,RECOVERY_RESUME= \
  controlled_degree2/recipe_a_recovery_v2.slurm
# After JOINT_SMOKE_OK and a successful exit:
sbatch --exclude=25a-hgpn144 --export=ALL,RECIPE_SMOKE=0,RECOVERY_RESUME= \
  controlled_degree2/recipe_a_recovery_v2.slurm
```

Outputs: `work_dirs/recipe_a_joint_JOBID`; logs:
`recipe-a-recovery-JOBID.{out,err}`. Smoke uses the real source and classifier,
two joint updates at batch 128/GPU, source-invariant validation on save and
an all-site finite-prefix Conv backward. Smoke demonstrates execution, not
that all 25 stages converge. Local tests also exercise gate rejection,
exact replay transformations, tensor update caps, full-scan retry and
ready-checkpoint resume/handoff with a lightweight synthetic runner.

## Validation (2026-09-12)

- 62 local tests passed across the A, legacy recovery, A-v2 and controlled
  quadratic suites. Shell syntax and whitespace checks passed. Scheduler
  signal/requeue and node routing are tested with mocked cluster commands;
  an actual wall-time requeue has not been exercised end to end.
- H200 x16 smoke **378111** completed with exit 0, using the real epoch-8
  source, full classifier, batch 128/GPU and exact stage-entry replay.
  Two joint updates changed the stem convolution by at most 0.000139058.
  All 209 trainable Conv/BN tensors participate in the optimizer; frozen
  tensor checks passed on save. Initial stage-1 max/fit ratio was 5.1513,
  correctly above the new advancement limit, despite zero non-finite rows.
- H200 x16 smoke **378114** loaded the newly saved A-v2 checkpoint at site 1,
  step 2, and completed steps 3–4 with exit 0; stem-convolution maximum delta
  was 0.000138387. These short runs validate execution and checkpoint reuse,
  not convergence, final accuracy or all-site stability.
