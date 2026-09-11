# MS1MV3-only unclipped recovery for recipe A

Recipe-A job 376833 completed epoch 8 with activation clipping, then aborted
on non-finite embeddings when all clipping was removed at epoch 9. Its
`last.pt` remains the original recovery source; do not resume that checkpoint
directly into the old unclipped transition.

## Preservation and data boundary

`recipe_a_recovery.py` requires a new, separate output directory. It makes an
exact SHA-256-verified, read-only `source_epoch8.pt` copy and copies the original
preparation and split artifacts. It never writes the source directory. The
source path/checksum and code revision are recorded in `recovery_source.json`.

Only the **original MS1MV3 training identity split** supplies images for
recovery gradients, augmentation, tail replay, stage gates and the final full
scan. Active classifier IDs must match those training identities and be
disjoint from development identities. Neither development images nor IJB-C
images/results are used to advance recovery. The original PReLU teacher and
epoch-8 classifier are reused for an identity-preservation objective.

All 25 functions remain fixed channelwise quadratics `c0+c1*x+c2*x^2`, whose
approximation target is the original PReLU on `[-lam_fit[c], lam_fit[c]]`.
`lam_fit`, `lam_reg`, coefficients and PReLU slopes are unchanged. Each
activation still uses one square level. Only the 25 immediately preceding
BatchNorm layers' weight/bias parameters train during recovery. Other
weights and all BN running statistics remain fixed. The affine updates are
learned on training data, not manually selected using IJB-C results.

## Recovery procedure

1. Audit the checkpoint's actual full-unclipped graph on a fixed sample of
   training rows and five deterministic variants; record initial failures.
2. Probe candidate prefixes in five groups: stem, layer1, layer2, layer3,
   layer4. Candidate-prefix activations have no clipping; later groups retain
   temporary clipping. Before the first polynomial whose input exceeds its
   original fitting interval, stop the entire batch forward **before the
   square**. Penalize each row's maximum excess above `0.9*lam_fit`, in FP64.
   This loss comes from finite activations, with no downstream NaN graph.
3. In that probe only, detach BN inputs so the gradient reaches the local
   preceding BN affine and cannot traverse the deep quadratic recurrence.
   Separately compute cosine teacher KD plus 0.1 times the frozen classifier's
   ArcFace loss on a clipped anchor graph, without those detach hooks. This
   temporary anchor constrains identity drift; it is not an unclipped gate.
   Pathological rows (.02) contribute to the range objective but not KD or
   classification. Replay up to 1/16 of a batch from a rank-local cache of
   the most recent escaping augmented training batch, retaining masks/labels.
4. Average **all** affine gradients across ranks, supplying explicit zeros for
   affines not reached on a rank. SGD LR 1e-4, momentum .9, no weight decay,
   norm clip 1. No finite-loss/gradient failures are silently skipped.
5. Every 250 updates, gate the candidate prefix on 8,192 fixed training rows
   with clean, flip, lowres20, shift4 and dark variants. Advance only if both
   embedding/nonzero-norm checks and all prefix interval checks pass. A group
   has a 2,000-update budget. Exhaustion saves `recovery_last.pt` and stops;
   it does not declare the model safe or automatically proceed.
6. After all five groups pass, scan **every training row, both orientations**
   on the exact full-unclipped graph. Any non-finite or zero-norm embedding
   prevents the handoff. Full-corpus range escapes are reported separately:
   unlike the smaller conservative stage gate, this final gate requires
   finite nonzero embeddings, not strict containment of every corpus value.

Diagnostic failure records contain training source indices and variants,
capped at 4,096 per rank per gate. Counts in gate reports are uncapped.
`groupN.pt` and `recovery_last.pt` remain diagnostic until the final gate passes.
`recovered.pt` is written only on success. Zero failures on these scans is
an empirical gate, not a proof for all possible augmentations or IJB-C inputs.

## Accuracy continuation

By default, successful recovery resumes recipe A at epoch 9 in the **new**
output directory, with the repaired weights and original classifier. The
conversion optimizer momentum is cleared and backbone/head learning rates
are multiplied by 0.1 (base values .0004/.002) before the existing cosine
schedule. No coefficients or scale buffers change. All accuracy-training
losses then use the full-unclipped graph. The existing development protocol
selects `student_best.pt` during this subsequent accuracy phase only.

Use `--no-continue-training` for recovery alone. Insufficient stage containment
or a failed full scan stops with diagnostics and never launches continuation.
There is no automatic restart of a failed recovery: retain its artifacts for
analysis and explicitly choose the next repair policy instead of rerunning
over the same directory. Gate results are not IJB-C accuracy measurements.

## H200 ×16

```bash
# First validate the actual epoch-8 checkpoint and 16-rank gradients:
sbatch --exclude=25a-hgpn144 --time=00:20:00 \
  --export=ALL,RECIPE_SMOKE=1 controlled_degree2/recipe_a_recovery.slurm
# After successful RECOVERY_SMOKE_OK and zero exit status:
sbatch --exclude=25a-hgpn144 --export=ALL,RECIPE_SMOKE=0 \
  controlled_degree2/recipe_a_recovery.slurm
```

Two H200 nodes, eight GPUs each, per-GPU batch 128. Node 144 is excluded in
these commands because the earlier recipe-A smoke saw a CUDA-unavailable
failure there. The job uses its own `work_dirs/recipe_a_recovery_JOBID` and
`recipe-a-recovery-JOBID.{out,err}`. It does not rerun initial calibration.

Smoke uses the **real epoch-8 checkpoint and full classifier**, audits the
full-unclipped graph, performs two actual affine updates, and exercises a
finite-prefix backward with all five groups open. It does not advance stage
gates, write `recovered.pt`, or continue accuracy training. CPU tests cover
early termination, gradient isolation, flag/hook restoration, identity
isolation, gate conditions, and source snapshot preservation.

## Validation (2026-09-11)

- 47 tests passed across `test_recipe_a_recovery.py`, `test_recipe_a.py` and
  `test_controlled_degree2.py`; shell syntax checks passed.
- H200 ×16 smoke job 377574 completed with exit code 0. All 160 initial
  full-unclipped probe rows were non-finite. The guarded repair nevertheless
  completed two updates at batch 128 per GPU and a full-prefix backward;
  maximum affine change was 0.000204325. This is execution validation, not
  evidence that numerical recovery has converged.
- Comparing the smoke checkpoint with the original snapshot found exactly
  50 changed BN affine tensors and **no other changed tensors**, including
  convolution weights, coefficients, intervals and BN running statistics.
  The original epoch-8 checkpoint SHA-256 was unchanged.
