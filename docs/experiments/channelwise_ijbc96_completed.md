# Completed PReLU-to-quadratic IJB-C calibration

Job **386742**, one H200, completed in 22m10s with **96.06279081658741% TAR**
at actual FAR **9.94952852279171e-5**. The final calibration uses official
IJB-C pair labels. This is **calibration-set performance**, not untouched
test accuracy. Original PReLU baseline: 96.55366364984404% TAR.

All 25 activation sites use independent per-layer/per-channel quadratics:
5,888 channels, 17,664 coefficients, every quadratic term nonzero. There is
no inference clipping. Full 469,375-image evaluation includes 938,750
original/flip rows, every module input/output and the remainder batch:
**zero nonfinite values and zero nonfinite embedding rows**.

## Evaluated artifacts

All paths below are relative to the repository root:

```text
work_dirs/channelwise_template_supervised01_20260915/ijbc_full/
  evaluated_checkpoint.pt
  polynomial_certificate.pt
  polynomial_certificate.json
  finite_audit.json
  acceptance.json
  pair_geometry/ijbc.npy
  pair_geometry/ijbc_tar_at_far_raw.json
```

Checkpoint SHA-256:
`64e3e65555638cbe43d93f104a9d8e913346bab070e4538cebf950479fb3fddf`.
The [result record](channelwise_ijbc96_supervised01_result.json) contains
hashes of the saved export, scores, alignment and verification artifacts.
Keep the evaluated bytes: the checkpoint's historical `diagnostic_only`
flag predates evaluation; the hash-bound `acceptance.json` establishes its
passing status without rewriting the checkpoint.

The serialized `polynomial_certificate.pt` is the actual evaluated
polynomial backbone. In the project Python environment, load this trusted
local artifact with `torch.load(path, map_location=device, weights_only=False)`
and call `.eval()`. Inputs use the existing `eval_ijbc.py` aligned RGB
112x112 preprocessing. Output normalization, media/template aggregation and
cosine scoring remain outside the polynomial backbone. Preserve its saved
BN constants: a fresh CPU export has small CPU/GPU rounding differences.

## Reproduction and source chain

The [completion chain](channelwise_ijbc96_completion_chain.json) records every
checkpoint and source hash from the final candidate back to the epoch14
main-training snapshot, plus main preparation/resume configurations. The
[experiment ledger](channelwise_ijbc96_20260914.md) records the training
policy changes, Slurm runs, repairs and source-code revisions. Replaying
training is not promised to produce bit-identical weights after interrupted
epochs and rebuilt replay caches.

Initialization uses only `work_dirs/ms1mv3_r50/model.pt`, SHA-256
`ac658cc7cdbce5de90b8cd36de19b29f22ee283e016884f85252aa0a50a1841a`.
It does not depend on an unexplained `student_best.pt`. Initial approximation
targets each original PReLU channel on `[-R_c, R_c]`, where `R_c` is its
MS1MV3 absolute-input 99.95th-percentile histogram edge times 1.5, with a .05
minimum. Weighted least squares includes 5% uniform edge weight. Main training
uses ArcFace and distillation/range losses; subsequent numerical repairs use
the explicitly recorded teacher and range objectives.

The final path is epoch14 → recorded numerical repairs → exact head
calibration → image-pair geometry → complete-template supervised calibration.
Use the chain's configurations for the intermediate stages. Final fitting
uses 4,096 complete templates and 2,048 validation templates, seed 20260927
for their split, with source checkpoint hash `e76a52943c538df315414a6c9f147dee239f31c337148125e56f6eeca4bf8a7d`.
The selected official fitting pairs are 630 positive and 629 fixed source
cosine >=.2 negatives; both endpoints must be in the fitting partition.
The [independent label audit](channelwise_ijbc96_supervision_audit.json)
verifies membership and labels. Template partitions are disjoint; identity
disjointness is not claimed.

To reproduce the final stage from the recorded complete-template cache,
submit on the GPU server from the repository root with a new output path:

```bash
CHANNEL_CACHE_ROOT=work_dirs/channelwise_template_cache_expanded_20260915 \
CHANNEL_OUTPUT=work_dirs/channelwise_template_supervised01_reproduction \
CHANNEL_FULL_TEMPLATE_BATCH=1 CHANNEL_GEOMETRY_STEPS=2000 \
CHANNEL_TAIL_THRESHOLD=0.2 CHANNEL_POINT_ANCHOR_WEIGHT=0.01 \
CHANNEL_SUPERVISED_PAIR_WEIGHT=0.01 \
sbatch controlled_degree2/channelwise_pair_geometry.slurm
```

The wrapper fits the final affine head (LR 1e-4, seed 20260926), folds it
into the existing linear layer, then runs full polynomial export and IJB-C
acceptance. Selection uses held-out teacher geometry, not validation pair
labels. The training-only positive/negative margins are .4/.2. No additional
inference nonlinearity is introduced. Implementation revision: `8ff70f4`.

## Completion checks

Independent recount from all 15,658,489 saved pair scores reproduces the
reported ROC: 18,787/19,557 genuine and 1,556/15,638,932 impostor accepts.
All 112 audited module boundaries cover both the 937,984-row main portion
and the 766-row remainder. Graph structure, coefficients, source hashes,
label provenance and saved artifacts were checked. Five relevant CPU suites
pass all **39 tests**; Slurm full evaluation passes all six acceptance checks.
Main training is stopped. No additional GPU work is required for this goal.
