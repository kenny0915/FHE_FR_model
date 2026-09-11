# Recipe A: teacher-only full quadratic iResNet50

The original protocol below is preserved for provenance. The explicit
[A-v2 recovery revision](recipe_a_recovery_v2.md) changes post-epoch-8
optimization while keeping data isolation, fixed quadratics/intervals and
BN moments. Report it as a revised training recipe, not an unchanged A/B arm.

Starts only from `work_dirs/ms1mv3_r50/model.pt` (the original 25-PReLU
iResNet50). It never reads run10, student_best, IJB-C images, their range
buffers, or their evaluation results. The old controlled experiment remains
available separately.

## Fixed experimental policy

- Seed 20260911; hold out 2% of MS1MV3 identities before calibration, class
  centers, augmentation or replay. The teacher may have seen these identities
  in its original training, so this is student-development data, not a claim
  of an unseen-teacher or final benchmark evaluation.
- Fresh calibration on 100,000 training images, per-channel absolute-input
  histogram (512 logarithmic bins; fixed spatial stride 4). Approximation
  target: original channelwise PReLU on `[-lam_fit[c], lam_fit[c]]`.
  The 0.9995 quantile upper bin edge times a global 1.5 guard band defines
  `lam_fit`; minimum 0.05, `lam_reg=0.6*lam_fit`. Weighted least squares uses
  the existing histogram fitter with 5% uniform edge weight. These are
  predeclared initial hyperparameters, not established optimal values.
- All 25 activations are `c0+c1*x+c2*x^2`; coefficients stay frozen. Each
  activation contributes one square level. Total CKKS cost is not benchmarked.
- Initialize class centers from up to four teacher embeddings per training
  identity. No compatible saved baseline classification head was available.
  The full ArcFace classifier is replicated with DDP; no identity sampling or
  gradient accumulation is used. Batch 128 per GPU, total 2048 on 16 GPUs.
- FP32 forward/backward. Frozen teacher; pretrained BN running statistics
  remain fixed throughout, while student BN affine parameters may train.
- Epoch 0 warms the classification head (student LR zero). Five groups
  (stem, stages 1–4) transition PReLU to quadratic over epochs 1–9 with
  overlapping two-epoch ramps. Epochs 9–20 use all-quadratic, unclipped
  forwards for **all** losses. Clipping only assists the transition.
- SGD/Nesterov, momentum .9, weight decay .0005, backbone LR .004, head LR
  .02, cosine decay to 5%, global gradient norm clip 5. ArcFace scale 64 and
  margin ramp to .5; cosine KD weight 1; four stage-output hints decay .3 to
  zero. Range weight ramps to 1 over two epochs; per-layer mean hinge-square
  plus .01 times per-image maximum hinge-square, averaged over layers.
- Crop .1, lowres .2, photo .2, stress .1; pathological fraction .02 is
  excluded from identity and KD losses. A 32-image per-rank cache replays
  approximately 1/16 of each batch from actual augmented training tails,
  retaining labels and excluding pathological rows. Resume rebuilds this
  transient cache rather than claiming bit-exact continuation.
- Non-finite embeddings/loss/gradients abort the job synchronously before an
  optimizer update; no silent skipped steps or sanitized training embeddings.

## Development and artifacts

Use up to six images per development identity. After each completed
unclipped-training epoch, scan clean, flip, lowres20, shift4 and dark variants.
Report non-finite rows, embedding maximum, and activation/fit-range maximum.
The unchanged PReLU teacher is evaluated with the same development protocol
once at training start, saved as `teacher_development.json`.
Verification uses clean+flip fused embeddings and a separate lowres20 score;
positive pairs are between distinct source images. Two million seeded
impostor draws set a sampled FAR=1e-4 threshold. Select the minimum of clean
and lowres TAR, only when all five variants are finite. This development
proxy is not the IJB-C template protocol and does not guarantee IJB-C TAR.
Repeated sampled pairs are allowed; the number of draws is not a count of
independent identities or comparisons.

`prepared.pt`, `split.npz`, `provenance.json` record fresh calibration, centers,
source policy and teacher/split SHA-256. `train_config.json` records training
settings. `last.pt` includes model, classifier and optimizer for epoch-boundary
resume. During conversion it is diagnostic: `pure_quadratic=False`.
`student_best.pt` is only emitted after actual full-quadratic unclipped
training and a finite development gate. It loads with the existing
`r50_controlled_d2` evaluator. `metrics.jsonl` stores development results.

The checkpoint retains frozen-statistics BN modules; they are affine at
inference and can be folded later. No new data-dependent normalization,
clipping, classification head or training loss is exported in the backbone.
Full-corpus stability, BN-folded equivalence, CKKS execution and final IJB-C
accuracy remain separate evaluations; zero development failures is empirical,
not a guarantee for arbitrary inputs.

## H200 submission

From repository root, after local tests:

```bash
sbatch --job-name=r50-d2-a-smoke --time=00:20:00 \
  --export=ALL,RECIPE_SMOKE=1 controlled_degree2/recipe_a.slurm
# Confirm SMOKE_OK and successful exit before submitting the full job:
sbatch controlled_degree2/recipe_a.slurm
```

The job uses partition `16gpus`, two nodes with eight H200 GPUs each,
one torchrun launcher per node and a two-day limit. Preparation and training
run sequentially in the allocation. Each job has a fresh output directory
`work_dirs/recipe_a_JOBID` and `recipe-a-JOBID.{out,err}` logs.
NCCL bootstrap binds to the interface resolving the Slurm node hostname
(the default IPoIB bootstrap failed on this cluster); IB remains enabled for
data transport. DataLoader uses spawn to avoid importing MXNet in a forked
CUDA process.

Smoke preparation is restricted to 64 training identities and marked as such;
production training rejects smoke preparation. It checks two real distributed
optimizer steps at the production batch size and checkpoint export, not
convergence or final stability. Training rows may repeat in this short test
to supply two global batches. The smoke classification head has 64 identities;
the full training head has all training identities. Smoke advances to early
conversion and verifies that the backbone weights actually change.
Unit tests additionally exercise an unclipped quadratic backward/update.
Do not point smoke and production jobs at the same output directory.

Resume by launching `recipe_a train --output ORIGINAL_OUTPUT --resume
ORIGINAL_OUTPUT/last.pt` under the same 16-process torchrun configuration and
unchanged hyperparameters; do not rerun preparation over existing artifacts.

## Implementation validation (2026-09-11)

- 40 tests passed: `tests/test_recipe_a.py` and
  `tests/test_controlled_degree2.py`; shell syntax checks passed.
- H200 smoke job 376829, nodes 25a-hgpn162–163: preparation succeeded;
  16 ranks × batch 128 completed two optimizer updates. First loss 3.4747,
  maximum stem-convolution weight change 0.000677; checkpoint exported.
- This validates execution, not convergence, later-phase stability or IJB-C
  accuracy. The all-quadratic unclipped backward path also has a lightweight
  unit test; it is not a substitute for a converged full iResNet50 evaluation.
