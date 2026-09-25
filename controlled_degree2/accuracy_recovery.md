# Accuracy-first quadratic recovery (2026-09-25)

Objective: improve recognition accuracy of the original iResNet-50 converted
to 25 channelwise quadratic activations. A small number of nonfinite inference
outputs is acceptable; zero failures and BTS input bounds are not the primary
training objective. No improved accuracy is claimed until GPU training and
full evaluation finish.

Start every arm from the **unscaled progressive/student_best.pt**, not
progressive/student_final.pt, later tail-refined checkpoints, or a BTS-scaled
inference checkpoint. Teacher: `work_dirs/ms1mv3_r50/model.pt`.
The source SHA-256 is
`3b3298217fcc54a368b045f50d85e8e0d9c845f305e7f1212640838c160f6b39`.
Verify the server copy with `sha256sum` before running.

The [matched baseline](../docs/experiments/progressive_unscaled_comparison_20260919/README.md)
has strict IJB-C TAR 95.13729% at FAR 1e-4 and 92.01820% at FAR 1e-5,
with 14 nonfinite source images out of 469,375 (about 0.00298%). BTS scaling
plus its range filter additionally rejects seven finite images, with no change
at these two FARs. This comparison does not isolate scaling roundoff from filtering.

## Hypotheses and prepared arms

The existing recipe uses range weight 1, causal-tail weight 1, extensive
stress/pathological augmentation, clipped training activations, and fixed
quadratic coefficients. These may trade accuracy for stability; this is a
hypothesis, not an established cause of the accuracy gap.

| Arm | control | fixed | coefficients |
| --- | --- | --- | --- |
| Purpose | Existing polish recipe from best | Accuracy-oriented recovery | Isolate coefficient adaptation versus fixed |
| Global learning rate | .002 | .0004 | .0004; coefficients .00004 |
| BN running statistics | Adaptive | Frozen | Frozen |
| Hint weight | 1 to .3 | .1 to .03 | .1 to .03 |
| Range weight | 1 | .01 | .01 |
| Causal-tail / operator weights | 1 / .0001 | 0 / 0 | 0 / 0 |
| Lowres / photo / crop probabilities | .2 / .2 / .1 | .05 / .05 / .05 | .05 / .05 / .05 |
| Stress / pathology probabilities | .4 / .05 | 0 / 0 | 0 / 0 |
| Tail replay fraction | .25 | 0 | 0 |
| Polynomial coefficients | Frozen | Frozen | Trainable, no weight decay |

Common: 3 epochs, FP32, SGD/Nesterov, global batch 2048, per-GPU microbatch
128, seed 20260925, cosine embedding KD weight 1, all activations quadratic
from step zero. Four GPUs per arm accumulate four microbatches. Three Slurm
array tasks use at most 12 GPUs. Control versus fixed changes a bundle of
settings; it cannot attribute a gain to an individual setting. Fixed versus
coefficients changes only coefficient optimization.

The approximation target at initialization is each teacher channel's PReLU
on the saved `[-lam_fit[c], lam_fit[c]]`; no interval widening or recalibration
is performed. `lam_reg` remains .6 times `lam_fit`. Learned coefficients are
task-adapted quadratics, no longer claimed to be the original least-squares
PReLU fit. Deployment remains `c0+c1*x+c2*x^2`, with one ciphertext
multiplication level per activation and no new encrypted-path operators.
Training still clips inputs to the fitting interval; evaluation does not.
The resulting training/deployment mismatch is a limitation of this first
experiment, so activation range logs and unclipped evaluation are mandatory.

## Run on the H200 server

Only lightweight tests run in this checkout. Preview a command without training:

```bash
python -m controlled_degree2.accuracy_recovery --arm coefficients \
  --canary-root faces_webface_112x112
```

From the repository root on the GPU server:

```bash
sha256sum work_dirs/controlled_degree2_tail_ms1mv3_20260907/progressive/student_best.pt
sbatch controlled_degree2/accuracy_recovery.slurm
```

Override `CHECKPOINT`, `TEACHER`, `DATASET_ROOT`, `CANARY_ROOT`, and `OUTPUT_ROOT` through
the submission environment if needed. The launcher rejects existing arm
directories to avoid overwriting a run. Save the server git revision and
checkpoint hashes with the run records. The 4-hour cap is a resource limit,
not a prediction that training will finish within it.

The server's MS1MV3 directory lacks CPLFW. The Slurm launcher therefore uses
`faces_webface_112x112` for LFW/CPLFW only; training remains MS1MV3. The
`accuracy_recovery_smoke.slurm` preflight runs all three arms on four H200s,
two accumulated updates each, with full canaries. Its `completed.json` verifies
finite checkpoint tensors, coefficient updates only in the trainable arm,
and unchanged BN running buffers in both accuracy-oriented arms.

### Launcher correction after initial submission

All tasks of array `435836` failed before training, with
`ModuleNotFoundError: No module named 'numpy'`. The initial smoke used the
inherited interpreter, while production loaded a module and activated Conda;
these different startup paths invalidated the runtime check.

Both scripts now source `accuracy_runtime.sh` and invoke the explicit
`ACCURACY_PYTHON` executable (default `$HOME/.conda/envs/face_recog/bin/python`).
The runtime logs its executable and NumPy/PyTorch/MXNet versions, checks
NumPy compatibility and four visible GPUs. There is no module/Conda activation.
Use `ACCURACY_MODE=smoke` with **the production array script** to exercise the
same startup and all three arms for two updates plus full canaries; use a
separate fresh `OUTPUT_ROOT`. Production uses `ACCURACY_MODE=train` (default).
The retry retains the original input checkpoint hashes and training settings.

## Evaluation and decision rule

Every epoch is saved as `epochN.pt` before canary evaluation, even if a canary
fails. These are diagnostic checkpoints, not zero-failure certifications.
The legacy `student_best.pt` still uses its strict finite canary gate; do not
use that alias as the only candidate for this experiment. Accepting inference
failures does not relax checks against nonfinite optimizer gradients.

Use MS1MV3 and non-IJB canaries for development. Predeclare epoch 3 as the
primary comparison, report all three arms, and use IJB-C only for final
reporting, not coefficient fitting, selecting epochs, or repeated tuning.
Evaluate the PReLU teacher with the same protocol to establish the remaining
conversion gap; no matched teacher score has been established by this change.

Run the existing full IJB-C evaluator for each fixed final checkpoint:

```bash
python eval_ijbc.py \
  --model-prefix work_dirs/accuracy_recovery_20260925/coefficients/epoch3.pt \
  --network r50_controlled_d2 --target IJBC \
  --image-path ijb/IJBC/loose_crop \
  --result-dir work_dirs/accuracy_recovery_20260925/coefficients/ijbc \
  --bts-failure-report work_dirs/accuracy_recovery_20260925/coefficients/failures \
  --ignore-bts-range
```

Match the archived baseline's remaining evaluator settings, including
alignment, FP32/TF32-disabled inference, flip fusion and faceness weighting.
Use **strict measured FAR <= requested FAR**, not nearest-FAR interpolation.
Either view failing zeros both complete embeddings before aggregation; keep
all 469,375 sources and all 15,658,489 pairs. Never score finite-only subsets
or replace individual nonfinite coordinates. Report failure count/rate next
to TAR at 1e-4 and 1e-5, plus activation input/radius maxima and ordinary
range exceedance rates from training. Finite out-of-range BTS values pass.

There is no invented acceptance threshold for “a small number.” Report the
accuracy/failure tradeoff against the 14-image baseline; a large increase
must not be labeled acceptable just because TAR improves. After selecting a
recipe using non-IJB development evidence, BTS scaling can be recalibrated
on MS1MV3 and its equivalence checked separately.

If these arms do not recover accuracy, the next experiment should address
unclipped training and identity/relational supervision. Neither is silently
included here, so the first comparison stays interpretable.
