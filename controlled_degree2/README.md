# Controlled direct degree-2 experiment

This experiment tests whether run10's stability came from degree 4 itself or
from its training/range-control recipe.  It lives in the main repository and
does not modify `poly_run10/`.

## Controlled variable

| Item | run10 reference | this experiment |
|---|---|---|
| Teacher | confirmed work_dirs iResNet-50 | exact same checkpoint |
| Activations | all 25 PReLUs replaced | all 25 PReLUs replaced |
| Interval | per-channel deployed `lam_fit/lam_reg` | copied exactly from checkpoint buffers |
| Fit weighting | teacher activation histogram + 5% uniform edge weight | same |
| Distillation | embedding cosine + all-block normalized hints | same |
| Stability loss | per-image summed hinge, beta 1.0, `lam_reg=0.6*lam_fit` | same |
| Coverage | crop 0.1, low-res 0.2, photo 0.2, stress 0.4, pathological 0.05 | same probabilities/semantics |
| Final polish | 3 epochs, SGD/Nesterov, hints 1.0 to 0.3 | same |
| Polynomial | `c0+c1*x+c2*x^2+c4*x^4` | **`c0+c1*x+c2*x^2`** |
| Mult. depth / activation | 2 (`x^2`, then `x^4`) | **1 (`x^2`)** |

The approximation target is channel-wise PReLU on the symmetric interval
`[-lam_fit[c], lam_fit[c]]`.  Calibration fits only the even `abs(x)` component;
the PReLU linear component is exact.  Deployment has no clamp, comparison,
division, or data-dependent branch.

## Run on the GPU server

The checkout is only for code validation; do not run full training here.
The launcher checks that the `face_recog` environment still has NumPy 1.x,
as pinned by `environment.yml`; PyTorch 2.1/MXNet 1.9 in this project are not
usable with a NumPy 2.x environment.

```bash
export TEACHER_CKPT=/path/to/work_dirs/.../backbone.pt
export RUN10_CKPT=/path/to/run10/student_best.pt
export DATASET_ROOT=/path/to/ms1m-retinaface-t1
export OUTPUT_ROOT=/path/to/work_dirs/controlled_direct_degree2

# Individual stages are restart-friendly.
bash controlled_degree2/run_controlled_degree2.sh calibrate
bash controlled_degree2/run_controlled_degree2.sh convert
bash controlled_degree2/run_controlled_degree2.sh train
```

Or submit `controlled_degree2/job.slurm` after exporting the same variables.
The four-GPU trainer accumulates four microbatches per GPU, preserving run10's
effective global batch of 2048 and linearly scaled learning rate 0.002.

Calibration requires the run10 checkpoint only as a read-only source of its
actual deployed range buffers and recorded widening factors.  It deliberately
ignores the degree-4 coefficients, fits a quadratic on the corresponding
pre-widened interval using 100k teacher images, then applies the same
`q_s(x)=s*q(x/s)` widening.  This preserves run10's calibration order as well
as its final intervals.

## Validity gates and comparison

`student_best.pt` is written only after all 25 activations are pure quadratic,
evaluation clipping is off, and both LFW/CPLFW canaries produce finite
embeddings.  `last.pt` and `student_final.pt` are diagnostic/training outputs;
they are not automatically safe deployment checkpoints.

Run IJB-B and IJB-C with the repository's ordinary evaluator and a hard
non-finite gate:

```bash
python eval_ijbc.py \
  --model-prefix "$OUTPUT_ROOT/polish/student_best.pt" \
  --network r50_controlled_d2 \
  --image-path /path/to/ijb/IJBC/loose_crop \
  --result-dir "$OUTPUT_ROOT/ijbc" \
  --target IJBC \
  --fail-on-nonfinite
```

Use exactly the same IJB protocol, image lists, flip fusion and FAR points as
the run10 report.  Record at minimum: non-finite image count, first escaping
activation if any, IJB-B/C TAR at FAR 1e-4 and 1e-5, and LFW/CPLFW canaries.

Interpretation:

- Degree 2 reaches zero non-finite and comparable accuracy: degree 4 was not
  required for numerical stability; the run10 range/training recipe was the
  important factor.
- Degree 2 reaches zero non-finite but loses accuracy: degree 4 mainly bought
  approximation capacity, not stability.
- Degree 2 remains non-finite under this recipe: do not yet conclude degree 4
  is intrinsically stable.  Inspect the first escape and range ratios first.

Because historical run10 arrived through several exploratory warm starts, the
strongest causal accuracy comparison would rerun a degree-4 arm from the same
teacher with this exact two-stage harness.  For the narrower question “can
direct degree 2 also eliminate non-finite values under the run10 controls?”,
this degree-2 arm is sufficient.
