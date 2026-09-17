# Progressive best: residual graph scaling for six BTS boundaries

Source: `work_dirs/controlled_degree2_tail_ms1mv3_20260907/progressive/student_best.pt`.
Output: `work_dirs/progressive_best_bts_scaled_20260917/student_scaled.pt`.
The source is unchanged. The output is a standard `r50_controlled_d2` checkpoint,
loadable with `load_controlled_checkpoint`; weights remain local under ignored
`work_dirs/`. [Raw measurements](progressive_best_bts_scaling_20260917.json)
include the source SHA-256 and the exact scaling factors.

## Transformation

This is an offline parameter transformation of the existing unfused model, with
no retraining, runtime scaling modules, clipping, or actual bootstrapping.
The polynomial remains degree 2 (one ciphertext multiplication level per
activation); the inference graph is unchanged.

An identity shortcut ties the input/output coordinate scale of each residual
block. Accordingly, all blocks in each stage share one scale. Stage 3 uses the
largest absolute activation over its three selected boundaries. Transition
blocks have projection shortcuts, which allow stage scales to differ.

| Stage | Coordinate scale s (new tensor = s × original tensor) |
| --- | ---: |
| 1 (including stem) | 0.23847179880215555 |
| 2 | 0.25873041600052066 |
| 3 | 0.07465125006418492 |
| 4 | 0.7509274743512693 |

For input scale a and output scale b, a convolution is transformed as
`W' = (b/a) W`, `bias' = b bias`. With frozen BN running mean mu, variance v,
and `d = sqrt(v + eps)`, BN becomes
`gamma' = (b/a) gamma`,
`beta' = b beta + b (1/a - 1) gamma mu/d`.
Running buffers and epsilon stay unchanged. This gives `BN'(a*x) = b*BN(x)`
in real arithmetic.

Each quadratic uses the same input/output scale s within its stage:
`q'(u) = s*c0 + c1*u + (c2/s)*u^2`, hence `q'(s*x) = s*q(x)`.
The approximation target remains per-channel PReLU, with per-channel intervals
changed from `[-lam_fit, lam_fit]` to `[-s*lam_fit, s*lam_fit]`;
`lam_reg` scales by s as well. Both residual branches use the same output scale.
The final spatial BN restores scale 1 before flattening and the embedding head.

## Validation

CPU FP32, eval mode, four threads, batch size 4, IJB-C five-point alignment,
RGB normalized to [-1,1], no flip. The first 100 readable images in metadata
order select scales with target absolute maximum 0.8 (only downscaling).
The next 100 images are held out from scale selection. This is a lightweight
range and equivalence check, not an IJB-C accuracy benchmark or full-dataset
range certification.

The exported checkpoint was reloaded using the existing model loader. On both
splits, every one of the 24 residual block outputs was compared to its original
output times the stage scale. Final embeddings were compared without scaling.
All comparisons passed `torch.testing.assert_close(atol=2e-5, rtol=2e-4)`.

| BTS boundary (0-based) | Scaled calibration [min,max] | Scaled held-out [min,max] |
| --- | --- | --- |
| layer1.2 | [-0.642935, 0.800000] | [-0.505095, 0.795361] |
| layer2.3 | [-0.800000, 0.486863] | [-0.781281, 0.400976] |
| layer3.3 | [-0.170435, 0.226713] | [-0.158218, 0.207171] |
| layer3.7 | [-0.174999, 0.409921] | [-0.154006, 0.387433] |
| layer3.11 | [-0.184824, 0.800000] | [-0.156203, 0.737990] |
| layer4.1 | [-0.800002, 0.624272] | [-0.898656, 0.694335] |

| Equivalence metric | Calibration | Held-out |
| --- | ---: | ---: |
| Maximum embedding absolute difference | 2.20537e-5 | 2.52649e-5 |
| Maximum per-image embedding relative L2 difference | 8.27333e-6 | 9.10324e-6 |
| Minimum embedding cosine similarity (FP64 reduction) | 0.999999999967879 | 0.999999999963132 |
| Maximum block difference after undoing coordinate scale | 1.53780e-5 | 1.26362e-5 |

All measured boundary tensors and embeddings are finite. All six boundaries
stay inside [-1,1] on both splits. Source calibration ranges reproduce the
previous progressive/best 100-image measurement.

## Reproduction

Run from the repository root, choosing a fresh output directory:

```bash
python -m controlled_degree2.rescale_residual_graph \
  --checkpoint work_dirs/controlled_degree2_tail_ms1mv3_20260907/progressive/student_best.pt \
  --output-dir work_dirs/progressive_best_bts_scaled_20260917 \
  --calibration-images 100 --validation-images 100 \
  --target-absmax 0.8 --batch-size 4 --threads 4

python -m pytest tests/test_rescale_residual_graph.py tests/test_layer_statistics.py -q
```

Tests: 10 passed. The graph tests cover nontrivial BN statistics and biases,
projection and identity shortcuts, parameter serialization, every block output,
interval transformation, unchanged module structure, and invalid scale/train
mode rejection. The real-checkpoint run also validates export/reload and the
calibration/held-out split.

## Scope

Equivalence applies to frozen-statistics inference; it is not bitwise equality.
Train-mode BN is not covered by this transformation. This is an inference
checkpoint, not a training resume checkpoint with optimizer state. If used as a
progressive2 initialization, BN training behavior and subsequent range drift
must be evaluated separately.

The stage-wide scale intentionally leaves the first two Stage 3 boundaries
well below magnitude 1. Independently scaling them would require altering the
identity-shortcut graph. The measured range is not a bound for unseen images;
full calibration and actual CKKS/BTS error/level measurements remain necessary
for deployment. Increasing c2 under a smaller coordinate scale is an algebraic
change of coordinates, not evidence of improved polynomial stability or FHE
precision.
