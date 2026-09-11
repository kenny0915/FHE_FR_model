# Folded degree-2 iResNet-50 student

This archive is self-contained. Keep these files together:

- `model.pt`: weights, polynomial coefficients, calibration ranges, and metadata.
- `backbone.py`: the exact iResNet-50 topology used to create the checkpoint.
- `loader.py`: reconstructs the BatchNorm-folded degree-2 graph and loads it strictly.
- `example.py`: minimal inference example.
- `requirements.txt`: minimum runtime dependencies.

Do not load `model.pt` into an ordinary iResNet-50. This checkpoint has
`bn_folded=True`: post-convolution BatchNorms are absorbed into convolution
weights/biases, residual pre-activation BatchNorms are channel affine layers,
and the final feature BatchNorm is folded into the fully connected layer.

## Input and output

- Input: aligned RGB faces, shape `[N, 3, 112, 112]`.
- Pixel scaling: `pixel / 255`, then `(pixel - 0.5) / 0.5`, giving `[-1, 1]`.
- Output: raw 512-dimensional embeddings.
- Verification: L2-normalize embeddings before cosine comparison.
- Activation target: the original per-channel PReLU on its calibrated symmetric
  interval `[-lam_fit[c], lam_fit[c]]`.
- Activation polynomial: `c0 + c1*x + c2*x^2`, with one multiplicative level.
- Deployment path: no clipping, comparison, division, or data-dependent branch.

## Load for inference or quantization

```python
from loader import load_model

model, metadata = load_model("model.pt", map_location="cuda")
```

The loader rejects a checkpoint unless it is marked degree 2 and every legacy
`c4` storage value is exactly zero. It intentionally does not evaluate `x^4`.

For post-training quantization, calibrate with representative aligned RGB faces.
Keep the polynomial explicit and measure each activation's observed input range
against its `lam_fit` buffer. Standard ReLU-only quantization recipes are not a
drop-in replacement for this graph.

## Fine-tuning

By default polynomial coefficients are frozen. To train them too:

```python
model, metadata = load_model(
    "model.pt", map_location="cuda", trainable_polynomials=True
)
model.train()
```

This deployment-only package does not add training-time clipping or range-loss
branches. If coefficients or earlier layers are fine-tuned, monitor polynomial
input ranges and non-finite embeddings on the target validation data. Preserve
the pure quadratic deployment graph when exporting.

