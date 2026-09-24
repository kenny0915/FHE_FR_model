# PILLAR polynomial iResNet-50

This experiment adapts the training method in `reference/Fast and Private
Inference of Deep Neural Networks by Co-designing Activation Functions.pdf`
to the repository's 112×112 face-recognition iResNet-50. The paper evaluates a
classification ResNet-50; this repository keeps its established iResNet-50
residual topology and embedding head so results remain comparable with the
existing face baseline.

## Encrypted activation

Every PReLU is replaced by the paper's p=10 quantization-aware approximation
to **ReLU on `[-5, 5]`**:

```text
q(x) = 0.314453125 + 0.5 x + 0.15625 x² - 0.0029296875 x⁴
```

The coefficients are exact multiples of `2^-10`. A balanced evaluation first
computes `x²`, then `x⁴ = x²·x²`, so the activation has algebraic degree 4 and
two sequential ciphertext-ciphertext multiplication levels. Coefficient
multiplications are plaintext operations. BatchNorm is affine at inference
and can be folded into adjacent linear operations. The backbone contains no
max-pooling.

## Training stabilization

For each activation input `x`, the model records the layer mean of
`(x / 4.8)^gamma`; the trainer averages those values over all 25 activation
sites and adds `beta` times that result to the face-recognition loss. The
regularization interval `[-4.8, 4.8]` is deliberately tighter than the fit
interval to provide an inference buffer.

The first four epochs use the paper's warm-up:

| Epoch | gamma | beta multiplier |
|---:|---:|---:|
| 0 | 4 | 1/100 |
| 1 | 6 | 1/50 |
| 2 | 8 | 1/10 |
| 3 | 10 | 1/5 |
| 4+ | 10 | 1 |

During training only, inputs are clipped to `[-5, 5]` after the penalty is
computed. Clipping is absent from evaluation and encrypted inference. Logged
range metrics include the maximum activation input and the fractions outside
both the regularization and approximation intervals. For a representative
evaluation pass, call `model.set_pillar_range_tracking(True)`, run validation,
read `model.pillar_range_summary()`, and disable tracking again before export.
These metrics must be checked before exporting a model because a polynomial
can diverge rapidly outside its fit interval.

## Configurations

- `configs/ms1mv3_r50_pillar.py`: production MS1Mv3 run, with `beta=5e-5`,
  cosine decay, five warm-up epochs, SGD, and FP32.
- `configs/casia_r50_pillar_smoke.py`: two-epoch, 100-step-per-epoch pipeline
  check for the GPU server. It is not intended to measure accuracy.

Run the smoke job first on the GPU server, then the production job. Compare
verification accuracy against `configs/ms1mv3_r50.py`, and inspect the
`PILLAR/*` range summaries throughout training. Full training must not be run
in this checkout environment.
