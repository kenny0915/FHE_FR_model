# Activation stability study

This report isolates one activation replacement at a time. It uses the trained checkpoint, not random model weights.

> **Smoke-result warning:** the local checkout has no face images. These measurements use deterministic normalized synthetic inputs. Use them to validate wiring and identify mechanisms, not to choose a production interval or claim recognition accuracy.

## Run

- Model: `ms1mv3_r50`
- Checkpoint: `work_dirs/ms1mv3_r50/model.pt`
- Dataset: `synthetic`
- Samples: 16 in 2 batch(es)
- Activations: 25
- Replacement family: `herpn`
- Monitored interval: `[-6.0, 6.0]` (monitoring only; no FHE-unfriendly clamp)

## Per-layer replacement data

| # | activation | outside | input std | input absmax | local rel. RMSE | negative poly derivative | embedding cosine | pairwise cosine MAE | next std ratio |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | `prelu` | 0.00% | 0.1756 | 2.945 | 1.297 | 0.09% | 0.345819 | 0.1648 | 1.167 |
| 1 | `layer1.0.prelu` | 0.00% | 0.4128 | 3.381 | 0.5232 | 3.61% | 0.768705 | 0.156 | 0.9845 |
| 2 | `layer1.1.prelu` | 0.00% | 0.3337 | 1.814 | 0.8256 | 1.25% | 0.72393 | 0.1831 | 1.382 |
| 3 | `layer1.2.prelu` | 0.00% | 0.2194 | 1.807 | 1.705 | 0.05% | 0.587352 | 0.05062 | 1.563 |
| 4 | `layer2.0.prelu` | 0.00% | 0.2081 | 1.607 | 1.873 | 0.26% | 0.112507 | 0.2431 | 2.323 |
| 5 | `layer2.1.prelu` | 0.00% | 0.09288 | 0.6581 | 4.388 | 0.00% | 0.0117222 | 0.1795 | 8.361 |
| 6 | `layer2.2.prelu` | 0.00% | 0.09497 | 0.6265 | 5.678 | 0.00% | -0.0653493 | 0.199 | 13.57 |
| 7 | `layer2.3.prelu` | 0.00% | 0.1166 | 0.6239 | 3.253 | 0.00% | 0.146363 | 0.2044 | 4.075 |
| 8 | `layer3.0.prelu` | 0.00% | 0.1005 | 1.129 | 4.15 | 0.00% | 0.0182437 | 0.2157 | 5.335 |
| 9 | `layer3.1.prelu` | 0.00% | 0.05381 | 0.3103 | 7.917 | 0.00% | -0.0498535 | 0.2441 | 10.79 |
| 10 | `layer3.2.prelu` | 0.00% | 0.07153 | 0.5297 | 8.495 | 0.00% | -0.000258915 | 0.2394 | 12.51 |
| 11 | `layer3.3.prelu` | 0.00% | 0.08025 | 0.5885 | 7.54 | 0.00% | 0.0251159 | 0.236 | 13.73 |
| 12 | `layer3.4.prelu` | 0.00% | 0.06353 | 0.5901 | 8.265 | 0.00% | 0.0197231 | 0.228 | 7.495 |
| 13 | `layer3.5.prelu` | 0.00% | 0.04947 | 0.3267 | 8.067 | 0.00% | 0.136874 | 0.1379 | 4.283 |
| 14 | `layer3.6.prelu` | 0.00% | 0.05055 | 0.4146 | 10.74 | 0.00% | 0.299237 | 0.1756 | 2.408 |
| 15 | `layer3.7.prelu` | 0.00% | 0.04943 | 0.3422 | 9.701 | 0.00% | 0.026557 | 0.2081 | 2.697 |
| 16 | `layer3.8.prelu` | 0.00% | 0.05506 | 0.3737 | 10.1 | 0.00% | -0.00333286 | 0.2188 | 4.202 |
| 17 | `layer3.9.prelu` | 0.00% | 0.07249 | 0.582 | 9.835 | 0.00% | -0.0130506 | 0.243 | 6.898 |
| 18 | `layer3.10.prelu` | 0.00% | 0.0671 | 0.4178 | 7.85 | 0.00% | 0.0501767 | 0.1856 | 6.205 |
| 19 | `layer3.11.prelu` | 0.00% | 0.07032 | 0.5638 | 8.455 | 0.00% | 0.175334 | 0.1963 | 3.874 |
| 20 | `layer3.12.prelu` | 0.00% | 0.07644 | 0.5958 | 7.233 | 0.00% | 0.120241 | 0.2247 | 4.886 |
| 21 | `layer3.13.prelu` | 0.00% | 0.08848 | 0.6414 | 6.406 | 0.00% | 0.30662 | 0.2283 | 2.96 |
| 22 | `layer4.0.prelu` | 0.00% | 0.08275 | 0.5462 | 5.574 | 0.00% | 0.195555 | 0.2154 | 5.35 |
| 23 | `layer4.1.prelu` | 0.00% | 0.08122 | 0.8846 | 7.87 | 0.00% | 0.233263 | 0.241 | 17.28 |
| 24 | `layer4.2.prelu` | 0.00% | 0.04829 | 0.405 | 9.352 | 0.00% | 0.294902 | 0.2358 | 3.366 |

`outside` is the fraction of baseline activation inputs outside the monitored interval. `next std ratio` compares the next activation's input standard deviation after versus before this one replacement.

The smallest sampled input standard deviation is `0.04829` at `layer4.2.prelu`; its replacement constant-term absolute mean is `0.2352`. The largest local relative error is `10.74` at `layer3.6.prelu`, and the largest next-layer standard-deviation multiplier is `17.28` from `layer4.1.prelu`.

## Approximation interval sweep

This sweep refits a uniform-L2 quadratic for every interval. It is separate from HerPN, whose Gaussian weighting does not change when its monitoring interval changes.

| interval | mean outside | inside RMSE | outside RMSE | worst outside error | mean observed rel. RMSE |
|---|---:|---:|---:|---:|---:|
| `[-0.25, 0.25]` | 7.66% | 0.012 | 0.157 | 15.24 | 0.8808 |
| `[-0.5, 0.5]` | 1.98% | 0.02839 | 0.154 | 6.875 | 1.093 |
| `[-1.0, 1.0]` | 0.21% | 0.07028 | 0.2429 | 2.731 | 2.511 |
| `[-2.0, 2.0]` | 0.00% | 0.1662 | 0.2644 | 0.7368 | 5.895 |
| `[-4.0, 4.0]` | 0.00% | 0.3691 | n/a | n/a | 12.94 |
| `[-6.0, 6.0]` | 0.00% | 0.5758 | n/a | n/a | 20.05 |

The lowest sampled mean error in this interval family is `[-0.25, 0.25]`. This is calibration evidence only, especially for synthetic data.

## All activations replaced

- Embedding cosine mean: `0`
- Embedding relative RMSE: `1`
- Pairwise cosine MAE: `0.7441`
- Non-finite embedding fraction: `100.00%`
- Layers receiving non-finite inputs: `['layer3.8.prelu', 'layer3.9.prelu', 'layer3.10.prelu', 'layer3.11.prelu', 'layer3.12.prelu', 'layer3.13.prelu', 'layer4.0.prelu', 'layer4.1.prelu', 'layer4.2.prelu']`
- First interval violation: `layer3.0.prelu`
- First input absmax above 100: `layer3.3.prelu`
- First non-finite activation input: `layer3.8.prelu`

This cumulative probe uses no retraining. Compare it with the isolated rows to see how repeated distribution shifts compound.

## Design and training insights

1. **Treat interval selection as a tail-risk problem.** A narrow fit reduces central error but increases the quadratic coefficient, so rare outliers create large outputs and derivatives. A wider fit lowers tail growth but spends approximation capacity on values the layer may rarely see. Select intervals from real per-layer quantiles, then check held-out tail errors rather than only uniform-grid error.
2. **Do not interpret HerPN's `[-R, R]` label as a clamp or minimax guarantee.** This HerPN is Gaussian-weighted. Changing only `R` changes the warning threshold, not its coefficients. Refit or rescale the polynomial if the interval is meant to affect approximation behavior.
3. **Polynomial instability comes from value and derivative amplification.** Degree-2 tails grow as `x^2`; their derivative grows linearly and can reverse sign on negative inputs. Residual blocks then carry this shift into later BatchNorm statistics, and repeated replacements compound a small local error into embedding drift.
4. **Use progressive conversion.** On this calibration run, begin with `layer1.0.prelu`, `layer1.1.prelu`, `layer1.2.prelu`, `prelu`, `layer3.13.prelu`; defer `layer2.2.prelu`, `layer3.1.prelu`, `layer3.9.prelu`, `layer3.8.prelu`, `layer3.2.prelu` or give them longer transitions. Re-rank with real faces because this order is data-dependent.
5. **Train each student locally before relying on task loss.** Initialize coefficients by regression on that layer's recorded inputs, retain a teacher-activation distillation loss, convert small groups, refresh normalization statistics, lower the backbone learning rate, clip gradients, and retain interval/tail penalties during training. Such penalties may use non-polynomial operations because they are outside encrypted inference.
6. **Validate behavior, not only activation MSE.** After every group, check embedding cosine, pairwise-score drift, LFW/CFP-FP/AgeDB accuracy, non-finite gradients, activation quantiles, and the final folded polynomial graph.

## Recommended GPU-server sequence

1. Run this study on 5k-20k held-out normalized faces.
2. Choose per-stage or per-layer intervals from train calibration data; verify tails on a disjoint set.
3. Fit all students locally with the baseline frozen.
4. Convert the lowest-impact groups progressively with distillation.
5. Recalibrate BatchNorm after each group and evaluate verification sets.
6. Fully fold the polynomial model and repeat range/accuracy checks.

