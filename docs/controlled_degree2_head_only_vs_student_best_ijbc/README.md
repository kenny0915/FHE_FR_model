# Controlled degree-2 head-only v3 vs. student_best_ijbc

## Compared artifacts

- `head-only v3`: `work_dirs/controlled_degree2_accuracy_recovery_head_only_v3_20260908/train/student_best.pt`
- `student_best_ijbc`: the evaluated model is
  `work_dirs/student_best_degree2_eval_state.pt`, converted from
  `work_dirs/student_best.pt`.  The converter only removes the identically-zero
  fourth coefficient column; every deployed tensor otherwise matches exactly
  (258/258 keys, maximum absolute difference 0).

There are 25 activation sites and 5,888 channelwise polynomials in each model.
Every channel uses

```text
q_c(x) = c0_c + c1_c*x + c2_c*x^2
```

to approximate its PReLU target on the symmetric, per-channel interval
`[-lam_fit_c, lam_fit_c]`. `lam_reg` is a training/control boundary, not the
approximation interval. Inference is unclipped, so neither interval is a clamp
in the deployed graph.

## Main finding

The raw-x functions are substantially different, but most of the difference is
a scale transformation rather than a new normalized polynomial shape:

```text
q_student(x) approximately equals r * q_head(x/r)
r = lam_fit_student / lam_fit_head
```

Across all 5,888 paired channels:

- `lam_fit_student / lam_fit_head`: median 0.4225, P10-P90 0.3231-0.4971,
  full range 0.2556-0.6507. The student interval is typically 2.37x narrower.
- `c0_student / c0_head`: median 0.4225.
- `c2_student / c2_head`: median 2.3661.
- `c1` is bit-identical in every paired channel; both keep the same exact
  linear PReLU component.
- After normalizing each channel with `u=x/lam_fit` and `v=q(x)/lam_fit`, the
  paired curve RMSE has median 0.000633 and P90 0.01057. Thus the visible raw-x
  curvature change is mainly caused by the narrower interval.
- Actual deployed `lam_reg/lam_fit` is exactly 0.6 for head-only v3 and 0.4 for
  student_best_ijbc.

The older `student_best.pt` has stale `poly_calib[*].lam_reg` descriptive
metadata at all 25 sites. The deployed state buffers are authoritative and are
what these plots use. Its `lam_fit` metadata is consistent with the buffers.

## Architecture/checkpoint differences

| Item | head-only v3 | student_best_ijbc |
|---|---:|---:|
| Runtime graph | standard controlled iResNet50 | BatchNorm-folded controlled iResNet50 |
| Nonlinear depth | 25 | 25 |
| Polynomial degree | 2 | 2 (zero-padded fourth storage column removed for eval) |
| State entries | 550 | 258 |
| BN running-mean/variance entries | 79/79 | 0/0 |
| Training provenance | controlled degree-2 distillation; layer1-layer4 and stem frozen, head-only recovery | `origin=distill`, step 9000; checkpoint contains no full training config |
| IJB-C non-finite augmented rows | 0 | 0 |
| IJB-C TAR@FAR=1e-4 | 93.42% | 94.74% |

The folded graph absorbs post-convolution BatchNorm into convolution weights and
biases, keeps pre-activation normalization as a frozen channel affine, and
folds the final feature BatchNorm into the FC. This changes checkpoint layout
but not polynomial degree or the number/location of nonlinearities.

## Per-site medians

Arrows show `head-only v3 -> student_best_ijbc`. The complete per-channel
quantiles and curve errors are in `layer_summary.csv`; the 25-page PDF includes
the explicit polynomial of one representative paired channel on every page.

| Site | C | median lam_fit head | median lam_fit student | interval ratio | median c0 | median c2 |
|---|---:|---:|---:|---:|---:|---:|
| `prelu` | 64 | 0.4072 | 0.1524 | 0.374 | 0.0037 -> 0.0014 | 1.4961 -> 3.9974 |
| `layer1.0.prelu` | 64 | 5.3073 | 1.9400 | 0.374 | 0.2850 -> 0.1044 | 0.1246 -> 0.3383 |
| `layer1.1.prelu` | 64 | 4.5254 | 1.6953 | 0.374 | 0.2288 -> 0.0853 | 0.1632 -> 0.4423 |
| `layer1.2.prelu` | 64 | 3.4111 | 1.2695 | 0.375 | 0.2053 -> 0.0770 | 0.2079 -> 0.5624 |
| `layer2.0.prelu` | 128 | 3.7913 | 1.4995 | 0.392 | 0.2383 -> 0.0938 | 0.1988 -> 0.5078 |
| `layer2.1.prelu` | 128 | 2.9210 | 1.1584 | 0.392 | 0.1969 -> 0.0782 | 0.2467 -> 0.6251 |
| `layer2.2.prelu` | 128 | 3.0105 | 1.2690 | 0.433 | 0.2562 -> 0.1032 | 0.2097 -> 0.5270 |
| `layer2.3.prelu` | 128 | 2.8055 | 1.1053 | 0.392 | 0.2015 -> 0.0803 | 0.2598 -> 0.6601 |
| `layer3.0.prelu` | 256 | 3.9718 | 1.2832 | 0.323 | 0.2573 -> 0.0831 | 0.1843 -> 0.5705 |
| `layer3.1.prelu` | 256 | 2.5916 | 0.9023 | 0.356 | 0.1750 -> 0.0590 | 0.2654 -> 0.7810 |
| `layer3.2.prelu` | 256 | 2.4390 | 0.8726 | 0.362 | 0.2074 -> 0.0703 | 0.2654 -> 0.7725 |
| `layer3.3.prelu` | 256 | 2.4334 | 0.8621 | 0.354 | 0.1957 -> 0.0665 | 0.2725 -> 0.7938 |
| `layer3.4.prelu` | 256 | 2.5187 | 1.1022 | 0.416 | 0.1816 -> 0.0766 | 0.2744 -> 0.6453 |
| `layer3.5.prelu` | 256 | 2.4867 | 1.4323 | 0.569 | 0.1685 -> 0.0973 | 0.2849 -> 0.4921 |
| `layer3.6.prelu` | 256 | 2.3743 | 0.7793 | 0.323 | 0.1608 -> 0.0522 | 0.2986 -> 0.9119 |
| `layer3.7.prelu` | 256 | 2.3931 | 0.7854 | 0.323 | 0.1639 -> 0.0530 | 0.2984 -> 0.9130 |
| `layer3.8.prelu` | 256 | 2.4334 | 1.1278 | 0.447 | 0.1743 -> 0.0806 | 0.2887 -> 0.6328 |
| `layer3.9.prelu` | 256 | 2.4159 | 1.1999 | 0.497 | 0.1948 -> 0.0925 | 0.2815 -> 0.5867 |
| `layer3.10.prelu` | 256 | 2.4891 | 1.1506 | 0.447 | 0.1770 -> 0.0806 | 0.2843 -> 0.6222 |
| `layer3.11.prelu` | 256 | 2.5788 | 1.1642 | 0.446 | 0.1819 -> 0.0819 | 0.2744 -> 0.6089 |
| `layer3.12.prelu` | 256 | 2.5162 | 1.2109 | 0.491 | 0.1974 -> 0.0908 | 0.2725 -> 0.5821 |
| `layer3.13.prelu` | 256 | 2.4446 | 1.1226 | 0.446 | 0.1774 -> 0.0803 | 0.2877 -> 0.6252 |
| `layer4.0.prelu` | 512 | 2.2585 | 0.9608 | 0.423 | 0.1659 -> 0.0703 | 0.3100 -> 0.7272 |
| `layer4.1.prelu` | 512 | 2.4195 | 1.0202 | 0.423 | 0.1910 -> 0.0809 | 0.2763 -> 0.6561 |
| `layer4.2.prelu` | 512 | 2.6670 | 1.1257 | 0.423 | 0.1430 -> 0.0604 | 0.2847 -> 0.6734 |

## Files

- `quadratic_shapes_by_layer.png`: all 25 sites in normalized coordinates,
  showing channel median and P10-P90 bands.
- `approximation_intervals_by_layer.png`: per-layer `lam_fit` P10/median/P90
  and `lam_reg` median in raw activation units.
- `quadratic_comparison_by_layer.pdf`: one detailed page per site, including a
  representative channel's explicit equations and raw-x curves, normalized
  all-channel shapes, and every channel's interval.
- `layer_summary.csv`: machine-readable per-layer coefficient, interval, and
  curve-difference quantiles.
- `comparison_summary.json`: checkpoint metadata, state layouts, and all summary
  values.

Regenerate with:

```bash
/home/u8798807/.conda/envs/face_recog/bin/python \
  -m controlled_degree2.compare_quadratic_checkpoints \
  --left work_dirs/controlled_degree2_accuracy_recovery_head_only_v3_20260908/train/student_best.pt \
  --right work_dirs/student_best.pt \
  --left-label 'head-only v3' \
  --right-label 'student_best_ijbc' \
  --output-dir docs/controlled_degree2_head_only_vs_student_best_ijbc
```
