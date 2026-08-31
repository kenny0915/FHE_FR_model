# Practicality of Low-Degree Polynomial Activations in iResNet-50 on IJB-C

## Executive decision

Low-degree polynomial activation is **practical as a selective/hybrid research
model**, but the experiments do **not** yet support a fully polynomial,
end-to-end FHE iResNet-50 deployment at the required accuracy.

The strongest strict result replaces 8 of the 13 retained PReLUs in the
`nl13` iResNet-50 with degree-2 polynomials. It processes all 469,375 IJB-C
images with zero non-finite augmented embedding rows and reaches 90.01% TAR at
FAR `1e-4`. This technically passes the requested gate, but:

- five PReLUs remain in the encrypted backbone;
- the accuracy margin is only 0.01 percentage point;
- a seven-polynomial checkpoint is 0.85 point more accurate;
- attempts to add a ninth polynomial caused a large recognition-accuracy
  shock even after its input range was successfully conditioned; and
- all reported evaluation is ordinary FP32 PyTorch inference, not execution
  under an FHE approximation, noise budget, or coefficient quantization.

The appropriate current decision is therefore:

| Intended use | Decision | Reason |
|---|---|---|
| Selective polynomial research/prototyping | **Go** | Seven- and eight-site models meet the strict IJB-C gate with finite outputs. |
| Hybrid private system where five PReLUs are outside the encrypted path or handled separately | **Conditional go** | System design and leakage implications must be specified and measured. |
| Fully polynomial encrypted iResNet-50 at 90% TAR | **No-go today** | No all-polynomial checkpoint meets the accuracy and numerical-safety gate. |
| Production deployment | **No-go today** | Accuracy margin, seed robustness, approximate-arithmetic behavior, latency, and memory remain unvalidated. |

## Acceptance criterion and approximation

A checkpoint qualifies only when strict inference over **all 469,375 IJB-C
images** has:

1. zero non-finite augmented embedding rows; and
2. TAR at FAR `1e-4` of at least 90%.

At activation site `i`, the degree-2 polynomial approximates that site's
frozen channel-wise PReLU on a measured public interval `[-S_i, S_i]`:

```text
P_i(x) = S_i * HerPN(x / S_i) = A_i x^2 + B_i x + C_i
```

The inference expression needs one ciphertext-ciphertext square at each
replaced site. Thus each quadratic contributes one multiplicative level on a
path through that site. The exact end-to-end FHE depth depends on which sites
are sequential on the residual graph and has not yet been compiled or
benchmarked in an FHE backend.

## Measured accuracy and safety frontier

The following are strict IJB-C results. They show that training procedure and
site selection matter as much as raw replacement count.

| Polynomial sites | Training/selection condition | Non-finite rows | TAR @ FAR `1e-4` | Result |
|---:|---|---:|---:|---|
| 3 | forward conversion | 0 | 95.20% | pass |
| 4 | forward conversion | 0 | 92.85% | pass |
| 5 | forward conversion | 0 | 91.79% | pass |
| 6 | fast schedule | 0 | 84.72% | fail |
| 6 | slow blend + fixed-graph recovery | 0 | 92.69% | pass |
| 7 | forward, fast schedule | 0 | 77.52% | fail |
| 7 | forward, extended recovery | 0 | 87.36% | fail |
| 7 | selective terminal `layer4.2`, epoch 7 | 0 | **90.86%** | pass |
| 8 | boundary-seeded | 0 | 88.62% | fail |
| 8 | recovered-seven source | 0 | 89.50% | fail |
| 8 | fresh PartialFC head, low-LR distillation | 0 | 88.79% | fail |
| 8 | fresh PartialFC head, high-LR distillation | 0 | 87.17% | fail |
| 8 | class-head-preserving resume, epoch 9 | 0 | 89.75% | fail |
| 8 | class-head-preserving resume, epoch 10 | 0 | **90.01%** | pass |

The complete experiment table and artifact index are in
[`nl13_polynomial_replacement_results.md`](nl13_polynomial_replacement_results.md).

### Maximum-count passing checkpoint

```text
work_dirs/ms1mv3_r50_nl13_prelu_herpn_selective8_resume_distill/
model_selective8_resume_epoch10_ijbc_90p01.pt
```

- SHA-256:
  `45d6e0e29401b2d2597918d3e791645e1a8c12a123a9c331c55701f7e4de957f`
- strict TAR at FAR `[1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]`:
  `[75.72, 83.49, 90.01, 94.15, 97.00, 98.82]%`
- strict IJB-C non-finite augmented embedding rows: `0 / 469,375`
- public site scales in conversion order:
  `[1.2140, 3.0112, 2.9724, 3.9498, 8.9278, 29.6929, 1.6235, 3.6565]`

The scale range is about 24.5-fold (`29.6929 / 1.2140`). This is direct
evidence that one global interval such as `[-5,5]` is unsuitable for this
face backbone.

The successful graph contains 8 quadratics, 5 PReLUs, and 12 intentional
identities across the original 25 nonlinear slots. Identity removal is
FHE-friendly, but it is not a polynomial approximation of PReLU. The five
remaining PReLUs are:

```text
layer3.3.prelu
layer3.6.prelu
layer3.9.prelu
layer3.13.prelu
layer4.0.prelu
```

### Higher-margin selective checkpoint

The seven-site fallback reaches 90.86% TAR with zero non-finite rows:

```text
work_dirs/ms1mv3_r50_nl13_prelu_herpn_selective7_layer42/
model_selective7_layer42_ijbc_90p86.pt
```

Its 0.86-point margin above the gate is much safer than the eight-site
model's 0.01-point margin. If one remaining non-polynomial activation can be
handled elsewhere, this is the more credible research checkpoint.

## Why the replacements fail

The large dataset exposes the problem; it is not the root mathematical cause.

### 1. Polynomial extrapolation is unbounded

A nonconstant quadratic grows as `x^2` outside its fitted interval, whereas
PReLU remains piecewise linear. A rare activation tail that is harmless under
PReLU can therefore become very large after replacement. Subsequent
polynomials and residual blocks compound that error. The 5.18-million-image
training set and 469,375-image IJB-C evaluation make rare tails more likely to
appear than a small calibration set does.

One attempted forward site had robust magnitude `39.04` but observed magnitude
`22,564.06`; covering the tail required a 39.74-fold interval expansion.
Earlier all-polynomial reduced-topology attempts reached validation embedding
magnitudes of `8.49947e22` and `4.93499e7`, and a frozen-BatchNorm variant
still produced a non-finite row.

### 2. Finiteness and face-recognition quality are different constraints

Several checkpoints had zero non-finite rows but failed badly on low-FAR
recognition: 84.72%, 77.52%, 87.36%, 88.62%, and 89.50% TAR. A polynomial can
be numerically controlled while still rotating or compressing embeddings
enough to destroy the extreme-tail separation used by IJB-C verification.

The attempted ninth site, `layer4.0.prelu`, demonstrates this clearly. Two
branches reduced provisional scales around `9e4` to strict scales `6.3372` and
`6.4419` without calibration violations. Nevertheless, full blend drove
CFP-FP and AgeDB accuracy to about 77--78%. The weaker branch later recovered
only 87.01% CFP-FP and 89.60% AgeDB and was stopped before an expensive IJB-C
run. This was an accuracy/representation boundary, not a non-finite boundary.

### 3. Progressive conversion changes downstream distributions

Every accepted polynomial changes the inputs seen by later BatchNorm and
activation sites. Intervals measured on the original PReLU network become
stale. Causal per-site calibration, slow blending, downstream BatchNorm
recalibration, and fixed-graph recovery were all required. Even then, the
best replacement order was selective rather than the natural forward order.

### 4. Face-training state is unusually important

The eighth site only crossed 90% when the trained distributed PartialFC class
head, optimizer momentum, and scheduler state were preserved. Fresh-head
distillation scored only 87.17--88.79%, even when LFW and CFP-FP proxy results
looked good. Proxy benchmarks therefore cannot safely select an IJB-C
low-FAR checkpoint.

### 5. The result is fragile relative to deployment arithmetic

The passing eight-site result is one checkpoint only 0.01 point above the
gate. There is no multi-seed confidence interval. It also has not been tested
with fixed-point polynomial coefficients, CKKS rescaling error, ciphertext
noise, or the normalization and template-scoring protocol that a real private
system would use. A deterministic FP32 pass is necessary evidence, not a
deployment guarantee.

## Why PILLAR-ESPN does not establish practicality for this task

PILLAR-ESPN successfully trains an ImageNet torchvision ResNet-50 using the
fixed quartic

```text
q(x) = 0.314453125 + 0.5x + 0.15625x^2 - 0.0029296875x^4
```

to approximate ReLU on `[-5,5]`. It has degree 4 and multiplicative depth 2.
The reported ImageNet result is 77.7% plaintext and 77.3% private accuracy,
versus 80.8% for its ReLU comparison. That is strong evidence that activation
co-design can work, but it does not transfer directly here:

- the reference task is ImageNet classification, not open-set face
  verification at FAR `1e-4`;
- it uses a torchvision post-activation ResNet-50 with scalar ReLU, whereas
  this project uses a pre-activation iResNet with channel-wise PReLU,
  embedding BatchNorm, ArcFace, and distributed PartialFC;
- PILLAR trains for 600 epochs with range-only warm-up, summed high-order range
  penalties, richer augmentation, and EMA;
- its empirical test set is not a universal bound outside `[-5,5]`; the paper
  itself treats escaping activations as a core problem; and
- the reported private system uses CrypTen/ESPN 64-bit-ring MPC, not an FHE
  deployment. The polynomial is FHE-compatible, but the published runtime is
  not an FHE runtime.

The exact PILLAR face adaptation and failed all-polynomial experiments are
documented in [`pillar_espn_face_experiment.md`](pillar_espn_face_experiment.md).
The official sources are the
[USENIX Security 2024 paper](https://www.usenix.org/system/files/usenixsecurity24-diaa.pdf)
and [PILLAR-ESPN implementation](https://github.com/LucasFenaux/PILLAR-ESPN).

## Practicality assessment

| Dimension | Evidence | Assessment |
|---|---|---|
| Degree/depth per activation | Degree 2; one square level | favorable locally |
| Numerical safety | 0 non-finite rows for selected seven/eight sites | demonstrated selectively |
| IJB-C accuracy | 90.86% at seven sites; 90.01% at eight | meets gate, weak margin at maximum count |
| Full polynomial encrypted path | 5 PReLUs remain | not achieved |
| Training stability | large dependence on schedule, order, BN, and head state | fragile and expensive |
| Tail coverage | site scales differ 24.5-fold; rare extremes observed | high operational risk |
| Reproducibility | one maximum-count passing checkpoint | insufficient for production |
| Actual FHE execution | no compiled depth/noise/runtime/memory result | unvalidated |

## Recommended next decision gates

Do not spend another large run merely extending the same sequential
replacement schedule. A credible fully polynomial program should first meet
these gates:

1. **Architecture co-design:** train an iResNet variant from the start with
   fewer activation depths, residual scaling, and site-specific quadratics,
   instead of converting a mature PReLU embedding geometry one site at a time.
2. **Repeatability:** reproduce at least three independent training seeds and
   require a buffer above the target (for example, 90.5% rather than 90.01%).
3. **Approximate-arithmetic simulation:** quantize polynomial coefficients and
   simulate the chosen CKKS scale/rescale schedule before another full IJB-C
   run.
4. **Encrypted-path audit:** remove or explicitly place outside encryption all
   remaining PReLU, division, normalization, max, data-dependent branching,
   template aggregation, and score-normalization operations.
5. **FHE systems measurement:** report compiled multiplicative depth,
   ciphertext levels, bootstraps, latency, peak memory, and any accuracy delta
   on the exact strict checkpoint.
6. **Use the seven-site model as the control:** it gives up one replacement but
   retains 0.85 percentage point more TAR than the maximum-count checkpoint.

Until these gates are passed, the defensible claim is:

> Degree-2 selective replacement in iResNet-50 can satisfy strict IJB-C
> numerical safety and 90% TAR, but a fully polynomial, production-practical
> FHE face-recognition model has not yet been demonstrated.
