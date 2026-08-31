# Epoch-10 eight-HerPN activation analysis and ninth-site plan

## Conclusion

`work_dirs/ms1mv3_r50_herpn/model_epoch_10.pt` is numerically safe, but its
internal activation distribution is **not** close to the original PReLU
iResNet50. Most deep activations are substantially narrower. The following
failure is a rare-tail distribution shift: epoch 11 updates the already
quadratic Layer2 prefix, one rare face leaves its normal range, and four new
Layer3 quadratics turn that finite tail into a repeated-square overflow.

The next experiment therefore preserves the eight accepted legacy quadratics
exactly and adds only the terminal `layer4.2.prelu` as a scaled, PReLU-aware
quadratic. It does not continue the failed four-at-once forward schedule.

## Evidence

### Checkpoint and schedule state

All tensors in epochs 10 through 15 are finite. The graph changes are:

| Checkpoint | Fully converted sites | Partially converted sites |
|---|---:|---|
| epoch 10 | stem + Layer1 + Layer2 = 8 | none |
| epoch 11 | same 8 | `layer3.0` through `layer3.3`, blend about 0.5 |
| epoch 12 | same 8 | `layer3.0` through `layer3.3`, blend about 1.0 |

Epoch 10 completed strict IJB-C inference over all 469,375 augmented rows with
zero non-finite rows and reached 95.18% TAR at FAR `1e-4`. Epoch 11 first
failed at IJB-C batch 13; epochs 12--15 also produced non-finite embeddings.

### Distribution comparison with the original ResNet50

The table profiles the first 1,400 aligned CFP-FP images, without test-time
flip, in FP32 on the same device. Statistics cover activation **inputs**.
Extrema cover every value; percentiles use a bounded sample.

| Tensor | Original std | Epoch-10 std | ratio | Original absmax | Epoch-10 absmax |
|---|---:|---:|---:|---:|---:|
| stem PReLU | 0.1653 | 0.04869 | 0.294 | 2.639 | 1.552 |
| `layer1.0.prelu` | 0.1945 | 0.06806 | 0.350 | 6.119 | 1.961 |
| `layer2.0.prelu` | 0.1425 | 0.07666 | 0.538 | 3.445 | 1.902 |
| `layer3.0.prelu` | 0.1379 | 0.02543 | 0.184 | 2.332 | 0.509 |
| `layer3.7.prelu` | 0.08881 | 0.01058 | 0.119 | 0.700 | 0.106 |
| `layer4.2.prelu` | 0.08658 | 0.01682 | 0.194 | 1.365 | 0.255 |
| embedding output | 0.9109 | 0.4458 | 0.489 | 4.476 | 2.196 |

Thus the good recognition result does not imply that the hidden
representations match. The converted model has learned a much lower-amplitude
operating regime, especially after Layer2. A continuation must protect this
new regime rather than reuse intervals measured on the original network.

The comparison is reproducible with:

```bash
python eval/compare_activation_distributions.py \
  --baseline-checkpoint work_dirs/ms1mv3_r50/model.pt \
  --candidate-checkpoint work_dirs/ms1mv3_r50_herpn/model_epoch_10.pt \
  --baseline-network r50 \
  --candidate-network r50_no_relu \
  --data-bin ms1m-retinaface-t1/cfp_fp.bin \
  --max-images 1400 \
  --output-prefix work_dirs/herpn_epoch10_vs_r50_cfp1400
```

### Reproduced tail cascade

CFP-FP encoded image index 704 is finite in epoch 10 and non-finite in epoch
11. Per-activation output maxima are:

| Activation | epoch 10 | epoch 11 |
|---|---:|---:|
| `layer2.0.prelu` | 0.959 | 7.09 |
| `layer2.1.prelu` | 0.892 | 19.43 |
| `layer2.2.prelu` | 0.273 | 503.1 |
| `layer2.3.prelu` | 0.586 | 1.63e5 |
| `layer3.0.prelu` | 0.274 | 8.95e8 |
| `layer3.1.prelu` | 0.0507 | 4.28e15 |
| `layer3.2.prelu` | 0.0683 | 2.37e29 |
| `layer3.3.prelu` | 0.165 | `inf` |

Small CUDA-kernel/batch-shape rounding changes the final large digits, as is
expected after repeated squaring, but not the first runaway site or outcome.
The epoch-11 Layer2 folded quadratic coefficients have `|A|max` between 0.40
and 0.55. They are not themselves non-finite or singular; once an input
reaches several units, the `A*x^2` term dominates and each following
quadratic amplifies the previous tail.

The first visible runaway is in the **already converted** Layer2 prefix,
before the newly blended Layer3 sites. Therefore the direct cause is upstream
weight/BatchNorm/coefficient drift during the four-site transition, followed
by an unbounded quadratic cascade. It is not an optimizer tensor NaN.

The old range objective cannot guarantee safety. It is averaged over all
pixels and batches, its per-sample maximum term has weight 0.1, the fixed
`[-6, 6]` interval is only a soft penalty, and inference correctly has no
non-polynomial clipping. A rare face can remain invisible to training loss
while still failing a 469,375-row IJB-C scan.

## Improved ninth-site experiment

The configuration is
`configs/ms1mv3_r50_herpn_epoch10_selective9_layer42.py`.

1. The first eight wrappers remain the original direct HerPN modules and load
   epoch-10 parameters exactly. A smoke comparison gives zero output
   difference before the ninth conversion.
2. Only `layer4.2.prelu` is converted. Being the final activation, it cannot
   inject a tail into another polynomial activation.
3. For its frozen channel-wise PReLU slope `a`, the new target on the public
   interval `[-S,S]` is
   `a*x + (1-a)*S*HerPN_ReLU(x/S)`. Calibration uses `S = 1.25 * observed
   absmax` over 2,048 augmented batches per rank.
4. The backbone is frozen during the local-fit epoch. Blend and recovery use
   1% of the base backbone learning rate and a frozen epoch-10 embedding
   teacher, limiting drift in the known-finite prefix.
5. Immediately before blending, the interval is measured again. The code
   temporarily enables the complete singleton and profiles its embedding
   boundary for the same 2,048 batches per rank. Non-finite output or absmax
   above 1,000 aborts before the first blend update.
6. Every ordinary validation pass is strict. Completion also saves a
   downstream-only BatchNorm-recalibrated group checkpoint.

No clamp, division by encrypted data, or branch is added to inference. After
folding, the new activation is `A*x^2+B*x+C`; it adds one ciphertext square
level. The nine-site path therefore has nine quadratic activation levels.

## Acceptance gates and next replacement

The ninth site is accepted only if:

- LFW, CFP-FP, and AgeDB validation have zero non-finite embeddings throughout;
- the completed-group checkpoint passes all 469,375 augmented IJB-C rows with
  zero non-finite values; and
- IJB-C TAR at FAR `1e-4` is at least 94.68%, no more than 0.50 percentage point
  below the 95.18% epoch-10 baseline.

If numerical safety passes but accuracy does not, recover the fixed nine-site
graph before attempting another conversion. If both pass, screen each
remaining zero-blend site on the accepted nine-site graph and choose the next
site by full-boundary tail and embedding-cosine shock; do not assume forward
order is optimal.
