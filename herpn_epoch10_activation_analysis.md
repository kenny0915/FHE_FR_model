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
3. Its student is the channel-wise degree-one polynomial `m_c*x+b_c`, locally
   distilled against the frozen PReLU before blending. It introduces no
   ciphertext square and needs no approximation interval.
4. The backbone is frozen during the local-fit epoch. Blend and recovery use
   1% of the base backbone learning rate and a frozen epoch-10 embedding
   teacher, limiting drift in the known-finite prefix.
5. Every ordinary validation pass is strict. Completion also saves a
   downstream-only BatchNorm-recalibrated group checkpoint.

No clamp, division by encrypted data, or branch is added to inference. After
folding, the new activation is exactly `m_c*x+b_c`; it adds zero ciphertext
square levels. The nine-site graph therefore removes nine PReLUs while retaining
only the original eight quadratic activation levels.

The first Nano4 submission (`321291`) tested a stronger photometric stress
domain. It failed at global step zero while observing the input of
`layer4.2.prelu`, before the ninth polynomial was enabled. The accepted
epoch-10 graph's existing eight-polynomial prefix therefore overflowed on a
synthetically stressed input. That augmentation is outside this fixed
baseline's stable domain and cannot be repaired by rescaling the later ninth
site. The revised run retains the fail-fast checks but uses the natural
training distribution; it does not hide the failure with clipping.

The complete natural-data follow-up (`321306`) found a second, stronger
counterexample. The graph remained finite, but one ordinary training image
reached `3.431934e23` at the input of `layer4.2.prelu` (rank 7, batch 2931,
sample 115; RecordIO index 86052). Replaying both orientations showed the
cascade begins in the already-converted prefix: about `8.5` after
`layer1.2.prelu`, `1.1e3` after `layer2.0`, `6.7e5` after `layer2.1`, `2.2e12`
after `layer2.2`, and `6.3e24` after `layer2.3`. The image bytes are valid.

Consequently, adding any nonzero quadratic at the ninth site is not globally
safe: even a zero quadratic coefficient evaluated naively as `0*x^2` becomes
NaN when `x^2` overflows. The revised degree-one student and its folded module
never evaluate a square. This directly addresses the observed failure instead
of choosing a robust quantile that hides it.

The first linear run (`321376`) reproduced the source checkpoint at blend zero:
LFW was 99.717% and CFP-FP was 97.957%, with finite maxima 3.03 and 3.54.
AgeDB then exposed another source-checkpoint tail at row 1317: magnitude
1006.59 in one orientation and 1.57239e6 after horizontal flip. The original
drift thresholds were therefore below the baseline they were intended to
guard. The hard guard is set to 1e12 for the linear run, while non-finite
values remain fatal and logged maxima are compared to the blend-zero value.
This headroom is not appropriate for an added quadratic, but the degree-one
student cannot superlinearly amplify this tail.

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
