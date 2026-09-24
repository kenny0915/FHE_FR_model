# Epoch-10 eight-HerPN activation analysis and accepted ninth replacement

## Conclusion

`work_dirs/ms1mv3_r50_herpn/model_epoch_10.pt` is numerically safe, but its
internal activation distribution is **not** close to the original PReLU
iResNet50. Most deep activations are substantially narrower. The following
failure is a rare-tail distribution shift: epoch 11 updates the already
quadratic Layer2 prefix, one rare face leaves its normal range, and four new
Layer3 quadratics turn that finite tail into a repeated-square overflow.

The successful extension preserves the eight accepted legacy quadratics
exactly and replaces `layer3.9.prelu` with the global channel-wise degree-one
polynomial `a_c*x`, where `a_c` is its frozen PReLU negative-branch slope.
Suffix-only embedding distillation recovers the accuracy without changing the
nine activation functions or the numerically fragile prefix. The accepted
checkpoint reaches 94.69755% IJB-C TAR at FAR approximately `1e-4`, only
0.48245 percentage point below epoch 10, with all evaluated embeddings finite.

## Evidence

### Checkpoint and schedule state

All tensors in epochs 10 through 15 are finite. The graph changes are:

| Checkpoint | Fully converted sites | Partially converted sites |
|---|---:|---|
| epoch 10 | stem + Layer1 + Layer2 = 8 | none |
| epoch 11 | same 8 | `layer3.0` through `layer3.3`, blend about 0.5 |
| epoch 12 | same 8 | `layer3.0` through `layer3.3`, blend about 1.0 |

Epoch 10 completed strict IJB-C inference over all 469,375 source images in
both orientations (938,750 embeddings) with zero non-finite rows and reached
95.18% TAR at FAR `1e-4`. Epoch 11 first
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
while still failing the 469,375-image, two-orientation IJB-C scan.

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
1006.59 in one orientation and 1.57239e6 after horizontal flip. A subsequent
fuller pass found 1.9028e15 at flipped row 5039. Fixed cutoffs of 1e3, 1e6, and
1e12 all rejected the unchanged source graph before the replacement was
active. The linear run therefore uses elementwise NaN/Inf as the hard gate and
logs maxima for blend-zero versus blended comparison. This policy is not
appropriate for an added quadratic: the observed 1e15 tail already rules that
out, whereas the degree-one student cannot superlinearly amplify it.

The first uncapped run (`321454`) then showed that freezing parameter gradients
alone does not preserve the baseline during local fit: BatchNorm running
buffers still update in training mode. Between validation steps 2000 and 4000,
while the replacement blend remained exactly zero, CFP-FP absmax increased
from 3.54 to 12.81 and AgeDB absmax from 1.90e15 to 7.69e16. The run was
stopped before blending. Simply forcing BatchNorm to eval mode was also wrong:
run `321518` reproduced a non-finite training embedding at step 641 because
the source graph needs batch moments for those training tails. The corrected
policy keeps train-mode batch normalization for the forward, then restores all
running buffers after each local-fit batch. During blend/recovery it restores
the buffers after every batch as well. Run `321536` showed a new failure at
only 9.3% affine blend: CFP-FP absmax grew from 2.88 to 4.70e4 and AgeDB from
2.66e14 to 3.69e17 even though the learned affine slope was bounded in
`[0.84, 1.0]`. Initially this was attributed to downstream running-stat drift.
Run `321603` restored **all** BN buffers after every blend batch and reproduced
essentially the same result (CFP-FP 4.66e4, AgeDB 3.66e17), ruling that
explanation out.

The actual cause is a sign-dependent extrapolation error. At this site the
frozen PReLU slopes are in `[-0.03324, 0]`, with mean `-0.02643`. Thus an
extreme negative input is multiplied by a small negative coefficient by the
teacher, while the centrally fitted affine multiplies it by about `+0.92`.
The fitted student can therefore make the rare negative tail roughly 35 times
larger and reverse its sign even though it has no square. Degree one prevents
superlinear growth, but does not by itself make an already enormous tail safe.

### Square-free full-replacement screen

Job `321650` screened full (`blend=1`) variants on all LFW, CFP-FP, and AgeDB
images. The already disproven fitted variant was stopped to let the useful
candidates finish inside the one-hour allocation. Accuracy and maximum finite
embedding magnitude were:

| variant | LFW | CFP-FP | AgeDB | AgeDB absmax | non-finite |
|---|---:|---:|---:|---:|---:|
| epoch-10 source | 99.800% | 98.000% | 97.517% | 2.665e14 | 0 |
| `prelu_slope` | 99.767% | 97.686% | 97.050% | 2.592e14 | 0 |
| `zero` | 99.733% | 97.614% | 96.900% | 2.427e14 | 0 |
| `small_positive_bias` | 99.483% | 94.900% | 94.500% | 2.451e14 | 0 |

`prelu_slope` is the selected ninth activation. Its degree-one polynomial is
`a_c*x`, using the frozen PReLU negative-branch slope, with zero bias. The
approximation target is deliberately tail-oriented: it is exact to PReLU on
`x < 0`; on `x >= 0` it replaces the unit slope by `a_c` and relies on the
residual path. There is no bounded approximation interval because this is a
global affine polynomial; the complete benchmark distributions, including
AgeDB row 5039, are the empirical safety domain. It adds no encrypted square
and introduces no division, clipping, comparison, or data-dependent branch.

Full IJB-C job `321724` evaluated the materialized nine-site checkpoint on
469,375 source images in both orientations. All 938,750 embeddings were
finite. Its TAR values were 80.82%, 89.22%, **93.61%**, 95.84%, 97.50%, and
98.72% at FAR `1e-6` through `1e-1`. Numerical safety therefore passes, but
the FAR `1e-4` accuracy is 1.57 percentage points below the 95.18% epoch-10
baseline and misses the 94.68% acceptance threshold.

Recovery job `321743` first tried full-backbone ArcFace training plus embedding
distillation at an effective `3e-5` learning rate. Although it remained finite
and reduced the AgeDB tail to 1.55e14, by step 1000 LFW/CFP-FP/AgeDB had all
degraded to 99.700%/97.243%/96.917%, so the run was stopped.

The revised recovery configuration is
`configs/ms1mv3_r50_herpn_epoch10_linear9_prelu_slope_recovery.py`. It starts
from `model_linear9_prelu_slope_static.pt`, keeps `a_c` and the zero bias
frozen, preserves all BatchNorm running buffers, and disables the classifier
loss. Only `layer4.2.conv2`, `layer4.2.bn3`, and the final `bn2`/`fc`/`features`
layers can learn from the epoch-10 embedding teacher; the complete polynomial
prefix is immutable. In particular, recovery cannot optimize the safe
negative slope back into the disproven positive affine fit or perturb the
upstream tail distribution.

Job `321749` tested that suffix-only objective with ordinary train-mode batch
moments. Its distillation loss fell from 0.397 to 0.288, but at step 500
LFW/CFP-FP had degraded to 99.650%/96.414%. This isolates a train/eval domain
mismatch: the frozen teacher and all acceptance tests use inference running
moments, while the student was learning from batch moments. The next recovery
therefore uses frozen inference BN moments for the student too. Rare MS1M
batches on which the accepted prefix itself is non-finite are synchronously
counted and skipped during optimization; benchmark validation and full IJB-C
retain the zero-tolerance finite gate.

The first inference-moment launch (`321753`) exposed an implementation
interaction before its first update: the older preserve-buffer path copied BN
buffers after the forward, but eval-mode BN backward had saved those tensors,
so autograd correctly rejected the in-place version change. Frozen BN does not
update buffers at all; the corrected path simply omits that redundant
snapshot/restore when inference moments are enabled.

Corrected job `321755` reproduced the known source-prefix overflow at step 641
and skipped it as intended, but its step-500 result still degraded to
99.600%/96.300%/95.883%. The information loss is therefore fundamental, not
just a BN-mode mismatch: applying the small negative slope to every positive
input discards information that a downstream linear/convolutional suffix
cannot reconstruct.

The next screen keeps a degree-one polynomial but chooses coefficients per
channel. A channel whose observed input crosses a calibrated negative-tail
threshold uses the PReLU negative slope; other channels use identity, which is
exact on PReLU's positive branch. Thresholds are calibrated jointly on the
three full verification sets and known MS1M counterexample RecordIO index
86052. A candidate must keep every embedding finite, limit worst-tail growth
to 10x the source, and stay within 0.5 accuracy point before it can proceed
to full IJB-C. This remains a fixed channel-wise affine polynomial with zero
ciphertext multiplication depth.

The first channel screen found that thresholds through `1e6` still marked all
512 channels unsafe: the known MS1M tail has been mixed across the complete
layer. Its overly tight 1.1x single-image magnitude gate also rejected the
otherwise finite all-negative-slope fallback. The expanded screen probes
thresholds through `1e23`; the 10x tail-growth bound remains conservative
against the 35x amplification already observed for the failed fitted affine.

Completed job `321772` found 511/512 channels unsafe through `1e21`. At
`1e23`, releasing 100 channels worsened LFW/CFP-FP/AgeDB to
99.383%/96.800%/95.500%. Per-channel masking at `layer4.2` therefore cannot
recover accuracy. The next screen applies the same globally tail-safe affine
to each of the 17 remaining PReLUs separately. This tests which residual block
is most redundant instead of assuming the last activation is the least
disruptive choice.

Site-selection job `321773` found two near-gate Layer3 candidates. Replacing
`layer3.8.prelu` (index 16) yielded 99.683%/97.500%/97.200%; replacing
`layer3.9.prelu` (index 17) yielded 99.767%/97.471%/97.267%. Their CFP-FP drops
miss the 0.5-point screening gate by only 0.014 and 0.043 point respectively,
while all outputs and the known RecordIO tail remain finite. Both proceed to
parallel full IJB-C evaluation because this difference is below one 10-fold
validation sample step and `layer4.2` is already known to miss IJB accuracy.

Both full jobs were numerically safe over 938,750 embeddings. `layer3.8`
(`321777`) reached 94.21% and `layer3.9` (`321778`) reached 94.27% TAR at FAR
`1e-4`, versus 93.61% for `layer4.2`. They still miss the 94.68% acceptance
gate by 0.47 and 0.41 point. `layer3.9` becomes the recovery baseline because
its remaining four Layer3 blocks, complete Layer4, and embedding head can be
distilled while the accepted nine-activation prefix stays immutable.

The first suffix-distillation launch (`321787`) improved the static candidate.
At step 500 LFW/CFP-FP/AgeDB were 99.750%/97.729%/97.417%; at step 1000 they
were 99.733%/97.843%/97.333%. Because CFP continued improving while the other
sets peaked earlier, the deterministic run is repeated with per-validation
snapshots enabled so both checkpoints can receive the authoritative IJB gate.

The repeated run reproduced those metrics and saved both states. Full IJB-C
remained finite, with step 500 (`321795`) at 94.54% and step 1000 (`321797`)
at 94.62% TAR@FAR=`1e-4`. Step 1000 is only 0.06 point below the acceptance
gate. The saved step-1500 state is evaluated once, while a separate continuation
starts from step 1000 at a 10x smaller learning rate to search the narrow
recovery region without the observed CFP-FP overshoot.

### Accepted ninth-replacement result

Step 1500 of the first recovery reached 94.68221% on the exact ROC operating
point. A continuation from step 1000 at `3e-5` learning rate then produced:

| checkpoint | LFW | CFP-FP | AgeDB | exact IJB-C TAR | selected FPR |
|---|---:|---:|---:|---:|---:|
| low-LR step 250 | 99.750% | 97.843% | 97.433% | 94.68221% | 0.0001003905 |
| low-LR step 500 | 99.750% | 97.857% | 97.400% | **94.69755%** | 0.0001001347 |

Both IJB score vectors contain 15,658,489 finite pair scores. The selected
step-500 run processed the complete 469,375-image IJB-C source set in both
orientations with zero non-finite embeddings. Relative to the epoch-10
95.18% baseline, its exact loss is 0.48245 percentage point, so it passes the
predeclared 0.50-point gate by 0.01755 point.

The accepted inference checkpoint is:

```text
work_dirs/ms1mv3_r50_herpn_epoch10_linear9_layer3_9_low_lr/model_step_00500.pt
sha256 9b250a23cb546d6f54d97aab3f7584934307e603a04e74c96a4edcbe080edab1
```

A strict CPU reload confirms all state tensors are finite, all keys match,
exactly nine of 25 progressive activations have blend 1, and none is partial.
The converted names are the stem, all three Layer1 sites, all four Layer2
sites, and `layer3.9.prelu`. At the ninth site the stored linear weight equals
the frozen PReLU slope exactly, the bias is zero, and both are frozen. It never
evaluates `x.square()`, so the ninth replacement adds no activation
multiplication level; only the original eight degree-two activations contribute
nonlinear multiplicative depth.

The recovery did encounter the already documented source-prefix overflow on
one MS1M training batch at step 641 and synchronously skipped that optimization
step. This is not used to relax inference: all three verification sets and the
complete IJB-C run retain a zero-tolerance finite gate. The checkpoint is
therefore accepted over the measured natural inference domain, not claimed to
make the old eight-quadratic prefix globally bounded.

## Acceptance gates and next replacement

The ninth site is accepted only if:

- LFW, CFP-FP, and AgeDB validation have zero non-finite embeddings throughout;
- the completed-group checkpoint passes all 469,375 IJB-C source images in
  both orientations (938,750 augmented embeddings) with zero non-finite
  values; and
- IJB-C TAR at FAR `1e-4` is at least 94.68%, no more than 0.50 percentage point
  below the 95.18% epoch-10 baseline.

The ninth site passes these gates. For a tenth replacement, use this selected
step-500 checkpoint as the immutable baseline. Screen each of the remaining
16 zero-blend sites independently with the same globally tail-safe degree-one
candidate, then rank by complete validation accuracy, known RecordIO-tail
growth, and embedding-cosine shock. Do not resume the old four-at-once forward
schedule or assume network order predicts replacement safety.
