# PILLAR-ESPN all-polynomial iResNet-50 experiment

## Objective

Train a ResNet-50-family face backbone whose activation sites are all fixed
low-degree polynomials, then require strict inference over all 469,375 IJB-C
images to produce zero non-finite augmented embedding rows and at least 90%
TAR at FAR `1e-4`.

The encrypted activation approximates ReLU on `[-5, 5]`:

```text
q(x) = 0.314453125 + 0.5 x + 0.15625 x^2 - 0.0029296875 x^4
```

It has degree 4 and multiplicative depth 2 (`x^2`, then `x^4`). Training-only
clipping and range regularization are removed from the inference graph.

## Authoritative implementation audit

Reference repository:
`https://github.com/LucasFenaux/PILLAR-ESPN`, audited at commit
`b3522ca3f290c48475887b03714e944bdd715935`.

The failed `ms1mv3_r50_pillar_d4` run copied the polynomial but did not copy
several decisive behaviors from the released implementation:

| Item | Released ImageNet command/code | Failed face run |
|---|---:|---:|
| epochs | 600 | 24 |
| batch per GPU | 256 (4 GPUs, total 1024) | 128 (4 GPUs, total 512) |
| initial LR | 0.03 | 0.013 |
| range coefficient | `1e-4` | `5e-5` |
| per-layer penalty | sum over every activation element | elementwise mean |
| epoch 0 task loss | exactly zero; range penalty only | ArcFace enabled |
| convolution weight decay | `2e-5` | `1e-4` |
| normalization decay | zero | `1e-4` |
| augmentation | TrivialAugment, erasing, MixUp, CutMix, repeated augmentation | horizontal flip |
| EMA | enabled | disabled |

The penalty reduction is the most important implementation mismatch. Released
`get_penalty` flattens `(x / 4.8)^gamma` and applies an L1 norm. Because gamma
is even, this is an unnormalized sum, not a mean. On early high-resolution
feature maps, changing it to a mean weakens the range term by millions of
times. The old training loss could therefore decrease while the clipping-free
validation graph produced embedding norms around `1e34` and later collapsed
to chance accuracy.

The released recipe also uses an ordinary torchvision bottleneck ResNet-50
with post-addition ReLUs for ImageNet classification. This project uses the
pre-activation face iResNet topology, 25 channel-wise PReLU sites, an embedding
BatchNorm head, and ArcFace/PartialFC. Therefore, the released hyperparameters
are a strong starting point rather than proof that the face model will meet the
IJB-C low-FAR gate.

The paper's claim is empirical rather than a bound over every possible input.
Its ImageNet ResNet-50 result is one trained run: 77.7% plaintext PILLAR
accuracy and 77.3% encrypted accuracy, versus 80.8% for the ReLU/CryptGPU
model. The authors explicitly call the same runaway behavior the “escaping
activation problem” and address it with the combination of range
regularization, training-only clipping, a four-epoch beta/gamma warm-up, and
BatchNorm. Their test demonstrates stability on the ImageNet evaluation they
ran; it does not prove that the quartic is bounded outside `[-5,5]` or that no
sample in a much larger out-of-distribution face corpus can escape. PILLAR's
reported private runtime uses 64-bit-ring MPC (CrypTen/ESPN/HoneyBadger), not
leveled FHE. The degree-4 inference polynomial is nevertheless compatible
with the polynomial portion of an FHE graph.

## Implemented face adaptation

`configs/ms1mv3_r50_pillar_espn.py` keeps the exact fixed quartic, `[-5,5]`
training clip, `[-4.8,4.8]` penalty interval, summed layer penalty, beta/gamma
warm-up, range-only epoch 0, FP32 execution, local batch 256, no normalization
weight decay, and strict post-warm-up validation. Bounded face photometric
range augmentation replaces MixUp/CutMix because mixing two identity labels
is incompatible with the current ArcFace/PartialFC target.

Early unclipped validation is deliberately deferred through the five-epoch
range/LR warm-up. It is then fail-fast for any non-finite embedding or
embedding magnitude above `1e4`.

## Eight-GPU smoke result

Slurm job `316069` ran six 200-step epochs on eight H200 GPUs using CASIA.

- All training embeddings and gradients remained finite.
- Range-only epoch 0 reduced training activation absmax from `19.47` at step
  10 to `11.94` at step 200.
- At the final step, activation absmax was `6.03`; only `1.65e-6` of activation
  elements exceeded the polynomial interval.
- The first strict post-warm-up LFW pass had zero non-finite outputs,
  embedding absmax `6.43482`, XNorm `23.981727`, and 94.967% flip accuracy.

The earlier 200-step checkpoint was intentionally tested once before the
deferred-validation policy was added. It still had one non-finite LFW row,
showing why all penalty warm-up phases must finish before judging the method.

## Production and IJB-C jobs

The first MS1Mv3 production attempt, Slurm `316092`, completed the released
four-epoch penalty warm-up with finite embeddings and gradients. At step
`10466`, shortly after beta reached its full `1e-4` value, a rare internal
iResNet activation made the direct FP32 `z^10` penalty non-finite. The
activation output itself was still hidden by the training-only clip, so the
failure occurred in the range loss before strict inference validation. The
clean epoch-4 model, PartialFC, optimizer, and scheduler states were preserved.

The resumed face adaptation keeps the exact released `z^10` penalty through
`|z|=2` (`|x|=9.6`) and continues with its tangent line for larger training
outliers. This makes the loss and its restoring gradient continuous while
preventing FP32 power overflow. A gradient-norm cap of `5` provides a second
guard against a single summed-penalty spike. Both changes are training-only;
the encrypted inference activation remains the same unclipped fixed quartic.

Recovery Slurm `316300` passed the former range-loss failure but found one
non-finite LFW embedding row at the first unclipped validation, step `15000`.
This is a rare-tail failure rather than broad collapse: the same training
batch had activation absmax `6.67` and only `4.03e-7` outside `[-5,5]`.
Post-warm-up verification now runs diagnostically through epoch 11 so this
single row cannot terminate further range conditioning; strict fail-fast
validation starts at epoch 12, and every full IJB-C evaluation remains strict.

Recovery Slurm `316391` quantified the tail rather than hiding it. At step
`15000`, the unclipped graph produced 2 non-finite augmented LFW rows, 81
CFP-FP rows, and 87 AgeDB-30 rows. At step `17500`, the respective counts were
0, 63, and 79. Finite embeddings remained ordinary in magnitude (maximum
about 6), and LFW flip accuracy was 99.117% at step `17500`. More range
conditioning therefore helps, but the failure count is not falling quickly
enough to assume that scale-1 inference will pass all 469,375 IJB-C images.

Layer hooks on the epoch-6 model identified the causal cascade. A persistent
CFP-FP sample had activation-input maxima
`5.54 -> 6.13 -> 18.93 -> 1279.57 -> 8.63e9`; the value `18.93` entered
`layer1.1.prelu`, whose quartic output became `-329.7`. The next residual
block amplified that finite tail until `layer2.0.prelu` overflowed. Seven of
the first eight failed CFP-FP rows began at `layer1.1.prelu` with maxima
between 16.53 and 18.93; the eighth began at `layer1.2.prelu` at 21.38. Thus,
large dataset coverage is an exposure mechanism, but the root cause is the
unbounded quartic outside its fitted interval combined with a train/eval
mismatch: training clips each activation after recording its penalty, whereas
inference must evaluate the polynomial exactly.

## Scaled-interval branch

For a positive scale `s`, the implementation now supports

```text
q_s(x) = s q(x / s)
```

which approximates ReLU on `[-5s, 5s]`. This transformation retains degree 4
and multiplicative depth 2; it only changes public polynomial coefficients.
The corresponding training penalty interval is `[-4.8s, 4.8s]`. The scale is
stored in checkpoints, while legacy scale-1 checkpoints still load strictly.

An inference-only test that widened just `layer1.1.prelu` and
`layer1.2.prelu` contained the original failure but shifted the first unsafe
input to `layer2.0.prelu`. Applying scale 4 to all sites with old scale-1
BatchNorm statistics also failed, because the larger polynomial constant
shifted residual distributions. These controls show that interval scaling
must be co-trained or followed by BatchNorm adaptation, not patched into a
finished checkpoint. Slurm `316527` resumes the epoch-7 optimizer state with
uniform `q_4`, target interval `[-20,20]`, on eight additional H200 GPUs. Its
first training batches were finite; initial activation maxima were 16.23 to
26.43 and the outside-interval fraction was below `4e-8`. Three adaptation
epochs are scheduled before diagnostic verification.

The scale-1 control did not converge monotonically toward safety. Its
CFP-FP/AgeDB non-finite counts moved from `31/49` at step `20000` to `40/66`
at step `22500`, then abruptly expanded to `2212/2251` at step `25000`; LFW
simultaneously jumped from zero to 3889 failed rows. Training-batch activation
statistics still looked ordinary. Slurm `316391` was stopped after this
decisive recurrence, with its checkpoints preserved.

Uniform scale 4 recovered from stale BatchNorm statistics after one adaptation
epoch, but its verification failures plateaued:

| checkpoint | LFW | CFP-FP | AgeDB-30 |
|---|---:|---:|---:|
| scale-4 epoch 8 | 0 | 51 | 79 |
| scale-4 epoch 9 | 0 | 42 | 71 |
| scale-4 epoch 10 | 0 | 53 | 74 |

The counts combine both flip passes. A diagnostic replay that enabled the
training clip while keeping all BatchNorm layers in evaluation mode localized
the remaining problem. For a failed CFP-FP row, raw stem and `layer1.0`
inputs were 22.46 and 26.72; clipping those values to the scale-4 interval
kept every later site below 21 and produced a finite embedding. A failed
AgeDB row likewise became completely finite when its `layer1.0` input 23.59
was contained; the next-layer input fell from 31.87 to 17.25. Consequently,
the next branch keeps 24 sites at scale 4 and widens only
`layer1.0.prelu` to scale 6, target interval `[-30,30]`. Inserting even this
single change into a frozen checkpoint invalidates BatchNorm statistics, so
Slurm `316640` co-trains it from the scale-4 epoch-9 optimizer state.

Final finite counts, TAR values, and accepted checkpoint will be added after
these jobs complete.
