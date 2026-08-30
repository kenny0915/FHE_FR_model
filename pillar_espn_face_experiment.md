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

- MS1Mv3 all-polynomial training: Slurm `316092`, eight H200 GPUs, 32 epochs.
- Strict full IJB-C evaluation: Slurm `316106`, dependent on successful
  completion of `316092`.

Final finite counts, TAR values, and accepted checkpoint will be added after
these jobs complete.
