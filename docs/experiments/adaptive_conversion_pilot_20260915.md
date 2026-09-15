# MS1MV3-only adaptive conversion pilot

User request: test a more general route to finite quadratic activations with
retained recognition quality, based on the completed IJB-C campaign.

This paired pilot starts each arm from the original `ms1mv3_r50/model.pt`,
never from the prior polynomial student. It reuses the hash-verified MS1MV3
identity split and teacher identity centers from the earlier preparation.
The fixed arm also reuses the original 100,000-image teacher histogram fit.
The adaptive arm refits each upcoming site on 2,048 training images through
the current student prefix. Target: original channelwise PReLU, interval
`[-R_c,R_c]`, R_c = .9995 histogram quantile upper edge times 1.5, minimum .05;
weighted least squares uses .05 uniform edge weight. Fit statistics and
coefficients are saved. Neither arm reads IJB-C images or labels.

Both arms use the same seeded sampled training sequence and augmentation
policy. The pilot tests the first two activation sites, not a deployable
25-site model. It has no head warmup; both heads start from the same original
teacher identity centers. The comparison changes two coupled features
(current-student fitting and quality-gated timing), so it does not isolate
their individual effects and is not equal-compute if adaptive stalls.

Each site blends through alpha .25/.5/.75/1. The fixed arm takes 100 updates
per phase. The adaptive arm takes 100–300, checking every 25 updates and
requiring two consecutive passes. The fixed 512-image development probe
includes original/flip and every module boundary. Pass requires zero
nonfinite values, cosine KD <= site-entry baseline + .02, and maximum
active-prefix input/radius ratio <= max(4, 1.25 * site-entry baseline).
The probe selects conversion timing, so it is not independent final testing.
These are preregistered pilot thresholds, not empirically established optima.

One H200, FP32, batch64, SGD momentum .9/Nesterov; backbone/head/coefficient
LR .001/.005/.00002; weight decay .0005 except coefficients (zero). BN
moments stay fixed; BN affine and convolution/linear parameters train.
Loss = ArcFace(m=.5,s=64) + embedding cosine KD + .3 relative four-stage MSE
+ activation tail penalty. Tail penalty uses mean squared excess above .9R
and .01 times per-image maximum squared excess. Only activated-prefix sites
participate. Gradient norm cap5. Crop/lowres/photo probabilities .1/.2/.2;
no artificial pathological/stress inputs. Hard-tail replay holds 32 images
and replaces up to 1/16 of each batch. No pair loss is added in this first
mechanism test, to keep the recognition objectives shared between arms.

Both training and probes use the identical unclipped hybrid forward, with
BN/dropout evaluation behavior. A training escape beyond 32R or a nonfinite
loss/gradient/update stops that arm and restores the pre-phase backbone and
alpha. It does not silently sanitize activations or resume a bad optimizer.
A failure or stalled gate is an experimental result, not permission to
extend training automatically. Saved checkpoints are explicitly hybrid,
diagnostic-only, `pure_quadratic=false`, with per-site alphas.

Validation before GPU execution: 21 CPU tests pass (adaptive gate/hybrid/tail
gradient tests and existing channelwise suite); Python compilation, shell
syntax and whitespace checks pass. Run a tiny GPU integration smoke first,
then the bounded paired pilot. Full IJB-C accuracy cannot be inferred from
this pilot; successful prefix conversion would justify a separately recorded
full-network experiment and final independent evaluation.

## GPU integration smoke

Job 387373 completed in 5m43s. Both arms converted the stem through all four
blend values in 8 updates and audited all 128 original/flip probe rows with
zero nonfinite values. Final KD fixed=.01986130, adaptive=.01992518; this is
an integration smoke, not evidence of an accuracy advantage.
[Smoke record](adaptive_conversion_smoke_20260915.json).

Before the longer pilot, deterministic gate images are cached in CPU memory
to avoid restarting RecordIO workers at each check. No image or probe arithmetic
changes. Added an export rejection for incomplete hybrids; 22 CPU tests pass.

## First bounded pilot result and next controlled hypothesis

Job 387411 (recipe 3dd1350) completed in 4m31s. At the same stem-alpha=1,
400-update point, fixed KD=.02002189 and adaptive KD=.02017700; both are
finite. Adaptive does not pass the .02 KD gate, and extending to 600 updates
worsens KD to .02117412. Fixed advances without that gate and finishes two
sites with KD=.02916465. These final depths are different and cannot establish
an adaptive precision advantage. `converted_sites=0` in the original adaptive
report counts gate-accepted sites: its stem is actually alpha=1.
[Recorded comparison](adaptive_conversion_pilot_result_20260915.json).

This does not yet test refitting after a changed prefix: adaptive stalled at
the stem. Waiting longer alone did not restore teacher agreement. A second
controlled pilot will change embedding KD weight from 1 to 5 in **both** arms,
keeping all gates, LRs and seeds fixed, and allow up to eight sites so a
successful stem can test refitting deeper student distributions. Maximum
phase updates stay 300; do not loosen the .02 gate after seeing results. This
is a new loss-weight hypothesis, not a claimed improvement. It remains one
H200 / 45 minutes, MS1MV3-only, without IJB-C or an independent accuracy claim.

## Stronger-KD paired result: no demonstrated adaptive advantage

Job **387435**, recipe **8265083**, completed in **11m58s**. Both arms used
embedding KD weight5. Fixed completed five sites and stopped during site6
alpha=.25, after 2,033 successful updates. Adaptive completed four sites and
stopped during site5 alpha=1, after 1,917 updates; rollback retains alpha=.75
at that site. Both failures are `activation escaped finite conversion guard`.

The guard rejects either a nonfinite input/radius or a finite value >32R.
Its original generic message does not preserve the offending layer, exact
ratio, or input. Thus the evidence does **not** prove a measured Inf or locate
the first causal layer; it establishes that the predefined safety guard
stopped training. Both restored checkpoints' 1,024-row clean/flip probes
remain finite, highlighting the limited coverage of those probes relative
to augmented/replayed training batches. They are hybrid diagnostics, not
full-quadratic deployment candidates.

At matched four-site / 1,600-update endpoints, fixed KD is .03706870 and
adaptive .03855590. The adaptive arm has no demonstrated embedding-quality
advantage. Dynamic fit intervals also mean cross-arm input/radius ratios
are not directly comparable absolute-activation measurements. Stronger KD
helped the stem meet the quality gate in both arms but did not prevent later
safety stops. All four matched prefix comparisons and hashes are recorded in
[the verified result](adaptive_conversion_kd5_result_20260915.json).

Stop this pilot here; do not loosen gates or launch a long full-network run
on this evidence. The experiment tests a partial proposed pipeline: student
refitting, consecutive gates, shared ArcFace/KD/tail objectives and replay.
It does not test automated finite-prefix repair after a safety stop, direct
pair-loss preservation, a matched-compute ablation, multiple seeds, or an
independent final accuracy evaluation. Therefore it cannot reject every
possible adaptive method or establish generalized zero-nonfinite behavior.
A future variant needs failure-localized capture and gate coverage aligned
with the training augmentation distribution before another scale-up.

Total GPU use for smoke + two paired pilots: one H200, 22m12s summed job
runtime. Existing 96.06279% IJB-C calibration model is unchanged. No further
GPU submission follows these results. Code validation: 22 CPU tests, Python
compilation, shell syntax, whitespace and result JSON/hash checks pass.
