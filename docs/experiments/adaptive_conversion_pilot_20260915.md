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
