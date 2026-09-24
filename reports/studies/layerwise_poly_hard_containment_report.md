# Hard-containment layerwise polynomial training

## Objective

Rebuild the iResNet50 PReLU-to-polynomial conversion so that a replacement is
never enabled unless a deterministic scan of the complete MS1Mv3 training
domain observes every input to the pending activation inside its immutable
polynomial approximation interval.  The encrypted path remains polynomial:
the safety protocol adds no clamp, comparison, division, or data-dependent
branch to inference.

The empirical domain is precisely the 5,179,510 aligned training records in
both canonical and horizontally flipped orientations (10,359,020 inputs).
Those are all transforms used by the natural-data training configuration.  A
passing scan is evidence for this finite domain, not a mathematical bound on
arbitrary images.

## Why the epoch-10 model cannot be the hard-containment starting state

The accepted epoch-10 legacy HerPN model replaces the stem, Layer1, and
Layer2 PReLUs (eight sites) and reaches 95.18% IJB-C TAR at FAR approximately
`1e-4`.  Its accepted ninth-site continuation reaches 94.69755%.  However,
the legacy eight-quadratic prefix is already known to produce roughly `1e24`
tails and occasional non-finite embeddings on natural MS1Mv3 training
batches.  It therefore cannot satisfy the new full-training-domain invariant.

The hard-containment experiment consequently starts from the original PReLU
R50 and rebuilds a safe polynomial prefix one singleton at a time.  The old
eight/nine-site models remain accuracy and failure-mode references; finite
IJB-C evaluation must not be confused with full-MS1Mv3 containment.

## Protocol

For each channel-wise PReLU target on `[-S, S]`, the degree-two student is

```text
linear*x + even*x^2/S + beta2*(1 - (x/S)^2)
```

and folds to `c0 + c1*x + c2*x^2` for inference.  At `beta2=0` it initializes
to the unique quadratic through PReLU at `-S`, `0`, and `S`.  The trainable
endpoint-vanishing `beta2` basis preserves the two endpoints while allowing
the interior, including `x=0`, to move under local PReLU distillation.  It
costs one sequential ciphertext square and has no non-polynomial inference
safeguard.

The training protocol is:

1. Scan every training record in both orientations with the current eval
   graph.  Record the global maximum, its source/orientation/coordinate, and
   the globally worst 512 source records.
2. Set `S` once from `2 * q99.95`; do not enlarge it to hide a rare tail.
3. If any observed input exceeds `S`, keep the pending activation at PReLU,
   replay the rare-tail records in every local-fit batch, and optimize a
   normalized per-sample top-tail loss together with ArcFace, local PReLU
   distillation, and a frozen baseline embedding teacher.
4. Immediately before blend, scan the complete domain again with the same
   immutable `S`.  Any `observed_absmax/S > 1` rejects the replacement.
5. Temporarily enable the complete polynomial singleton and scan its immediate
   downstream boundary for non-finite or catastrophic values before the first
   blend update.
6. After blend completion, refresh downstream BatchNorm statistics, scan the
   complete accepted prefix against every immutable interval, and only then
   save the completed-group checkpoint.
7. Continue replaying measured tails and applying the strong prefix
   containment loss during blend and final hold.  After the last optimizer
   update, repeat the complete-domain prefix audit, run finite validation, and
   save a separately named final-audited checkpoint.  A safe intermediate
   boundary therefore cannot silently drift into an unsafe final `model.pt`.

Every replacement is a singleton in forward order.  Later conditioning also
audits the complete already-accepted prefix, preventing upstream updates from
invalidating an earlier interval.

## Implementation and local verification

The implementation is committed in `81e5f0c`; CPU/GPU scan utilization and
progress logging are corrected in `24a8db0`.  Commit `0440f79` closes the
chunked-resume gap: expanding an accepted conversion frontier now performs a
provisional full-domain scan of the newly exposed singleton before local-fit,
while an ordinary mid-group resume does not prematurely calibrate later
groups.  Commit `4eee87f` removes the former relative `1e-6` containment
tolerance, so one representable value above `S` is rejected.  Commit
`b4fd609` keeps tail guards active after local-fit and requires a final
post-update audit plus finite validation before writing
`model_herpn_final_containment_audited.pt`.  The Nano4 launcher reserves eight
V100 GPUs, 64 CPU cores, and 500 GB RAM.  Eight loader workers per rank use the
complete CPU allocation.

Commit `c27df8b` adds the non-diluted linear batch-L-infinity containment loss
and deterministic priority replay of the worst manifest sources.  Commit
`44af8cf` carries the same hard-max, eval-BN conditioning policy into the
second singleton configuration, so a continuation cannot silently fall back
to the earlier squared top-k objective.

The first strict rejection exposed one additional resume invariant: a
checkpoint retained the fixed interval but the trainer did not reconstruct
the persisted top-512 replay stream.  The recovery implementation now loads
and validates every calibrated-prefix manifest, checks its interval against
the checkpoint, and merges old-prefix and newly scanned tail indices instead
of replacing the former with the latter.

Relevant local tests cover the containment-top-k loss, configuration safety
invariants, exact zero-tolerance interval comparison, contiguous-prefix guards,
deterministic canonical/flip enumeration, and existing BatchNorm freeze
behavior.  The current combined suite completed with 39 passes and one known
unrelated degree-eight initialization test deselected.

## Nano4 stem proof

Job `322025` uses
`configs/ms1mv3_r50_layerwise_poly_hard_containment_stem.py`.  The initial
10,359,020-input scan completed in 3 minutes 15 seconds after increasing the
loader from two to eight workers per rank.  It measured:

| quantity | value |
|---|---:|
| robust `q99.95` absmax | 1.218544 |
| immutable interval radius `S` | 2.43708872795105 |
| complete-domain observed absmax | 2.734803 |
| initial containment ratio | 1.122160 |
| worst source index | 2,471,019 |
| worst orientation | canonical |
| replay source count | 512 |

The initial state is deliberately provisional: it exceeds the fixed interval
by 12.2%, so no polynomial blend is permitted yet.  Through step 9000, the
model retained 99.833% LFW, CFP-FP rose from 98.843% to 98.914%, and AgeDB-30
rose from 98.150% to 98.267%.  All validation embeddings remained finite.  The
epoch-1 snapshot provides a direct replay check on the original globally
worst source 2,471,019: its canonical stem maximum fell from 2.734803 to
2.454535 (a 10.25% reduction).  This is still 1.007159 times the immutable
radius, so one conditioning epoch is demonstrably insufficient even though it
has removed most of the original rail tail.  The flipped orientation moved
similarly, from 2.734456 to 2.454234.
An explicit epoch-1 replay-set audit of all 512 recorded sources in both
orientations found only those two orientations of source 2,471,019 still
outside `S` (2/1,024); the next largest value was 2.436505, already inside by
0.000584.  This confirms that conditioning has not merely moved the maximum to
another recorded tail source, while still leaving the complete-domain scan as
the authoritative check for unrecorded sources.

Across the four successive 1000-step windows, the largest logged
training-batch stem inputs
were 2.494719, 2.469125, 2.491171, and 2.482128.  The last occurred at step
4450, where only 3.89e-8 of tensor elements were outside the interval and the
normalized range penalty was 2.13e-5.  Step 5000 itself had maximum 2.284883
and zero range penalty.  These sampled batches show that ordinary batches are
well controlled, but their remaining rare excursions also demonstrate why
they cannot substitute for the mandatory complete-domain scan.

The authoritative epoch-2 strict scan rejected the stem before any blend:

| quantity | value |
|---|---:|
| immutable interval radius `S` | 2.43708872795105 |
| strict observed absmax | 2.444153 |
| strict containment ratio | 1.002899 |
| strict holdout absmax | 2.425858 |
| worst source/orientation | 2,471,019 / canonical |

Thus two full conditioning epochs reduced the original excess from 12.216%
to 0.2899% while preserving finite validation (step 10,000: LFW 99.833%,
CFP-FP 98.914%, AgeDB-30 98.200%), but did not meet the exact invariant.  The
failure is a useful strict-gate result: the activation remained PReLU and `S`
was not widened.  Recovery resumes `model_epoch_02.pt` plus the distributed
epoch-2 state, restores the recorded extrema, uses only a half-epoch extension,
and reruns the full gate at epoch 2.5.  Its reproducible entry points are
`configs/ms1mv3_r50_layerwise_poly_hard_containment_stem_recovery.py` and
`scripts/slurm_train_r50_layerwise_poly_hard_containment_stem_recovery.sh`.

That first half-epoch recovery was also rejected.  Despite restoring the
correct top-512 manifest, its eval-mode maximum regressed to 2.461008
(`ratio=1.009815`; holdout maximum 2.442533), again at source 2,471,019.  This
isolates a train/eval mismatch rather than an unidentified source: range loss
was computed with train-mode BatchNorm batch moments, while every strict scan
uses inference running statistics.  More epochs under that mismatched graph
are not justified.  The next recovery therefore resumes the still-unchanged
epoch-2 PReLU checkpoint, freezes BatchNorm running statistics while leaving
affine parameters trainable, and conditions against the exact normalization
graph used by inference.  It retains the immutable interval and the same
half-epoch gate.  Reproducible entry points are
`configs/ms1mv3_r50_layerwise_poly_hard_containment_stem_evalbn.py` and
`scripts/slurm_train_r50_layerwise_poly_hard_containment_stem_evalbn.sh`.

The eval-BN recovery removed that graph mismatch but was likewise rejected:
its exact maximum reached 2.481986 (`ratio=1.018422`; holdout 2.463453), with
the same source and coordinate.  Telemetry explains why: the normalized
squared top-k objective selected 16 samples per rank, so a lone 1.84% rail
pixel contributed only about 2e-5 after the sample mean, and the squared hinge
lost gradient near the boundary.  The next recovery replaces this training-
only objective with a linear batch `L-infinity` hinge and repeats each of the
eight worst ordered manifest sources 64 times in the replay population.  It
still trains on the complete ordinary data stream, retains ArcFace/teacher
losses, freezes inference BN statistics, keeps the original `S`, and cannot
blend before the same exact full-domain gate.  Its entry points are
`configs/ms1mv3_r50_layerwise_poly_hard_containment_stem_hardmax.py` and
`scripts/slurm_train_r50_layerwise_poly_hard_containment_stem_hardmax.sh`.

Hard-max recovery job `322933` passed the exact pre-blend gate without changing
the immutable interval.  Its complete-domain maximum was `2.429538`
(`ratio=0.996900`), leaving 0.007551 absolute headroom below
`S=2.43708872795105`.  The worst input remained source 2,471,019 in canonical
orientation, so the improvement is directly attributable to conditioning the
known rail rather than a change in which sample happened to be largest.  With
the stem temporarily fully quadratic, the complete-domain scan of its
immediate `layer1.0.prelu` boundary was finite with maximum `6.723052`; the
half-epoch blend was therefore allowed to start.

After blend completion and 1,000-batch downstream BatchNorm recalibration, the
mandatory prefix audit again covered all 10,359,020 inputs.  The stem maximum
was `2.434547` (`ratio=0.998960`) with no violation, and the audited group-1
checkpoint was saved as `model_herpn_group_01_bnrecalibrated.pt`.  Validation
remained finite during the blend; at step 14,000 it reached 99.783% LFW,
98.914% CFP-FP, and 98.033% AgeDB-30.  The job is now retaining the fully
quadratic stem for one guarded hold epoch before the required final
post-update audit.

That final post-update audit also passed: its complete-domain maximum fell to
`2.412438` (`ratio=0.989890`) under the same immutable interval.  The hold
remained finite through step 20,000, with 99.783% LFW, 98.943% CFP-FP, and
98.133% AgeDB-30.  Job `322933` then exposed a bookkeeping-only defect after
the successful scan: the final checkpoint metadata referenced the configured
orientation flag through an uninitialized local name.  The epoch-4
distributed state and audit evidence are intact.  The flag is now initialized
from the config before use; an audit-only resume repeats the final scan and
validation before naming the final checkpoint.  Audit-only job `323381`
completed successfully and reproduced the exact `2.412438` maximum.  Its
final validation was finite at 99.783% LFW, 98.886% CFP-FP, and 98.133%
AgeDB-30.  It wrote
`model_herpn_final_containment_audited.pt` at global step 20,232 with
`scan_both_orientations=True`; the SHA-256 digest is
`dc848c11b6f3b01da7d9d883a7bb98a9391d8dc2a405d8f8a183fa9a6f8c16d5`.

With the stem accepted, the reproducible next run is
`configs/ms1mv3_r50_layerwise_poly_hard_containment_group02.py`, launched by
`scripts/slurm_train_r50_layerwise_poly_hard_containment_group02.sh`.  It
resumes the epoch-4 distributed state, scans `layer1.0.prelu`, and first spends
only half an epoch on tail conditioning.  A cheap full-domain gate then either
permits a half-epoch blend or stops for a longer recovery window.  Epoch 5
performs completion/BN audit and one guarded hold epoch.  This adaptive probe
reduces a successful second-site run from four training epochs to two without
relaxing any immutable-interval check.

Group-2 probe job `323409` measured a much heavier initial rail at
`layer1.0.prelu`: robust `q99.95=2.492338`, immutable `S=4.984676`, and
complete-domain maximum `8.882543` (`ratio=1.781970`) at flipped source
1,443,713.  Its half-epoch hard-max probe nevertheless reduced the strict
maximum to `4.994925` (`ratio=1.002056`).  The exact gate rejected the
remaining 0.2056% excess and never enabled the polynomial.  The recovery
therefore restarts from the audited epoch-4 stem state, keeps the same interval
and tail manifest, conditions for one full epoch, blends for one full epoch,
and holds for one epoch.  Its entry points are
`configs/ms1mv3_r50_layerwise_poly_hard_containment_group02_recovery.py` and
`scripts/slurm_train_r50_layerwise_poly_hard_containment_group02_recovery.sh`.

One full conditioning epoch in recovery job `323542` did not solve that last
rail: the strict maximum was `4.997492` (`ratio=1.002571`) and migrated again,
this time to flipped source 3,005,115.  The two post-conditioning maxima are
not among the initial top-512 manifest, and inspection found a replay-ordering
bug: persistent stem indices preceded the pending group's indices, so the
eight priority-repeat slots continued to favor stem rails.  Later singleton
results now take priority over the older prefix while all prefix indices remain
in the replay population.  Sources 3,005,115 and 4,498,665 are explicitly
seeded from the two authoritative failed scans.

The second recovery also adds a training-only guard band: the linear
L-infinity hinge begins at `0.98*S` while the immutable approximation and
acceptance interval remains exactly `S=4.984676`.  Values between `0.98*S`
and `S` therefore continue receiving an inward gradient instead of handing
the maximum to the next unconditioned source as soon as they cross the rail.
This changes no inference operation or polynomial coefficient.  Reproducible
entry points are
`configs/ms1mv3_r50_layerwise_poly_hard_containment_group02_guardband.py` and
`scripts/slurm_train_r50_layerwise_poly_hard_containment_group02_guardband.sh`.
The prior recovery preserved its conditioned PReLU state in the epoch-5
distributed checkpoint before the gate, so this run moves the blend boundary
to epoch 6: epoch 5 receives the new guard-band objective, epoch 6 blends, and
epoch 7 holds before the final audit.

Guard-band recovery job `323754` completed all authoritative gates with exit
code zero.  Its epoch-5 complete-domain pre-blend scan covered all 10,359,020
canonical/flip inputs without widening either interval.  The accepted-prefix
results were:

| activation | immutable `S` | observed absmax | ratio | result |
|---|---:|---:|---:|---|
| `prelu` | 2.437089 | 2.342288 | 0.96110 | no violation |
| `layer1.0.prelu` | 4.984676 | 4.884321 | 0.97987 | no violation |

The second singleton's maximum moved to flipped source 1,791,077, but unlike
the preceding failures it retained approximately 2% guard-band headroom.
With `layer1.0.prelu` temporarily set to its fully quadratic path, the
complete-domain causal scan of downstream boundary `layer1.1.prelu` was also
finite and passed with maximum `3.739572`.  The second one-epoch blend was
therefore allowed to proceed.

After the second activation became fully quadratic and BatchNorm was
recalibrated for 1,000 batches, the mandatory accepted-prefix scan again
passed: the stem maximum was `2.345515` (`ratio=0.96243`) and the second-site
maximum was `4.889443` (`ratio=0.98089`).  The resulting intermediate
checkpoint is `model_herpn_group_02_bnrecalibrated.pt`.  A full guarded hold
epoch then kept both polynomial activations active while continuing the range
objective and prioritized rare-tail replay.

The final post-update complete-domain audit passed with more headroom than the
post-BN audit:

| activation | immutable `S` | final absmax | ratio | result |
|---|---:|---:|---:|---|
| `prelu` | 2.437089 | 2.303439 | 0.94516 | no violation |
| `layer1.0.prelu` | 4.984676 | 4.852946 | 0.97357 | no violation |

Final verification at global step 40,464 was finite: 99.783% LFW, 98.757%
CFP-FP, and 97.917% AgeDB-30.  Relative to the audited stem-only checkpoint,
the changes are 0.000, -0.129, and -0.216 percentage points respectively.
Thus the second replacement is usable and does not exhibit the former
non-finite-inference failure, although AgeDB-30 shows the largest measurable
accuracy cost.  Job `323754` completed in 2:13:55 and wrote the two-group
`model_herpn_final_containment_audited.pt` with `herpn_group=2`, global step
40,464, and `scan_both_orientations=True`.  Its SHA-256 digest is
`7028194cfb665924cb8a46b89c1798fa2edb2d86352b4f954b66643882a7b6d1`.

### Completed authoritative gates

- second singleton (`layer1.0.prelu`) immutable-interval pre-blend scan;
- fully-quadratic downstream-boundary scan;
- post-BN and final accepted-prefix complete-domain audits;
- final finite verification and reproducible two-group checkpoint/hash.
