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

The first strict rejection exposed one additional resume invariant: a
checkpoint retained the fixed interval but the trainer did not reconstruct
the persisted top-512 replay stream.  The recovery implementation now loads
and validates every calibrated-prefix manifest, checks its interval against
the checkpoint, and merges old-prefix and newly scanned tail indices instead
of replacing the former with the latter.

Relevant local tests cover the containment-top-k loss, configuration safety
invariants, exact zero-tolerance interval comparison, contiguous-prefix guards,
deterministic canonical/flip enumeration, and existing BatchNorm freeze
behavior.  The current combined suite completed with 35 passes and one known
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

If the recovery stem gates pass, the reproducible next run is
`configs/ms1mv3_r50_layerwise_poly_hard_containment_group02.py`, launched by
`scripts/slurm_train_r50_layerwise_poly_hard_containment_group02.sh`.  It
resumes the epoch-4 distributed state, scans `layer1.0.prelu`, and first spends
only half an epoch on tail conditioning.  A cheap full-domain gate then either
permits a half-epoch blend or stops for a longer recovery window.  Epoch 5
performs completion/BN audit and one guarded hold epoch.  This adaptive probe
reduces a successful second-site run from four training epochs to two without
relaxing any immutable-interval check.

### Pending authoritative gates

- recovery pre-blend complete-domain
  `observed_absmax <= 2.43708872795105` (the first attempt was correctly
  rejected at 2.444153);
- fully quadratic stem downstream-boundary scan is finite;
- post-blend/BatchNorm complete-domain containment audit;
- post-hold final complete-domain audit (or an explicitly audited safe group
  checkpoint for this already-running pre-`b4fd609` job);
- final validation accuracy and finite embeddings;
- final checkpoint/hash and reproducible continuation instructions.
