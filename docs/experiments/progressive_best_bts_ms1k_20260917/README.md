# MS1MV3-only BTS range calibration and full IJB-C evaluation

Source checkpoint: `work_dirs/controlled_degree2_tail_ms1mv3_20260907/progressive/student_best.pt`.
Scaled checkpoint: `work_dirs/progressive_best_bts_ms1k_20260917/scaled/student_scaled.pt`.
The source checkpoint is unchanged. This experiment supersedes the earlier
IJB-C-calibrated range experiment for the user's requested evaluation.

## Calibration

Scale selection uses exactly 1,000 MS1MV3 training images, sampled without
replacement using Python `random.Random(42).sample`, then sorted for RecordIO
access. Only the original image orientation is used, without random training
augmentation. [The image manifest](calibration_images.json) records both the
zero-based dataset indices and RecordIO indices. No IJB-C images or labels
participate in scale selection, fitting, or checkpoint selection.

The six BTS boundaries remain `layer1.2`, `layer2.3`, `layer3.3`, `layer3.7`,
`layer3.11`, and `layer4.1`. Each stage shares a coordinate scale to preserve
identity shortcuts without new inference nodes. Scales are selected so the
largest calibration absolute value among the selected boundaries in each
stage becomes 0.8 (with no upscaling). The margin is fixed before IJB-C.

| Stage | Scale |
| --- | ---: |
| 1 including stem | 0.18875518698808677 |
| 2 | 0.2592094453179323 |
| 3 | 0.06956599620400201 |
| 4 | 0.40212774779961513 |

The same offline reparameterization as
[the initial equivalence experiment](../progressive_best_bts_scaling_20260917.md)
is used: weights, frozen BN affine parameters, quadratic coefficients, and
polynomial intervals are transformed consistently. There is no training or
runtime activation scaling/clipping. The approximation target remains
per-channel PReLU on `[-s*lam_fit, s*lam_fit]`; degree 2 and its one multiplication
level per activation are unchanged. Equivalence is for eval mode with frozen
BN statistics, not train-mode BatchNorm.

The exported checkpoint was reloaded and checked on all 1,000 calibration
images (CUDA FP32, TF32 disabled). Maximum embedding absolute error was
`3.38554e-5`, maximum relative L2 error `8.69195e-6`, and minimum cosine similarity
`0.9999999999628982`. All six boundaries remained within [-1,1].
[Calibration measurements and hashes](calibration.json) give full precision.

## IJB-C failure policy

Evaluate every image in `ijbc_name_5pts_score.txt`, with both original and
horizontal-flip views, using the existing alignment and evaluation protocol.
A source image fails if **either view** has any of:

- A value strictly below -1 or above 1 at any of the six boundaries. Exactly
  -1 and 1 pass; no numerical tolerance is added to this failure criterion.
- A nonfinite value at any boundary.
- A nonfinite value in the final embedding.

On failure, **both entire view embeddings are assigned zero**, even if only
one view failed. Assignment is used rather than multiplication so NaN/Inf
cannot survive masking. This happens before flip fusion, detector-score
weighting, media averaging, template aggregation, and pair scoring. Source
images, media entries, templates, and verification pairs are not removed.
A template whose contributions are all zero remains a zero vector through
normalization. TAR@FAR therefore includes the failed samples' effect.

`bts.failures.jsonl` contains one record for every failed source image with its
index/name, per-view failure flags, all six per-view min/max values and
nonfinite/range flags, and the embedding-zeroed flag. Nonfinite extrema are
encoded as JSON null, with explicit nonfinite flags. The summary includes
image/view counts and per-boundary failures; reason counts may overlap.
Reported global finite-row extrema omit rows with nonfinite extrema, and do
not certify a finite range for the nonfinite rows.

IJB-C does not change the calibration scales. This is a range-gated plaintext
simulation of BTS failure, not execution of encrypted bootstrapping.

## Full IJB-C results

Job 397290 completed successfully (exit 0, elapsed 10m12s). All **469,375
source images / 938,750 views** were processed, including the 127-image
remainder batch. **21 source images failed (0.00447403%)**:

- 14 had nonfinite final embeddings (and also boundary failures).
- 7 had finite embeddings but out-of-range boundary values.
- All 21 had at least one range failure; counts across reasons are overlapping.
- 24 individual views failed. Both views of each failed source were zeroed:
  **42 zeroed view embeddings**, producing 21 zero source embeddings.

All images and all **15,658,489 verification pairs** remain in the evaluation.
All pair scores were confirmed finite; [validation details](validation.json). See the
[complete failure annotations](bts.failures.jsonl) and [failure summary](bts.summary.json).

| Boundary | Failed source images (range or nonfinite) | Out-of-range views | Nonfinite views |
| --- | ---: | ---: | ---: |
| layer1.2 | 7 | 10 | 0 |
| layer2.3 | 14 | 14 | 0 |
| layer3.3 | 14 | 10 | 4 |
| layer3.7 | 14 | 0 | 14 |
| layer3.11 | 14 | 0 | 14 |
| layer4.1 | 14 | 0 | 14 |

TAR after zeroing failed embeddings:

| Requested FAR | Existing evaluator: nearest ROC FAR, TAR (%) | Strict measured FAR <= requested, TAR (%) |
| --- | ---: | ---: |
| 1e-06 | 84.68068 | 83.55576 |
| 1e-05 | 92.01820 | 92.01820 |
| 0.0001 | 95.14240 | 95.13729 |
| 0.001 | 96.86557 | 96.86046 |
| 0.01 | 98.15411 | 98.14900 |
| 0.1 | 98.96201 | 98.96201 |

The distinction matters particularly at FAR=1e-6: the nearest ROC point has
actual FAR=1.02309e-6, slightly above the request, and TAR=84.68068%; the strict
point has actual FAR=9.59145e-7 and TAR=83.55576%. Both variants are retained
in the [raw metrics](ijbc_tar_at_far_raw.json); the [CSV](ijbc_tar_at_far.csv)
preserves the existing evaluator's nearest-FAR convention. At FAR=1e-4 the
strict TAR is 95.13729% (95.14% rounded).

This result demonstrates the requested failure-and-zero policy; 1,000-image
MS1MV3 calibration does not guarantee every IJB-C boundary is bounded. No
post-evaluation scale adjustment was performed. Existing polynomial numerical
failures are not repaired by coordinate scaling.

Failure IDs/names, all per-boundary and per-view counts, zeroed-row count,
unique 1,000-image calibration manifest, unchanged source checkpoint, and
scaled checkpoint hashes were verified using `verify_results.py` from the
repository root. Full pair-score count and finiteness were checked separately.

## Reproduction

From the repository root, with a fresh run directory:

```bash
sbatch --output=work_dirs/NEW_RUN/slurm-%j.out \
  --error=work_dirs/NEW_RUN/slurm-%j.err \
  --export=ALL,RUN_DIR=work_dirs/NEW_RUN \
  controlled_degree2/job_bts_ms1mv3_ijbc.slurm
```

Create `work_dirs/NEW_RUN` before submission. The script first calibrates on
MS1MV3, then runs full IJB-C. To reuse an already calibrated checkpoint, set
`SCALED_CHECKPOINT` and optionally `EVAL_SUBDIR`; this does not recalibrate.

This run calibrated under Slurm job 397284. Its initial serial-reader evaluation
was cancelled to use bounded parallel decoding/alignment. Full evaluation job
397290 reused the exact same scaled checkpoint under `ijbc_full/`. Image order,
preprocessing, batch size, scales, and failure policy were unchanged; the
failure records in the shared completed prefix were identical. Prefetch has
at most `2*workers` pending tasks, and decode failures abort rather than silently
skip images. Both jobs used one H200 GPU; the final job uses eight image threads
and 256 source images (512 views) per full batch.

Tests cover unit-interval endpoints, flip-only failure, NaN/Inf boundary and
embedding failures, actual `Embedding.forward_db` zeroing before reshape,
full/remainder audit accumulation, old-hook removal, deterministic calibration
sampling, shared stage scales, ordered bounded prefetch, decode error
propagation, residual-graph equivalence, and layer statistics. Test command:

```bash
python -m pytest tests/test_bts_failure_audit.py tests/test_ordered_prefetch.py \
  tests/test_rescale_residual_graph.py tests/test_layer_statistics.py -q
```

Result: 19 passed.
