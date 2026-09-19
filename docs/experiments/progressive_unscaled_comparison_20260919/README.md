# Original progressive/best versus MS1MV3-scaled BTS-gated evaluation

The original checkpoint is
`work_dirs/controlled_degree2_tail_ms1mv3_20260907/progressive/student_best.pt`,
SHA-256 `3b3298217fcc54a368b045f50d85e8e0d9c845f305e7f1212640838c160f6b39`.
This is the same unchanged source used by the
[MS1MV3-scaled experiment](../progressive_best_bts_ms1k_20260917/README.md).
Older results for polish/final or `work_dirs/student_best.pt` are different
checkpoints and are not used for this comparison.

The unscaled model is evaluated on the full IJB-C metadata using the same
FP32/TF32-disabled inference, alignment, original/flip processing, batch size
256, eight image readers, faceness weighting, template aggregation, and all
verification pairs as the scaled run. There is **no BTS range threshold** for
the unscaled run: finite values outside [-1,1] pass. Nonfinite boundary or
embedding outputs fail; either view failing zeros both entire embeddings.
This matches the scaled run's nonfinite policy. The scaled run additionally
zeros source images with an out-of-range BTS boundary.

Therefore the measured difference compares **original + nonfinite-only
zeroing** with **equivalent scaling + BTS-range/nonfinite zeroing**. It is not
an isolated estimate of the numerical effect of scaling alone. Frozen-BN
mathematical equivalence does not remove the extra failure-filtering effect.
No checkpoint training, calibration, or selection was performed on IJB-C.

## Results

Full evaluation job 403247 completed with exit 0 in 9m25s. Coverage is
469,375 source images, 938,750 original/flip views, and 15,658,489 verification
pairs. All pair scores are finite. Both runs have the **same 14 nonfinite
source-image IDs**. The scaled run additionally zeros 7 finite source images
for BTS-range failures (21 failed sources versus 14 in the baseline).

At FAR=1e-4, both strict TAR values are **95.13729099555147%**; the difference
is **0 percentage points**. No decrease appears at any of the six reported
FAR targets. These results do not isolate parameter-scaling roundoff from the
additional seven-image failure filter.

Strict TAR uses the best empirical ROC point with actual FAR not exceeding
the requested FAR. Delta is scaled/gated minus original/nonfinite-only,
in **percentage points**, not relative percent.

| Requested FAR | Original TAR (%) | Scaled + BTS failure filter TAR (%) | Delta (pp) |
| --- | ---: | ---: | ---: |
| 1e-06 | 83.54553 | 83.55576 | +0.01023 |
| 1e-05 | 92.01820 | 92.01820 | +0.00000 |
| 0.0001 | 95.13729 | 95.13729 | +0.00000 |
| 0.001 | 96.86046 | 96.86046 | +0.00000 |
| 0.01 | 98.14900 | 98.14900 | +0.00000 |
| 0.1 | 98.96201 | 98.96201 | +0.00000 |

For comparison with older CSVs, the existing evaluator's nearest-FAR method
produces the following values. Its selected actual FAR can exceed the
requested target; this is why the strict table above is preferred.

| Requested FAR | Original nearest-FAR TAR (%) | Scaled nearest-FAR TAR (%) | Delta (pp) |
| --- | ---: | ---: | ---: |
| 1e-06 | 84.67556 | 84.68068 | +0.00511 |
| 1e-05 | 92.01820 | 92.01820 | +0.00000 |
| 0.0001 | 95.14240 | 95.14240 | +0.00000 |
| 0.001 | 96.86046 | 96.86557 | +0.00511 |
| 0.01 | 98.15411 | 98.15411 | +0.00000 |
| 0.1 | 98.96201 | 98.96201 | +0.00000 |

Artifacts: [comparison](comparison.json), [original raw ROC metrics](unscaled_tar_at_far_raw.json),
[original failure summary](unscaled_failure_summary.json), and
[original failure annotations](unscaled_failures.jsonl).
The score arrays and logs remain under the corresponding `work_dirs/` runs.

## Reproduction

From the repository root, create the fresh output directory named in
[run.slurm](run.slurm), then submit that script with Slurm stdout/stderr paths.
The completed unscaled evaluation uses job 403247. Its key option is:

```text
--bts-failure-report PREFIX --ignore-bts-range
```

Without `--ignore-bts-range`, the existing six-boundary range policy remains
unchanged. The option requires a failure report so the filtering policy is
explicit and auditable.

After evaluation, regenerate archived metrics and the difference table:

```bash
python docs/experiments/progressive_unscaled_comparison_20260919/summarize.py
```

The summary verifies full source/view and pair-score coverage, finite scores,
failure-record IDs/names, zeroed-row counts, the original checkpoint hash, and
strict FAR thresholds. It records whether both runs have the same nonfinite
failure IDs, and which source images are zeroed only in the scaled run.

Tests: `python -m pytest tests/test_bts_failure_audit.py tests/test_ordered_prefetch.py -q`
(10 passed). The new test proves that large finite boundary values are accepted
in nonfinite-only mode while nonfinite embeddings still zero both source views.
