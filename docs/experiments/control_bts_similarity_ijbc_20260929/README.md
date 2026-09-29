# BTS6 cosine versus raw inner product on IJB-C

Checkpoint: `work_dirs/control_bts_layouts_20260925/exports/bts6/student_scaled.pt`.
SHA-256: `ec1eada0fc33a81f75cb9c3bec5fc56296c350948e2b57eecc94cceae2dba4fd`.

Slurm job: `454235`, one H200, code revision `ce4bc9671337aa7e9e6fcf6af32a35fc46479aee`.
Submission and fixed code snapshot paths are recorded in [submission.json](submission.json).

## Protocol

Use `eval_ijbc.py --similarity both --bts-failure-report ...`.
Either original or flipped view failing any checkpoint BTS boundary outside
inclusive [-1,1], or producing nonfinite values, zeros both full embeddings.
Keep all source images and verification pairs. Both scoring methods share the
same feature extraction, flip sum, detector weighting, video-media mean,
and sum over media in each template.

- Cosine: L2-normalize the final template, then take its pairwise inner product.
- Dot: directly take the inner product of the unnormalized final templates.

Image embedding norms are already preserved by the existing evaluator.
Removing final template normalization retains both embedding magnitude and
media-count effects. This experiment measures their combined effect under the
existing IJB-C aggregation protocol; it does not retrain the model.

Primary comparison uses the best empirical TAR at actual FAR <= each requested
FAR. Positive `drop_percentage_points` means cosine TAR minus dot TAR is positive.
Each method gets its own ROC threshold. The legacy CSV also retains the existing
nearest-FAR reporting convention; use `ijbc_similarity_comparison.json` for the
strict paired comparison.

## Outputs

Runtime directory: `work_dirs/control_bts_similarity_ijbc_20260929`.
Scores and reports: `results/bts6_zero_failed/`.
- `ijbc_cosine.npy`, `ijbc_dot.npy`: full pair scores.
- `ijbc_similarity_comparison.json`: strict TAR comparison and protocol metadata.
- `ijbc_tar_at_far_raw.json`, `ijbc_tar_at_far.csv`, `ijbc.pdf`: ROC reports.
- `results/failures.summary.json`, `results/failures.failures.jsonl`: BTS audit.

## Completed result

Job 454235 completed with exit 0 in 10m57s. Evaluated all 469,375 source images,
23,124 templates and 15,658,489 pairs. Five source images were zeroed (ten views),
matching the prior BTS6 evaluation. Cosine TAR exactly matches the prior baseline
at all six reported strict FAR operating points.

| FAR | Cosine TAR (%) | Raw dot TAR (%) | Drop (percentage points) |
| --- | ---: | ---: | ---: |
| 1e-06 | 83.5711 | 1.0022 | 82.5689 |
| 1e-05 | 92.2688 | 3.3236 | 88.9451 |
| 0.0001 | 95.2447 | 11.5662 | 83.6785 |
| 0.001 | 96.8911 | 33.8191 | 63.0720 |
| 0.01 | 98.2155 | 67.3161 | 30.8994 |
| 0.1 | 98.9927 | 93.3681 | 5.6246 |

At FAR 1e-4, omitting final template normalization reduces TAR from 95.2447% to
11.5662%, a loss of 83.6785 percentage points. Raw inner products are highly
sensitive to template magnitude under this aggregation. The run does not isolate
how much of the loss comes from per-image embedding norms versus media counts.

The [paired report](ijbc_similarity_comparison.json) contains unrounded TAR and
actual FAR values; [BTS audit](failures.summary.json) records the failure counts.


## Validation and reproduction

19 lightweight tests passed across `tests/test_ijb_similarity.py` and
`tests/test_bts_failure_audit.py`; Python compilation and shell syntax checks passed.
Tests cover actual media/template aggregation, raw versus normalized scores,
zero templates, unchanged default normalization, signed percentage-point changes,
and the existing BTS failure policy.

```bash
export CHECKPOINT="$PWD/work_dirs/control_bts_layouts_20260925/exports/bts6/student_scaled.pt"
export IJBC_ROOT="$PWD/ijb/IJBC"
export SIMILARITY_OUTPUT="$PWD/work_dirs/bts6_similarity_rerun"
sbatch controlled_degree2/control_bts_similarity_ijbc.slurm
```
