# Activation stability analysis

This directory measures how polynomial activation replacement changes a trained
R50 or PoolFormer backbone. The primary study replaces **one activation at a
time**, then replaces all activations together to expose compounding. It does not
perform optimizer updates.

For every activation, the study records:

- baseline input/output moments, quantiles, range violations, and NaN/Inf;
- local polynomial value error and derivative error;
- embedding MSE, cosine drift, norm drift, and pairwise-similarity drift;
- the distribution shift at the next activation;
- uniform-L2 quadratic fits over several approximation intervals.

The R50 HerPN replacement preserves every channel's trained PReLU slope. HerPN
is a degree-2 Gaussian-weighted approximation. Its configured interval is a
monitoring range, not a clamp and not a uniform minimax guarantee. One quadratic
adds one sequential ciphertext-ciphertext multiplication. PoolFormer remains an
analysis target only: its normalization and other operations are not thereby
made FHE-compatible.

## Layerwise study

Use real, normalized face images whenever possible:

```bash
python -m stability_analysis.study \
  --model ms1mv3_r50 \
  --checkpoint work_dirs/ms1mv3_r50/model.pt \
  --dataset ms1m-retinaface-t1 \
  --device cuda:0 \
  --batch-size 32 \
  --max-batches 160 \
  --max-samples 5120 \
  --interval-sweep 0.25 0.5 1 2 4 6 \
  --output-prefix stability_analysis/results/r50_herpn_real
```

This writes JSON with all measurements, a flat per-layer CSV, and a Markdown
report with rankings and training recommendations. The dataset may be an
InsightFace `train.rec`/`train.idx` directory, a class-subdirectory ImageFolder,
or `synthetic`. Never select an approximation interval or claim recognition
accuracy from synthetic inputs.

For PoolFormer:

```bash
python -m stability_analysis.study \
  --model ms1mv3_poolformer_s24 \
  --checkpoint work_dirs/ms1mv3_poolformer_s24/model.pt \
  --dataset ms1m-retinaface-t1 \
  --device cuda:0 \
  --output-prefix stability_analysis/results/poolformer_s24_herpn_real
```

Use `--replacement uniform-quadratic` to make the actual isolated replacements
use a uniform-L2 interval fit. The interval sweep is always reported separately
so it remains comparable across replacement families.

## Interpreting interval results

A narrow interval usually lowers error near the high-density center but raises
the quadratic coefficient. Inputs outside that interval can then produce large
values and derivatives. A wider interval lowers tail growth but increases error
where a tightly concentrated layer spends most of its time. Select intervals
per layer or stage from training calibration quantiles, then validate outliers on
a disjoint held-out set. Do not clamp in encrypted inference.

## Training sequence

1. Profile a frozen baseline on representative real faces.
2. Regress each polynomial on that layer's observed inputs and preserve PReLU's
   channel-wise slope where applicable.
3. Locally distill every student activation while the backbone is frozen.
4. Convert low-impact layers in small groups using a teacher-activation loss.
5. Refresh BatchNorm statistics after each group; use a lower backbone learning
   rate, gradient clipping, and training-only range/tail penalties.
6. Evaluate LFW/CFP-FP/AgeDB, pairwise-score drift, activation ranges, and
   non-finite gradients after each group and after final polynomial folding.

The generated ranking is data-dependent. Recompute it with real faces before
changing the conversion schedule in `configs/ms1mv3_r50_prelu_herpn.py`.

## Compact all-at-once check

The older runner replaces all activations and optionally executes a backward
proxy in one pass:

```bash
python -m stability_analysis.run \
  --model ms1mv3_r50 \
  --checkpoint work_dirs/ms1mv3_r50/model.pt \
  --dataset ms1m-retinaface-t1 \
  --device cuda:0 \
  --output r50_all_replaced.json
```

## User-designed activation

Copy `custom_activation_example.py`, state the target and design interval, and
define:

```python
def make_activation(name, original_module):
    return your_torch_module
```

Then pass `--activation-file path/to/my_activation.py`. The module should expose
`interval`, `target`, and `degree` attributes so the JSON remains auditable.
The CLI's `--input-scale` controls the global warning interval; set it to match
the custom design interval, or call the Python API for per-layer policies.

Start with a small number of batches, inspect the first unsafe layer, then expand
the calibration set. Full training belongs on the 4x V100 GPU server, not this
checkout.

## Included smoke result

`results/r50_herpn_synthetic_smoke.md` is a reproducible wiring result from the
trained R50 checkpoint and 16 deterministic synthetic normalized inputs. It
demonstrates the reporting workflow and all-at-once numerical blow-up. Its own
warning and limitations are part of the result and must remain visible.
