# Polynomial activation stability workflow

This directory evaluates a trained face-recognition backbone after activation
replacement **without training or optimizer updates**. It supports the
`ms1mv3_r50` (PReLU) and `ms1mv3_poolformer_s24` (GELU) baselines.

The default is HerPN: a degree-2 Gaussian-Hermite approximation to PReLU on
`[-6, 6]`. For PoolFormer GELU it uses the ReLU-limit Hermite proxy and labels
that fact in the report. The polynomial is not clamped. One quadratic activation
adds one sequential ciphertext-ciphertext multiplication; the report does not
claim that PoolFormer's normalization operations are FHE-compatible.

## Workflow

1. Load the trained baseline checkpoint strictly.
2. Deep-copy it and replace PReLU/ReLU/GELU modules.
3. Run both models on a representative calibration subset.
4. Record every polynomial input's range, moments, interval violations, NaN/Inf,
   embedding MSE, and cosine similarity against the baseline.
5. Optionally backpropagate an embedding-energy proxy once per batch, without
   stepping an optimizer, to find non-finite parameter gradients.
6. Save the complete JSON report. A sampled pass is evidence, not a mathematical
   proof that future training will remain stable.

Use real, normalized face images whenever possible:

```bash
python -m stability_analysis.run \
  --model ms1mv3_r50 \
  --checkpoint work_dirs/ms1mv3_r50/model.pt \
  --dataset ms1m-retinaface-t1 \
  --device cuda:0 \
  --output r50_herpn_stability.json
```

The dataset may be an InsightFace `train.rec`/`train.idx` directory, a standard
class-subdirectory image folder, or `synthetic` for a wiring-only smoke test.
Do not use synthetic results to select an approximation interval.

For PoolFormer:

```bash
python -m stability_analysis.run \
  --model ms1mv3_poolformer_s24 \
  --checkpoint work_dirs/ms1mv3_poolformer_s24/model.pt \
  --dataset ms1m-retinaface-t1 \
  --no-backward \
  --output poolformer_s24_herpn_stability.json
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
the calibration set. Full training belongs on the GPU server, not this checkout.
