# ArcFace Torch Experiments

This repository contains PyTorch training and evaluation code for face recognition experiments based on ArcFace-style backbones and margin losses. It includes ResNet, MobileFaceNet, PoolFormer, Patch-CNN, and custom non-linearity variants.

## Repository Layout

- `backbones/`: model architecture implementations, including ResNet, MobileFaceNet, PoolFormer, Patch-CNN, and custom variants.
- `configs/`: experiment configuration files. Each config defines the dataset path, model architecture, optimizer settings, loss settings, validation targets, and output directory.
- `docs/`: upstream documentation for installation, dataset preparation, evaluation, model zoo notes, and hyperparameter references.
- `work_dirs/`: default location for training outputs, checkpoints, TensorBoard logs, and evaluation results.
- `train_v2.py`: distributed training entry point.
- `eval_ijbc.py`: IJBB/IJBC evaluation entry point.
- `README_arcface_author.md`: original ArcFace notes, including dataset download links.

## Experiment Naming

Use the following naming convention for configs and output folders:

```text
dataset_modelarchitecture_setting
```

Examples:

- `casia_r50`: train an R50 model on CASIA/WebFace.
- `ms1mv3_poolformer_s24`: train a PoolFormer-S24 model on MS1MV3.
- `ms1mv3_poolformer_s24_no_ln`: train PoolFormer-S24 on MS1MV3 with layer normalization replaced by another operation.
- `ms1mv3_poolformer_s24_no_ln_x2_act`: train the MLP-ratio-2 PoolFormer-S24
  with progressive RepBatchNorm and NAFNet-style SimpleGate. Despite the
  historical config name, the gate is `x1 * x2`, not a scalar `x**2`.
- `ms1mv3_r50_no_relu`: train an R50 variant with ReLU/PReLU removed or replaced.

Keep names short but specific enough to identify the dataset, backbone, and main experimental change.

## Data and Outputs

Download the required face datasets before training. Dataset download links are available in `README_arcface_author.md`.

Training configs expect MXNet RecordIO-style face datasets. For example, `configs/casia_r50.py` currently points to:

```text
./faces_webface_112x112/
```

That directory should contain the training record files and validation targets used by the config, such as `lfw`, `cfp_fp`, and `agedb_30`.

Training outputs are written to the config output directory. Typical files include:

- `model.pt`: exported backbone weights for evaluation.
- `checkpoint_gpu_*.pt`: resumable distributed training checkpoints.
- `tensorboard/`: TensorBoard logs.
- evaluation folders such as `ijbc_result/`.

## Training

Launch training with `torchrun` and pass the config path without the `.py` suffix:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun \
  --master_port=29500 \
  --nproc_per_node=2 \
  train_v2.py configs/casia_r50
```

Important training settings are controlled by the config file:

- `network`: backbone name passed to `backbones.get_model`.
- `rec`: training dataset directory.
- `num_classes` and `num_image`: dataset statistics.
- `batch_size`: per-GPU batch size.
- `num_epoch` and `warmup_epoch`: training schedule.
- `loss`: margin loss type, such as `adaface`.
- `fp16`: enable mixed precision training.
- `output`: output directory. If `None`, the config loader may derive it from the config name.

To resume training, set `config.resume = True` in the config and make sure the corresponding `checkpoint_gpu_*.pt` files exist in the output directory.

For the SimpleGate/RepBatchNorm experiment, epochs 0-8 use GELU while the
normalization transition finishes. BatchNorm is then recalibrated and verified
before six contiguous block groups progressively blend from GELU into
SimpleGate during epochs 8-20. The second projection half is initialized from
the local GELU expansion and warmed by sampled distillation and range losses.
Epochs 20-25 use only SimpleGate. At the end of training, BatchNorm statistics
are reset and recalibrated through the final inference graph, a layer-wise gate
profile is written to `simple_gate_final_profile.json`, and the recalibrated
model is verified and saved. Per-layer operand, product, gradient, correlation,
range, blend, teacher-error, and residual-scale measurements are available
under `SimpleGate/` in TensorBoard.

To determine whether a progressive SimpleGate checkpoint fails only in FP16,
run its saved epoch model through the standard verification sets in full FP32:

```bash
CUDA_VISIBLE_DEVICES=0 python eval_poolformer_checkpoint_fp32.py \
  --checkpoint work_dirs/ms1mv3_poolformer_s24_no_ln_x2_act/model_epoch_10.pt \
  --epoch 10 \
  --batch-size 32
```

For `model_epoch_XX.pt`, `--epoch` can be omitted because it is inferred from
the filename. The script reconstructs the gate blends for that epoch and exits
with a failure if any requested validation embedding is non-finite.

## IJBB/IJBC Evaluation

After training, evaluate a saved model with `eval_ijbc.py`:

```bash
CUDA_VISIBLE_DEVICES=1 python eval_ijbc.py \
  --model-prefix work_dirs/casia_r50/model.pt \
  --image-path ijb/IJBC \
  --result-dir work_dirs/casia_r50/ijbc_result \
  --batch-size 256 \
  --job casia_r50 \
  --target IJBC \
  --network r50
```

Argument notes:

- `--model-prefix`: path to the exported `model.pt`.
- `--image-path`: root directory of the IJBB or IJBC dataset.
- `--result-dir`: directory for extracted features, scores, and plots.
- `--target`: use `IJBB` or `IJBC`.
- `--network`: backbone name. This must match the architecture used during training.
- `--job`: label used in result filenames and logs.

## TinyFace Evaluation

The uploaded TinyFace tree can be evaluated with `eval_tinyface.py`. The script reads
`tinyface/Testing_Set/*_img_ID_pairs.mat`, extracts features for `Gallery_Match`,
`Probe`, and `Gallery_Distractor`, writes MATLAB-compatible feature files, and reports
TinyFace mAP/CMC metrics in Python.

```bash
CUDA_VISIBLE_DEVICES=0 python eval_tinyface.py \
  --model-prefix work_dirs/casia_r50/model.pt \
  --data-dir tinyface \
  --result-dir work_dirs/casia_r50/tinyface_result \
  --batch-size 256 \
  --job casia_r50 \
  --network r50
```

Outputs are written under `work_dirs/.../tinyface_result/<job>/`, including
`features/gallery.mat`, `features/probe.mat`, `features/distractor.mat`,
`tinyface_metrics.json`, `tinyface_ap.npy`, and `tinyface_first_ranks.npy`.

## Useful Documentation

Additional setup and data preparation notes are available in:

- `docs/install.md`
- `docs/prepare_custom_dataset.md`
- `docs/prepare_webface42m.md`
- `docs/eval.md`
- `docs/hyperparameters_and_backbone.md`
- `docs/modelzoo.md`
