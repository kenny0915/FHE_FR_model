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
- `ms1mv3_poolformer_s24_fully_gated_fp32`: train a diagnostic PoolFormer-S24
  from scratch in FP32 with all 24 SimpleGates active, NAFNet channel-wise
  LayerNorm, exact `C -> 2C -> gate -> C` expansion, and zero residual scales.
  This variant retains non-FHE LayerNorm to isolate gate-training stability.
- `ms1mv3_r50_no_relu`: train an R50 variant with ReLU/PReLU removed or replaced.
- `ms1mv3_r50_precise_relu_s8`: fine-tune the pretrained R50 with every
  PReLU's ReLU component replaced by `PreciseReLUAlpha10` on `[-8, 8]`, then
  transition through independently fitted degrees 16, 8, and 4. The
  `ms1mv3_r50_precise_relu_s16` variant uses `[-16, 16]`.
- `ms1mv3_r50_precise_relu_alpha7_s16`: fine-tune through an Alpha10-to-Alpha7
  curriculum on the fixed interval `[-16, 16]`. Scale-24 and scale-32 configs
  provide wider-range comparisons; all finish with the paper's two-component
  Alpha7 polynomial at nonlinear multiplicative depth 7.
- `ms1mv3_r50_herpn_residual_scale`: train IResNet50 from scratch with pure
  degree-2 HerPN activations and one learnable residual scalar per block,
  initialized to `1/sqrt(24)`. This recipe has no PReLU teacher or
  distillation path; its public scales and HerPN normalization fold into the
  FHE inference graph.
- `ms1mv3_iresnet_nf12_fp32`: train a 12-block normalization-free IResNet
  whose bounded quadratic branches start exactly linear. It uses SWS during
  training, one ciphertext product per block, and a frozen R50 embedding
  teacher; all SWS/scalar/head-normalization state folds for deployment.

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

For the progressive PreciseReLU R50 experiment, first ensure the baseline
checkpoint exists at `work_dirs/ms1mv3_r50/model.pt`, then launch one scale:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
  --nproc_per_node=4 \
  train_v2.py configs/ms1mv3_r50_precise_relu_s8
```

All 25 activations use Alpha10 initially. At epochs 2, 6, and 10 the whole
network begins a one-epoch transition to degree 16, 8, and 4 respectively;
BatchNorm statistics are recalibrated after each transition. TensorBoard logs
`PreciseReLU/Input Abs Max` and `PreciseReLU/Outside Range Fraction`. Values
outside the declared interval are especially important because polynomial
error can grow rapidly there. Degree 4 is the final target (nonlinear
multiplicative depth 2); Alpha10's composed algebraic degree is 638 and is only
the accurate starting teacher. Use the `s16` config when the scale-8 range
statistics show meaningful tails outside `[-8, 8]`.

The four-GPU recipe uses 64 samples per GPU for a global batch of 256 and a
linearly scaled learning rate of 0.0025. Convolutions run under FP16 autocast,
while polynomial activations evaluate and differentiate internally in FP32.
Its custom backward saves only the activation input and recomputes the fixed
polynomial derivatives, rather than retaining every full-sized Horner
intermediate. The polynomial forward is exact, but training uses a ReLU
straight-through derivative because Alpha10's exact derivative is unstable at
the approximation boundary. Together with the learned channel slope, this is
the ordinary PReLU derivative. This surrogate is training-only and does not
alter inference or the FHE graph. If a 32 GB GPU still runs out of memory, use
batch size 32 and reduce the learning rate to 0.00125.

For the Alpha7 accuracy experiment, start with scale 16:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
  --nproc_per_node=4 \
  train_v2.py configs/ms1mv3_r50_precise_relu_alpha7_s16
```

Epochs 0-2 use Alpha10, epochs 3-5 blend to Alpha7, and the remaining epochs
fine-tune Alpha7 exclusively. A frozen baseline R50 embedding teacher, fixed
interval range penalty, finite checks, gradient clipping, and post-transition
BatchNorm recalibration protect face-recognition geometry. Compare the `s24`
and `s32` configs only if TensorBoard shows meaningful activation mass outside
`[-16, 16]`; their in-range Alpha7 error bounds increase from 0.125 to 0.1875
and 0.25 respectively.

For the SimpleGate/RepBatchNorm experiment, epochs 0-8 use GELU while the
normalization transition finishes. BatchNorm is then recalibrated and verified
before six contiguous block groups progressively blend from GELU into
SimpleGate during epochs 8-20. The second projection half is initialized from
the local GELU expansion and warmed by sampled distillation and range losses.
After each group reaches blend 1.0, BatchNorm statistics are reset and
recalibrated through that exact inference graph before verification and a
`model_simple_gate_group_XX_bnrecalibrated.pt` snapshot. Only then does the
next group begin. Group snapshots include the exact blend tuple and calibration
metadata, so the evaluation tools reconstruct the saved inference graph
automatically. Epochs 20-25 use only SimpleGate. At the end of training,
BatchNorm statistics are reset and recalibrated through the final inference
graph, a layer-wise gate profile is written to
`simple_gate_final_profile.json`, and the recalibrated model is verified and
saved. Per-layer operand, product, gradient, correlation, range, blend,
teacher-error, and residual-scale measurements are available under
`SimpleGate/` in TensorBoard.

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
