# PyTorch ArcFace Hyperparameters and Backbone Editing

This note is for `recognition/arcface_torch`, the PyTorch ArcFace training code.

## Main Files

| Purpose | File |
| --- | --- |
| Training entrypoint | `train_v2.py` |
| Base defaults shared by all configs | `configs/base.py` |
| Per-experiment configs | `configs/*.py` |
| Config loader and merge logic | `utils/utils_config.py` |
| Backbone registry | `backbones/__init__.py` |
| Default IResNet backbone | `backbones/iresnet.py` |
| Editable backbone copy added for experiments | `backbones/iresnet_custom.py` |
| Example custom config added for experiments | `configs/ms1mv3_custom_r50.py` |

`train_v2.py` requires the config argument to start with `configs/`, for example:

```bash
cd recognition/arcface_torch
torchrun --nproc_per_node=8 train_v2.py configs/ms1mv3_r50
```

For the editable backbone copy:

```bash
cd recognition/arcface_torch
torchrun --nproc_per_node=8 train_v2.py configs/ms1mv3_custom_r50
```

## Where to Modify Hyperparameters

Most training hyperparameters live in `configs/base.py` and are overridden by the selected file in `configs/*.py`. The loader imports `configs.base`, then updates it with the job config.

Change these in the job config first, not in `base.py`, if you want reproducible experiments:

| Hyperparameter | Config key | Effect |
| --- | --- | --- |
| Backbone choice | `config.network` | Passed to `backbones.get_model(...)`. Examples: `r50`, `r100`, `mbf`, `vit_b`, `custom_r50`. |
| Embedding dimension | `config.embedding_size` | Output feature size of the backbone and PartialFC input size. Default is `512`. |
| ArcFace margin | `config.margin_list` | Passed to `CombinedMarginLoss`. Stock ArcFace uses `(1.0, 0.5, 0.0)`. |
| Dataset path | `config.rec` | InsightFace record folder containing training data and validation `.bin` files. |
| Number of identities | `config.num_classes` | Class count for PartialFC. Must match the dataset. |
| Number of training images | `config.num_image` | Used to compute scheduler step counts. Must match the dataset. |
| Epochs | `config.num_epoch` | Total training epochs. |
| Warmup epochs | `config.warmup_epoch` | Used to compute `cfg.warmup_step`. |
| Per-GPU batch size | `config.batch_size` | Effective batch is `batch_size * world_size * gradient_acc`. |
| Gradient accumulation | `config.gradient_acc` | Optimizer steps every N batches. Useful when memory is limited. |
| Optimizer | `config.optimizer` | `sgd` or `adamw`. |
| Learning rate | `config.lr` | Optimizer learning rate. Scale carefully when changing global batch size. |
| Momentum | `config.momentum` | Intended SGD momentum value. Note: current `train_v2.py` hardcodes `momentum=0.9` in the SGD optimizer. |
| Weight decay | `config.weight_decay` | Optimizer weight decay. |
| Mixed precision | `config.fp16` | Enables autocast in the backbone and gradient scaling in training. |
| PartialFC sample rate | `config.sample_rate` | Fraction of negative class centers sampled. `1.0` uses all classes. |
| Inter-class filtering | `config.interclass_filtering_threshold` | Passed to `CombinedMarginLoss` for noisy large-scale datasets. |
| Validation targets | `config.val_targets` | Validation sets read from `config.rec`, such as `lfw`, `cfp_fp`, `agedb_30`. |
| Validation interval | `config.verbose` | Runs verification every this many global steps. |
| Logging interval | `config.frequent` | Training log print interval. Default comes from `base.py`. |
| DALI loader | `config.dali`, `config.dali_aug` | Optional NVIDIA DALI input pipeline and augmentation. |
| DataLoader workers | `config.num_workers` | Worker count for PyTorch DataLoader when DALI is off. |
| Resume training | `config.resume` | Loads checkpoint files from `config.output`. |
| Save full state | `config.save_all_states` | Saves optimizer, scheduler, backbone, and PartialFC checkpoint files. |
| Output directory | `config.output` | If `None`, becomes `work_dirs/<config_name>`. |

## Backbone Editing Workflow

The current stock ResNet-style architecture is in `backbones/iresnet.py`. I copied it to:

```text
recognition/arcface_torch/backbones/iresnet_custom.py
```

The custom copy is registered in `backbones/__init__.py` with these names:

```text
custom_r18
custom_r34
custom_r50
custom_r100
custom_r200
```

To modify the architecture, edit `backbones/iresnet_custom.py`, then select it from your config:

```python
config.network = "custom_r50"
```

Good starting places inside `iresnet_custom.py`:

| What to change | Location |
| --- | --- |
| Block internals | `class IBasicBlock` |
| Stem convolution / first BN / PReLU | `IResNet.__init__` near `self.conv1`, `self.bn1`, `self.prelu` |
| Stage widths and strides | `IResNet.__init__` calls to `self._make_layer(...)` |
| Number of blocks per stage | `iresnet18/34/50/100/200(...)` layer lists |
| Feature head | `self.bn2`, `self.dropout`, `self.fc`, `self.features` |
| Forward path | `IResNet.forward(...)` |

If your architecture changes the final spatial shape or channel count, update:

```python
IResNet.fc_scale
self.fc = nn.Linear(..., num_features)
```

The training code expects the backbone forward pass to return a tensor shaped:

```text
[batch_size, config.embedding_size]
```

## Training and Testing Commands

Train with the stock R50 config:

```bash
cd recognition/arcface_torch
torchrun --nproc_per_node=8 train_v2.py configs/ms1mv3_r50
```

Train with the editable custom R50 config:

```bash
cd recognition/arcface_torch
torchrun --nproc_per_node=8 train_v2.py configs/ms1mv3_custom_r50
```

Single-GPU quick run:

```bash
cd recognition/arcface_torch
python train_v2.py configs/ms1mv3_custom_r50
```

Run simple inference after training:

```bash
cd recognition/arcface_torch
python inference.py --weight work_dirs/ms1mv3_custom_r50/model.pt --network custom_r50 --img path/to/aligned_112x112_face.jpg
```

Run IJB-C evaluation after training:

```bash
cd recognition/arcface_torch
CUDA_VISIBLE_DEVICES=0,1 python eval_ijbc.py \
  --model-prefix work_dirs/ms1mv3_custom_r50/model.pt \
  --image-path IJB_release/IJBC \
  --result-dir work_dirs/ms1mv3_custom_r50/ijbc \
  --batch-size 128 \
  --job ms1mv3_custom_r50 \
  --target IJBC \
  --network custom_r50
```

## Important Notes

- Pretrained weights are architecture-specific. If you change layer names, channel counts, or block counts, old `r50` weights may not load.
- `train_v2.py` saves the backbone weights as `model.pt` under `config.output`.
- `eval_ijbc.py` loads the same backbone registry as training, so use `--network custom_r50` for models trained with the custom config.
- Dataset values (`rec`, `num_classes`, `num_image`) must match the dataset you actually prepared.
