"""Train FHEPoolFormer-NF12 from scratch with an embedding teacher.

The twelve backbone blocks have no activation normalization.  Each block has
one exact degree-2 gate, so the nonlinear multiplicative depth is 12 along the
longest path.  Scaled weight standardization and bounded scalar
parameterizations are training-time operations on plaintext parameters and
are materialized/folded before FHE export.

The accepted LayerNorm S24 model is used only as a frozen embedding teacher.
It does not appear in the student inference graph and is not a warm start.
"""

from easydict import EasyDict as edict


config = edict()
config.network = "poolformer_nf12"
config.resume = False
config.output = "work_dirs/ms1mv3_poolformer_nf12_progressive_fp32"
config.embedding_size = 512
config.sample_rate = 1.0

# Frozen representation teacher. The checkpoint may be either an inference
# state dict or a training checkpoint containing ``state_dict_backbone``.
config.embedding_teacher_network = "poolformer_fully_gated_s24"
config.embedding_teacher_checkpoint = (
    "work_dirs/ms1mv3_poolformer_s24_fully_gated_fp32/model.pt")
config.embedding_distill_weight = 1.0

# Stable NF block initialization. tau mixes only 10% local average; alpha is
# initially 0.05 and cannot exceed 0.1; the quadratic modulator starts at zero,
# making every gate exactly linear (u * 1) on the first forward pass.
config.nf_ws_eps = 1e-4
config.nf_learnable_ws_gain = False
config.nf_tau_init = 0.1
config.nf_alpha_init = 0.05
config.nf_alpha_max = 0.1
config.nf_input_gain_init = 1.0
config.nf_input_gain_min = 0.5
config.nf_input_gain_max = 1.25
config.nf_modulator_scale_max = 0.02
config.nf_initial_modulation_progress = 0.0

# Train an exactly linear twelve-block network first. Starting at epoch 2,
# activate one quadratic coefficient per epoch from block 11 back to block 0.
# A new gate therefore enters a graph whose remaining inactive blocks are
# still linear; all gates are fully active after epoch 14.
config.nf_modulation_group_epochs = tuple(
    2.0 + block_index for block_index in range(12))
config.nf_modulation_transition_epochs = 1.0
config.nf_modulation_order = "reverse"
config.nf_require_full_modulation = True
config.nf_range_limit = 6.0
config.nf_range_sample_size = 16384
config.nf_range_loss_weight = 1.0
config.nf_stats_interval = 50

# FP32 is intentional for the first stability run.  Clip the backbone only so
# PartialFC's much larger classifier cannot dominate the norm.
config.fp16 = False
config.gradient_clip = 0.5
config.gradient_clip_type = "norm"
config.gradient_clip_scope = "backbone"
config.stable_gradient_clip = True
config.gradient_norm_warning_threshold = 100.0
config.gradient_norm_warning_interval = 100
config.check_finite_grads = True
config.fail_on_nonfinite_val = True
config.max_nonfinite_embedding_skips = 0

config.optimizer = "adamw"
config.lr = 1e-4
config.weight_decay = 0.01
config.selective_weight_decay = True
config.momentum = 0.9
config.batch_size = 128
config.warmup_epoch = 3
config.num_epoch = 25

# Only the foldable face head contains BatchNorm.
config.sync_bn = True
config.broadcast_buffers = True
config.ddp_fp16_compress = False

config.save_all_states = True
config.checkpoint_interval_epochs = 1
config.save_epoch_models = True
config.epoch_model_interval = 1
config.verbose = 2000
config.frequent = 10
config.dali = False
config.dali_aug = False
config.gradient_acc = 1

config.rec = "./ms1m-retinaface-t1"
config.num_classes = 93431
config.num_image = 5179510
config.interclass_filtering_threshold = 0
config.margin_list = (1.0, 0.0, 0.4)
config.val_targets = ["lfw", "cfp_fp", "agedb_30"]
