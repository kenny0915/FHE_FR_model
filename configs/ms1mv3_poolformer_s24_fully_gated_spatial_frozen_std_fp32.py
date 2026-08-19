"""Safely remove LayerNorm scale from the accepted gated PoolFormer.

This replaces the failed scalar hard-switch experiment.  Each site first runs
as exact LayerNorm while collecting a per-spatial-position EMA of both the
batch-mean and batch-maximum channel standard deviation.  At conversion, the
tail map is checked, multiplied by a safety margin, frozen as a plaintext
constant, and blended from exact LayerNorm over 400 optimizer steps.  Only one
site is allowed to transition at a time.

The final normalization path uses channel centering and multiplication by a
fixed [1, 1, H, W] reciprocal map.  It therefore has no encrypted variance,
square root, or data-dependent reciprocal.  Spatially varying plaintext
constants do not add multiplicative depth, but do cost more constants than the
unsafe one-scalar-per-site experiment.
"""

from easydict import EasyDict as edict


config = edict()
config.margin_list = (1.0, 0.0, 0.4)
config.network = "poolformer_fully_gated_spatial_frozen_std_s24"
config.resume = False
config.output = (
    "work_dirs/ms1mv3_poolformer_s24_fully_gated_spatial_frozen_std_fp32")
config.embedding_size = 512
config.sample_rate = 1.0

# Start only from the accepted all-SimpleGate model with real LayerNorm.  Do
# not reuse a checkpoint, optimizer, EMA, or schedule state from either failed
# frozen-std run.
config.backbone_init = (
    "work_dirs/ms1mv3_poolformer_s24_fully_gated_fp32/model.pt")

# Exact-LN warmup supplies 10,116 observations before conversion starts.
# Momentum 0.9 follows the earlier implementation, while the batch maximum and
# 1.25 margin protect positions whose scale is above the batch mean.
config.frozen_std_momentum = 0.9
config.frozen_std_initial = 1.0
config.frozen_std_spatial_margin = 1.25
# A large tail/mean ratio means the conservative tail map attenuates typical
# samples at that position; it does not amplify the tail.  Keep this ratio in
# the transition diagnostics, but do not abort on it.  The absolute magnitude
# cap below remains the numerical guard against the scale explosion observed
# in the scalar hard-switch run.  Set a positive value for fidelity ablations.
config.frozen_std_max_tail_to_mean_ratio = 0.0
config.frozen_std_max_value = 1e4

# With 4 V100s and 128 samples/GPU there are about 10,116 optimizer steps per
# epoch.  A group transitions for 400 steps and the next starts every 600, so
# transitions never overlap.  Group 49 finishes at step 39,317 (epoch 3.89),
# leaving more than three epochs to fine-tune the fully converted network.
config.frozen_std_start_epoch = 1.0
config.frozen_std_group_gap_steps = 600
config.frozen_std_transition_steps = 400
config.frozen_std_require_full_conversion = True
config.frozen_std_aux_loss_weight = 0.0
config.final_verification_after_frozen_std = True

# The quadratic backbone is kept in FP32.  Use a lower LR than the failed
# hard-switch experiment and clip only the backbone before non-finite gradients
# can reach the optimizer.
config.fp16 = False
config.gradient_clip = 0.01
config.gradient_clip_type = "norm"
config.gradient_clip_scope = "backbone"
config.check_finite_grads = True
config.fail_on_nonfinite_val = True

config.optimizer = "adamw"
config.lr = 2e-5
config.weight_decay = 0.001
config.momentum = 0.9
config.batch_size = 128
config.warmup_epoch = 0
config.num_epoch = 7

# Dynamic spatial maps and blend buffers are part of every rank's checkpoint.
# Keep DDP from overwriting each rank's local EMA with rank 0 before every
# forward.  At transition time a collective averages the mean map and takes the
# cross-rank maximum of the tail map.  SyncBatchNorm still synchronizes the two
# BatchNorm layers in the face head.
config.sync_bn = True
config.broadcast_buffers = False
config.ddp_fp16_compress = False
config.save_all_states = True
config.checkpoint_interval_epochs = 1
config.save_epoch_models = True
config.epoch_model_interval = 1

config.prepbn_decay_steps = 0
config.prepbn_require_full_transition = False
config.prepbn_bn_stat_epochs = 0
config.final_verification_after_prepbn = False

config.verbose = 2000
config.frequent = 10
config.dali = False
config.dali_aug = False
config.gradient_acc = 1

config.rec = "./ms1m-retinaface-t1"
config.num_classes = 93431
config.num_image = 5179510
config.interclass_filtering_threshold = 0
config.val_targets = ["lfw", "cfp_fp", "agedb_30"]
