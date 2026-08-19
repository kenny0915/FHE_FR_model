"""Remove LayerNorm standard deviations from the accepted gated PoolFormer.

This follows the hard-switch fine-tuning protocol from ``removing-layer-norm``:
each wrapper is exactly LayerNorm at initialization, tracks an EMA of its input
standard deviation, and is switched permanently to channel centering followed
by multiplication with a frozen reciprocal standard deviation.  The 24 norm2
sites are converted shallow-to-deep first, then the 24 norm1 sites, and finally
the output norm.  SimpleGate remains the exact degree-2 activation throughout.

The final encrypted path therefore retains only fixed-count channel centering
and fixed affine multiplication at normalization sites.  It has no variance,
square root, or encrypted-data-dependent reciprocal.  The 24 sequential
SimpleGates remain the dominant multiplicative-depth cost.
"""

from easydict import EasyDict as edict


config = edict()
config.margin_list = (1.0, 0.0, 0.4)
config.network = "poolformer_fully_gated_frozen_std_s24"
config.resume = False
config.output = (
    "work_dirs/ms1mv3_poolformer_s24_fully_gated_frozen_std_fp32")
config.embedding_size = 512
config.sample_rate = 1.0

# Strictly warm-start the accepted all-SimpleGate LayerNorm model.  Its outputs
# are bit-exact before the first hard switch; only EMA/schedule buffers are new.
config.backbone_init = (
    "work_dirs/ms1mv3_poolformer_s24_fully_gated_fp32/model.pt")

# Track one scalar standard deviation per normalization site.  The paper/code
# use EMA momentum 0.9; the initial value is only a guard until data is seen.
config.frozen_std_momentum = 0.9
config.frozen_std_initial = 1.0

# Four V100s with batch_size=128 give 512 images/optimizer step and about
# 10,116 steps/epoch.  After one exact-LN warmup epoch, switch one of the 49
# sites every 200 steps.  All conversions finish around epoch 1.95, leaving
# just over three epochs to optimize the fully converted network.
config.frozen_std_start_epoch = 1.0
config.frozen_std_group_gap_steps = 200
config.frozen_std_require_full_conversion = True

# The paper applies its variance-concentration auxiliary objective at the input
# to the final LayerNorm.  This implementation uses the same objective, with a
# detached global mean across DDP workers.
config.frozen_std_aux_loss_weight = 0.1
config.final_verification_after_frozen_std = True

# Conservative FP32 warm-start fine-tuning for V100 stability.  AMP/bfloat16 is
# intentionally disabled because V100 has no native bfloat16 support and the
# fully gated quadratic backbone is already known to need FP32 here.
config.fp16 = False
config.gradient_clip = 0.01
config.gradient_clip_type = "norm"
config.gradient_clip_scope = "backbone"
config.check_finite_grads = True
config.fail_on_nonfinite_val = True

config.optimizer = "adamw"
config.lr = 5e-5
config.weight_decay = 0.001
config.momentum = 0.9
config.batch_size = 128
config.warmup_epoch = 0
config.num_epoch = 5

# The wrappers have scalar buffers but no BatchNorm.  Buffer broadcast plus the
# one-time all-reduce at each hard switch makes the stored constants identical
# across ranks.  Keep a resumable checkpoint and inference snapshot each epoch.
config.sync_bn = True
config.broadcast_buffers = True
config.ddp_fp16_compress = False
config.save_all_states = True
config.checkpoint_interval_epochs = 1
config.save_epoch_models = True
config.epoch_model_interval = 1

# Disable other normalization conversion mechanisms explicitly.
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
