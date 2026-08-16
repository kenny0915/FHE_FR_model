"""Groupwise fine-tune PoolFormer-S24 from LayerNorm to fixed affine maps.

The all-at-once conversion is numerically unsafe for this quadratic backbone:
small errors at all 49 sites compound through 24 sequential SimpleGates.  This
recipe converts one block at a time in forward order.  Before each group
starts, representative inputs refit only that group's affine maps against the
current upstream graph.  Later LayerNorms remain active as stability barriers.

Twenty-five groups transition for one epoch each, followed by two pure-affine
fine-tuning epochs.  At inference the affine maps contain only plaintext
per-channel scales and biases.

The 24 SimpleGates are unchanged and remain exact degree-2 products.
"""

from easydict import EasyDict as edict


config = edict()
config.margin_list = (1.0, 0.0, 0.4)
config.network = "poolformer_fully_gated_affine_s24"
config.resume = False
config.output = "work_dirs/ms1mv3_poolformer_s24_fully_gated_affine_grouped_fp32"
config.embedding_size = 512
config.sample_rate = 1.0

# Strict warm start from the accepted fully gated LayerNorm backbone.
config.backbone_init = (
    "work_dirs/ms1mv3_poolformer_s24_fully_gated_fp32/model.pt")

# One block per group prevents an affine-driven residual shift from entering a
# second affine SimpleGate before a still-exact downstream LayerNorm can bound
# its branch. The 24 blocks plus final norm give 25 groups.
# Each group is refit after all preceding groups have reached pure affine.
config.affine_blocks_per_group = 1
config.affine_group_epochs = tuple(range(25))
config.affine_group_transition_epochs = 1.0
config.affine_group_calibration_batches = 50
config.affine_group_require_full_conversion = True
config.affine_calibration_batches = 0
config.affine_calibration_ridge = 1e-6

# Disable the old global schedule; group blends are updated independently.
config.prepbn_decay_steps = 0
config.prepbn_require_full_transition = False
config.validate_after_prepbn_transition = False
config.prepbn_bn_stat_epochs = 0
config.final_verification_after_prepbn = True

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
config.num_epoch = 27

# The head still contains BatchNorm, so preserve the established synchronized
# training behavior.  The 49 converted backbone normalization sites do not.
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
config.val_targets = ["lfw", "cfp_fp", "agedb_30"]
