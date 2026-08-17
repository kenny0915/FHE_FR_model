"""Train a pure degree-2 HerPN IResNet50 with scaled residuals from scratch.

There is deliberately no ``backbone_init`` and no PReLU teacher.  All 25
nonlinearities target the AESPA/Hermite degree-2 ReLU approximation, monitored
on [-6, 6].  The 24 residual branches start at 1/sqrt(24), which limits early
forward/Jacobian growth without blocking branch gradients as alpha=0 would.

The resulting public residual scalars and calibrated HerPN BatchNorms can be
folded into an FHE graph containing affine operations and one square per
activation.
"""

import math

from easydict import EasyDict as edict


config = edict()
config.margin_list = (1.0, 0.5, 0.0)
config.network = "r50_herpn_residual_scale"
config.resume = False
config.output = "work_dirs/ms1mv3_r50_herpn_residual_scale"
config.embedding_size = 512
config.sample_rate = 1.0

# Polynomial branches are more sensitive to range/rounding excursions.  Use
# FP32 for the first full run; FP16 should be a separate ablation after this
# recipe is stable.
config.fp16 = False
config.batch_size = 128
config.lr = 0.02
config.momentum = 0.9
config.weight_decay = 5e-4
config.selective_weight_decay = True
config.gradient_clip = 1.0
config.gradient_clip_scope = "backbone"
config.warmup_epoch = 2

# IResNet50 has 3+4+14+3 = 24 residual blocks.  Scales are trainable public
# scalars, excluded from weight decay, and folded into bn3 for inference.
config.residual_scale_init = 1.0 / math.sqrt(24.0)
config.residual_scale_trainable = True

# Pure HerPN from step zero: no conversion schedule and no distillation loss.
# The interval penalty is a training-only guard; it introduces no encrypted
# ReLU or branching into the exported inference graph.
config.herpn_initial_progress = 5.0
config.herpn_bn_eps = 1e-4
config.herpn_range_limit = 6.0
config.herpn_range_loss_weight = 0.05
config.herpn_distill_loss_weight = 0.0
config.herpn_stage_epochs = ()
config.herpn_conversion_groups = ()

config.sync_bn = True
config.broadcast_buffers = True
# Keep synchronized gradients in FP32 for the first stability run.
config.ddp_fp16_compress = False
config.check_finite_grads = True
config.fail_on_nonfinite_val = True
config.max_validation_embedding_abs = 1e6
config.save_validation_snapshots = True
config.validation_batch_size = 128
config.save_all_states = True
config.checkpoint_interval_epochs = 1
config.save_epoch_models = True
config.epoch_model_interval = 1

config.verbose = 2000
config.dali = False
config.rec = "./ms1m-retinaface-t1"
config.num_classes = 93431
config.num_image = 5179510
# Longer than the 20-epoch PReLU baseline because all polynomial features and
# residual scales are learned from scratch rather than converted from a teacher.
config.num_epoch = 30
config.val_targets = ["lfw", "cfp_fp", "agedb_30"]
