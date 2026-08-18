"""Train a normalization-free degree-2 IResNet-NF12 from scratch.

Every residual block is linear at initialization and has one learned quadratic
product.  Bounded residual and quadratic coefficients prevent the unrestricted
polynomial cascade observed in the all-HerPN R50 experiment.  A frozen PReLU
R50 supplies embedding targets but is absent from student inference.
"""

from easydict import EasyDict as edict


config = edict()
config.network = "iresnet_nf12"
config.resume = False
config.output = "work_dirs/ms1mv3_iresnet_nf12_twostream_fp32"
config.embedding_size = 512
config.sample_rate = 1.0

# Architecture-independent representation teacher. This is distillation, not
# a warm start, so the NF student retains its stable linear initialization.
config.embedding_teacher_network = "r50"
config.embedding_teacher_checkpoint = "work_dirs/ms1mv3_r50/model.pt"
config.embedding_distill_weight = 1.0

# Twelve blocks, one bounded u*(1+beta*v) product per block. Independent u/v
# projections avoid the accumulating positive mean of a self-square. Both the
# residual and quadratic paths start conservative and remain bounded.
config.nf_ws_eps = 1e-4
config.nf_learnable_ws_gain = False
config.nf_alpha_init = 0.02
config.nf_alpha_max = 0.1
config.nf_input_gain_init = 1.0
config.nf_input_gain_min = 0.5
config.nf_input_gain_max = 2.0
config.nf_quadratic_scale_max = 0.1
config.nf_range_limit = 6.0
config.nf_range_sample_size = 16384
config.nf_range_loss_weight = 0.05
config.nf_stats_interval = 100

config.fp16 = False
config.gradient_clip = 1.0
config.gradient_clip_type = "norm"
config.gradient_clip_scope = "backbone"
config.check_finite_grads = True
config.fail_on_nonfinite_val = True
config.max_validation_embedding_abs = 1e6
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
config.save_validation_snapshots = True
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
