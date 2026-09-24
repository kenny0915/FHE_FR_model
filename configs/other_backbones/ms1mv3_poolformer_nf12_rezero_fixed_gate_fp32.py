"""Train the conservative ReZero + fixed-gate NF12 PoolFormer.

This is a fresh normalization-free student, not another conversion of the
scale-indeterminate LayerNorm checkpoint.  Every residual block starts as the
exact identity through a zero-initialized public ReZero scalar constrained to
[-1/12, 1/12].  The quadratic term is

    u + 0.02 * progress * (u * v),

so its scale is fixed and cannot grow to cancel a later normalization.  Scaled
weight standardization and bounded plaintext input/token-mixing gains provide
the remaining NFNet-style signal control.  All parameterizations are folded to
ordinary convolutions and public scalar multiplications for FHE deployment.
"""

from easydict import EasyDict as edict


config = edict()
config.network = "poolformer_nf12_rezero_fixed_gate"
config.resume = False
config.output = "work_dirs/ms1mv3_poolformer_nf12_rezero_fixed_gate_fp32"
config.embedding_size = 512
config.sample_rate = 1.0
config.margin_list = (1.0, 0.0, 0.4)

# The accepted S24 LayerNorm model is a frozen representation teacher only.
# Its scale-indeterminate residual parameters are deliberately not loaded into
# the normalization-free student.
config.embedding_teacher_network = "poolformer_fully_gated_s24"
config.embedding_teacher_checkpoint = (
    "work_dirs/ms1mv3_poolformer_s24_fully_gated_fp32/model.pt")
config.embedding_distill_weight = 1.0

# NFNet-style public-weight parameterization.  Weight gains stay fixed so the
# optimizer cannot undo the residual and gate bounds with another scalar.
config.nf_ws_eps = 1e-4
config.nf_learnable_ws_gain = False
config.nf_tau_init = 0.1
config.nf_input_gain_init = 1.0
config.nf_input_gain_min = 0.75
config.nf_input_gain_max = 1.25

# ReZero residuals are exactly zero at initialization and symmetrically bounded
# by the empirical 1/L scale reported for trained ReZero networks.
config.nf_residual_mode = "rezero"
config.nf_alpha_init = 0.0
config.nf_alpha_max = 1.0 / 12.0

# Keep the quadratic coefficient fixed.  The existing reverse curriculum only
# changes its public progress multiplier from zero to one.
config.nf_fixed_modulator_scale = 0.02
config.nf_modulator_scale_max = 0.02
config.nf_initial_modulation_progress = 0.0
config.nf_modulation_group_epochs = tuple(
    2.0 + block_index for block_index in range(12))
config.nf_modulation_transition_epochs = 1.0
config.nf_modulation_order = "reverse"
config.nf_require_full_modulation = True

# Training-only range control.  It introduces no encrypted activation or
# data-dependent branch into the deployed graph.
config.nf_range_limit = 6.0
config.nf_range_sample_size = 16384
config.nf_range_loss_weight = 1.0
config.nf_stats_interval = 50

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
config.max_validation_embedding_abs = 1e4

config.optimizer = "adamw"
config.lr = 1e-4
config.weight_decay = 0.01
config.selective_weight_decay = True
config.momentum = 0.9
config.batch_size = 128
config.warmup_epoch = 3
config.num_epoch = 30

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
