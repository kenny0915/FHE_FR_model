"""Conflict-aware full-backbone recovery from the phase-1 epoch-23 model.

Approximation target: PReLU over [-6, 6].  All convolution weights, ordinary
BN affine parameters/running statistics, and original HerPN weight/bias
coefficients can adapt.  FC and the final embedding BN stay fixed because the
measured non-finite cascade occurs before them.

Clean MS1Mv3 batches update BN and supply the frozen-PReLU teacher objective.
The 244 mined catastrophic orientations run separately with inference-mode BN
and a training-only straight-through bound.  Their loss selects each sample's
earliest unsafe boundary, so later repeated squares cannot dominate it.  The
saved/eval/FHE graph remains the exact unclipped degree-2 HerPN network.
"""

from easydict import EasyDict as edict

from configs.ms1mv3_r50_no_relu import config as _base_config


config = edict(_base_config.copy())
config.output = (
    "work_dirs/"
    "ms1mv3_r50_herpn_full_conversion_phase2_conflict_aware_recovery")
config.backbone_init = (
    "work_dirs/ms1mv3_r50_herpn_full_conversion_phase1/model_epoch_23.pt")
config.herpn_initial_progress = 5.0
config.backbone_init_herpn_progress = 5.0
config.herpn_conversion_groups = ()
config.herpn_group_epochs = ()
config.herpn_stage_epochs = ()
config.herpn_bn_recalibration_batches = 0

# A guard at 80% of the fitted interval leaves headroom before the first
# square.  The ordered list covers the measured onset from layer1.2 through
# layer3.0; later activations are not allowed to drown out an earlier cause.
config.herpn_range_penalty_mode = "sample_max_tail"
config.herpn_range_guard_ratio = 0.8
config.tail_causal_activation_names = (
    "layer1.2.prelu",
    "layer2.0.prelu",
    "layer2.1.prelu",
    "layer2.2.prelu",
    "layer2.3.prelu",
    "layer3.0.prelu",
)
config.herpn_training_stabilization_limit = 6.0
config.herpn_training_stabilization_names = ()

config.fixed_tail_replay_file = (
    "work_dirs/ms1mv3_r50_herpn_full_conversion_phase2_tail_mining/"
    "epoch23_prefix_tails.json")
config.fixed_tail_replay_orientations_key = "output_nonfinite"
config.fixed_tail_replay_batch_size = 4
config.fixed_tail_replay_workers = 2
config.fixed_tail_replay_interval = 8

config.embedding_teacher_network = "r50"
config.embedding_teacher_checkpoint = "work_dirs/ms1mv3_r50/model.pt"
config.clean_distill_weight = 1.0
config.tail_range_loss_weight = 1.0

# SGD retains actual gradient scale.  PCGrad-style conflict projection and
# per-tensor update caps prevent the AdamW drift seen in the preceding run.
config.optimizer = "sgd"
config.lr = 1e-4
config.min_lr_ratio = 0.1
config.momentum = 0.0
config.weight_decay = 0.0
config.tail_to_clean_gradient_ratio = 1.0
config.max_step_update_ratio = 1e-5
config.parameter_trust_region_ratio = 0.005
config.parameter_trust_region_interval = 100
config.parameter_scale_floor = 1.0

config.num_epoch = 3
config.fp16 = False
config.sync_bn = True
config.tensorboard = False
config.frequent = 100
config.save_epoch_models = True
