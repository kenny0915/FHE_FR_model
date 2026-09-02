"""Joint numerical recovery of the epoch-23 fully polynomial R50.

All 25 HerPN activations remain degree-2 approximations of PReLU on [-6, 6].
The epoch-23 function is the exact initialization: no quadratic coefficient is
manually attenuated.  BN running statistics and affine parameters are frozen,
while every convolution and all polynomial coefficients are optimized jointly.

The training-only straight-through bounds keep catastrophic MS1Mv3 replay
faces differentiable.  Evaluation and FHE inference use the exact unclipped
Ax^2 + Bx + C graph, with one ciphertext square per activation and no added
multiplicative depth.
"""

from easydict import EasyDict as edict

from configs.ms1mv3_r50_no_relu_phase2_tail_recovery import (
    config as _phase2_config,
)


config = edict(_phase2_config.copy())
config.output = (
    "work_dirs/ms1mv3_r50_herpn_full_conversion_phase2_joint_recovery")

# This adds independent trainable H0/H1/H2 channel scales initialized to one.
# Loading epoch23 therefore preserves its output exactly before the first step.
config.herpn_independent_basis_scales = True
config.herpn_basis_anchor_loss_weight = 1.0

_stage_blocks = (
    ("layer1", 3),
    ("layer2", 4),
    ("layer3", 14),
    ("layer4", 3),
)
config.backbone_trainable_prefixes = (
    "conv1",
    "prelu.herpn",
    *tuple(
        prefix
        for stage, block_count in _stage_blocks
        for index in range(block_count)
        for prefix in (
            f"{stage}.{index}.conv1",
            f"{stage}.{index}.conv2",
            f"{stage}.{index}.prelu.herpn",
            *((f"{stage}.{index}.downsample.0",) if index == 0 else ()),
        )
    ),
)
config.herpn_range_loss_names = (
    "prelu",
    *(f"{stage}.{index}.prelu"
      for stage, block_count in _stage_blocks
      for index in range(block_count)),
)

# Freeze every BN tensor.  Stabilization is a training surrogate on all 25
# HerPNs; it is absent from saved-model evaluation and encrypted inference.
config.freeze_batchnorm_running_stats = True
config.freeze_batchnorm_affine = True
config.herpn_training_stabilization_limit = 6.0
config.herpn_training_stabilization_names = ()
config.herpn_range_loss_weight = 2.0
config.herpn_distill_loss_weight = 0.0

# Replay exactly the 244 catastrophic source/orientation rows found with the
# epoch-23 graph.  Avoid random photometric perturbations in this controlled
# joint-recovery pass.
config.fixed_tail_replay_file = (
    "work_dirs/ms1mv3_r50_herpn_full_conversion_phase2_tail_mining/"
    "epoch23_prefix_tails.json")
config.fixed_tail_replay_batch_size = 16
config.fixed_tail_replay_workers = 2
config.fixed_tail_replay_priority_count = 0
config.fixed_tail_replay_priority_repeats = 1
config.fixed_tail_replay_orientations_key = "output_nonfinite"
config.range_augmentation = {"enabled": False}

# Preserve ordinary embedding geometry while the rare-tail range loss changes
# the end-to-end convolution/polynomial system.  PartialFC is not used.
config.embedding_teacher_network = "r50"
config.embedding_teacher_checkpoint = "work_dirs/ms1mv3_r50/model.pt"
config.embedding_distill_weight = 1.0
config.task_loss_weight = 0.0

# The previous broad-prefix run showed that 5e-5 is too aggressive.  Use a
# low-LR, zero-momentum joint update and save both epochs for separate gates.
config.lr = 1e-6
config.momentum = 0.0
config.weight_decay = 1e-5
config.gradient_clip = 1.0
config.stable_gradient_clip = True
config.warmup_epoch = 0
config.num_epoch = 2
config.save_epoch_models = True
config.epoch_model_interval = 1
