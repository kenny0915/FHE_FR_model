"""Whole-model recovery with tensor-local gradients and sparse tail replay.

Every Conv and HerPN tensor is trainable in every end-to-end update.  Tensor-
local clipping prevents an early layer from consuming its family's entire
gradient budget, while low-LR AdamW makes small FP32 updates representable.
Exact catastrophic orientations are replayed intermittently instead of being
inserted into every batch, preserving the ordinary MS1Mv3/teacher signal.

The target remains PReLU on [-6, 6].  BN is immutable, H2 starts at one, and
evaluation/FHE inference remains an unclipped degree-2 polynomial graph.
"""

from easydict import EasyDict as edict

from configs.ms1mv3_r50_no_relu_phase2_joint_grouped_recovery import (
    config as _grouped_config,
)


config = edict(_grouped_config.copy())
config.output = (
    "work_dirs/"
    "ms1mv3_r50_herpn_full_conversion_phase2_joint_tensor_recovery")

# AdamW's per-element state makes updates representable even when a tensor's
# gradient is much smaller than another tensor in the same stage.  Gradients
# are clipped before Adam moments are accumulated, preventing FP32 v overflow.
config.optimizer = "adamw"
config.lr = 1e-7
config.herpn_lr_multiplier = 10.0
config.conv_herpn_gradient_clip_granularity = "tensor"
config.conv_gradient_clip = 1.0
config.herpn_gradient_clip = 0.1
config.weight_decay = 1e-5

# Four exact orientations per rank every eighth update average 0.39% of the
# global training stream.  This is still about 160x their natural frequency,
# but avoids the previous 12.5% hard-tail concentration and ~2,600 replays per
# catastrophic orientation per epoch.
config.fixed_tail_replay_batch_size = 4
config.fixed_tail_replay_interval = 8

config.num_epoch = 2
