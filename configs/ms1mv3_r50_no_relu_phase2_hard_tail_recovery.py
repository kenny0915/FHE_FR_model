"""Replay exact MS1Mv3 tails to recover the epoch-23 polynomial R50.

The approximation target remains PReLU on [-6, 6].  Unlike the first recovery
attempt, this run uses the exact unclipped polynomial during training and
replays real MS1Mv3 images mined from that same inference graph.  The complete
stem-through-Layer3 prefix is constrained and trainable because the Stage3
overflow can be the end of amplification that began in Stage1/2.  BN remains
bitwise frozen.  Encrypted inference stays degree 2 with one square per
activation and gains no clamp or branch.
"""

from easydict import EasyDict as edict

from configs.ms1mv3_r50_no_relu_phase2_tail_recovery import (
    config as _phase2_config,
)


config = edict(_phase2_config.copy())
config.output = (
    "work_dirs/ms1mv3_r50_herpn_full_conversion_phase2_hard_tail_recovery")

_stage_blocks = (("layer1", 3), ("layer2", 4), ("layer3", 14))
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

# Exact forward: do not clamp earlier HerPNs and thereby change the Layer3
# distribution that the range loss is supposed to condition.
config.herpn_training_stabilization_limit = None
config.herpn_range_loss_weight = 2.0
config.fixed_tail_replay_file = (
    "work_dirs/ms1mv3_r50_herpn_full_conversion_phase2_tail_mining/"
    "epoch23_prefix_tails.json")
config.fixed_tail_replay_batch_size = 16
config.fixed_tail_replay_workers = 2
config.fixed_tail_replay_priority_count = 256
config.fixed_tail_replay_priority_repeats = 16

config.lr = 5e-5
config.num_epoch = 2
