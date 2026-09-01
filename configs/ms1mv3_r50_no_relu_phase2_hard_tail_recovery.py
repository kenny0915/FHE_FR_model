"""Replay exact MS1Mv3 tails to recover the epoch-23 polynomial R50.

The approximation target remains PReLU on [-6, 6].  Unlike the first recovery
attempt, this run uses the exact unclipped polynomial during training and
replays real MS1Mv3 images mined from that same inference graph.  All Layer3
pre-HerPN inputs are constrained because fixing the first four merely moved
some failures deeper.  BN remains bitwise frozen.  Encrypted inference stays
degree 2 with one square per activation and gains no clamp or branch.
"""

from easydict import EasyDict as edict

from configs.ms1mv3_r50_no_relu_phase2_tail_recovery import (
    config as _phase2_config,
)


config = edict(_phase2_config.copy())
config.output = (
    "work_dirs/ms1mv3_r50_herpn_full_conversion_phase2_hard_tail_recovery")

config.backbone_trainable_prefixes = tuple(
    prefix
    for index in range(14)
    for prefix in (
        f"layer3.{index}.conv1",
        f"layer3.{index}.conv2",
        f"layer3.{index}.prelu.herpn",
        *((f"layer3.{index}.downsample.0",) if index == 0 else ()),
    )
)
config.herpn_range_loss_names = tuple(
    f"layer3.{index}.prelu" for index in range(14))

# Exact forward: do not clamp earlier HerPNs and thereby change the Layer3
# distribution that the range loss is supposed to condition.
config.herpn_training_stabilization_limit = None
config.herpn_range_loss_weight = 2.0
config.fixed_tail_replay_file = (
    "work_dirs/ms1mv3_r50_herpn_full_conversion_phase2_tail_mining/"
    "epoch23_layer3_tails.json")
config.fixed_tail_replay_batch_size = 16
config.fixed_tail_replay_workers = 2
config.fixed_tail_replay_priority_count = 256
config.fixed_tail_replay_priority_repeats = 16

config.lr = 5e-5
config.num_epoch = 2
