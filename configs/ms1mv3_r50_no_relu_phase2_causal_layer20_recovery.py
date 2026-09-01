"""Causally condition the first measured runaway boundary, layer2.0.

Exact MS1Mv3 prefix mining found maxima 1.79 (stem), 5.94 (end of Layer1),
then 115.6 at the input of ``layer2.0.prelu``.  This run therefore adjusts
the stem/Layer1/layer2.0 prefix and penalizes only that first unsafe boundary.
The training-only stabilization begins at layer2.0 so exact non-finite replay
sources remain usable.  Evaluation/FHE inference is still the exact unclipped
degree-2 graph targeting PReLU on [-6, 6], with unchanged depth.
"""

from easydict import EasyDict as edict

from configs.ms1mv3_r50_no_relu_phase2_tail_recovery import (
    config as _phase2_config,
)


config = edict(_phase2_config.copy())
config.output = (
    "work_dirs/ms1mv3_r50_herpn_full_conversion_phase2_causal_layer20")

_stage_blocks = (("layer1", 3), ("layer2", 1))
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
config.herpn_range_loss_names = ("layer2.0.prelu",)

# Preserve the exact path through the measured boundary input. Stabilize only
# that activation and its suffix so a catastrophic replay source can still
# deliver a range gradient to the causal prefix.
config.herpn_training_stabilization_limit = 6.0
config.herpn_training_stabilization_names = (
    *(f"layer2.{index}.prelu" for index in range(4)),
    *(f"layer3.{index}.prelu" for index in range(14)),
    *(f"layer4.{index}.prelu" for index in range(3)),
)
config.herpn_range_loss_weight = 2.0

config.fixed_tail_replay_file = (
    "work_dirs/ms1mv3_r50_herpn_full_conversion_phase2_tail_mining/"
    "epoch23_prefix_tails.json")
config.fixed_tail_replay_batch_size = 16
config.fixed_tail_replay_workers = 2
config.fixed_tail_replay_priority_count = 158
config.fixed_tail_replay_priority_repeats = 32

config.lr = 5e-5
config.num_epoch = 1
