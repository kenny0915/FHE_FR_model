from easydict import EasyDict as edict

from configs.ms1mv3_poolformer_s24_no_ln_x2_act_reverse_1block import (
    config as one_block_config,
)


config = edict(one_block_config.copy())

# Preserve both the original six-group experiment and the slower one-block
# reverse run. This directory is exclusively for the two-block schedule.
config.output = (
    "work_dirs/ms1mv3_poolformer_s24_no_ln_x2_act_reverse_2block")
config.resume = False

# Convert two adjacent blocks at a time from the output end toward the input:
# (network.6.2, network.6.3), (network.6.0, network.6.1), ..., then
# (network.0.0, network.0.1). Twelve groups receive three epochs each.
# Epoch 44 performs the final completion/recalibration boundary and provides
# one fully converted hold epoch.
config.simple_gate_grouping = "two_blocks_reverse"
config.simple_gate_group_epochs = tuple(range(8, 44, 3))
config.simple_gate_transition_epochs = 3.0
config.num_epoch = 45

# Preserve the approximate update budget per block from the one-block run:
# five epochs * 0.1 LR = three epochs * (1/6) LR.
config.simple_gate_lr_multiplier = 1.0 / 6.0
