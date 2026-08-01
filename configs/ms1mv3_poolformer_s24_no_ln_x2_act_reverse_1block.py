from easydict import EasyDict as edict

from configs.ms1mv3_poolformer_s24_no_ln_x2_act import config as base_config


config = edict(base_config.copy())

# Keep every checkpoint, log, and evaluation artifact separate from the
# original six-group x2 activation experiment.
config.output = (
    "work_dirs/ms1mv3_poolformer_s24_no_ln_x2_act_reverse_1block")
config.resume = False

# Convert one of the 24 MLP activations at a time, starting at network.6.3 and
# walking backward to network.0.0. RepBN finishes at epoch 8; each gate then
# receives a five-epoch transition. Epoch 128 performs the final completion
# boundary/recalibration and provides one pure-SimpleGate hold epoch.
config.simple_gate_grouping = "one_block_reverse"
config.simple_gate_group_epochs = tuple(range(8, 128, 5))
config.simple_gate_transition_epochs = 5.0
config.num_epoch = 129

# Only the next/current block retains its GELU teacher and multiplier
# auxiliary tensors. The main task optimizer uses one tenth of the scheduled
# LR once SimpleGate conversion begins; the scheduler itself remains intact.
config.simple_gate_current_group_auxiliary = True
config.simple_gate_lr_multiplier = 0.1
