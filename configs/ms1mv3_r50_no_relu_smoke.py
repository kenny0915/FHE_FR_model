from copy import deepcopy

from configs.ms1mv3_r50_no_relu import config as training_config


config = deepcopy(training_config)
config.output = "work_dirs/ms1mv3_r50_herpn_smoke"
config.resume = False
config.sync_bn = False
config.broadcast_buffers = True
config.batch_size = 16
config.num_image = 32
config.num_epoch = 6
config.warmup_epoch = 0
config.max_steps_per_epoch = 2
config.verbose = 1000000
config.frequent = 1
config.val_targets = []
config.save_all_states = False

# Exercise every block group and post-group BN recalibration quickly.
config.herpn_stage_epochs = ()
config.herpn_group_epochs = tuple(index * 0.25 for index in range(
    len(config.herpn_conversion_groups)))
config.herpn_transition_epochs = 0.25
config.herpn_bn_recalibration_batches = 1
