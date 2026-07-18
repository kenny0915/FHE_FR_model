from copy import deepcopy

from configs.ms1mv3_r50_no_relu import config as training_config


config = deepcopy(training_config)
config.output = "work_dirs/ms1mv3_r50_no_relu_progressive_smoke"
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

# Exercise every transition and every post-stage BN recalibration quickly.
config.cheby_stage_epochs = (0, 1, 2, 3, 4)
config.cheby_transition_epochs = 1.0
config.cheby_bn_recalibration_batches = 1
