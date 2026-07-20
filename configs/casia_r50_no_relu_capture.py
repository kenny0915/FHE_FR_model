from copy import deepcopy

from configs.casia_r50_no_relu import config as training_config


config = deepcopy(training_config)
config.resume = True
config.output = "work_dirs/casia_r50_herpn_capture8000"
config.num_epoch = 4
config.max_steps_per_epoch = 336
config.verbose = 1000000
config.val_targets = []
