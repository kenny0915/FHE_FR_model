"""Six-epoch stability probe for the exact released PILLAR penalty."""

import copy

from configs.ms1mv3_r50_pillar_espn import config as production_config


config = copy.deepcopy(production_config)
config.output = "work_dirs/casia_r50_pillar_espn_smoke"
config.rec = "./faces_webface_112x112"
config.num_classes = 10572
# With eight GPUs and batch 256 this gives 200 scheduled optimizer steps per
# epoch, matching max_steps_per_epoch and keeping the cosine schedule honest.
config.num_image = 409600
config.num_epoch = 6
config.max_steps_per_epoch = 200
config.verbose = 200
config.pillar_log_interval = 10
config.checkpoint_interval_epochs = 1
config.val_targets = ["lfw"]
