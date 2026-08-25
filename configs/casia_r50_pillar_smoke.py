"""Short GPU-server smoke run for the PILLAR polynomial iResNet-50.

This preserves the production polynomial, interval, regularization, and
warm-up settings while limiting the data and steps. It is not an accuracy run.
"""

import copy

from configs.ms1mv3_r50_pillar import config as production_config


config = copy.deepcopy(production_config)
config.output = "work_dirs/casia_r50_pillar_smoke"
config.rec = "./faces_webface_112x112/"
config.num_classes = 10572
config.num_image = 25600
config.num_epoch = 2
config.warmup_epoch = 1
config.max_steps_per_epoch = 100
config.verbose = 1000000
config.val_targets = []
config.save_all_states = False
config.save_epoch_models = False
