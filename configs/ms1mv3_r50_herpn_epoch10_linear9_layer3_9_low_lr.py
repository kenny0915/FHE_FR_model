"""Continue the near-accepted step-1000 recovery with a 10x smaller LR."""

from easydict import EasyDict as edict

from configs.ms1mv3_r50_herpn_epoch10_linear9_layer3_9_recovery import (
    config as recovery_config,
)


config = edict(recovery_config.copy())
config.resume = False
config.output = (
    "work_dirs/ms1mv3_r50_herpn_epoch10_linear9_layer3_9_low_lr")
config.backbone_init = (
    "work_dirs/ms1mv3_r50_herpn_epoch10_linear9_layer3_9_recovery/"
    "model_step_01000.pt")
config.lr = 3.0e-5
config.num_epoch = 1
config.verbose = 250

