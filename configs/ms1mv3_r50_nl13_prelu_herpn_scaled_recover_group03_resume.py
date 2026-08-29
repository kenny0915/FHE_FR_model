"""Resume full-epoch conditioning from the group-3 recovery directory."""

from easydict import EasyDict as edict

from configs.ms1mv3_r50_nl13_prelu_herpn_scaled_recover_group03 import (
    config as recovery_config,
)


config = edict(recovery_config.copy())
config.resume = True
