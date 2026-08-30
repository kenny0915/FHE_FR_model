"""Resume the guarded all-polynomial PILLAR run from its latest epoch."""

from easydict import EasyDict as edict

from configs.ms1mv3_r50_pillar_espn import config as _base_config


config = edict(_base_config.copy())
config.resume = True
