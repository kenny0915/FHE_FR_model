"""Continue scale-4 PILLAR training on the exact inference graph.

The range penalty and gradient guard remain training-only, but activation
clipping is disabled.  Training therefore evaluates the same degree-4,
depth-2 polynomial used for inference on every forward pass.
"""

from easydict import EasyDict as edict

from configs.ms1mv3_r50_pillar_espn_scale4 import config as _base_config


config = edict(_base_config.copy())
config.output = "work_dirs/ms1mv3_r50_pillar_espn_d4_scale4_unclipped"
config.pillar_training_clip = False
