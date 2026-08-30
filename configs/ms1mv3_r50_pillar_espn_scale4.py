"""Uniform scale-4 PILLAR polynomial for robust all-polynomial inference.

Every activation uses ``4 * q(x / 4)``.  The target is ReLU on [-20, 20],
with polynomial degree 4 and multiplicative depth 2.
"""

from easydict import EasyDict as edict

from configs.ms1mv3_r50_pillar_espn import config as _base_config


config = edict(_base_config.copy())
config.output = "work_dirs/ms1mv3_r50_pillar_espn_d4_scale4"
config.resume = True
config.pillar_input_scale = 4.0
config.pillar_input_scale_overrides = {}
# The source checkpoint's BatchNorm statistics describe scale 1.  Permit
# three full adaptation epochs (7--9) before evaluating the scale-4 graph.
config.pillar_skip_verification_epochs = 10
config.pillar_strict_verification_epoch = 14
