"""Widen the two empirically unstable early PILLAR activation intervals.

The base polynomial q approximates ReLU on [-5, 5].  These two sites use
``4 * q(x / 4)``, which approximates ReLU on [-20, 20] while retaining degree
4 and multiplicative depth 2.  All other 23 activation sites remain q(x).
"""

from easydict import EasyDict as edict

from configs.ms1mv3_r50_pillar_espn import config as _base_config


config = edict(_base_config.copy())
config.output = "work_dirs/ms1mv3_r50_pillar_espn_d4_scale4_early"
config.resume = True
config.pillar_input_scale = 1.0
config.pillar_input_scale_overrides = {
    "layer1.1.prelu": 4.0,
    "layer1.2.prelu": 4.0,
}
