"""Scale-4 PILLAR with only the first residual activation at scale 6.

Diagnostic clipping on both CFP-FP and AgeDB showed that containing
``layer1.0.prelu`` was sufficient to stop the downstream quartic cascade.
This variant minimizes the BatchNorm distribution shift by leaving the stem
and the other 23 activation sites at scale 4.
"""

from easydict import EasyDict as edict

from configs.ms1mv3_r50_pillar_espn_scale4 import config as _base_config


config = edict(_base_config.copy())
config.output = "work_dirs/ms1mv3_r50_pillar_espn_d4_scale4_layer10_scale6"
config.pillar_input_scale_overrides = {
    "layer1.0.prelu": 6.0,
}
