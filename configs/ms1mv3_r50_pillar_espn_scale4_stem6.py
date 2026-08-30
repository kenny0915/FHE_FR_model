"""Scale-4 PILLAR with wider intervals at the two causal input sites.

The stem and first residual activation use ``6 * q(x / 6)`` and target ReLU
on [-30, 30].  The remaining 23 activations use ``4 * q(x / 4)`` and target
ReLU on [-20, 20].  Every site remains degree 4, depth 2.
"""

from easydict import EasyDict as edict

from configs.ms1mv3_r50_pillar_espn_scale4 import config as _base_config


config = edict(_base_config.copy())
config.output = "work_dirs/ms1mv3_r50_pillar_espn_d4_scale4_stem6"
config.pillar_input_scale_overrides = {
    "prelu": 6.0,
    "layer1.0.prelu": 6.0,
}
