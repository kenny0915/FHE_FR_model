"""Selective seventh quadratic at ``layer4.1.prelu``.

This is the matched alternative to the final-block ``layer4.2`` experiment.
It uses the same accepted six-polynomial source, degree-2 PReLU target,
causal public interval, two-epoch blend, and fixed-seven recovery schedule.
"""

from easydict import EasyDict as edict

from backbones.iresnet_nl13_prelu_herpn import NL13_ACTIVATION_NAMES
from configs.ms1mv3_r50_nl13_prelu_herpn_selective7_layer42 import (
    config as _layer42_config,
)


config = edict(_layer42_config.copy())
config.output = (
    "work_dirs/ms1mv3_r50_nl13_prelu_herpn_selective7_layer41")
prefix = NL13_ACTIVATION_NAMES[:6]
target = "layer4.1.prelu"
remainder = tuple(
    name for name in NL13_ACTIVATION_NAMES[6:] if name != target)
config.herpn_conversion_groups = tuple((name,) for name in (
    *prefix, target, *remainder))
