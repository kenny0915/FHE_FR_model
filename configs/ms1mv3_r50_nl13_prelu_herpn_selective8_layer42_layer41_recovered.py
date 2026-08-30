"""Add an eighth quadratic to the fully recovered selective-seven model.

This is the causal counterpart to the conversion-boundary control.  It starts
from the final recovered ``layer4.2`` model, then converts ``layer4.1.prelu``
while preserving the accepted six-polynomial prefix.  The target is the
frozen channel-wise PReLU on a newly calibrated public ``[-S, S]`` interval;
inference remains a folded degree-2 polynomial with one square level.
"""

from easydict import EasyDict as edict

from configs.ms1mv3_r50_nl13_prelu_herpn_selective8_layer42_layer41 import (
    config as _boundary_config,
)


config = edict(_boundary_config.copy())
config.output = (
    "work_dirs/"
    "ms1mv3_r50_nl13_prelu_herpn_selective8_layer42_layer41_recovered")
config.backbone_init = (
    "work_dirs/ms1mv3_r50_nl13_prelu_herpn_selective7_layer42/model.pt")
