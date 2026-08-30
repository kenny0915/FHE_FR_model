"""Add a ninth quadratic to a fully recovered selective-eight checkpoint.

The source has the accepted six-polynomial prefix plus ``layer4.2`` and
``layer4.1``.  This run converts ``layer4.0.prelu`` after the eight-site graph
has completed fixed-graph recovery.  It targets the frozen channel-wise PReLU
on a causal public ``[-S, S]`` interval and retains degree 2 / one square
level for the added inference activation.
"""

from easydict import EasyDict as edict

from configs.ms1mv3_r50_nl13_prelu_herpn_selective9_layer42_layer41_layer40 import (
    config as _boundary_config,
)


config = edict(_boundary_config.copy())
config.output = (
    "work_dirs/"
    "ms1mv3_r50_nl13_prelu_herpn_selective9_layer42_layer41_layer40_recovered")
config.backbone_init = (
    "work_dirs/ms1mv3_r50_nl13_prelu_herpn_selective8_layer42_layer41/"
    "model.pt")
