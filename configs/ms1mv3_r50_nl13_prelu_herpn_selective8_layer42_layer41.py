"""Add ``layer4.1.prelu`` after selectively converting ``layer4.2.prelu``.

The source is the BN-recalibrated seven-polynomial boundary from the stronger
terminal-activation experiment.  The eighth target is its frozen channel-wise
PReLU on a causally calibrated public interval ``[-S, S]``.  Its inference
graph remains one folded degree-2 polynomial and adds one square level.
"""

from easydict import EasyDict as edict

from backbones.iresnet_nl13_prelu_herpn import NL13_ACTIVATION_NAMES
from configs.ms1mv3_r50_nl13_prelu_herpn_selective7_layer42 import (
    config as _layer42_config,
)


config = edict(_layer42_config.copy())
config.output = (
    "work_dirs/"
    "ms1mv3_r50_nl13_prelu_herpn_selective8_layer42_layer41")
config.backbone_init = (
    "work_dirs/ms1mv3_r50_nl13_prelu_herpn_selective7_layer42/"
    "model_herpn_group_07_bnrecalibrated.pt")
config.backbone_init_herpn_progress = 0.0

prefix = NL13_ACTIVATION_NAMES[:6]
first_target = "layer4.2.prelu"
target = "layer4.1.prelu"
remainder = tuple(
    name for name in NL13_ACTIVATION_NAMES[6:]
    if name not in (first_target, target))
config.herpn_conversion_groups = tuple((name,) for name in (
    *prefix, first_target, target, *remainder))
config.herpn_group_epochs = (
    -14.0, -12.0, -10.0, -8.0, -6.0, -4.0, -2.0,
    1.0,
    100.0, 102.0, 104.0, 106.0, 108.0,
)
config.layerwise_poly_training_group_limit = 8
config.num_epoch = 8
