"""Convert all three layer-4 PReLUs after the accepted six-polynomial prefix.

The source is the BN-recalibrated eight-polynomial boundary ordered as
``layer4.2`` then ``layer4.1``.  This run adds ``layer4.0.prelu`` as the ninth
quadratic.  It targets the frozen channel-wise PReLU on a causally calibrated
public ``[-S, S]`` interval and retains degree 2 / one square level per site.
"""

from easydict import EasyDict as edict

from backbones.iresnet_nl13_prelu_herpn import NL13_ACTIVATION_NAMES
from configs.ms1mv3_r50_nl13_prelu_herpn_selective8_layer42_layer41 import (
    config as _layer42_layer41_config,
)


config = edict(_layer42_layer41_config.copy())
config.output = (
    "work_dirs/"
    "ms1mv3_r50_nl13_prelu_herpn_selective9_layer42_layer41_layer40")
config.backbone_init = (
    "work_dirs/"
    "ms1mv3_r50_nl13_prelu_herpn_selective8_layer42_layer41/"
    "model_herpn_group_08_bnrecalibrated.pt")
config.backbone_init_herpn_progress = 0.0

prefix = NL13_ACTIVATION_NAMES[:6]
converted_targets = ("layer4.2.prelu", "layer4.1.prelu")
target = "layer4.0.prelu"
remainder = tuple(
    name for name in NL13_ACTIVATION_NAMES[6:]
    if name not in (*converted_targets, target))
config.herpn_conversion_groups = tuple((name,) for name in (
    *prefix, *converted_targets, target, *remainder))
config.herpn_group_epochs = (
    -16.0, -14.0, -12.0, -10.0, -8.0, -6.0, -4.0, -2.0,
    1.0,
    100.0, 102.0, 104.0, 106.0,
)
config.layerwise_poly_training_group_limit = 9
config.num_epoch = 8
