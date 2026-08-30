"""Add a selective seventh quadratic at the final NL13 activation.

The accepted source has a degree-2 polynomial prefix through
``layer3.0.prelu``. This run leaves ``layer3.3--layer4.1`` as PReLUs and
converts ``layer4.2.prelu`` on their actual output distribution. Its target is
the frozen channel-wise PReLU on the causally calibrated public interval
``[-S, S]``; the folded activation remains one quadratic / depth-one square.
"""

from easydict import EasyDict as edict

from backbones.iresnet_nl13_prelu_herpn import NL13_ACTIVATION_NAMES
from configs.ms1mv3_r50_nl13_prelu_herpn_scaled_group06_accuracy_recovery import (
    config as _group6_config,
)


config = edict(_group6_config.copy())
config.resume = False
config.output = (
    "work_dirs/ms1mv3_r50_nl13_prelu_herpn_selective7_layer42")
config.backbone_init = (
    "work_dirs/ms1mv3_r50_nl13_prelu_herpn_scaled_group06_accuracy_recovery/"
    "model_group06_ijbc_92p69.pt")
config.backbone_init_herpn_progress = 0.0

prefix = NL13_ACTIVATION_NAMES[:6]
target = "layer4.2.prelu"
remainder = tuple(
    name for name in NL13_ACTIVATION_NAMES[6:] if name != target)
config.herpn_conversion_groups = tuple((name,) for name in (
    *prefix, target, *remainder))
config.herpn_group_epochs = (
    -12.0, -10.0, -8.0, -6.0, -4.0, -2.0,
    1.0,
    100.0, 102.0, 104.0, 106.0, 108.0, 110.0,
)
config.herpn_transition_epochs = 2.0
config.herpn_require_full_conversion = False
config.layerwise_poly_allow_selective_order = True
config.layerwise_poly_training_group_limit = 7
config.layerwise_poly_max_tail_scale_expansion = 5.1
# The measured public interval already contains the late-stage distribution.
# Keep a numerically positive value for the provisional-schedule invariant,
# but let the frozen local PReLU fit and recognition losses drive adaptation;
# even a 0.01 relative-tail weight was dominated by very rare extreme values.
config.layerwise_poly_conditioning_range_loss_weight = 1.0e-8
# The normalized polynomial coefficients are dimensionless, but their output
# is multiplied by the calibrated S (about 6e4--8e4 here).  Slow this optimizer
# group so one coefficient step cannot create an S-amplified output jump.
config.layerwise_poly_optimizer_lr_scale = 1.0e-4
# Late NL13 PReLUs safely receive large finite values because their teacher is
# linear outside zero.  Preserve those measured values through public input
# normalization instead of forcing an unsafe small quadratic interval.
config.layerwise_poly_max_input_scale = 1.0e6
config.num_epoch = 8
config.fail_on_nonfinite_val = True
