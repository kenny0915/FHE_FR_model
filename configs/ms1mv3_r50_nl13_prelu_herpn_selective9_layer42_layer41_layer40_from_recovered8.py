"""Add a ninth quadratic to the stronger recovered-source eighth model.

Unlike the boundary-seeded control, this source converted ``layer4.1`` from
the screened selective-seven epoch and then completed fixed-graph recovery.
This run converts ``layer4.0.prelu`` on a newly calibrated public ``[-S, S]``
interval.  The added inference activation is degree 2 and costs one square
level.
"""

from easydict import EasyDict as edict

from configs.ms1mv3_r50_nl13_prelu_herpn_selective9_layer42_layer41_layer40_recovered import (
    config as _control_config,
)


config = edict(_control_config.copy())
config.output = (
    "work_dirs/"
    "ms1mv3_r50_nl13_prelu_herpn_selective9_layer42_layer41_layer40_from_recovered8")
config.backbone_init = (
    "work_dirs/"
    "ms1mv3_r50_nl13_prelu_herpn_selective8_layer42_layer41_recovered/"
    "model.pt")
