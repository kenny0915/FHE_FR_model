"""Warm-start targeted NL13 tail conditioning from accepted group 3.

The source snapshot contains the fully polynomial stem, layer1.0 and
layer1.2 activations with recalibrated BatchNorm statistics.  A fresh optimizer
is intentional because staged conditioning changes optimizer parameter groups
and therefore cannot safely consume the earlier full-state optimizer.
"""

from easydict import EasyDict as edict

from configs.ms1mv3_r50_nl13_prelu_herpn_scaled_resume import (
    config as recovery_config,
)


config = edict(recovery_config.copy())
config.resume = False
config.output = "work_dirs/ms1mv3_r50_nl13_prelu_herpn_scaled_recover_group03"
config.backbone_init = (
    "work_dirs/ms1mv3_r50_nl13_prelu_herpn_scaled/"
    "model_herpn_group_03_bnrecalibrated.pt")

# NL13 stage progress 2 restores the stem and both retained layer1
# activations. The first three singleton groups are therefore already done.
config.backbone_init_herpn_progress = 2.0
config.layerwise_poly_initial_calibration_provisional = True

# Give layer2.0 one full local-fit epoch. Every later activation receives one
# local-fit epoch after its predecessor completes, followed by a half-epoch
# blend. Four fully converted epochs remain at the end.
config.herpn_group_epochs = (
    -2.5,
    -1.5,
    -0.5,
    1.0,
    *tuple(3.0 + 1.5 * index for index in range(9)),
)
config.num_epoch = 20
