"""Recover group 8 after the first full-epoch schedule reached group 7.

The epoch-7 distributed state contains six recorded completed groups and a
fully blended seventh group; startup repeats group-7 BN recalibration and then
calibrates ``layer3.6.prelu``.  That pending activation receives two complete
conditioning epochs.  Its observed tail required a 2.862x robust-interval
expansion, so this recovery raises only that generic safety cap from 2x to 3x.
The absolute scale and same-stage growth guards remain unchanged.
"""

from easydict import EasyDict as edict

from configs.ms1mv3_r50_nl13_prelu_herpn_scaled_recover_group03_resume import (
    config as resume_config,
)


config = edict(resume_config.copy())
config.layerwise_poly_max_tail_scale_expansion = 3.0

# Preserve starts already represented by the checkpoint. Delay group 8 from
# 7.5 to 9.0; use two-epoch spacing thereafter so calibration at an integer
# checkpoint boundary is followed by a full local-fit epoch.
config.herpn_group_epochs = (
    -2.5,
    -1.5,
    -0.5,
    1.0,
    3.0,
    4.5,
    6.0,
    9.0,
    11.0,
    13.0,
    15.0,
    17.0,
    19.0,
)
config.num_epoch = 24
