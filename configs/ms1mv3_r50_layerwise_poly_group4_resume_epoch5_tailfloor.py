"""Resume epoch 5 and use bounded tail-safe intervals before group-2 blend.

The epoch-5 full checkpoints contain two completed zero-blend conditioning
epochs for group 2. Strict calibration keeps the robust q=0.999 interval unless
the representative maximum requires a larger scale to meet the tail limit.
Any such increase receives a 1.1 safety margin and may not exceed 2x the robust
scale. The group schedule and total epoch count remain unchanged from the
epoch-4 recovery run.
"""

from easydict import EasyDict as edict

from configs.ms1mv3_r50_layerwise_poly_group4_resume_epoch4 import (
    config as resume_epoch4_config,
)


config = edict(resume_epoch4_config.copy())
