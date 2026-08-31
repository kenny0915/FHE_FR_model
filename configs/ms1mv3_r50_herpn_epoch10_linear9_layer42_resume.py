"""Resume the accepted epoch-0 affine fit with all BN buffers preserved.

The rank checkpoints in ``config.output`` were written at global step 5058,
before the first nonzero blend update. They contain the fitted degree-one
student, optimizer/PartialFC state, and the unchanged epoch-10 BatchNorm
buffers. The canceled exploratory blend is not part of those checkpoints.
"""

from easydict import EasyDict as edict

from configs.ms1mv3_r50_herpn_epoch10_selective9_layer42 import (
    config as base_config,
)


config = edict(base_config.copy())
config.resume = True
