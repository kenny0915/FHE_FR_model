"""Resume the stable frozen-std run from its clean epoch checkpoint.

Use this after the former auxiliary-loss configuration failed during epoch 1.
The epoch-1 checkpoint is at global step 10,116, immediately before the first
frozen-std switch at step 10,117, so it contains the accepted exact-LayerNorm
model and can safely continue with the auxiliary objective disabled.  If that
legacy checkpoint contains an infinite EMA from the old centered-square
collector, the first resumed finite batch now replaces it with the stable RMS
observation before any affected group is frozen.
"""

from easydict import EasyDict as edict

from configs.ms1mv3_poolformer_s24_fully_gated_frozen_std_fp32 import (
    config as stable_config,
)


config = edict(stable_config.copy())
config.resume = True
