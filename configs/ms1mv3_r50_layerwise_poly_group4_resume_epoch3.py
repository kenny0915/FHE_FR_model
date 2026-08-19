"""Resume the corrected group-4 conversion from full epoch-3 checkpoints.

Place ``checkpoint_gpu_0.pt`` through ``checkpoint_gpu_3.pt`` in the output
directory configured by the base experiment. The checkpoints contain epoch 3,
optimizer, scheduler, PartialFC, and per-rank state, so training continues at
the group-1 completion boundary without replaying earlier epochs.
"""

from easydict import EasyDict as edict

from configs.ms1mv3_r50_layerwise_poly_group4 import config as base_config


config = edict(base_config.copy())
config.resume = True
