"""Fast recovery after the unsafe simultaneous group-2 conversion.

The accepted group-1 BatchNorm-recalibrated checkpoint is used as a warm
start.  Group 1 is represented as already complete at epoch zero; the
remaining six groups retain the original four-activation schedule.  Strict
calibration is causal within each group, so every downstream interval observes
the actual polynomial prefix instead of the all-PReLU teacher graph.  A final
full-group representative pass rejects finite but catastrophic cascades.

The degree-2 target remains each frozen channel-wise PReLU on its measured
``[-S_i, S_i]`` interval.  Causal calibration changes only plaintext training
and does not add operations or multiplicative depth to FHE inference.
"""

from easydict import EasyDict as edict

from configs.ms1mv3_r50_layerwise_poly_group4 import config as base_config


config = edict(base_config.copy())
config.resume = False
config.output = (
    "work_dirs/ms1mv3_r50_layerwise_poly_group4_d2_causal_recovery")
config.backbone_init = (
    "work_dirs/ms1mv3_r50_layerwise_poly_group4_d2/"
    "model_herpn_group_01_bnrecalibrated.pt")

# The checkpoint contains the stem/layer1 group fully converted.  Progress 2.0
# preserves exactly those stages while the negative schedule entry records
# that their transition completed before this recovery run starts.
config.herpn_initial_progress = 0.0
config.backbone_init_herpn_progress = 2.0
config.herpn_group_epochs = (-1, 1, 3, 5, 7, 9, 11)

# The final group completes at epoch 12, followed by six joint-tuning epochs.
config.num_epoch = 18
