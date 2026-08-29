"""Resume causal NL13 conversion with targeted tail conditioning.

The first three converted activations remained numerically bounded.  The
fourth activation had a single observed tail 6.72 times the robust 99.95th
percentile interval.  Permit tails up to 8, condition that pending activation
for 1.5 epochs, then floor its strict interval from the observed tail before
blending.  The floor is capped at 2x the robust interval so a genuinely broad
runaway distribution still fails closed.
"""

from easydict import EasyDict as edict

from configs.ms1mv3_r50_nl13_prelu_herpn_scaled import config as base_config


config = edict(base_config.copy())
config.resume = True
config.layerwise_poly_max_tail_ratio = 8.0
config.layerwise_poly_staged_training = True
config.layerwise_poly_freeze_backbone_during_local_fit = False
config.layerwise_poly_allow_provisional_tail_conditioning = True
config.layerwise_poly_conditioning_backbone_lr_scale = 0.01
config.layerwise_poly_conditioning_range_loss_weight = 1.0
config.layerwise_poly_strict_tail_scale_floor = True
config.layerwise_poly_tail_scale_floor_margin = 1.1
config.layerwise_poly_max_tail_scale_expansion = 2.0
config.layerwise_poly_blend_backbone_lr_scale = 0.1
config.layerwise_poly_final_backbone_lr_scale = 0.1

# Preserve the completed prefix schedule. Give layer2.0 one extra conditioning
# epoch, then retain the original half-local-fit/half-blend cadence.
config.herpn_group_epochs = (0.5, 1.5, 2.5) + tuple(
    index + 4.5 for index in range(10))
config.num_epoch = 18
