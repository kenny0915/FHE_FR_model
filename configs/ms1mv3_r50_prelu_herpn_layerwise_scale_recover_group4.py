"""Recover the singleton HerPN conversion from the accepted group-4 graph.

The source artifact is a backbone-only, BatchNorm-recalibrated snapshot, so
this is a warm start rather than a full optimizer/PartialFC resume.  The stem
and three layer1 activations remain fully polynomial.  ``layer2.0.prelu`` is
calibrated provisionally at blend zero, receives one full epoch of upstream
range conditioning, and must pass a fresh strict calibration before blending.

The approximation target remains the frozen channel-wise PReLU on each public
``[-S_i, S_i]`` interval.  Conditioning, interval fitting, and tail checks are
plaintext-only training operations and do not change FHE multiplicative depth.
"""

from easydict import EasyDict as edict

from configs.ms1mv3_r50_prelu_herpn_layerwise_scale import (
    config as base_config,
)


config = edict(base_config.copy())
config.resume = False
config.output = (
    "work_dirs/"
    "ms1mv3_r50_prelu_herpn_layerwise_scale_range_aug_recover_group4")
config.backbone_init = (
    "work_dirs/"
    "ms1mv3_r50_prelu_herpn_layerwise_scale_range_aug_one_epoch/"
    "model_herpn_group_04_bnrecalibrated.pt")

# The checkpoint contains the complete stem and layer1 polynomial prefix.  The
# coarse progress setter restores exactly those two architectural stages after
# the strict checkpoint load; later activations remain PReLU at blend zero.
config.backbone_init_herpn_progress = 2.0

# The failed layer2.0 interval had a representative tail ratio of 17.12.  Keep
# it provisional while blend is zero so the pending layer's range loss can
# condition upstream convolution/BatchNorm parameters.  A strict pass at the
# blend boundary may floor S toward the observed tail, but an increase beyond
# 2x the robust scale is still rejected as a broad runaway distribution.
config.layerwise_poly_initial_calibration_provisional = True
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

# Groups 1-4 completed before recovery epoch zero.  Give layer2.0 one full
# conditioning epoch before its half-epoch blend.  Later singleton groups keep
# the original half-epoch local-fit and half-epoch blend cadence; their starts
# remain half an epoch after the previous integer completion boundary so BN
# refresh and provisional calibration happen first.
config.herpn_group_epochs = (
    -3.5,
    -2.5,
    -1.5,
    -0.5,
    1.0,
    *tuple(index + 2.5 for index in range(20)),
)

# The last activation completes at recovery epoch 22.  Retain four fully
# converted epochs for joint fine-tuning, matching the original experiment.
config.num_epoch = 26
