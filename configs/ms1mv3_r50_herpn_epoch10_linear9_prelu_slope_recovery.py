"""Recover low-FAR accuracy with a fixed, tail-safe ninth activation.

The source checkpoint fully replaces ``layer4.2.prelu`` by ``a_c*x``, where
``a_c`` is the frozen PReLU negative-branch slope.  Full IJB-C inference is
finite, but its static TAR@FAR=1e-4 is 93.61%.  Keep those coefficients fixed
and adapt only the ordinary backbone/embedding with an epoch-10 embedding
teacher.  The eight legacy quadratic activations and every BatchNorm running
buffer remain fixed to limit movement of the known extreme tails.
"""

from easydict import EasyDict as edict

from configs.ms1mv3_r50_herpn_epoch10_selective9_layer42 import (
    config as base_config,
)


config = edict(base_config.copy())
config.resume = False
config.output = (
    "work_dirs/ms1mv3_r50_herpn_epoch10_linear9_prelu_slope_recovery")
config.backbone_init = (
    "work_dirs/ms1mv3_r50_herpn_epoch10_linear9_layer42/"
    "model_linear9_prelu_slope_static.pt")
config.backbone_init_herpn_progress = 0.0  # Checkpoint blends take precedence.
config.prelu_herpn_linear_trainable = False

# All nine selected activations are already fully converted at epoch zero.
# The remaining sixteen PReLUs stay outside this recovery run.
config.herpn_group_epochs = (
    -18.0, -16.0, -14.0, -12.0, -10.0, -8.0, -6.0, -4.0, -2.0,
    *tuple(100.0 + 2.0 * index for index in range(16)),
)
config.herpn_transition_epochs = 2.0
config.layerwise_poly_training_group_limit = 9

# The fixed affine has no local approximation objective.  Use a conservative
# 3e-5 effective backbone LR and retain the epoch-10 embedding geometry.
config.herpn_distill_loss_weight = 0.0
config.embedding_distill_weight = 5.0
config.layerwise_poly_final_backbone_lr_scale = 0.1
config.num_epoch = 4
config.warmup_epoch = 0
config.verbose = 1000

