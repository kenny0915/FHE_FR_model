"""Recover the finite layer3.9 ninth replacement by suffix distillation."""

from easydict import EasyDict as edict

from configs.ms1mv3_r50_herpn_epoch10_selective9_layer42 import (
    config as base_config,
)


config = edict(base_config.copy())
config.resume = False
config.output = (
    "work_dirs/ms1mv3_r50_herpn_epoch10_linear9_layer3_9_recovery")
config.backbone_init = (
    "work_dirs/ms1mv3_r50_herpn_epoch10_linear_sites/"
    "model_linear9_layer3_9.pt")
config.backbone_init_herpn_progress = 0.0
config.prelu_herpn_linear_indices = (17,)
config.prelu_herpn_linear_trainable = False

legacy_prefix = (
    "prelu",
    "layer1.0.prelu", "layer1.1.prelu", "layer1.2.prelu",
    "layer2.0.prelu", "layer2.1.prelu", "layer2.2.prelu",
    "layer2.3.prelu",
)
target = "layer3.9.prelu"
remaining = tuple(
    name for name in (
        *(f"layer3.{index}.prelu" for index in range(14)),
        "layer4.0.prelu", "layer4.1.prelu", "layer4.2.prelu",
    ) if name != target
)
config.herpn_conversion_groups = tuple(
    (name,) for name in (*legacy_prefix, target, *remaining))
config.herpn_group_epochs = (
    -18.0, -16.0, -14.0, -12.0, -10.0, -8.0, -6.0, -4.0, -2.0,
    *tuple(100.0 + 2.0 * index for index in range(len(remaining))),
)
config.herpn_transition_epochs = 2.0
config.layerwise_poly_training_group_limit = 9
config.layerwise_poly_allow_selective_order = True

# The nine activation functions and the complete prefix through layer3.9 are
# immutable. Reconstruct the epoch-10 embedding using only the residual suffix
# that still has enough capacity to compensate this removed nonlinearity.
config.herpn_distill_loss_weight = 0.0
config.task_loss_weight = 0.0
config.embedding_distill_weight = 1.0
config.backbone_trainable_prefixes = (
    "layer3.9.conv2",
    "layer3.9.bn3",
    "layer3.10",
    "layer3.11",
    "layer3.12",
    "layer3.13",
    "layer4",
    "bn2",
    "fc",
    "features",
)
config.layerwise_poly_final_backbone_lr_scale = 1.0
config.freeze_batchnorm_running_stats = True
config.freeze_batchnorm_affine = False
config.max_nonfinite_embedding_skips = 100
config.num_epoch = 2
config.warmup_epoch = 0
config.verbose = 500

