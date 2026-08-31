"""Replace a ninth PReLU without adding another unstable square.

The source checkpoint has the direct legacy HerPN polynomial at the stem and
all Layer1/Layer2 activation sites (8 of 25 PReLUs). Those eight modules are
preserved exactly. A complete MS1Mv3 eval-mode scan found rare inputs already
reaching 3.43e23 before ``layer4.2.prelu``. Any nonzero quadratic coefficient
at this site can therefore overflow even though the accepted benchmark domain
is finite.

The ninth target is the last activation, ``layer4.2.prelu``. Its student is

    m_c*x + b_c

with learned plaintext channel coefficients. It is locally distilled against
the frozen PReLU before blending. This is a degree-one polynomial, introduces
no encrypted square, and cannot amplify tails quadratically.
"""

from easydict import EasyDict as edict


config = edict()
config.margin_list = (1.0, 0.5, 0.0)
config.network = "r50_prelu_herpn"
config.resume = False
config.output = "work_dirs/ms1mv3_r50_herpn_epoch10_linear9_layer42"
config.embedding_size = 512
config.sample_rate = 1.0
config.fp16 = False
config.momentum = 0.9
config.weight_decay = 5e-4
config.selective_weight_decay = True
config.gradient_clip = 0.5
config.gradient_clip_scope = "backbone"
config.batch_size = 128
config.lr = 3.0e-4
config.verbose = 2000
config.dali = False
config.tensorboard = False

config.backbone_init = "work_dirs/ms1mv3_r50_herpn/model_epoch_10.pt"
config.backbone_init_herpn_progress = 3.0
config.prelu_herpn_legacy_prefix = 8
# Activation index 24 is layer4.2.prelu in iResNet50.
config.prelu_herpn_linear_indices = (24,)
config.prelu_herpn_layerwise_scale = True
config.prelu_herpn_initial_scale = 1.0
config.prelu_herpn_distill_eps = 1e-4
config.herpn_initial_progress = 0.0
config.herpn_bn_eps = 1e-4
config.herpn_range_limit = 1.0

# Preserve the accepted epoch-10 representation while fitting the new local
# student. During the blend/recovery phases the ordinary backbone receives
# only 1% of the already conservative base learning rate.
config.embedding_teacher_network = "r50_no_relu"
config.embedding_teacher_checkpoint = config.backbone_init
config.embedding_distill_weight = 5.0
config.herpn_distill_loss_weight = 1.0
config.herpn_range_loss_weight = 0.0
config.layerwise_poly_staged_training = True
config.layerwise_poly_freeze_backbone_during_local_fit = True
config.layerwise_poly_blend_backbone_lr_scale = 0.01
config.layerwise_poly_final_backbone_lr_scale = 0.01
# The two affine student parameters are the only trainable backbone tensors in
# local fit, so retain the full conservative base LR for them.
config.layerwise_poly_optimizer_lr_scale = 1.0

# The new degree-one site needs no interval calibration. Validation remains
# fail-fast; no clamp or data-dependent branch is added to inference.
config.layerwise_poly_strict_recalibrate_before_blend = False
config.layerwise_poly_causal_strict_calibration = False
config.layerwise_poly_verify_singleton_boundary = False

legacy_prefix = (
    "prelu",
    "layer1.0.prelu", "layer1.1.prelu", "layer1.2.prelu",
    "layer2.0.prelu", "layer2.1.prelu", "layer2.2.prelu",
    "layer2.3.prelu",
)
target = "layer4.2.prelu"
remaining = (
    *(f"layer3.{index}.prelu" for index in range(14)),
    "layer4.0.prelu", "layer4.1.prelu",
)
config.herpn_conversion_groups = tuple(
    (name,) for name in (*legacy_prefix, target, *remaining))
config.herpn_group_epochs = (
    -16.0, -14.0, -12.0, -10.0, -8.0, -6.0, -4.0, -2.0,
    1.0,
    *tuple(100.0 + 2.0 * index for index in range(len(remaining))),
)
config.herpn_transition_epochs = 2.0
config.herpn_require_full_conversion = False
config.layerwise_poly_allow_selective_order = True
config.layerwise_poly_training_group_limit = 9
config.herpn_bn_recalibration_batches = 500
config.herpn_save_after_group = True

config.sync_bn = True
config.broadcast_buffers = True
config.ddp_fp16_compress = False
config.check_finite_grads = True
config.fail_on_nonfinite_val = True
# The unchanged epoch-10 graph spans extreme but finite validation tails (at
# least 1.90e15 on AgeDB). A guessed absolute cutoff rejects the source model
# before the replacement is active. Keep elementwise NaN/Inf fail-fast and log
# maxima for blend-zero versus blended comparison instead.
config.max_validation_embedding_abs = None
config.save_all_states = True
config.checkpoint_interval_epochs = 1
config.save_epoch_models = True
config.epoch_model_interval = 1

config.rec = "./ms1m-retinaface-t1"
config.num_classes = 93431
config.num_image = 5179510
config.num_epoch = 8
config.warmup_epoch = 1
config.val_targets = ["lfw", "cfp_fp", "agedb_30"]
