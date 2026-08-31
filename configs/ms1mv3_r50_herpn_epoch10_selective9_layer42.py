"""Add one safe quadratic to the known-finite eight-HerPN epoch-10 graph.

The source checkpoint has the direct legacy HerPN polynomial at the stem and
all Layer1/Layer2 activation sites (8 of 25 PReLUs). Those eight modules are
preserved exactly. The ninth target is the terminal ``layer4.2.prelu`` because
it cannot feed another polynomial site and therefore minimizes runaway-square
cascade risk.

For its frozen channel-wise PReLU slope ``a`` and calibrated public scale
``S``, the new student targets

    PReLU_a(x), x in [-S, S]

with ``a*x + (1-a)*S*HerPN_ReLU(x/S)``. ``S`` is 1.5 times the largest input
seen across the complete natural MS1Mv3 range-calibration shards and is checked
again at the blend boundary. The folded inference activation is still
``A*x^2+B*x+C``: degree 2 and one ciphertext square at this site.
"""

from easydict import EasyDict as edict


config = edict()
config.margin_list = (1.0, 0.5, 0.0)
config.network = "r50_prelu_herpn"
config.resume = False
config.output = "work_dirs/ms1mv3_r50_herpn_epoch10_selective9_layer42_natural"
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
config.herpn_range_loss_weight = 0.2
config.layerwise_poly_staged_training = True
config.layerwise_poly_freeze_backbone_during_local_fit = True
config.layerwise_poly_blend_backbone_lr_scale = 0.01
config.layerwise_poly_final_backbone_lr_scale = 0.01
config.layerwise_poly_optimizer_lr_scale = 0.01

# Approximation interval: [-S, S], with S fitted to the maximum rather than a
# central quantile. A full distributed loader pass covers the natural MS1Mv3
# training distribution. The earlier stress-augmentation attempt was rejected
# by this same fail-fast pass: the accepted epoch-10 graph overflowed before
# layer4.2 on an augmented input, before the ninth polynomial was enabled.
# Therefore stress samples are not a valid calibration domain for this fixed
# baseline. No clipping or other non-polynomial guard enters inference.
config.layerwise_poly_range_calibration_batches = 0
config.layerwise_poly_range_margin = 1.5
config.layerwise_poly_min_scale = 1e-3
config.layerwise_poly_range_quantile = 1.0
config.layerwise_poly_quantile_samples = 65536
config.layerwise_poly_range_holdout_fraction = 0.1
config.layerwise_poly_max_tail_ratio = 1.0
config.layerwise_poly_max_scale_growth = 0.0
config.layerwise_poly_max_input_scale = 1.0e3
config.layerwise_poly_strict_recalibrate_before_blend = True
config.layerwise_poly_causal_strict_calibration = True
config.layerwise_poly_verify_singleton_boundary = True
config.layerwise_poly_calibration_log_interval = 128

config.range_augmentation = {"enabled": False}

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
config.max_validation_embedding_abs = 1.0e3
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
