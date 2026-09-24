"""Causally scaled degree-2 conversion of the thirteen NL13 activations.

Each public interval ``S_i`` is calibrated on the graph containing the
already-converted prefix.  Inference evaluates ``S_i * poly(x / S_i)``, which
folds to one quadratic and adds no depth compared with unscaled HerPN.
"""

from easydict import EasyDict as edict

from backbones.iresnet_nl13_prelu_herpn import NL13_ACTIVATION_NAMES


config = edict()
config.margin_list = (1.0, 0.5, 0.0)
config.network = "r50_nl13_prelu_herpn"
config.arch_config = "nl13"
config.resume = False
config.output = "work_dirs/ms1mv3_r50_nl13_prelu_herpn_scaled"
config.embedding_size = 512
config.sample_rate = 1.0
config.fp16 = False
config.batch_size = 128
config.lr = 0.001
config.momentum = 0.9
config.weight_decay = 5e-4
config.selective_weight_decay = True
config.gradient_clip = 1.0
config.gradient_clip_scope = "backbone"
config.warmup_epoch = 1
config.dali = False
config.verbose = 2000
config.tensorboard = False

config.backbone_init = "work_dirs/ms1mv3_r50_nl13/model.pt"
config.embedding_distill_weight = 0.0
config.herpn_initial_progress = 0.0
config.herpn_bn_eps = 1e-4
config.herpn_range_limit = 1.0
config.prelu_herpn_distill_eps = 1e-4
config.prelu_herpn_layerwise_scale = True
config.prelu_herpn_initial_scale = 1.0

# Fit a robust central interval, then reject it if a held-out tail is too far
# outside. The 2x margin gives room beyond the empirical 99.95th percentile.
config.layerwise_poly_range_calibration_batches = 512
config.layerwise_poly_range_margin = 2.0
config.layerwise_poly_min_scale = 1e-3
config.layerwise_poly_range_quantile = 0.9995
config.layerwise_poly_quantile_samples = 65536
config.layerwise_poly_range_holdout_fraction = 0.1
config.layerwise_poly_max_tail_ratio = 6.0
config.layerwise_poly_max_scale_growth = 16.0
config.layerwise_poly_max_input_scale = 1e4
config.layerwise_poly_strict_recalibrate_before_blend = True
config.layerwise_poly_causal_strict_calibration = True
config.layerwise_poly_calibration_log_interval = 128

config.range_augmentation = {
    "enabled": True,
    "probability": 0.70,
    "contrast": (0.45, 1.90),
    "gain": (0.65, 1.40),
    "bias": (-0.15, 0.15),
    "gamma": (0.65, 1.60),
    "noise_probability": 0.35,
    "noise_std": 0.05,
}
config.herpn_range_loss_weight = 0.2
config.herpn_distill_loss_weight = 1.0
config.herpn_bn_recalibration_batches = 300
config.herpn_save_after_group = True
config.herpn_require_full_conversion = True
config.herpn_stage_epochs = ()
config.herpn_conversion_groups = tuple(
    (name,) for name in NL13_ACTIVATION_NAMES)
config.herpn_group_epochs = tuple(
    index + 0.5 for index in range(len(NL13_ACTIVATION_NAMES)))
config.herpn_transition_epochs = 0.5

config.sync_bn = True
config.broadcast_buffers = True
config.ddp_fp16_compress = False
config.check_finite_grads = True
config.fail_on_nonfinite_val = True
config.max_validation_embedding_abs = 1e6
config.save_all_states = True
config.checkpoint_interval_epochs = 1
config.save_epoch_models = True
config.epoch_model_interval = 1

config.rec = "./ms1m-retinaface-t1"
config.num_classes = 93431
config.num_image = 5179510
config.num_epoch = 17
config.val_targets = ["lfw", "cfp_fp", "agedb_30"]
