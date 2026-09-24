"""PReLU-aware degree-2 HerPN with per-activation input scaling.

For activation ``i`` with calibrated public scale ``S_i``, the ReLU branch is
evaluated as

    S_i * HerPN_ReLU(x / S_i),  x in [-S_i, S_i].

The Hermite target interval is therefore ``[-1, 1]`` at every activation.
After BatchNorm folding the result is still one channel-wise degree-2
polynomial in ``x`` and adds no multiplicative depth relative to the original
HerPN experiment.

The training-only range augmentation deliberately samples strong but bounded
exposure, contrast, gamma, and high-frequency noise. It is not part of the
encrypted inference graph.
"""

from easydict import EasyDict as edict


config = edict()
config.margin_list = (1.0, 0.5, 0.0)
config.network = "r50_prelu_herpn"
config.resume = False
config.output = (
    "work_dirs/ms1mv3_r50_prelu_herpn_layerwise_scale_range_aug_one_epoch")
config.embedding_size = 512
config.sample_rate = 1.0
config.fp16 = False
config.momentum = 0.9
config.weight_decay = 5e-4
config.selective_weight_decay = True
config.gradient_clip = 0.5
config.batch_size = 128
config.lr = 0.005
config.verbose = 2000
config.dali = False

# Blend zero remains exactly the pretrained PReLU model. The frozen PReLU
# slopes are channel-wise plaintext constants in the final polynomial.
config.backbone_init = "work_dirs/ms1mv3_r50/model.pt"
config.herpn_initial_progress = 0.0
config.herpn_bn_eps = 1e-4
config.herpn_range_limit = 1.0
config.prelu_herpn_distill_eps = 1e-4
config.prelu_herpn_layerwise_scale = True
config.prelu_herpn_initial_scale = 1.0

# Robust, augmented-data interval calibration. Each scale is measured on the
# graph containing the already-converted prefix, then rechecked immediately
# before its blend begins. The 2x margin gives the polynomial room beyond the
# fitted 99.95th percentile while explicit tail checks reject unsafe fits.
config.layerwise_poly_range_calibration_batches = 2048
config.layerwise_poly_range_margin = 2.0
config.layerwise_poly_min_scale = 1e-3
config.layerwise_poly_range_quantile = 0.9995
config.layerwise_poly_quantile_samples = 65536
config.layerwise_poly_range_holdout_fraction = 0.1
config.layerwise_poly_max_tail_ratio = 6.0
config.layerwise_poly_max_scale_growth = 16.0
config.layerwise_poly_max_input_scale = 1e4
config.layerwise_poly_strict_recalibrate_before_blend = True
config.layerwise_poly_calibration_log_interval = 256

# Stronger input-range coverage than horizontal flip alone. All operations
# stay bounded in [0, 1] before the usual normalization to [-1, 1].
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
config.herpn_bn_recalibration_batches = 200
config.herpn_save_after_group = True
config.herpn_require_full_conversion = True
config.herpn_stage_epochs = ()

# Strict forward order is required: the scale for the next activation is
# measured only after the previous polynomial and downstream BN refresh are
# committed. IResNet50 contains 25 PReLU sites.
config.herpn_conversion_groups = (
    ("prelu",),
    ("layer1.0.prelu",),
    ("layer1.1.prelu",),
    ("layer1.2.prelu",),
    ("layer2.0.prelu",),
    ("layer2.1.prelu",),
    ("layer2.2.prelu",),
    ("layer2.3.prelu",),
    ("layer3.0.prelu",),
    ("layer3.1.prelu",),
    ("layer3.2.prelu",),
    ("layer3.3.prelu",),
    ("layer3.4.prelu",),
    ("layer3.5.prelu",),
    ("layer3.6.prelu",),
    ("layer3.7.prelu",),
    ("layer3.8.prelu",),
    ("layer3.9.prelu",),
    ("layer3.10.prelu",),
    ("layer3.11.prelu",),
    ("layer3.12.prelu",),
    ("layer3.13.prelu",),
    ("layer4.0.prelu",),
    ("layer4.1.prelu",),
    ("layer4.2.prelu",),
)

# Each activation now occupies one epoch: local fit at blend zero during the
# first half, then a linear blend during the second half. Group i completes at
# the next integer epoch boundary, where BN refresh and calibration of group
# i+1 run before its local-fit half. Four fully converted epochs finish the run.
config.herpn_group_epochs = tuple(index + 0.5 for index in range(25))
config.herpn_transition_epochs = 0.5

config.sync_bn = True
config.broadcast_buffers = True
config.check_finite_grads = True
config.ddp_fp16_compress = False
config.fail_on_nonfinite_val = True
config.save_all_states = True
config.checkpoint_interval_epochs = 1
config.save_epoch_models = True
config.epoch_model_interval = 1

config.rec = "./ms1m-retinaface-t1"
config.num_classes = 93431
config.num_image = 5179510
config.num_epoch = 29
config.warmup_epoch = 1
config.val_targets = ["lfw", "cfp_fp", "agedb_30"]
