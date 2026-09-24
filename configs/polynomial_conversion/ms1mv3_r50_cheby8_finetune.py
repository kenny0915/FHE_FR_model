"""Progressively fine-tune degree-8 ChebyReLU R50 on MS1Mv3.

The student starts from the original ``ms1mv3_r50`` PReLU checkpoint, the
fixed degree-8 minimax coefficients used by evaluation, and the 25 public
per-layer scales measured in ``cheby_relu_degree8_layer_scales.json``.

Training-only PReLU branches remain frozen teachers. Each polynomial begins as

    PReLU_a(x) ~= a*x + (1-a)*ChebyReLU8(x; S_i)

on ``[-S_i, S_i]``. Channel-wise residual amplitudes start at zero and are
locally distilled before each ordered group blends from PReLU to polynomial.
After conversion, eight epochs jointly fine-tune the polynomial backbone.
The PReLU branches, blends, losses, and range checks are absent after folding.
"""

from easydict import EasyDict as edict


config = edict()
config.margin_list = (1.0, 0.5, 0.0)
config.network = "r50_layerwise_poly"
config.resume = False
config.output = "work_dirs/ms1mv3_r50_cheby8_finetune"
config.embedding_size = 512
config.sample_rate = 1.0

# Degree-8 powers and the local teacher branch are memory-intensive. FP32
# avoids AMP overflow in x^8; accumulation keeps an effective global batch of
# 512 across four V100s (32 * 4 * 4).
config.fp16 = False
config.batch_size = 32
config.gradient_acc = 4
config.normalize_gradient_accumulation = True
config.lr = 0.001
config.momentum = 0.9
config.weight_decay = 5e-4
config.selective_weight_decay = True
config.gradient_clip = 0.25
config.check_finite_grads = True
config.ddp_fp16_compress = False

config.backbone_init = "work_dirs/ms1mv3_r50/model.pt"
config.layerwise_poly_scale_file = (
    "configs/scales/ms1mv3_r50_cheby_degree8_layer_scales.json")
config.herpn_initial_progress = 0.0
config.layerwise_poly_degree = 8
config.layerwise_poly_initial_scale = 1.0
config.layerwise_poly_distill_eps = 1e-4

# All public scales are loaded from the supplied JSON. These settings remain
# active as validation defaults if a future experiment omits that file.
config.layerwise_poly_range_calibration_batches = 0
config.layerwise_poly_range_margin = 2.0
config.layerwise_poly_min_scale = 1e-3
config.layerwise_poly_range_quantile = 0.999
config.layerwise_poly_quantile_samples = 65536
config.layerwise_poly_range_holdout_fraction = 0.05
config.layerwise_poly_max_tail_ratio = 8.0
config.layerwise_poly_max_scale_growth = 32.0
config.layerwise_poly_max_input_scale = 1e5

# Local activation fitting dominates while a group is still PReLU. During its
# blend and the final polynomial-only phase, update the pretrained backbone at
# one tenth of the classifier/polynomial learning rate.
config.herpn_range_loss_weight = 0.02
config.herpn_distill_loss_weight = 2.0
config.herpn_bn_recalibration_batches = 500
config.herpn_save_after_group = True
config.herpn_require_full_conversion = True
config.herpn_stage_epochs = ()
config.layerwise_poly_staged_training = True
config.layerwise_poly_freeze_backbone_during_local_fit = True
config.layerwise_poly_blend_backbone_lr_scale = 0.1
config.layerwise_poly_final_backbone_lr_scale = 0.1

# Early stages convert in groups of four. The sensitive 14-block third stage
# converts in pairs, and each final-stage activation converts alone. Every
# group gets one local-fit epoch and one one-epoch blend.
config.herpn_conversion_groups = (
    (
        "prelu",
        "layer1.0.prelu",
        "layer1.1.prelu",
        "layer1.2.prelu",
    ),
    (
        "layer2.0.prelu",
        "layer2.1.prelu",
        "layer2.2.prelu",
        "layer2.3.prelu",
    ),
    ("layer3.0.prelu", "layer3.1.prelu"),
    ("layer3.2.prelu", "layer3.3.prelu"),
    ("layer3.4.prelu", "layer3.5.prelu"),
    ("layer3.6.prelu", "layer3.7.prelu"),
    ("layer3.8.prelu", "layer3.9.prelu"),
    ("layer3.10.prelu", "layer3.11.prelu"),
    ("layer3.12.prelu", "layer3.13.prelu"),
    ("layer4.0.prelu",),
    ("layer4.1.prelu",),
    ("layer4.2.prelu",),
)
config.herpn_group_epochs = tuple(range(1, 24, 2))
config.herpn_transition_epochs = 1.0

config.sync_bn = True
config.broadcast_buffers = True
config.fail_on_nonfinite_val = True
config.max_nonfinite_embedding_skips = 0
config.save_all_states = True
config.checkpoint_interval_epochs = 1
config.save_epoch_models = True
config.epoch_model_interval = 1

config.rec = "./ms1m-retinaface-t1"
config.num_classes = 93431
config.num_image = 5179510
config.num_epoch = 32
config.warmup_epoch = 1
config.verbose = 2000
config.dali = False
config.num_workers = 8
config.val_targets = ["lfw", "cfp_fp", "agedb_30"]
