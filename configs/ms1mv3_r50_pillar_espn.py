"""All-polynomial iResNet-50 using the released PILLAR-ESPN recipe.

Approximation target and interval:
    ReLU(x) on [-5, 5]

Polynomial (degree 4, multiplicative depth 2):
    0.314453125 + 0.5*x + 0.15625*x^2 - 0.0029296875*x^4

The upstream ImageNet command trains for 600 epochs with an effective batch
of 1024. This is a staged face-recognition adaptation: it preserves the exact
unclipped inference polynomial, unnormalized range penalty, range-only first
epoch, coefficient/range settings, FP32 execution, normalization decay rule,
and local batch size. Face identity labels do not admit ImageNet MixUp/CutMix,
so bounded photometric range stress is used instead.
"""

from easydict import EasyDict as edict


config = edict()
config.margin_list = (1.0, 0.5, 0.0)
config.network = "r50_pillar"
config.resume = False
config.output = "work_dirs/ms1mv3_r50_pillar_espn_d4"
config.embedding_size = 512
config.sample_rate = 1.0

config.fp16 = False
config.optimizer = "sgd"
config.lr = 0.03
config.momentum = 0.9
config.weight_decay = 2e-5
config.selective_weight_decay = True
config.batch_size = 256
# Face iResNet can produce rare internal outliers during ArcFace training.
# The reference ImageNet ResNet does not clip gradients, but a norm cap is
# necessary here to keep one summed range-gradient spike recoverable.
config.gradient_clip = 5.0
config.gradient_norm_warning_threshold = 5.0
config.gradient_norm_warning_interval = 50
config.check_finite_grads = True

config.pillar_approximation_range = 5.0
config.pillar_regularization_range = 4.8
config.pillar_regularization_coefficient = 1e-4
config.pillar_regularization_exponent = 10
config.pillar_training_clip = True
config.pillar_regularization_warmup = True
# Exact behavior of the released PILLAR-ESPN code: L1 norm of the flattened
# even-power penalty, averaged over activation sites, with task loss disabled
# during epoch zero.
config.pillar_penalty_reduction = "sum"
# Preserve the exact z^10 range loss through |z|=2 (|x|=9.6), then continue
# with its tangent line. This avoids FP32 overflow on a rare internal outlier
# while retaining a nonzero restoring gradient. It is training-only.
config.pillar_penalty_tail_cap = 2.0
config.pillar_range_only_epochs = 1
config.pillar_log_interval = 50
# Early unclipped evaluations can overflow before the range/LR warm-up has
# finished. The strict finite/range gate starts immediately after warm-up.
config.pillar_skip_verification_epochs = 5

config.lr_scheduler = "cosine"
config.min_lr_ratio = 0.01
config.warmup_epoch = 5

config.range_augmentation = {
    "enabled": True,
    "probability": 0.50,
    "contrast": (0.55, 1.70),
    "gain": (0.70, 1.30),
    "bias": (-0.12, 0.12),
    "gamma": (0.70, 1.45),
    "noise_probability": 0.25,
    "noise_std": 0.035,
}

config.sync_bn = True
config.broadcast_buffers = True
config.ddp_fp16_compress = False
config.dali = False
config.num_workers = 8
config.verbose = 2500
config.frequent = 50
config.fail_on_nonfinite_val = True
config.max_validation_embedding_abs = 1e4
config.validation_batch_size = 256
config.save_all_states = True
config.checkpoint_interval_epochs = 1
config.save_epoch_models = True
config.epoch_model_interval = 1

config.rec = "./ms1m-retinaface-t1"
config.num_classes = 93431
config.num_image = 5179510
config.num_epoch = 32
config.val_targets = ["lfw", "cfp_fp", "agedb_30"]
