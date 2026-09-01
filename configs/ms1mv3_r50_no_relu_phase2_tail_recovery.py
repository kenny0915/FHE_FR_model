"""Numerical recovery of the epoch-23 fully polynomial R50.

The student starts from the exact 25/25 epoch-23 snapshot.  Every BatchNorm
running buffer and affine parameter is frozen.  Only the convolutions and
HerPN coefficients in layer3.0--layer3.3 are optimized against the four
dominant pre-square IJB-C failure boundaries.

The approximation target remains PReLU on ``[-6, 6]``.  A straight-through
clamp is used only in the training surrogate so rare stressed MS1Mv3 faces can
contribute a finite range gradient.  Evaluation and encrypted inference keep
the exact unclipped degree-2 polynomial, hence add no operation and retain one
ciphertext square (one multiplicative level) at each activation.
"""

from easydict import EasyDict as edict

from configs.ms1mv3_r50_no_relu import config as _base_config


config = edict(_base_config.copy())
config.resume = False
config.output = (
    "work_dirs/ms1mv3_r50_herpn_full_conversion_phase2_tail_recovery")
config.backbone_init = (
    "work_dirs/ms1mv3_r50_herpn_full_conversion_phase1/model_epoch_23.pt")
config.herpn_initial_progress = 5.0
config.backbone_init_herpn_progress = 5.0

# The graph is already 25/25.  Disable every conversion transition and all BN
# recalibration paths so the epoch-23 normalization state is immutable.
config.herpn_stage_epochs = ()
config.herpn_conversion_groups = ()
config.herpn_group_epochs = ()
config.herpn_bn_recalibration_batches = 0
config.herpn_require_full_conversion = True
config.freeze_batchnorm_running_stats = True
config.freeze_batchnorm_affine = True

# IJB-C tracing places 78.6% of epoch-23 failures at these first four Layer3
# activations.  Their conv weights and channel-wise quadratic coefficients are
# the only trainable backbone tensors in the first recovery pass.
config.backbone_trainable_prefixes = tuple(
    prefix
    for index in range(4)
    for prefix in (
        f"layer3.{index}.conv1",
        f"layer3.{index}.conv2",
        f"layer3.{index}.prelu.herpn",
        *((f"layer3.{index}.downsample.0",) if index == 0 else ()),
    )
)
config.herpn_range_loss_names = tuple(
    f"layer3.{index}.prelu" for index in range(4))

# Rare-tail rather than feature-map-average conditioning.  The plaintext
# training surrogate is bounded at the same [-6, 6] target interval; it is
# disabled automatically by eval mode and absent from FHE inference.
config.herpn_range_penalty_mode = "linear_tail"
config.herpn_range_topk_fraction = 0.001
config.herpn_range_bulk_weight = 0.01
config.herpn_training_stabilization_limit = 6.0
config.herpn_range_loss_weight = 2.0
config.herpn_distill_loss_weight = 0.0

# Photometric stress matches the darker, higher-contrast and clipped IJB-C
# failure population, but samples and identities still come only from MS1Mv3.
config.range_augmentation = {
    "enabled": True,
    "probability": 0.70,
    "contrast": (0.50, 1.80),
    "gain": (0.55, 1.30),
    "bias": (-0.18, 0.10),
    "gamma": (0.65, 1.70),
    "noise_probability": 0.25,
    "noise_std": 0.04,
}

# A frozen PReLU R50 teacher prevents ordinary MS1Mv3 embedding geometry from
# drifting while Layer3 is pushed back into the polynomial interval.  The
# random warm-start PartialFC head is intentionally excluded from the loss.
config.embedding_teacher_network = "r50"
config.embedding_teacher_checkpoint = "work_dirs/ms1mv3_r50/model.pt"
config.embedding_distill_weight = 1.0
config.task_loss_weight = 0.0

config.lr = 1e-4
config.weight_decay = 1e-5
config.gradient_clip = 1.0
config.stable_gradient_clip = True
config.warmup_epoch = 0
config.num_epoch = 2
config.verbose = 500
config.fp16 = False
config.ddp_fp16_compress = False
config.sync_bn = False
config.broadcast_buffers = True
config.check_finite_grads = True
config.max_nonfinite_embedding_skips = 1000
config.max_nonfinite_loss_skips = 1000
config.skip_nonfinite_gradients = True
config.max_nonfinite_gradient_skips = 1000
config.fail_on_nonfinite_val = False
config.tensorboard = False
config.save_all_states = True
config.checkpoint_interval_epochs = 1
config.save_epoch_models = True
config.epoch_model_interval = 1
