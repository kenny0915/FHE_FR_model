"""Sequential MS1Mv3 conversion to per-activation rescaled polynomials.

Approximation target for activation i and trained PReLU channel slope a_ic:

    PReLU_a(x), x in [-S_i, S_i]
    z = x / S_i
    student_ic(x) = S_i * q_ic(z)

``S_i`` is fitted from sampled 99.9th percentiles on the current partially
converted eval graph. Disjoint holdout batches and the complete-loader maximum
are tail checks, not clamps: calibration stops if they expose an unsafe range.
A 10% margin is added to the robust estimate. Exactly one activation converts
at a time in forward order; after it completes, globally synchronized
BatchNorm statistics are refreshed before measuring the next interval.

The default degree is 2 (one sequential ciphertext square). Set
``layerwise_poly_degree = 3`` for a two-level cubic comparison.
"""

from easydict import EasyDict as edict


config = edict()
config.margin_list = (1.0, 0.5, 0.0)
config.network = "r50_layerwise_poly"
# Set True on the GPU server to recover from checkpoint_gpu_<rank>.pt. Legacy
# theta2 checkpoints are migrated and their incompatible coefficient momentum
# is cleared automatically; group-09 inference snapshots must not be resumed.
config.resume = False
config.output = "work_dirs/ms1mv3_r50_layerwise_poly_d2"
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

# Blend zero is exactly the pretrained PReLU model. PReLU slopes are frozen
# teachers and remain channel-wise plaintext constants in the final student.
config.backbone_init = "work_dirs/ms1mv3_r50/model.pt"
config.herpn_initial_progress = 0.0
config.layerwise_poly_degree = 2
config.layerwise_poly_initial_scale = 1.0
config.layerwise_poly_distill_eps = 1e-4

# Zero means one complete representative-data pass on every distributed shard.
# Fit batches use a bounded sample per activation tensor so quantile estimation
# remains practical. Every twentieth batch is disjoint holdout data. The true
# global maximum is retained for diagnostics and tail rejection.
config.layerwise_poly_range_calibration_batches = 0
config.layerwise_poly_range_margin = 1.1
config.layerwise_poly_min_scale = 1e-3
config.layerwise_poly_range_quantile = 0.999
config.layerwise_poly_quantile_samples = 65536
config.layerwise_poly_range_holdout_fraction = 0.05

# Stop before a rare quadratic tail poisons the next public interval. These
# checks run only during plaintext calibration and add no encrypted operation.
config.layerwise_poly_max_tail_ratio = 8.0
config.layerwise_poly_max_scale_growth = 16.0
config.layerwise_poly_max_input_scale = 1e5

# The trainer's existing progressive-polynomial protocol supplies task,
# teacher-distillation, range-tail, finite-gradient, checkpoint, and BN-refresh
# handling. Distillation is teacher-energy normalized in this backbone.
config.herpn_range_loss_weight = 0.1
config.herpn_distill_loss_weight = 1.0
config.herpn_bn_recalibration_batches = 200
config.herpn_save_after_group = True
config.herpn_require_full_conversion = True
config.herpn_stage_epochs = ()

# Strict forward order is required because each next interval is measured from
# the already converted prefix. There are 25 PReLUs in IResNet50.
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

# Each activation gets one full local-fit epoch after its interval is measured,
# followed by a one-epoch blend. The stem gets two initial local-fit epochs.
config.herpn_group_epochs = tuple(range(2, 52, 2))
config.herpn_transition_epochs = 1.0

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
# Last conversion completes at epoch 51; retain four fully polynomial epochs.
config.num_epoch = 55
config.warmup_epoch = 1
config.val_targets = ["lfw", "cfp_fp", "agedb_30"]
