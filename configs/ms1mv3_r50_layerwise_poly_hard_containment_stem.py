"""Prove full-training-set containment before the first polynomial blend.

The approximation target is the trained channel-wise PReLU on the fixed
interval ``[-S, S]``.  ``S`` is chosen once from two-orientation MS1Mv3
inference using ``2 * q99.95``.  If any observed activation input lies outside
that interval, the interval is kept fixed and the stem remains PReLU while two
conditioning epochs optimize the ordinary backbone against a normalized
top-tail range loss, ArcFace, and the frozen baseline embedding teacher.

The strict pre-blend scan consumes every training image in both orientations
and requires ``observed_absmax <= S``.  It cannot enlarge ``S`` to pass.  The
512 worst source indices from calibration are replayed in every conditioning
batch.  This probe converts only the stem; later sites remain PReLU until this
containment protocol proves both numerical safety and accuracy retention.
"""

from easydict import EasyDict as edict

from configs.ms1mv3_r50_layerwise_poly import config as base_config


config = edict(base_config.copy())
config.resume = False
config.output = "work_dirs/ms1mv3_r50_layerwise_poly_hard_containment_stem"
config.backbone_init = "work_dirs/ms1mv3_r50/model.pt"
config.herpn_initial_progress = 0.0

# A complete deterministic source-index scan; the wrapper evaluates each
# transformed image and its horizontal mirror. No observed tail may remain
# outside the interval at the strict boundary.
config.layerwise_poly_range_calibration_batches = 0
config.layerwise_poly_scan_both_orientations = True
config.layerwise_poly_range_quantile = 0.9995
config.layerwise_poly_range_margin = 2.0
config.layerwise_poly_range_holdout_fraction = 0.05
config.layerwise_poly_max_tail_ratio = 0.0
config.layerwise_poly_max_scale_growth = 0.0
config.layerwise_poly_max_input_scale = 1.0e3
config.layerwise_poly_require_full_containment = True
config.layerwise_poly_freeze_containment_interval = True

# Tail-aware conditioning. The polynomial coefficients are fitted locally,
# while upstream weights may move slowly enough to place every observed input
# inside the already fixed interval. Replay replaces 1/8 of each rank's batch.
config.layerwise_poly_range_penalty_mode = "containment_topk"
config.layerwise_poly_range_topk_fraction = 0.125
config.layerwise_poly_range_bulk_weight = 0.01
config.layerwise_poly_tail_topk = 512
config.layerwise_poly_tail_replay_batch_size = 16
config.layerwise_poly_tail_replay_workers = 2
config.layerwise_poly_staged_training = True
config.layerwise_poly_freeze_backbone_during_local_fit = False
config.layerwise_poly_allow_provisional_tail_conditioning = True
config.layerwise_poly_initial_calibration_provisional = True
config.layerwise_poly_strict_recalibrate_before_blend = True
config.layerwise_poly_causal_strict_calibration = True
config.layerwise_poly_verify_singleton_boundary = True
config.layerwise_poly_strict_tail_scale_floor = False
config.layerwise_poly_conditioning_backbone_lr_scale = 0.05
config.layerwise_poly_conditioning_range_loss_weight = 10.0
config.layerwise_poly_blend_backbone_lr_scale = 0.05
config.layerwise_poly_final_backbone_lr_scale = 0.05
config.layerwise_poly_optimizer_lr_scale = 1.0

# Preserve recognition geometry while range-conditioning the PReLU graph.
config.embedding_teacher_network = "r50"
config.embedding_teacher_checkpoint = config.backbone_init
config.embedding_distill_weight = 1.0
config.task_loss_weight = 1.0
config.herpn_distill_loss_weight = 1.0
config.herpn_range_loss_weight = 0.1

# Two complete conditioning epochs, one blend epoch, and one hold epoch. Only
# the first singleton is in scope for this proof run.
config.herpn_group_epochs = tuple(2.0 + 3.0 * index for index in range(25))
config.herpn_transition_epochs = 1.0
config.layerwise_poly_training_group_limit = 1
config.herpn_require_full_conversion = False
config.num_epoch = 4
config.warmup_epoch = 0
config.verbose = 1000
config.herpn_bn_recalibration_batches = 1000
config.fail_on_nonfinite_val = True
config.fp16 = False

