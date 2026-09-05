"""WIDER-only range-margin recovery from the 25/25 phase1 epoch 23 model.

The degree-2 HerPN target remains channel-wise PReLU on [-6, 6].  WIDER
training images are deterministically divided at the scene-image level: fold
zero is a held-out numerical gate and the other nine folds supply calibration
crops.  No IJB image participates in gradients or checkpoint selection.
"""

from easydict import EasyDict as edict

from configs.ms1mv3_r50_no_relu_phase2_ms1mv3_robust_margin import (
    config as _robust_config,
)


config = edict(_robust_config.copy())
config.output = (
    "work_dirs/ms1mv3_r50_herpn_full_conversion_phase2_"
    "wider_robust_margin")
config.backbone_init = (
    "work_dirs/ms1mv3_r50_herpn_full_conversion_phase1/model_epoch_23.pt")
config.calibration_dataset = "wider"
config.wider_image_root = "WIDER_train/images"
config.wider_annotation_path = (
    "wider_face_split/wider_face_train_bbx_gt.txt")
config.wider_mining_split = "calibration"
config.wider_validation_modulo = 10
config.wider_validation_fold = 0
config.wider_min_face_size = 20
config.wider_crop_scale = 1.35

config.calibration_replay_manifests = (
    "work_dirs/ms1mv3_r50_herpn_full_conversion_phase2_"
    "wider_tail_mining/epoch23_wider_tails_top2048.json",
)
config.calibration_priority_manifests = config.calibration_replay_manifests
config.calibration_replay_activation_topk = 2048
config.ijbc_gate_failure_repeats = 4

# Fold zero is a held-out selection gate.  Its exact failing indices must not
# be fed back into training; all replay rows come only from the other folds.
config.replay_gate_failures = False

# Approximate PReLU on [-6, 6], train against a [-4, 4] safety guard, and keep
# the final inference graph unchanged.  The local attack is training-only.
config.herpn_range_limit = 6.0
config.herpn_range_guard_ratio = 2.0 / 3.0
config.numerical_range_gate_limit = 4.0
config.adversarial_tail_enabled = True
config.adversarial_tail_epsilon = 16.0 / 255.0
config.adversarial_tail_step_size = 4.0 / 255.0
config.adversarial_tail_steps = 3
config.adversarial_tail_random_start = True

config.ijbc_preservation_count = 8192
config.steps_per_epoch = 500
config.num_epoch = 5
