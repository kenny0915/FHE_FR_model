"""MS1Mv3 focus after held-out WIDER range-margin calibration.

This second half of the alternating, IJB-free protocol starts from the first
checkpoint that passes the held-out WIDER gate.  It replays only MS1Mv3
failures and a small fixed cross-layer MS1Mv3 background, then selects with a
full MS1Mv3 original-plus-flip numerical gate.
"""

from easydict import EasyDict as edict

from configs.ms1mv3_r50_no_relu_phase2_ms1mv3_robust_margin import (
    config as _ms1mv3_config,
)


config = edict(_ms1mv3_config.copy())
config.output = (
    "work_dirs/ms1mv3_r50_herpn_full_conversion_phase2_"
    "wider_ms1mv3_focus1")
config.backbone_init = (
    "work_dirs/ms1mv3_r50_herpn_full_conversion_phase2_"
    "wider_robust_margin/model_numerical_gate_zero.pt")
config.calibration_dataset = "ms1mv3"
config.calibration_replay_manifests = (
    "work_dirs/ms1mv3_r50_herpn_full_conversion_phase2_"
    "ms1mv3_tail_mining_top4096/epoch23_ms1mv3_tails_top4096.json",
)
config.calibration_priority_manifests = (
    "work_dirs/ms1mv3_r50_herpn_full_conversion_phase2_"
    "wider_robust_margin/ms1mv3_gate/"
    "model_numerical_gate_zero_ms1mv3_gate.json",
)
config.calibration_replay_activation_topk = 64
config.ijbc_gate_failure_repeats = 128

config.herpn_range_limit = 6.0
config.herpn_range_guard_ratio = 2.0 / 3.0
config.numerical_range_gate_limit = 4.0
config.adversarial_tail_enabled = True
config.adversarial_tail_epsilon = 16.0 / 255.0
config.adversarial_tail_step_size = 4.0 / 255.0
config.adversarial_tail_steps = 3
config.adversarial_tail_random_start = True

config.steps_per_epoch = 500
config.num_epoch = 5
