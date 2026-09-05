"""MS1Mv3-only numerical calibration from the 25/25 epoch-23 model.

Approximation target: channel-wise PReLU on [-6, 6].  This reproduces the
successful BN-frozen, causal-tail, priority-replay method without reading any
IJB image.  Full MS1Mv3 original/flip scans select numerical checkpoints;
IJB-C remains untouched until the final evaluation.
"""

from easydict import EasyDict as edict

from configs.ms1mv3_r50_no_relu_phase2_ijbc_numerical_calibration import (
    config as _calibration_config,
)


config = edict(_calibration_config.copy())
config.output = (
    "work_dirs/ms1mv3_r50_herpn_full_conversion_phase2_ms1mv3_calibration")
config.backbone_init = (
    "work_dirs/ms1mv3_r50_herpn_full_conversion_phase1/model_epoch_23.pt")
config.calibration_dataset = "ms1mv3"
config.calibration_replay_manifests = (
    "work_dirs/ms1mv3_r50_herpn_full_conversion_phase2_"
    "ms1mv3_tail_mining_all25/epoch23_ms1mv3_tails_all25.json",
)
config.calibration_priority_manifests = config.calibration_replay_manifests
config.calibration_replay_activation_topk = 64
config.ijbc_gate_failure_repeats = 16
config.ijbc_preservation_count = 8192
config.steps_per_epoch = 500
config.num_epoch = 5
