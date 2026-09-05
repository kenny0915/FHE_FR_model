"""Priority focus for residual MS1Mv3 adversarial range-margin failures."""

from easydict import EasyDict as edict

from configs.ms1mv3_r50_no_relu_phase2_ms1mv3_robust_margin import (
    config as _robust_config,
)


config = edict(_robust_config.copy())
config.output = (
    "work_dirs/ms1mv3_r50_herpn_full_conversion_phase2_"
    "ms1mv3_robust_margin_focus1")
config.backbone_init = (
    "work_dirs/ms1mv3_r50_herpn_full_conversion_phase2_"
    "ms1mv3_robust_margin/model_epoch_03.pt")
config.calibration_priority_manifests = (
    "work_dirs/ms1mv3_r50_herpn_full_conversion_phase2_"
    "ms1mv3_robust_margin/full_gate_epoch_03.json",
)

# Retain a small cross-layer background while making the 46 active numerical
# failures dominate each epoch.  This checkpoint and all selection gates are
# still MS1Mv3-only.
config.calibration_replay_activation_topk = 64
config.ijbc_gate_failure_repeats = 256
config.steps_per_epoch = 500
config.num_epoch = 5
