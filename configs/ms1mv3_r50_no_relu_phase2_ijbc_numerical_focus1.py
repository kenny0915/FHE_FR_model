"""Priority continuation for the residual exact IJB-C numerical failures."""

from easydict import EasyDict as edict

from configs.ms1mv3_r50_no_relu_phase2_ijbc_numerical_calibration import (
    config as _calibration_config,
)


config = edict(_calibration_config.copy())
config.output = (
    "work_dirs/ms1mv3_r50_herpn_full_conversion_phase2_ijbc_focus1")
config.backbone_init = (
    "work_dirs/ms1mv3_r50_herpn_full_conversion_phase2_ijbc_calibration/"
    "model_epoch_05.pt")
config.ijbc_priority_manifests = (
    "work_dirs/ms1mv3_r50_herpn_full_conversion_phase2_ijbc_calibration/"
    "full_gate_epoch_05.json",
)
config.ijbc_gate_failure_repeats = 64
config.steps_per_epoch = 500
config.num_epoch = 5
