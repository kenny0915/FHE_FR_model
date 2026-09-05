"""Cumulative residual focus after the first WIDER-to-MS1Mv3 pass.

Start from the lowest-failure checkpoint (focus1 epoch 4), replay the union of
every failure observed in focus1, and halve the per-step trust-region update.
The full MS1Mv3 original-plus-flip gate remains the checkpoint selector; no
IJB sample is used for gradients or selection.
"""

from easydict import EasyDict as edict

from configs.ms1mv3_r50_no_relu_phase2_wider_ms1mv3_focus1 import (
    config as _focus1_config,
)


config = edict(_focus1_config.copy())
config.output = (
    "work_dirs/ms1mv3_r50_herpn_full_conversion_phase2_"
    "wider_ms1mv3_focus2")
config.backbone_init = (
    "work_dirs/ms1mv3_r50_herpn_full_conversion_phase2_"
    "wider_ms1mv3_focus1/model_epoch_04.pt")
config.calibration_priority_manifests = tuple(
    "work_dirs/ms1mv3_r50_herpn_full_conversion_phase2_"
    f"wider_ms1mv3_focus1/full_gate_epoch_{epoch:02d}.json"
    for epoch in range(1, 6)
)

# The cumulative set is small enough that 256 copies plus the fixed top-64
# per-layer background are traversed in roughly one 300-step epoch.  Smaller
# tensor updates reduce boundary samples trading places between gate scans.
config.ijbc_gate_failure_repeats = 256
config.max_step_update_ratio = 5e-6
config.steps_per_epoch = 300
config.num_epoch = 5
