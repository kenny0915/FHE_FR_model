"""MS1Mv3-only adversarial range-margin recovery from phase1 epoch 23.

The degree-2 HerPN target remains channel-wise PReLU on [-6, 6].  Training
requires every MS1Mv3 original/flip pre-HerPN input to stay within [-4, 4],
providing a numerical margin rather than merely checking that FP32 did not
overflow.  IJB data is not used for gradients or checkpoint selection.
"""

from easydict import EasyDict as edict

from configs.ms1mv3_r50_no_relu_phase2_ms1mv3_numerical_calibration import (
    config as _calibration_config,
)


config = edict(_calibration_config.copy())
config.output = (
    "work_dirs/ms1mv3_r50_herpn_full_conversion_phase2_"
    "ms1mv3_robust_margin")
config.backbone_init = (
    "work_dirs/ms1mv3_r50_herpn_full_conversion_phase1/model_epoch_23.pt")
config.calibration_replay_manifests = (
    "work_dirs/ms1mv3_r50_herpn_full_conversion_phase2_"
    "ms1mv3_tail_mining_top4096/epoch23_ms1mv3_tails_top4096.json",
)
config.calibration_priority_manifests = config.calibration_replay_manifests
config.calibration_replay_activation_topk = 4096
config.ijbc_gate_failure_repeats = 16

# The approximation radius is still 6.  A 2/3 guard and exact full-dataset
# gate make 4 the maximum accepted pre-HerPN magnitude.
config.herpn_range_limit = 6.0
config.herpn_range_guard_ratio = 2.0 / 3.0
config.numerical_range_gate_limit = 4.0

# Inputs use the model's normalized [-1, 1] scale.  epsilon=16/255 therefore
# corresponds to an 8/255 perturbation in ordinary [0, 1] pixels.
config.adversarial_tail_enabled = True
config.adversarial_tail_epsilon = 16.0 / 255.0
config.adversarial_tail_step_size = 4.0 / 255.0
config.adversarial_tail_steps = 3
config.adversarial_tail_random_start = True

config.steps_per_epoch = 1000
config.num_epoch = 5
