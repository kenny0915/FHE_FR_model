"""BN-frozen deployment calibration for the 25/25 epoch-23 HerPN R50.

Approximation target: channel-wise PReLU on [-6, 6].  Every saved activation
is still an exact degree-2 polynomial, so this adds no FHE multiplicative
depth.  IJB-C inputs are used without identity labels solely to enforce the
zero-non-finite deployment gate; resulting accuracy is a calibrated diagnostic
rather than an untouched IJB-C benchmark.
"""

from easydict import EasyDict as edict

from configs.ms1mv3_r50_no_relu import config as _base_config


config = edict(_base_config.copy())
config.output = (
    "work_dirs/ms1mv3_r50_herpn_full_conversion_phase2_ijbc_calibration")
config.backbone_init = (
    "work_dirs/ms1mv3_r50_herpn_full_conversion_phase1/model_epoch_23.pt")
config.ijbc_root = "ijb/IJBC"
config.ijbc_target = "IJBC"
config.ijbc_replay_manifests = (
    "work_dirs/ms1mv3_r50_herpn_full_conversion_phase2_ijbc_tail_mining/"
    "epoch23_ijbc_tails.json",
)
config.ijbc_replay_activation_topk = 64
config.ijbc_preservation_count = 8192
config.ijbc_workers = 4
config.replay_batch_size = 8
config.preservation_batch_size = 32
config.full_gate_source_batch_size = 256

config.herpn_range_penalty_mode = "sample_max_tail"
config.herpn_range_limit = 6.0
config.herpn_range_guard_ratio = 0.75
config.herpn_training_stabilization_limit = 6.0
config.causal_range_reduction = "mean_max"
config.tail_range_loss_weight = 1.0
config.preservation_loss_weight = 1.0

config.lr = 1e-4
config.tail_to_clean_gradient_ratio = 10000.0
config.max_step_update_ratio = 1e-5
config.parameter_trust_region_ratio = 0.01
config.parameter_scale_floor = 1.0
config.steps_per_epoch = 500
config.num_epoch = 5
config.fp16 = False
config.frequent = 50
