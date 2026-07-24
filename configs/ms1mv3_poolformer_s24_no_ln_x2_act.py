from easydict import EasyDict as edict

config = edict()
config.margin_list = (1.0, 0.0, 0.4)
config.network = "poolformer_no_ln_x2_act_s24_mlp2"
config.resume = False
config.output = None
config.embedding_size = 512
config.sample_rate = 1.0
config.fp16 = True
config.amp_init_scale = 1024.0
config.amp_growth_interval = 1000
config.gradient_clip = 1.0
config.prepbn_decay_epochs = 8
config.prepbn_bn_stat_epochs = 1
config.prepbn_require_full_transition = True
config.validate_after_prepbn_transition = True
config.final_verification_after_prepbn = True

# NAFNet-style SimpleGate: split the projected channels and multiply the two
# halves. The MLP2 variant offsets the doubled pre-gate width. Range captures
# are sampled to keep the layer-wise p99.9 instrumentation affordable.
config.simple_gate_range_limit = 6.0
config.simple_gate_stats_interval = 100
config.simple_gate_stats_sample_size = 16384
config.simple_gate_final_profile_batches = 50
config.simple_gate_compute_fp32 = True
config.simple_gate_fail_on_nonfinite = True
# Convert six contiguous groups only after RepBatchNorm has reached its final
# graph. Each group blends for two epochs; epochs 20-25 are pure SimpleGate.
config.simple_gate_initial_blend = 0.0
config.simple_gate_group_epochs = (8, 10, 12, 14, 16, 18)
config.simple_gate_transition_epochs = 2.0
config.simple_gate_require_full_conversion = True
config.simple_gate_distill_loss_weight = 0.1
config.simple_gate_range_loss_weight = 0.01
config.simple_gate_repbn_recalibration_batches = 200
config.simple_gate_verify_after_repbn = True
# Each distributed rank consumes this many batches; the cumulative running
# statistics are merged before verification and checkpointing.
config.simple_gate_group_bn_recalibration_batches = 500
config.simple_gate_verify_after_group = True
config.simple_gate_save_after_group = True
config.fail_on_nonfinite_val = True
# Retain an inference-only backbone snapshot after every completed epoch so a
# stable earlier model remains available if a later conversion group fails.
config.save_epoch_models = True
config.epoch_model_interval = 1
config.momentum = 0.9
config.weight_decay = 0.1
config.batch_size = 256
config.lr = 0.001
config.verbose = 2000
config.dali = False
config.dali_aug = False

config.rec = "./ms1m-retinaface-t1"
config.num_classes = 93431
config.num_image = 5179510
config.num_epoch = 25
config.warmup_epoch = 2

config.interclass_filtering_threshold = 0
config.optimizer = "adamw"
config.gradient_acc = 1
config.frequent = 10
config.val_targets = ["lfw", "cfp_fp", "agedb_30"]
