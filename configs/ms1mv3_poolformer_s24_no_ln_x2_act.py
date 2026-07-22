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
config.prepbn_decay_epochs = 20
config.prepbn_bn_stat_epochs = 1
config.prepbn_require_full_transition = True
config.validate_after_prepbn_transition = True
config.final_verification_after_prepbn = True

# NAFNet-style SimpleGate: split the projected channels and multiply the two
# halves. The MLP2 variant offsets the doubled pre-gate width. Range captures
# are sampled to keep the layer-wise p99.9 instrumentation affordable.
config.simple_gate_range_limit = 6.0
config.simple_gate_stats_interval = 500
config.simple_gate_stats_sample_size = 16384
config.simple_gate_final_profile_batches = 50
config.simple_gate_compute_fp32 = True
config.simple_gate_fail_on_nonfinite = True
config.fail_on_nonfinite_val = True
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
