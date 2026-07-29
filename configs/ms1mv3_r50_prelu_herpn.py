"""Progressively convert an MS1Mv3 IResNet50 to PReLU-aware HerPN.

Approximation target:
    PReLU_a(x) = a*x + (1-a)*ReLU(x)
is approximated on the monitored interval [-6, 6] by
    a*x + (1-a)*HerPN_ReLU(x).

The frozen channel-wise slope ``a`` is loaded from the baseline checkpoint.
After BatchNorm calibration, each activation folds to one degree-2
polynomial and therefore needs one sequential ciphertext square.
"""

from easydict import EasyDict as edict


config = edict()
config.margin_list = (1.0, 0.5, 0.0)
config.network = "r50_prelu_herpn"
config.resume = False
config.output = "work_dirs/ms1mv3_r50_prelu_herpn"
config.embedding_size = 512
config.sample_rate = 1.0
config.fp16 = False
config.momentum = 0.9
config.weight_decay = 5e-4
config.selective_weight_decay = True
config.gradient_clip = 1.0
config.batch_size = 128
config.lr = 0.005
config.verbose = 2000
config.dali = False

# Start from the fully trained ordinary PReLU model. At blend=0 this new
# backbone is exactly the baseline graph; the PReLU slopes remain frozen.
config.backbone_init = "work_dirs/ms1mv3_r50/model.pt"
config.herpn_initial_progress = 0.0
config.herpn_bn_eps = 1e-4
config.herpn_range_limit = 6.0
config.herpn_range_loss_weight = 0.1
# Relative (teacher-energy-normalized) distillation stays active at blend=1.
config.prelu_herpn_distill_eps = 1e-4
config.herpn_distill_loss_weight = 1.0

# Convert all 25 activations in small, non-overlapping groups. Large Layer3
# is split into pairs because it was the unstable point in the earlier
# all-at-once schedule. Every transition lasts two epochs and is followed by
# recalibration before the next group starts.
config.herpn_stage_epochs = ()
config.herpn_conversion_groups = (
    ("prelu",),
    ("layer1.0.prelu", "layer1.1.prelu", "layer1.2.prelu"),
    ("layer2.0.prelu", "layer2.1.prelu"),
    ("layer2.2.prelu", "layer2.3.prelu"),
    ("layer3.0.prelu", "layer3.1.prelu"),
    ("layer3.2.prelu", "layer3.3.prelu"),
    ("layer3.4.prelu", "layer3.5.prelu"),
    ("layer3.6.prelu", "layer3.7.prelu"),
    ("layer3.8.prelu", "layer3.9.prelu"),
    ("layer3.10.prelu", "layer3.11.prelu"),
    ("layer3.12.prelu", "layer3.13.prelu"),
    ("layer4.0.prelu",),
    ("layer4.1.prelu",),
    ("layer4.2.prelu",),
)
config.herpn_group_epochs = tuple(range(1, 29, 2))
config.herpn_transition_epochs = 2.0
config.herpn_bn_recalibration_batches = 200
config.herpn_save_after_group = True
config.herpn_require_full_conversion = True
config.sync_bn = True
config.broadcast_buffers = True
config.check_finite_grads = True
config.save_all_states = True
config.checkpoint_interval_epochs = 1
config.save_epoch_models = True
config.epoch_model_interval = 1

config.rec = "./ms1m-retinaface-t1"
config.num_classes = 93431
config.num_image = 5179510
config.num_epoch = 33
config.warmup_epoch = 1
config.val_targets = ["lfw", "cfp_fp", "agedb_30"]
