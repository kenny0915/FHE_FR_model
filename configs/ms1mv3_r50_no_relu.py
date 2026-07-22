from easydict import EasyDict as edict

# make training faster
# our RAM is 256G
# mount -t tmpfs -o size=140G  tmpfs /train_tmp

config = edict()
config.margin_list = (1.0, 0.5, 0.0)
config.network = "r50_no_relu"
config.resume = False
config.output = "work_dirs/ms1mv3_r50_herpn_blockwise"
config.embedding_size = 512
config.sample_rate = 1.0
config.fp16 = False
config.momentum = 0.9
config.weight_decay = 5e-4
config.gradient_clip = 1.0
config.batch_size = 128
config.lr = 0.005
config.verbose = 2000
config.dali = False

# Convert one or two residual blocks at a time, from the network output back
# toward the stem. This avoids replacing all 14 Layer3 activations together.
config.herpn_initial_progress = 0.0
config.backbone_init = "work_dirs/ms1mv3_r50/model.pt"
config.herpn_bn_eps = 1e-4
config.herpn_quadratic_bn_eps = 1e-3
config.herpn_range_limit = 6.0
config.herpn_output_limit = 8.0
config.herpn_range_loss_weight = 0.1
config.herpn_output_loss_weight = 0.05
config.herpn_distill_loss_weight = 0.1
config.herpn_distillation_floor = 0.1
config.herpn_lr_multiplier = 0.1
config.herpn_teacher_momentum = 0.0
config.herpn_stage_epochs = ()
config.herpn_conversion_groups = (
    ("layer4.2.prelu", "layer4.1.prelu"),
    ("layer4.0.prelu",),
    ("layer3.13.prelu", "layer3.12.prelu"),
    ("layer3.11.prelu", "layer3.10.prelu"),
    ("layer3.9.prelu", "layer3.8.prelu"),
    ("layer3.7.prelu", "layer3.6.prelu"),
    ("layer3.5.prelu", "layer3.4.prelu"),
    ("layer3.3.prelu", "layer3.2.prelu"),
    ("layer3.1.prelu", "layer3.0.prelu"),
    ("layer2.3.prelu", "layer2.2.prelu"),
    ("layer2.1.prelu", "layer2.0.prelu"),
    ("layer1.2.prelu", "layer1.1.prelu"),
    ("layer1.0.prelu",),
    ("prelu",),
)
config.herpn_group_epochs = tuple(range(2, 16))
config.herpn_transition_epochs = 1.0
config.herpn_bn_recalibration_batches = 500
config.herpn_require_full_conversion = True
config.sync_bn = True
config.broadcast_buffers = True
config.check_finite_grads = True
config.save_all_states = True
config.checkpoint_interval_epochs = 1
config.checkpoint_keep_previous = True
config.fail_on_nonfinite_val = True
config.max_validation_embedding_abs = 1e4
config.save_validation_snapshots = True

config.rec = "./ms1m-retinaface-t1"
config.num_classes = 93431
config.num_image = 5179510
config.num_epoch = 22
config.warmup_epoch = 2
config.val_targets = ['lfw', 'cfp_fp', "agedb_30"]
