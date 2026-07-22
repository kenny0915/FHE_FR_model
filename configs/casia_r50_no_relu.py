from easydict import EasyDict as edict

# make training faster
# our RAM is 256G
# mount -t tmpfs -o size=140G  tmpfs /train_tmp

config = edict()
config.margin_list = (1.0, 0.5, 0.0)
config.network = "r50_no_relu"
config.resume = False
config.output = "work_dirs/casia_r50_herpn"
config.embedding_size = 512
config.sample_rate = 1.0
config.fp16 = False
config.sync_bn = True
config.broadcast_buffers = True
config.momentum = 0.9
config.weight_decay = 5e-4
config.batch_size = 128
config.lr = 0.05
config.gradient_clip = 5.0
config.check_finite_grads = True
config.fail_on_nonfinite_val = True
config.max_validation_embedding_abs = 1e6
config.save_validation_snapshots = True
config.validation_batch_size = 128
# Progressive normalized degree-2 HerPN training. Each tuple entry starts
# stem/layer1/layer2/layer3/layer4; progress 5 is polynomial-only.
config.herpn_initial_progress = 0.0
config.backbone_init = "work_dirs/casia_r50/model.pt"
config.herpn_bn_eps = 1e-4
config.herpn_range_limit = 6.0
config.herpn_range_loss_weight = 0.1
config.herpn_distill_loss_weight = 0.1
config.herpn_stage_epochs = ()
config.herpn_conversion_groups = (
    ("prelu",),
    ("layer1.0.prelu", "layer1.1.prelu", "layer1.2.prelu"),
    ("layer2.0.prelu", "layer2.1.prelu", "layer2.2.prelu", "layer2.3.prelu"),
    ("layer3.0.prelu", "layer3.1.prelu", "layer3.2.prelu", "layer3.3.prelu"),
    ("layer3.4.prelu", "layer3.5.prelu", "layer3.6.prelu", "layer3.7.prelu"),
    ("layer3.8.prelu", "layer3.9.prelu", "layer3.10.prelu", "layer3.11.prelu"),
    ("layer3.12.prelu", "layer3.13.prelu"),
    ("layer4.0.prelu", "layer4.1.prelu", "layer4.2.prelu"),
)
config.herpn_group_epochs = (2, 6, 10, 14, 16, 18, 20, 22)
config.herpn_transition_epochs = 2.0
config.herpn_bn_recalibration_batches = 100
config.herpn_require_full_conversion = True
config.checkpoint_interval_epochs = 1
config.save_epoch_models = True
config.epoch_model_interval = 1
# config.lr_steps = [20, 28, 32]
config.verbose = 2000
config.dali = False

config.rec = "./faces_webface_112x112/"
config.num_classes = 10572
config.num_image = 494149
config.num_epoch = 28
config.warmup_epoch = 1
config.val_targets = ['lfw', 'cfp_fp', "agedb_30"]

config.save_all_states = True
