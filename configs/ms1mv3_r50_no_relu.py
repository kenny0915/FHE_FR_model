from easydict import EasyDict as edict

# make training faster
# our RAM is 256G
# mount -t tmpfs -o size=140G  tmpfs /train_tmp

config = edict()
config.margin_list = (1.0, 0.5, 0.0)
config.network = "r50_no_relu"
config.resume = False
config.output = "work_dirs/ms1mv3_r50_herpn"
config.embedding_size = 512
config.sample_rate = 1.0
config.fp16 = False
config.momentum = 0.9
config.weight_decay = 5e-4
config.gradient_clip = 1.0
config.batch_size = 128
config.lr = 0.01
config.verbose = 2000
config.dali = False

# Warm the normalized degree-2 HerPN branches, then progressively convert
# stem -> layer4. The IResNet topology stays unchanged throughout training.
config.herpn_initial_progress = 0.0
config.backbone_init = "work_dirs/ms1mv3_r50/model.pt"
config.herpn_bn_eps = 1e-4
config.herpn_range_limit = 6.0
config.herpn_range_loss_weight = 0.1
config.herpn_distill_loss_weight = 0.1
config.herpn_stage_epochs = (1, 4, 7, 10, 13)
config.herpn_transition_epochs = 2.0
config.herpn_bn_recalibration_batches = 200
config.herpn_require_full_conversion = True
config.sync_bn = True
config.broadcast_buffers = True
config.check_finite_grads = True
config.save_all_states = True

config.rec = "./ms1m-retinaface-t1"
config.num_classes = 93431
config.num_image = 5179510
config.num_epoch = 20
config.warmup_epoch = 2
config.val_targets = ['lfw', 'cfp_fp', "agedb_30"]
