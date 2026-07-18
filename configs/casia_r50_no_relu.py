from easydict import EasyDict as edict

# make training faster
# our RAM is 256G
# mount -t tmpfs -o size=140G  tmpfs /train_tmp

config = edict()
config.margin_list = (1.0, 0.5, 0.0)
config.network = "r50_no_relu"
config.resume = False
config.output = "work_dirs/casia_r50_no_relu_stable_v8"
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
# Progressive polynomial training. Progress 0 uses PReLU; progress 5 is
# polynomial-only. Each tuple entry starts stem/layer1/layer2/layer3/layer4.
config.cheby_initial_progress = 0.0
config.backbone_init = "work_dirs/casia_r50/model.pt"
config.cheby_scales = {
    "stem": 8.0,
    "layer1": 8.0,
    "layer2": 7.0,
    "layer3": 6.5,
    "layer4": 6.5,
}
config.cheby_range_limit = 6.0
config.cheby_range_loss_weight = 0.01
config.cheby_stage_epochs = (2, 6, 10, 14, 18)
config.cheby_transition_epochs = 2.0
config.cheby_bn_recalibration_batches = 100
config.cheby_require_full_conversion = True
# config.lr_steps = [20, 28, 32]
config.verbose = 2000
config.dali = False

config.rec = "./faces_webface_112x112/"
config.num_classes = 10572
config.num_image = 494149
config.num_epoch = 24
config.warmup_epoch = 1
config.val_targets = ['lfw', 'cfp_fp', "agedb_30"]

config.save_all_states = True
