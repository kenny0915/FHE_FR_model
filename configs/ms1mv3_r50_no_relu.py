from easydict import EasyDict as edict

# make training faster
# our RAM is 256G
# mount -t tmpfs -o size=140G  tmpfs /train_tmp

config = edict()
config.margin_list = (1.0, 0.5, 0.0)
config.network = "r50_no_relu"
config.resume = False
config.output = None
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

# Progressively convert stem -> layer4 while keeping Cheby inputs inside the
# range-regularized [-6, 6] target. The final trained model is Cheby-only.
config.cheby_initial_progress = 0.0
config.backbone_init = "work_dirs/ms1mv3_r50/model.pt"
config.cheby_scales = {
    "stem": 8.0,
    "layer1": 8.0,
    "layer2": 7.0,
    "layer3": 6.5,
    "layer4": 6.5,
}
config.cheby_range_limit = 6.0
config.cheby_range_loss_weight = 0.01
config.cheby_stage_epochs = (1, 4, 7, 10, 13)
config.cheby_transition_epochs = 1.0
config.cheby_bn_recalibration_batches = 200
config.cheby_require_full_conversion = True

config.rec = "./ms1m-retinaface-t1"
config.num_classes = 93431
config.num_image = 5179510
config.num_epoch = 20
config.warmup_epoch = 2
config.val_targets = ['lfw', 'cfp_fp', "agedb_30"]
