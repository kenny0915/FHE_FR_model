from easydict import EasyDict as edict

config = edict()
config.margin_list = (1.0, 0.5, 0.0)
config.network = "patch_cnn"
config.input_size = 112
config.patch_size = 56
config.resume = False
config.output = None
config.embedding_size = 256
config.sample_rate = 1.0
config.fp16 = False
config.momentum = 0.9
config.weight_decay = 5e-4
config.gradient_clip = 1.0
config.gradient_clip_type = "value"
config.batch_size = 128
config.lr = 0.05
config.verbose = 2000
config.dali = False
config.dali_aug = False

config.rec = "./faces_webface_112x112/"
config.num_classes = 10572
config.num_image = 494149
config.num_epoch = 24
config.warmup_epoch = 1
config.val_targets = ["lfw", "cfp_fp", "agedb_30"]

config.interclass_filtering_threshold = 0
config.optimizer = "sgd"
config.gradient_acc = 1
config.frequent = 10
config.patch_cnn_jigsaw_weight = 0.005
