from easydict import EasyDict as edict

config = edict()
config.margin_list = (1.0, 0.0, 0.4)
config.network = "poolformer_s24_mlp2"
config.resume = False
config.output = None
config.embedding_size = 512
config.sample_rate = 1.0
config.fp16 = True
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
