from easydict import EasyDict as edict

# make training faster
# our RAM is 256G
# mount -t tmpfs -o size=140G  tmpfs /train_tmp

config = edict()
config.margin_list = (1.0, 0.5, 0.0) # 0.5 -> 0.4 since mbf is smaller than r50
config.network = "mbf"
config.resume = False
config.output = None
config.embedding_size = 512
config.sample_rate = 1.0
config.fp16 = True
config.momentum = 0.9
config.weight_decay = 2e-4
config.batch_size = 128
config.lr = 0.1
# config.lr_steps = [20, 28, 32]
config.verbose = 2000
config.dali = False

config.rec = "./faces_webface_112x112/"
config.num_classes = 10572
config.num_image = 494149
config.num_epoch = 40
config.warmup_epoch = 0
config.val_targets = ['lfw', 'cfp_fp', "agedb_30"]
