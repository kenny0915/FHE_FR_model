from easydict import EasyDict as edict


config = edict()
config.network = "r50_no_relu"
config.output = "work_dirs/casia_r50_herpn_smoke"
config.rec = "./faces_webface_112x112/"
config.num_classes = 10572
config.num_image = 25600
config.num_epoch = 2
config.warmup_epoch = 1
config.batch_size = 128
config.embedding_size = 512
config.sample_rate = 1.0
config.fp16 = False
config.sync_bn = True
config.broadcast_buffers = True
config.optimizer = "sgd"
config.lr = 0.02
config.momentum = 0.9
config.weight_decay = 5e-4
config.gradient_clip = 5.0
config.check_finite_grads = True
config.max_steps_per_epoch = 100
config.verbose = 1000000
config.frequent = 10
config.dali = False
config.val_targets = []
config.save_all_states = False
