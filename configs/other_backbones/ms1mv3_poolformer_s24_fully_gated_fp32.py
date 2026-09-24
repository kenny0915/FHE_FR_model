from easydict import EasyDict as edict


config = edict()
config.margin_list = (1.0, 0.0, 0.4)
config.network = "poolformer_fully_gated_s24"
config.resume = False
config.output = None
config.embedding_size = 512
config.sample_rate = 1.0

# NAFNet-style stability diagnostic: all 24 gates are active from the first
# step, true channel-wise LayerNorm remains in every block, and AMP is disabled.
config.fp16 = False
config.gradient_clip = 0.01
config.gradient_clip_type = "norm"
# NAFNet clips its single image-restoration network. Clip only this backbone so
# the much larger PartialFC classifier cannot dominate the shared norm.
config.gradient_clip_scope = "backbone"
config.check_finite_grads = True
config.fail_on_nonfinite_val = True

# AdamW settings follow the conservative end of the published NAFNet recipes.
# The per-GPU batch is halved from the FP16 PoolFormer runs to fit FP32
# activations on the 32 GB V100 training server.
config.optimizer = "adamw"
config.lr = 0.001
config.weight_decay = 0.001
config.momentum = 0.9
config.batch_size = 128

config.verbose = 2000
config.frequent = 10
config.dali = False
config.dali_aug = False
config.gradient_acc = 1

config.rec = "./ms1m-retinaface-t1"
config.num_classes = 93431
config.num_image = 5179510
config.num_epoch = 25
config.warmup_epoch = 2

config.interclass_filtering_threshold = 0
config.val_targets = ["lfw", "cfp_fp", "agedb_30"]
