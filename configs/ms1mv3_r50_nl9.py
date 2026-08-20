"""Fine-tune the 9-PReLU iResNet50 ablation on MS1MV3."""

from easydict import EasyDict as edict


config = edict()
config.margin_list = (1.0, 0.5, 0.0)
config.network = "r50_nl9"
config.arch_config = "nl9"
config.resume = False
config.output = "work_dirs/ms1mv3_r50_nl9"
config.embedding_size = 512
config.sample_rate = 1.0
config.fp16 = True
config.momentum = 0.9
config.weight_decay = 5e-4
config.batch_size = 128
config.lr = 0.01
config.verbose = 2000
config.dali = False

# Initialize directly from the same baseline as nl13 so the nonlinear-depth
# comparison does not depend on a previously fine-tuned ablation checkpoint.
config.backbone_init = "work_dirs/ms1mv3_r50/model.pt"

config.rec = "./ms1m-retinaface-t1"
config.num_classes = 93431
config.num_image = 5179510
config.num_epoch = 20
config.warmup_epoch = 0
config.val_targets = ["lfw", "cfp_fp", "agedb_30"]
