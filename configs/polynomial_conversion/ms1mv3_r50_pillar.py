"""Train an MS1Mv3 iResNet-50 with the paper's PILLAR PolyReLU recipe.

Approximation target and interval:
    ReLU(x) on [-5, 5]

Quantization-aware p=10 polynomial from the paper:
    0.314453125 + 0.5*x + 0.15625*x^2 - 0.0029296875*x^4

The tighter [-4.8, 4.8] regularization interval gives an inference safety
buffer.  Training clips to [-5, 5] only after recording the penalty; eval and
FHE inference do not clip.  The balanced x^2 -> x^4 evaluation has degree 4
and multiplicative depth 2.  This model is trained from scratch because its
target is ReLU, whereas the repository baseline uses channel-wise PReLU.
"""

from easydict import EasyDict as edict


config = edict()
config.margin_list = (1.0, 0.5, 0.0)
config.network = "r50_pillar"
config.resume = False
config.output = "work_dirs/ms1mv3_r50_pillar_d4"
config.embedding_size = 512
config.sample_rate = 1.0

# Polynomial cascades are sensitive to half-precision overflow. Keep the
# backbone in FP32 while establishing the activation distributions.
config.fp16 = False
config.optimizer = "sgd"
config.lr = 0.013
config.momentum = 0.9
config.weight_decay = 1e-4
config.batch_size = 128
config.gradient_clip = 5.0
config.check_finite_grads = True

# PILLAR hyperparameters reported in Section 6.1 of the reference paper.
config.pillar_approximation_range = 5.0
config.pillar_regularization_range = 4.8
config.pillar_regularization_coefficient = 5e-5
config.pillar_regularization_exponent = 10
config.pillar_training_clip = True
config.pillar_regularization_warmup = True

# Match the paper's cosine annealing and five-epoch linear LR warm-up. The
# minimum learning rate is 0.01 times the initial rate.
config.lr_scheduler = "cosine"
config.min_lr_ratio = 0.01
config.warmup_epoch = 5

config.sync_bn = True
config.broadcast_buffers = True
config.dali = False
config.verbose = 2000
config.frequent = 10
config.save_all_states = True
config.checkpoint_interval_epochs = 1
config.save_epoch_models = True
config.epoch_model_interval = 1

config.rec = "./ms1m-retinaface-t1"
config.num_classes = 93431
config.num_image = 5179510
# Four extra epochs over the ordinary 20-epoch face baseline allow the
# regularization warm-up to finish before the main optimization phase.
config.num_epoch = 24
config.val_targets = ["lfw", "cfp_fp", "agedb_30"]
