"""Fine-tune R50 with an Alpha10-to-degree-4 ReLU curriculum.

Every channel-wise PReLU keeps its learned slope ``a`` and uses
``a*x + (1-a)*poly_relu(x)``.  The approximation target is ReLU on [-8, 8].
Alpha10 is the accurate starting teacher; the independently fitted degree
16, 8, and 4 students reduce nonlinear multiplicative depth to 4, 3, and 2.
"""

from easydict import EasyDict as edict


config = edict()
config.margin_list = (1.0, 0.5, 0.0)
config.network = "r50_precise_relu"
config.resume = False
config.output = "work_dirs/ms1mv3_r50_precise_relu_s8_d4"
config.embedding_size = 512
config.sample_rate = 1.0

# Alpha10 and its analytical backward compute internally in FP32 even under
# autocast. Keep convolutions/BN activations in FP16 to fit a practical batch.
config.fp16 = True
config.momentum = 0.9
config.weight_decay = 5e-4
config.gradient_clip = 0.5
# batch_size is per GPU. On four GPUs, two normalized accumulation steps keep
# the original effective global batch: 64 * 4 * 2 = 512.
config.batch_size = 64
config.gradient_acc = 2
config.normalize_gradient_accumulation = True
config.lr = 0.005
config.verbose = 2000
config.dali = False

# Start from the successful ordinary PReLU R50. All 25 PReLUs are replaced at
# construction time; their trained channel-wise slopes load strictly.
config.backbone_init = "work_dirs/ms1mv3_r50/model.pt"
config.precise_relu_input_scale = 8.0
config.precise_relu_lower_degrees = (16, 8, 4)
config.precise_relu_initial_progress = 0.0

# Epochs 0-2 train the full Alpha10 replacement. Each listed epoch starts a
# smooth one-epoch whole-network transition to the next lower degree. Nine
# final epochs remain at degree 4 for adaptation.
config.precise_relu_stage_epochs = (2, 6, 10)
config.precise_relu_transition_epochs = 1.0
config.precise_relu_require_final_degree = True

# Penalize activation inputs outside the declared approximation interval.
# The penalty is plaintext training machinery and is not in the FHE graph.
config.precise_relu_range_loss_weight = 0.1
config.precise_relu_bn_recalibration_batches = 200

config.sync_bn = True
config.broadcast_buffers = True
config.check_finite_grads = True
config.ddp_fp16_compress = False
config.save_all_states = True
config.checkpoint_interval_epochs = 1
config.save_epoch_models = True
config.epoch_model_interval = 1

config.rec = "./ms1m-retinaface-t1"
config.num_classes = 93431
config.num_image = 5179510
config.num_epoch = 20
config.warmup_epoch = 1
config.val_targets = ["lfw", "cfp_fp", "agedb_30"]
