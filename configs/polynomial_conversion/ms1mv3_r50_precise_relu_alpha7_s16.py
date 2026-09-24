"""Fine-tune R50 through an Alpha10-to-Alpha7 ReLU curriculum.

Every channel-wise PReLU retains its learned plaintext slope ``a`` and uses
``a*x + (1-a)*poly_relu(x)``. The final encrypted-path activation is the
paper's Alpha7 composite on the fixed interval [-16, 16]. Its two degree-7
components have multiplicative depth 7 and absolute error bound 16/128=0.125
inside that interval.
"""

from easydict import EasyDict as edict


config = edict()
config.margin_list = (1.0, 0.5, 0.0)
config.network = "r50_precise_relu_alpha7"
config.resume = False
config.output = "work_dirs/ms1mv3_r50_precise_relu_alpha7_s16"
config.embedding_size = 512
config.sample_rate = 1.0

# The fixed polynomials and analytical backward recomputation use FP32. Keep
# convolutions and BN activations under autocast to fit batch 64 per V100.
config.fp16 = True
config.momentum = 0.9
config.weight_decay = 5e-4
config.gradient_clip = 0.5
config.batch_size = 64
config.gradient_acc = 1
config.lr = 0.0025
config.verbose = 2000
config.dali = False

# Strict baseline loading moves every trained PReLU slope into its curriculum
# wrapper and fills the fixed Alpha10/Alpha7 coefficient buffers.
config.backbone_init = "work_dirs/ms1mv3_r50/model.pt"
config.precise_relu_input_scale = 16.0
config.precise_relu_target_alphas = (7,)
config.precise_relu_lower_degrees = ()
config.precise_relu_initial_progress = 0.0
config.precise_relu_target_component_degrees = (7, 7)
config.precise_relu_target_multiplicative_depth = 7
config.precise_relu_approximation_error_bound = 0.125

# Preserve the exact polynomial forward while using the ordinary ReLU
# derivative. Combined with slope a, this is the baseline PReLU surrogate
# derivative and avoids ill-conditioned composite-polynomial gradients.
config.precise_relu_backward_mode = "relu_ste"

# Epochs 0-2 use Alpha10. Epochs 3-5 blend Alpha10 -> Alpha7 across the whole
# network; all remaining epochs fine-tune the final Alpha7-only forward graph.
config.precise_relu_stage_epochs = (3,)
config.precise_relu_transition_epochs = 2.0
config.precise_relu_require_final_stage = True

# These operations are plaintext training machinery and do not enter the FHE
# graph. Recalibration occurs once the Alpha7 transition is complete.
config.precise_relu_range_loss_weight = 0.1
config.precise_relu_bn_recalibration_batches = 200

# Keep the converted representation close to the successful PReLU checkpoint
# while PartialFC adapts the classifier and backbone to Alpha7 approximation.
config.embedding_teacher_network = "r50"
config.embedding_teacher_checkpoint = "work_dirs/ms1mv3_r50/model.pt"
config.embedding_distill_weight = 1.0

config.sync_bn = True
config.broadcast_buffers = True
config.check_finite_grads = True
config.ddp_fp16_compress = False
config.amp_init_scale = 1024.0
config.amp_growth_interval = 1000
config.fail_on_nonfinite_val = True
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
