import copy

import torch
from torch import nn

from backbones import get_model
from backbones.iresnet_precise_relu import ProgressivePrecisePReLU
from backbones.polynomial_relu import PreciseReLUAlpha7
from eval.non_linear_replacement import PreciseReLUAlpha7 as EvalAlpha7
from utils.utils_config import get_config


def test_memory_efficient_alpha7_matches_eval_forward_and_backward():
    torch.manual_seed(23)
    reference_input = (
        torch.empty(2, 3, 4, 5).uniform_(-15.0, 15.0).requires_grad_())
    efficient_input = reference_input.detach().clone().requires_grad_()
    output_weight = torch.randn_like(reference_input)

    reference_output = EvalAlpha7(16.0)(reference_input)
    efficient_output = PreciseReLUAlpha7(16.0)(efficient_input)
    (reference_output * output_weight).sum().backward()
    (efficient_output * output_weight).sum().backward()

    torch.testing.assert_close(
        efficient_output, reference_output, rtol=0, atol=0)
    torch.testing.assert_close(
        efficient_input.grad, reference_input.grad, rtol=2e-4, atol=2e-5)


def test_alpha7_autograd_saves_only_one_activation_sized_tensor():
    inputs = torch.randn(2, 3, 4, 5, requires_grad=True)
    saved_numels = []

    def pack(tensor):
        saved_numels.append(tensor.numel())
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        PreciseReLUAlpha7(16.0)(inputs).sum().backward()

    assert sum(numel >= inputs.numel() for numel in saved_numels) == 1


def test_alpha7_relu_ste_preserves_forward_and_uses_relu_backward():
    inputs = torch.tensor(
        [-16.2, -4.0, -0.25, 0.0, 0.25, 4.0, 16.2],
        requires_grad=True,
    )
    output_weight = torch.tensor([0.2, -0.4, 0.7, 1.0, -0.3, 0.6, -0.8])

    ste_output = PreciseReLUAlpha7(
        16.0, backward_mode="relu_ste")(inputs)
    exact_output = PreciseReLUAlpha7(16.0)(inputs.detach())
    torch.testing.assert_close(ste_output, exact_output, rtol=0, atol=0)
    (ste_output * output_weight).sum().backward()

    torch.testing.assert_close(
        inputs.grad,
        output_weight * (inputs.detach() > 0).to(output_weight.dtype),
    )


def test_progressive_prelu_transitions_from_alpha10_to_alpha7():
    activation = ProgressivePrecisePReLU(
        channels=2,
        input_scale=16.0,
        target_alphas=(7,),
        lower_degrees=(),
        progress=0.0,
    ).eval()
    with torch.no_grad():
        activation.prelu.weight.copy_(torch.tensor([0.1, 0.4]))
    inputs = torch.linspace(-15.0, 15.0, 66).reshape(1, 2, 3, 11)
    slope = activation.prelu.weight.reshape(1, 2, 1, 1)

    alpha10_expected = (
        slope * inputs + (1.0 - slope) * activation.alpha10(inputs))
    torch.testing.assert_close(activation(inputs), alpha10_expected)

    activation.set_degree_progress(1.0)
    alpha7_expected = (
        slope * inputs
        + (1.0 - slope) * PreciseReLUAlpha7(16.0)(inputs)
    )
    torch.testing.assert_close(activation(inputs), alpha7_expected)


def test_alpha7_r50_loads_baseline_strictly_and_ends_at_alpha7():
    baseline = get_model("r50", dropout=0, fp16=False)
    baseline_state = copy.deepcopy(baseline.state_dict())
    cfg = get_config("configs/ms1mv3_r50_precise_relu_alpha7_s16")
    model = get_model(
        cfg.network,
        dropout=0,
        fp16=False,
        precise_relu_input_scale=cfg.precise_relu_input_scale,
        precise_relu_target_alphas=cfg.precise_relu_target_alphas,
        precise_relu_lower_degrees=cfg.precise_relu_lower_degrees,
        precise_relu_progress=cfg.precise_relu_initial_progress,
        precise_relu_backward_mode=cfg.precise_relu_backward_mode,
    )
    model.load_state_dict(baseline_state, strict=True)

    activations = [
        module for module in model.modules()
        if isinstance(module, ProgressivePrecisePReLU)
    ]
    assert len(activations) == 25
    assert model.polynomial_transition_count() == 1
    assert model.polynomial_stage_names() == ("alpha10", "alpha7")
    assert all(isinstance(module.students[0], PreciseReLUAlpha7)
               for module in activations)
    assert {
        id(module) for module in model.modules()
        if isinstance(module, nn.PReLU)
    } == {id(activation.prelu) for activation in activations}
    torch.testing.assert_close(
        model.prelu.prelu.weight, baseline.prelu.weight)

    model.set_polynomial_progress(1.0)
    assert all(float(activation.progress) == 1.0
               for activation in activations)


def test_alpha7_configs_define_fixed_scale_comparison_runs():
    for config_name, expected_scale in (
            ("configs/ms1mv3_r50_precise_relu_alpha7_s16", 16.0),
            ("configs/ms1mv3_r50_precise_relu_alpha7_s24", 24.0),
            ("configs/ms1mv3_r50_precise_relu_alpha7_s32", 32.0)):
        cfg = get_config(config_name)
        assert cfg.network == "r50_precise_relu_alpha7"
        assert cfg.precise_relu_input_scale == expected_scale
        assert cfg.precise_relu_target_alphas == (7,)
        assert cfg.precise_relu_lower_degrees == ()
        assert cfg.precise_relu_target_component_degrees == (7, 7)
        assert cfg.precise_relu_target_multiplicative_depth == 7
        assert cfg.precise_relu_approximation_error_bound == (
            expected_scale / 128.0)
        assert cfg.precise_relu_backward_mode == "relu_ste"
        assert cfg.precise_relu_stage_epochs == (3,)
        assert cfg.precise_relu_transition_epochs == 2.0
        assert cfg.precise_relu_require_final_stage is True
        assert cfg.embedding_teacher_network == "r50"
        assert cfg.embedding_teacher_checkpoint == cfg.backbone_init
        assert cfg.embedding_distill_weight == 1.0
        assert cfg.precise_relu_range_loss_weight == 0.1
        assert cfg.precise_relu_bn_recalibration_batches == 200
        assert cfg.fp16 is True
        assert cfg.batch_size == 64
        assert cfg.batch_size * 4 == 256
        assert cfg.gradient_clip == 0.5
        assert cfg.precise_relu_stage_epochs[0] + (
            cfg.precise_relu_transition_epochs) < cfg.num_epoch
