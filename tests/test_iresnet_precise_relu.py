import copy

import torch
from torch import nn

from backbones import get_model
from backbones.iresnet_precise_relu import ProgressivePrecisePReLU
from backbones.polynomial_relu import (
    ChebyReLU,
    PreciseReLUAlpha10 as SharedPreciseReLUAlpha10,
)
from eval.non_linear_replacement import (
    ChebyReLU as EvalChebyReLU,
    PreciseReLUAlpha10 as EvalPreciseReLUAlpha10,
)
from utils.utils_config import get_config


def test_shared_alpha10_matches_successful_eval_implementation():
    inputs = torch.linspace(-8.0, 8.0, 257)
    expected = EvalPreciseReLUAlpha10(input_scale=8.0)(inputs)
    actual = SharedPreciseReLUAlpha10(input_scale=8.0)(inputs)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_memory_efficient_alpha10_backward_matches_reference():
    torch.manual_seed(17)
    reference_input = (
        torch.empty(2, 3, 4, 5).uniform_(-7.5, 7.5).requires_grad_())
    efficient_input = reference_input.detach().clone().requires_grad_()
    output_weight = torch.randn_like(reference_input)

    reference_output = EvalPreciseReLUAlpha10(8.0)(reference_input)
    efficient_output = SharedPreciseReLUAlpha10(8.0)(efficient_input)
    (reference_output * output_weight).sum().backward()
    (efficient_output * output_weight).sum().backward()

    torch.testing.assert_close(efficient_output, reference_output, rtol=0, atol=0)
    torch.testing.assert_close(
        efficient_input.grad,
        reference_input.grad,
        rtol=2e-4,
        atol=2e-5,
    )


def test_alpha10_autograd_saves_only_one_activation_sized_tensor():
    inputs = torch.randn(2, 3, 4, 5, requires_grad=True)
    saved_numels = []

    def pack(tensor):
        saved_numels.append(tensor.numel())
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        SharedPreciseReLUAlpha10(8.0)(inputs).sum().backward()

    assert sum(numel >= inputs.numel() for numel in saved_numels) == 1


def test_memory_efficient_cheby_backward_matches_reference():
    torch.manual_seed(19)
    for degree in (4, 8, 16):
        reference_input = (
            torch.empty(2, 3, 4, 5).uniform_(-7.5, 7.5).requires_grad_())
        efficient_input = reference_input.detach().clone().requires_grad_()
        output_weight = torch.randn_like(reference_input)

        reference_output = EvalChebyReLU(8.0, degree)(reference_input)
        efficient_output = ChebyReLU(8.0, degree)(efficient_input)
        (reference_output * output_weight).sum().backward()
        (efficient_output * output_weight).sum().backward()

        torch.testing.assert_close(
            efficient_output, reference_output, rtol=0, atol=0)
        torch.testing.assert_close(
            efficient_input.grad,
            reference_input.grad,
            rtol=2e-4,
            atol=2e-5,
        )


def test_cheby_autograd_saves_only_one_activation_sized_tensor():
    for degree in (4, 8, 16):
        inputs = torch.randn(2, 3, 4, 5, requires_grad=True)
        saved_numels = []

        def pack(tensor):
            saved_numels.append(tensor.numel())
            return tensor

        with torch.autograd.graph.saved_tensors_hooks(
                pack, lambda tensor: tensor):
            ChebyReLU(8.0, degree)(inputs).sum().backward()

        assert sum(numel >= inputs.numel() for numel in saved_numels) == 1


def test_memory_efficient_range_penalty_matches_reference_gradient():
    activation = ProgressivePrecisePReLU(
        channels=2, input_scale=2.0, progress=0.0).train()
    efficient_input = torch.tensor(
        [[[[3.0, -2.5], [0.5, -0.25]],
          [[-4.0, 2.25], [0.75, -1.0]]]],
        requires_grad=True,
    )
    reference_input = efficient_input.detach().clone().requires_grad_()

    activation(efficient_input)
    efficient_penalty = activation.range_penalty()
    reference_excess = torch.relu(reference_input.abs() - 2.0)
    reference_penalty = (
        reference_excess.square().mean()
        + 0.1 * reference_excess.flatten(1).amax(dim=1).square().mean()
    )
    efficient_penalty.backward()
    reference_penalty.backward()

    torch.testing.assert_close(efficient_penalty, reference_penalty)
    torch.testing.assert_close(
        efficient_input.grad, reference_input.grad, rtol=1e-6, atol=1e-7)


def test_relu_ste_keeps_polynomial_forward_and_uses_relu_backward():
    inputs = torch.tensor(
        [-8.2, -4.0, -0.25, 0.0, 0.25, 4.0, 8.2],
        requires_grad=True,
    )
    exact_inputs = inputs.detach().clone()
    output_weight = torch.tensor([0.2, -0.4, 0.7, 1.0, -0.3, 0.6, -0.8])

    ste = SharedPreciseReLUAlpha10(
        8.0, backward_mode="relu_ste")(inputs)
    exact_forward = SharedPreciseReLUAlpha10(8.0)(exact_inputs)
    torch.testing.assert_close(ste, exact_forward, rtol=0, atol=0)
    (ste * output_weight).sum().backward()
    torch.testing.assert_close(
        inputs.grad,
        output_weight * (inputs.detach() > 0).to(output_weight.dtype),
    )


def test_progressive_prelu_ste_has_ordinary_prelu_surrogate_gradient():
    activation = ProgressivePrecisePReLU(
        channels=2,
        input_scale=8.0,
        progress=0.0,
        backward_mode="relu_ste",
    ).eval()
    with torch.no_grad():
        activation.prelu.weight.copy_(torch.tensor([0.1, 0.4]))
    inputs = torch.tensor(
        [[[[-8.2, -0.5, 0.5, 8.2]],
          [[-8.2, -0.5, 0.5, 8.2]]]],
        requires_grad=True,
    )
    output_weight = torch.tensor(
        [[[[0.2, -0.3, 0.4, -0.5]],
          [[-0.6, 0.7, -0.8, 0.9]]]],
    )
    (activation(inputs) * output_weight).sum().backward()

    slope = activation.prelu.weight.detach().reshape(1, 2, 1, 1)
    expected_derivative = torch.where(
        inputs.detach() > 0, torch.ones_like(inputs), slope)
    torch.testing.assert_close(
        inputs.grad, output_weight * expected_derivative)


def test_progressive_activation_starts_at_alpha10_and_ends_at_degree4():
    activation = ProgressivePrecisePReLU(
        channels=2,
        input_scale=8.0,
        lower_degrees=(16, 8, 4),
        progress=0.0,
    ).eval()
    with torch.no_grad():
        activation.prelu.weight.copy_(torch.tensor([0.1, 0.4]))
    inputs = torch.linspace(-7.5, 7.5, 66).reshape(1, 2, 3, 11)
    slope = activation.prelu.weight.reshape(1, 2, 1, 1)

    alpha_expected = (
        slope * inputs
        + (1.0 - slope)
        * SharedPreciseReLUAlpha10(input_scale=8.0)(inputs)
    )
    torch.testing.assert_close(activation(inputs), alpha_expected)

    activation.set_degree_progress(3.0)
    degree4_expected = (
        slope * inputs
        + (1.0 - slope) * ChebyReLU(8.0, degree=4)(inputs)
    )
    torch.testing.assert_close(activation(inputs), degree4_expected)


def test_fractional_progress_blends_adjacent_independent_fits():
    activation = ProgressivePrecisePReLU(
        channels=1, input_scale=8.0, progress=1.25).eval()
    inputs = torch.linspace(-7.0, 7.0, 31).reshape(1, 1, 1, -1)
    slope = activation.prelu.weight.reshape(1, 1, 1, 1)
    relu16 = ChebyReLU(8.0, degree=16)(inputs)
    relu8 = ChebyReLU(8.0, degree=8)(inputs)
    expected_relu = 0.75 * relu16 + 0.25 * relu8
    expected = slope * inputs + (1.0 - slope) * expected_relu
    torch.testing.assert_close(activation(inputs), expected)


def test_r50_replaces_every_original_prelu_and_loads_baseline_strictly():
    baseline = get_model("r50", dropout=0, fp16=False)
    baseline_state = copy.deepcopy(baseline.state_dict())
    model = get_model(
        "r50_precise_relu",
        dropout=0,
        fp16=False,
        precise_relu_input_scale=8.0,
        precise_relu_lower_degrees=(16, 8, 4),
        precise_relu_progress=0.0,
    )
    model.load_state_dict(baseline_state, strict=True)

    activations = [
        module for module in model.modules()
        if isinstance(module, ProgressivePrecisePReLU)
    ]
    assert len(activations) == 25
    # ``train_v2.set_prepbn_progress`` reserves ``set_progress(step, total)``
    # for RepBatchNorm-style modules. PreciseReLU must not match that protocol.
    assert not any(hasattr(activation, "set_progress")
                   for activation in activations)
    assert {
        id(module) for module in model.modules()
        if isinstance(module, nn.PReLU)
    } == {id(activation.prelu) for activation in activations}
    torch.testing.assert_close(
        model.prelu.prelu.weight,
        baseline.prelu.weight,
    )
    assert model.polynomial_stage_names() == (
        "alpha10", "degree16", "degree8", "degree4")


def test_scale8_and_scale16_configs_reach_degree4_with_final_finetuning():
    for config_name, expected_scale in (
            ("configs/ms1mv3_r50_precise_relu_s8", 8.0),
            ("configs/ms1mv3_r50_precise_relu_s16", 16.0)):
        cfg = get_config(config_name)
        assert cfg.network == "r50_precise_relu"
        assert cfg.precise_relu_input_scale == expected_scale
        assert cfg.precise_relu_lower_degrees == (16, 8, 4)
        assert cfg.precise_relu_backward_mode == "relu_ste"
        assert cfg.fp16 is True
        assert cfg.batch_size == 64
        assert cfg.gradient_acc == 1
        assert cfg.batch_size * 4 == 256
        assert cfg.lr == 0.0025
        assert cfg.amp_init_scale == 1024.0
        assert len(cfg.precise_relu_stage_epochs) == 3
        assert all(
            right >= left + cfg.precise_relu_transition_epochs
            for left, right in zip(
                cfg.precise_relu_stage_epochs,
                cfg.precise_relu_stage_epochs[1:],
            )
        )
        final_transition = (
            cfg.precise_relu_stage_epochs[-1]
            + cfg.precise_relu_transition_epochs)
        assert final_transition < cfg.num_epoch
