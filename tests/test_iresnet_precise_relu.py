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
    PreciseReLUAlpha10 as EvalPreciseReLUAlpha10,
)
from utils.utils_config import get_config


def test_shared_alpha10_matches_successful_eval_implementation():
    inputs = torch.linspace(-8.0, 8.0, 257)
    expected = EvalPreciseReLUAlpha10(input_scale=8.0)(inputs)
    actual = SharedPreciseReLUAlpha10(input_scale=8.0)(inputs)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


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

    activation.set_progress(3.0)
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
