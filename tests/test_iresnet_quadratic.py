import copy
import math

import torch

from backbones import get_model
from backbones.iresnet_quadratic import (
    FoldedNormalizedQuadratic,
    NormalizedQuadratic,
    ProgressiveQuadraticActivation,
)
from utils.utils_config import get_config


def test_quadratic_initialization_targets_prelu_slope():
    activation = ProgressiveQuadraticActivation(
        channels=2,
        input_scale=6.0,
        abs_quadratic_coefficient=1.0 / math.sqrt(2.0 * math.pi),
        blend=0.0,
    )
    baseline_state = {"weight": torch.tensor([0.1, 0.4])}
    activation.load_state_dict(baseline_state, strict=True)

    slope = torch.tensor([0.1, 0.4]).reshape(2, 1, 1)
    expected_linear = 0.5 * (1.0 + slope)
    even = 0.5 * (1.0 - slope)
    abs_coefficient = 1.0 / math.sqrt(2.0 * math.pi)
    expected_quadratic = 6.0 * abs_coefficient * even
    expected_constant = abs_coefficient * even / 6.0
    torch.testing.assert_close(
        activation.quadratic.coefficient1, expected_linear)
    torch.testing.assert_close(
        activation.quadratic.coefficient2, expected_quadratic)
    torch.testing.assert_close(
        activation.quadratic.coefficient0,
        expected_constant,
    )


def test_progress_zero_strictly_loads_and_matches_prelu_backbone():
    torch.manual_seed(3)
    baseline = get_model("r18", dropout=0, fp16=False).eval()
    state = copy.deepcopy(baseline.state_dict())
    quadratic = get_model(
        "r18_quadratic",
        dropout=0,
        fp16=False,
        quadratic_progress=0.0,
    ).eval()
    quadratic.load_state_dict(state, strict=True)

    inputs = torch.randn(2, 3, 112, 112)
    with torch.no_grad():
        expected = baseline(inputs)
        actual = quadratic(inputs)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_folded_quadratic_is_exact():
    torch.manual_seed(5)
    quadratic = NormalizedQuadratic(3, input_scale=4.0).eval()
    with torch.no_grad():
        quadratic.coefficient2.uniform_(-0.4, 0.4)
        quadratic.coefficient1.uniform_(0.2, 1.0)
        quadratic.coefficient0.uniform_(-0.1, 0.1)
    folded = FoldedNormalizedQuadratic.from_quadratic(quadratic).eval()
    inputs = torch.randn(2, 3, 5, 5)
    torch.testing.assert_close(
        folded(inputs), quadratic(inputs), rtol=1e-6, atol=1e-6)


def test_distillation_remains_active_at_full_conversion():
    activation = ProgressiveQuadraticActivation(
        channels=2, input_scale=6.0, blend=1.0).train()
    inputs = torch.randn(4, 2, 3, 3)
    output = activation(inputs)
    loss = activation.distillation_loss()
    assert output.shape == inputs.shape
    assert loss is not None
    assert torch.isfinite(loss)
    assert float(loss) > 0.0
    loss.backward()
    assert activation.quadratic.coefficient2.grad is not None
    assert torch.isfinite(
        activation.quadratic.coefficient2.grad).all()


def test_r50_schedule_covers_every_quadratic_activation():
    cfg = get_config("configs/ms1mv3_r50_quadratic")
    model = get_model(
        cfg.network,
        dropout=0,
        fp16=False,
        quadratic_input_scale=cfg.quadratic_input_scale,
        quadratic_range_limit=cfg.quadratic_range_limit,
        quadratic_abs_init=cfg.quadratic_abs_init,
        quadratic_progress=cfg.herpn_initial_progress,
    )
    expected = {
        name for name, module in model.named_modules()
        if isinstance(module, ProgressiveQuadraticActivation)
    }
    scheduled = {
        name for group in cfg.herpn_conversion_groups for name in group
    }
    assert len(expected) == 25
    assert scheduled == expected
    assert len(cfg.herpn_group_epochs) == len(
        cfg.herpn_conversion_groups)
    assert all(
        right >= left + cfg.herpn_transition_epochs
        for left, right in zip(
            cfg.herpn_group_epochs, cfg.herpn_group_epochs[1:])
    )
    assert (
        cfg.herpn_group_epochs[-1] + cfg.herpn_transition_epochs
        <= cfg.num_epoch
    )
    assert cfg.num_epoch - (
        cfg.herpn_group_epochs[-1] + cfg.herpn_transition_epochs
    ) == 4
    assert cfg.output == "work_dirs/ms1mv3_r50_quadratic"
    assert cfg.ddp_fp16_compress is False
