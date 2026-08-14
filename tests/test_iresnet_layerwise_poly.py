import copy
import importlib.util
import sys
import types

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from backbones import get_model
from backbones.iresnet_layerwise_poly import (
    FoldedLayerwisePolynomial,
    LayerwisePolynomialActivation,
)


class _EasyDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


def _load_standalone_config(path):
    fake_easydict = types.ModuleType("easydict")
    fake_easydict.EasyDict = _EasyDict
    previous = sys.modules.get("easydict")
    sys.modules["easydict"] = fake_easydict
    try:
        spec = importlib.util.spec_from_file_location(
            "_test_layerwise_poly_config", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.config
    finally:
        if previous is None:
            del sys.modules["easydict"]
        else:
            sys.modules["easydict"] = previous


def _activation(degree=2, slopes=(0.1, 0.4), scale=6.0, blend=0.0):
    activation = LayerwisePolynomialActivation(
        len(slopes), degree=degree, blend=0.0)
    activation.load_state_dict(
        {"weight": torch.tensor(slopes)}, strict=True)
    activation.set_input_scale(scale)
    activation.set_blend(blend)
    return activation


def test_baseline_checkpoint_loads_strictly_and_blend_zero_is_exact():
    torch.manual_seed(3)
    baseline = get_model("r18", dropout=0, fp16=False).eval()
    state = copy.deepcopy(baseline.state_dict())
    polynomial = get_model(
        "r18_layerwise_poly",
        dropout=0,
        fp16=False,
        layerwise_poly_degree=2,
        layerwise_poly_progress=0.0,
    ).eval()
    polynomial.load_state_dict(state, strict=True)

    inputs = torch.randn(2, 3, 112, 112)
    with torch.no_grad():
        expected = baseline(inputs)
        actual = polynomial(inputs)
    assert torch.equal(actual, expected)
    assert len(polynomial.uncalibrated_layerwise_poly_names()) == 9


@pytest.mark.parametrize("degree", [2, 3])
def test_normalized_polynomial_always_matches_prelu_at_interval_endpoints(degree):
    slopes = torch.tensor([0.1, 0.4])
    activation = _activation(
        degree=degree, slopes=tuple(slopes), scale=6.0, blend=1.0).eval()
    with torch.no_grad():
        activation.theta2.copy_(torch.tensor([0.7, -1.2]).reshape(2, 1, 1))
        if activation.theta3 is not None:
            activation.theta3.copy_(
                torch.tensor([-0.8, 1.4]).reshape(2, 1, 1))

    inputs = torch.tensor([-6.0, 6.0]).reshape(2, 1, 1, 1).expand(-1, 2, 1, 1)
    expected = F.prelu(inputs, slopes)
    assert torch.allclose(
        activation(inputs), expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("degree", [2, 3])
def test_folded_polynomial_is_exact(degree):
    torch.manual_seed(7)
    activation = _activation(degree=degree, scale=4.5, blend=1.0).eval()
    with torch.no_grad():
        activation.theta2.uniform_(-1.0, 0.2)
        if activation.theta3 is not None:
            activation.theta3.uniform_(-0.3, 0.3)
    folded = activation.folded().eval()
    inputs = 4.0 * torch.randn(3, 2, 5, 5)

    assert isinstance(folded, FoldedLayerwisePolynomial)
    assert folded.degree == degree
    assert torch.allclose(
        folded(inputs), activation(inputs), rtol=1e-5, atol=1e-5)


def test_scale_is_required_before_conversion_and_is_checkpointed():
    activation = LayerwisePolynomialActivation(2, degree=2, blend=0.0)
    with pytest.raises(RuntimeError, match="Calibrate"):
        activation.set_blend(0.1)

    activation.set_input_scale(7.25)
    activation.set_blend(0.1)
    state = copy.deepcopy(activation.state_dict())
    restored = LayerwisePolynomialActivation(2, degree=2, blend=0.0)
    restored.load_state_dict(state, strict=True)

    assert bool(restored.scale_calibrated.item())
    assert float(restored.input_scale.item()) == pytest.approx(7.25)
    assert restored._blend == pytest.approx(0.1)


def test_relative_distillation_updates_coefficients_but_not_input():
    activation = _activation(degree=2, scale=6.0, blend=1.0).train()
    inputs = torch.randn(4, 2, 3, 3, requires_grad=True)
    activation(inputs)
    loss = activation.distillation_loss()

    assert torch.isfinite(loss)
    assert float(loss) > 0.0
    loss.backward()
    assert activation.theta2.grad is not None
    assert torch.isfinite(activation.theta2.grad).all()
    assert inputs.grad is None or torch.count_nonzero(inputs.grad) == 0


def test_batchnorm_refresh_keeps_measured_upstream_prefix_fixed():
    model = get_model(
        "r18_layerwise_poly", dropout=0, fp16=False,
        layerwise_poly_degree=2).train()
    state = model.begin_batchnorm_recalibration_after(
        "layer1.0.prelu", reset=False)
    selected = {id(module) for module, _, _ in state["batchnorm"]}

    assert isinstance(model.layer1[0].bn2, nn.BatchNorm2d)
    assert id(model.bn1) not in selected
    assert id(model.layer1[0].bn1) not in selected
    assert id(model.layer1[0].bn2) not in selected
    assert id(model.layer1[0].bn3) in selected
    assert id(model.layer1[1].bn1) in selected
    model.end_batchnorm_recalibration(state)
    assert model.training


def test_r50_config_converts_every_activation_singly_in_forward_order():
    cfg = _load_standalone_config(
        "configs/ms1mv3_r50_layerwise_poly.py")
    model = get_model(
        cfg.network,
        dropout=0,
        fp16=False,
        layerwise_poly_degree=cfg.layerwise_poly_degree,
        layerwise_poly_initial_scale=cfg.layerwise_poly_initial_scale,
        layerwise_poly_distill_eps=cfg.layerwise_poly_distill_eps,
        layerwise_poly_progress=cfg.herpn_initial_progress,
    )
    expected_order = model.layerwise_poly_activation_names()
    scheduled_order = [
        name for group in cfg.herpn_conversion_groups for name in group
    ]

    assert len(expected_order) == 25
    assert all(len(group) == 1 for group in cfg.herpn_conversion_groups)
    assert scheduled_order == expected_order
    assert cfg.layerwise_poly_range_calibration_batches == 0
    assert cfg.layerwise_poly_range_margin >= 1.0
    assert all(
        right >= left + cfg.herpn_transition_epochs
        for left, right in zip(
            cfg.herpn_group_epochs, cfg.herpn_group_epochs[1:])
    )
    final_conversion_epoch = (
        cfg.herpn_group_epochs[-1] + cfg.herpn_transition_epochs)
    assert cfg.num_epoch - final_conversion_epoch == 4
