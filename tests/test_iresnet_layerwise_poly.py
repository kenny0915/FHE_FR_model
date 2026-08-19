import copy
import importlib.util
import json
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
from eval.non_linear_replacement import PReLU_Approx


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
        activation.beta2.copy_(torch.tensor([0.7, -1.2]).reshape(2, 1, 1))
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
        activation.beta2.uniform_(-1.0, 0.2)
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
    assert activation.beta2.grad is not None
    assert torch.isfinite(activation.beta2.grad).all()
    assert inputs.grad is None or torch.count_nonzero(inputs.grad) == 0


def test_beta2_distillation_gradient_stays_bounded_at_extreme_scale():
    activation = _activation(
        degree=2, slopes=(0.1, 0.4), scale=1.0e12, blend=0.0).train()
    inputs = torch.tensor(
        [[[[-2.0, -0.5], [0.5, 2.0]],
          [[-3.0, -1.0], [1.0, 3.0]]]],
        dtype=torch.float32,
    )
    activation(inputs)
    loss = activation.distillation_loss()
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(activation.beta2.grad).all()
    assert float(activation.beta2.grad.abs().max()) < 100.0


def test_degree8_initializes_from_eval_chebyrelu_and_folds_exactly():
    slopes = torch.tensor([0.1, 0.4])
    activation = _activation(
        degree=8, slopes=tuple(slopes), scale=6.0, blend=1.0).eval()
    evaluation = PReLU_Approx(
        slopes, input_scale=6.0, polynomial_degree=8).eval()
    inputs = torch.linspace(-6.0, 6.0, 192).reshape(2, 2, 4, 12)

    expected = evaluation(inputs)
    actual = activation(inputs)
    folded = activation.folded().eval()(inputs)

    assert torch.allclose(actual, expected, rtol=1e-5, atol=5e-6)
    assert torch.allclose(folded, actual, rtol=1e-5, atol=5e-6)
    assert torch.count_nonzero(activation.cheby_residuals) == 0


def test_degree8_baseline_load_and_named_scale_initialization_are_exact():
    baseline = get_model("r18", dropout=0, fp16=False).eval()
    polynomial = get_model(
        "r18_layerwise_poly", dropout=0, fp16=False,
        layerwise_poly_degree=8, layerwise_poly_progress=0.0).eval()
    polynomial.load_state_dict(copy.deepcopy(baseline.state_dict()), strict=True)
    scales = {
        name: float(index + 1)
        for index, name in enumerate(polynomial.layerwise_poly_activation_names())
    }

    loaded = polynomial.load_layerwise_poly_input_scales({
        "polynomial_degree": 8,
        "scales": scales,
    })
    inputs = torch.randn(1, 3, 112, 112)

    assert loaded == len(scales)
    assert not polynomial.uncalibrated_layerwise_poly_names()
    assert torch.equal(polynomial(inputs), baseline(inputs))
    assert len(polynomial.layerwise_poly_parameters()) == len(scales)


def test_legacy_theta2_checkpoint_migrates_exactly_to_beta2():
    source = _activation(
        degree=2, slopes=(0.1, 0.4), scale=7.25, blend=0.6).eval()
    with torch.no_grad():
        source.beta2.copy_(torch.tensor([0.7, -1.2]).reshape(2, 1, 1))
    inputs = torch.linspace(-7.0, 7.0, 28).reshape(2, 2, 7, 1)
    expected = source(inputs)

    legacy = copy.deepcopy(source.state_dict())
    slope = legacy["prelu.weight"].reshape(-1, 1, 1)
    even = 0.5 * (1.0 - slope)
    legacy["theta2"] = legacy.pop("beta2") / legacy["input_scale"] - even

    restored = LayerwisePolynomialActivation(2, degree=2, blend=0.0).eval()
    restored.load_state_dict(legacy, strict=True)

    assert restored._loaded_legacy_theta2
    assert torch.allclose(restored.beta2, source.beta2, rtol=1e-5, atol=1e-5)
    assert torch.allclose(restored(inputs), expected, rtol=1e-5, atol=1e-5)


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


def test_polynomial_parameter_selection_is_group_local():
    model = get_model(
        "r18_layerwise_poly", dropout=0, fp16=False,
        layerwise_poly_degree=3).train()
    names = model.layerwise_poly_activation_names()
    selected = model.layerwise_poly_parameters(names[:2])
    expected = []
    activations = dict(model.named_progressive_activations())
    for name in names[:2]:
        expected.extend([
            activations[name].beta2,
            activations[name].theta3,
        ])

    assert {id(parameter) for parameter in selected} == {
        id(parameter) for parameter in expected}
    assert len(selected) == 4


def test_range_conditioning_penalty_can_select_only_pending_group():
    model = get_model(
        "r18_layerwise_poly", dropout=0, fp16=False,
        layerwise_poly_degree=2).train()
    activations = dict(model.named_progressive_activations())
    names = model.layerwise_poly_activation_names()[:2]
    for index, name in enumerate(names):
        activation = activations[name]
        activation.set_input_scale(1.0)
        channels = activation.prelu.num_parameters
        activation(torch.full((2, channels, 2, 2), 2.0 + index))

    first_penalty = model.herpn_range_penalty((names[0],))
    second_penalty = model.herpn_range_penalty((names[1],))
    combined_penalty = model.herpn_range_penalty(names)

    assert torch.equal(first_penalty, activations[names[0]].range_penalty())
    assert torch.equal(second_penalty, activations[names[1]].range_penalty())
    assert torch.allclose(
        combined_penalty, 0.5 * (first_penalty + second_penalty))
    with pytest.raises(ValueError, match="Unknown layerwise"):
        model.herpn_range_penalty(("missing.prelu",))


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
    assert 0.0 < cfg.layerwise_poly_range_quantile < 1.0
    assert 0.0 < cfg.layerwise_poly_range_holdout_fraction < 0.5
    assert cfg.layerwise_poly_max_tail_ratio > 1.0
    assert cfg.layerwise_poly_max_scale_growth > 1.0
    assert cfg.layerwise_poly_max_input_scale > 0.0
    assert all(
        right >= left + cfg.herpn_transition_epochs
        for left, right in zip(
            cfg.herpn_group_epochs, cfg.herpn_group_epochs[1:])
    )
    final_conversion_epoch = (
        cfg.herpn_group_epochs[-1] + cfg.herpn_transition_epochs)
    assert cfg.num_epoch - final_conversion_epoch == 4


def test_r50_group4_config_is_stage_aligned_and_finishes_with_joint_tuning():
    cfg = _load_standalone_config(
        "configs/ms1mv3_r50_layerwise_poly_group4.py")
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

    assert scheduled_order == expected_order
    assert [len(group) for group in cfg.herpn_conversion_groups] == [
        4, 4, 4, 4, 4, 2, 3]
    assert max(map(len, cfg.herpn_conversion_groups)) == 4
    assert cfg.layerwise_poly_staged_training
    assert not cfg.layerwise_poly_freeze_backbone_during_local_fit
    assert cfg.layerwise_poly_allow_provisional_tail_conditioning
    assert cfg.layerwise_poly_strict_recalibrate_before_blend
    assert cfg.layerwise_poly_conditioning_backbone_lr_scale == pytest.approx(
        0.01)
    assert cfg.layerwise_poly_conditioning_range_loss_weight == pytest.approx(
        1.0)
    assert cfg.layerwise_poly_blend_backbone_lr_scale == pytest.approx(0.1)
    assert cfg.layerwise_poly_final_backbone_lr_scale == pytest.approx(0.1)
    assert cfg.layerwise_poly_range_margin == pytest.approx(1.5)
    assert cfg.herpn_bn_recalibration_batches == 1000
    final_conversion_epoch = (
        cfg.herpn_group_epochs[-1] + cfg.herpn_transition_epochs)
    assert cfg.num_epoch - final_conversion_epoch == 7


def test_r50_group4_epoch3_resume_config_keeps_output_and_schedule():
    base_cfg = _load_standalone_config(
        "configs/ms1mv3_r50_layerwise_poly_group4.py")
    resume_cfg = _load_standalone_config(
        "configs/ms1mv3_r50_layerwise_poly_group4_resume_epoch3.py")

    assert resume_cfg.resume
    assert resume_cfg.output == base_cfg.output
    assert resume_cfg.herpn_conversion_groups == base_cfg.herpn_conversion_groups
    assert resume_cfg.herpn_group_epochs == base_cfg.herpn_group_epochs
    assert resume_cfg.num_epoch == base_cfg.num_epoch
    assert (resume_cfg.herpn_group_epochs[0]
            + resume_cfg.herpn_transition_epochs) == pytest.approx(3.0)
    assert resume_cfg.herpn_group_epochs[1] == pytest.approx(4.0)


def test_r50_cheby8_config_uses_pretrained_checkpoint_and_saved_scales():
    cfg = _load_standalone_config(
        "configs/ms1mv3_r50_cheby8_finetune.py")
    model = get_model(
        cfg.network,
        dropout=0,
        fp16=False,
        layerwise_poly_degree=cfg.layerwise_poly_degree,
        layerwise_poly_initial_scale=cfg.layerwise_poly_initial_scale,
        layerwise_poly_distill_eps=cfg.layerwise_poly_distill_eps,
        layerwise_poly_progress=cfg.herpn_initial_progress,
    )
    with open(cfg.layerwise_poly_scale_file) as scale_handle:
        scale_data = json.load(scale_handle)
    loaded = model.load_layerwise_poly_input_scales(scale_data)
    expected_order = model.layerwise_poly_activation_names()
    scheduled_order = [
        name for group in cfg.herpn_conversion_groups for name in group]

    assert cfg.backbone_init == "work_dirs/ms1mv3_r50/model.pt"
    assert cfg.layerwise_poly_degree == 8
    assert scale_data["polynomial_degree"] == 8
    assert loaded == 25
    assert scheduled_order == expected_order
    assert max(map(len, cfg.herpn_conversion_groups)) == 4
    assert len(cfg.herpn_conversion_groups) == 12
    assert cfg.layerwise_poly_staged_training
    assert cfg.layerwise_poly_freeze_backbone_during_local_fit
    assert cfg.fp16 is False
    assert cfg.gradient_acc == 4
    assert cfg.normalize_gradient_accumulation
    final_conversion_epoch = (
        cfg.herpn_group_epochs[-1] + cfg.herpn_transition_epochs)
    assert cfg.num_epoch - final_conversion_epoch == 8
