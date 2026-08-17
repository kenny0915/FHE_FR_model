import copy
import importlib.util
import math
import sys
import types

import pytest
import torch
from torch import nn

from backbones import get_model
from backbones.iresnet_herpn_residual_scale import (
    PolynomialHerPN,
    ResidualScaledIBasicBlock,
)
from utils.utils_optimizer import split_weight_decay_parameters


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
            "_test_herpn_residual_scale_config", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.config
    finally:
        if previous is None:
            del sys.modules["easydict"]
        else:
            sys.modules["easydict"] = previous


def test_r50_is_pure_herpn_with_depth_scaled_residual_initialization():
    model = get_model(
        "r50_herpn_residual_scale", dropout=0, fp16=False)
    activations = [
        module for module in model.modules()
        if isinstance(module, PolynomialHerPN)
    ]
    blocks = [
        module for module in model.modules()
        if isinstance(module, ResidualScaledIBasicBlock)
    ]

    assert len(activations) == 25
    assert len(blocks) == 24
    assert not any(isinstance(module, nn.PReLU) for module in model.modules())
    expected = 1.0 / math.sqrt(24.0)
    assert all(
        float(block.residual_scale.detach()) == pytest.approx(expected)
        for block in blocks)
    assert float(model.herpn_progress.item()) == pytest.approx(5.0)


def test_nonzero_residual_scale_allows_task_gradients_from_step_zero():
    block = ResidualScaledIBasicBlock(
        4,
        4,
        activation_factory=lambda channels: PolynomialHerPN(channels),
        residual_scale_init=0.2,
    ).train()
    inputs = torch.randn(2, 4, 8, 8, requires_grad=True)
    outputs = block(inputs)
    outputs.square().mean().backward()

    assert block.residual_scale.grad is not None
    assert torch.isfinite(block.residual_scale.grad)
    assert block.conv1.weight.grad is not None
    assert float(block.conv1.weight.grad.abs().sum()) > 0.0
    assert block.prelu.weight.grad is not None
    assert float(block.prelu.weight.grad.abs().sum()) > 0.0


def test_residual_scale_folds_exactly_into_final_batchnorm():
    torch.manual_seed(4)
    block = ResidualScaledIBasicBlock(
        4,
        4,
        activation_factory=lambda channels: PolynomialHerPN(channels),
        residual_scale_init=0.2,
    ).eval()
    with torch.no_grad():
        block.residual_scale.fill_(0.37)
        block.bn3.running_mean.uniform_(-0.3, 0.3)
        block.bn3.running_var.uniform_(0.5, 1.5)
        block.bn3.weight.uniform_(0.6, 1.4)
        block.bn3.bias.uniform_(-0.2, 0.2)
    inputs = torch.randn(2, 4, 8, 8)
    expected = block(inputs)

    folded = copy.deepcopy(block).fold_residual_scale_for_inference()
    actual = folded(inputs)

    assert folded.residual_scale is None
    assert folded._residual_scale_folded
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_complete_inference_fold_removes_herpn_and_residual_multiplies():
    model = get_model(
        "r18_herpn_residual_scale", dropout=0, fp16=False).eval()
    folded = copy.deepcopy(model).fold_herpn_for_inference()

    assert not any(
        isinstance(module, PolynomialHerPN) for module in folded.modules())
    assert all(
        block.residual_scale is None
        for block in folded.residual_blocks())


def test_selective_weight_decay_excludes_herpn_and_residual_scales():
    model = get_model(
        "r18_herpn_residual_scale", dropout=0, fp16=False)
    decay, no_decay = split_weight_decay_parameters(model)
    decay_ids = {id(parameter) for parameter in decay}
    no_decay_ids = {id(parameter) for parameter in no_decay}

    for block in model.residual_blocks():
        assert id(block.residual_scale) in no_decay_ids
    for activation in model.polynomial_activations():
        assert id(activation.weight) in no_decay_ids
        assert id(activation.bias) in no_decay_ids
    assert id(model.conv1.weight) in decay_ids
    assert decay_ids.isdisjoint(no_decay_ids)


def test_ms1mv3_config_is_from_scratch_without_distillation_or_conversion():
    cfg = _load_standalone_config(
        "configs/ms1mv3_r50_herpn_residual_scale.py")

    assert cfg.network == "r50_herpn_residual_scale"
    assert "backbone_init" not in cfg
    assert cfg.herpn_initial_progress == pytest.approx(5.0)
    assert cfg.herpn_distill_loss_weight == pytest.approx(0.0)
    assert cfg.herpn_conversion_groups == ()
    assert cfg.herpn_stage_epochs == ()
    assert cfg.residual_scale_init == pytest.approx(1.0 / math.sqrt(24.0))
    assert cfg.residual_scale_trainable
    assert not cfg.fp16
    assert cfg.lr < 0.1
    assert cfg.warmup_epoch >= 2
    assert cfg.num_epoch > 20
