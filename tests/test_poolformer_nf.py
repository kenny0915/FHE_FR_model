import importlib

import pytest
import torch
import torch.nn as nn

from backbones import get_model
from backbones.poolformer_nf import (
    BoundedScalar,
    NormFreeGatedBlock,
    NormFreePoolFormer,
    ScaledWSConv2d,
    SymmetricBoundedScalar,
)


def _tiny_model(num_classes=16):
    return NormFreePoolFormer(
        layers=(1, 1, 1, 1),
        embed_dims=(8, 16, 32, 64),
        num_classes=num_classes,
        fp16=False,
    )


def test_nf12_factory_has_twelve_gates_and_no_backbone_activation_norm():
    model = get_model("poolformer_nf12", num_features=32, fp16=False)
    assert len(model.nf_blocks()) == 12

    forbidden = (nn.LayerNorm, nn.GroupNorm, nn.modules.batchnorm._BatchNorm)
    backbone_norms = [
        name for name, module in model.named_modules()
        if not name.startswith("head") and isinstance(module, forbidden)
    ]
    assert backbone_norms == []


def test_gate_is_exactly_linear_at_initialization_and_coefficients_are_bounded():
    block = NormFreeGatedBlock(dim=8)
    block.set_range_tracking(True)
    inputs = torch.randn(2, 8, 7, 7)
    outputs = block(inputs)

    assert torch.equal(
        block._last_range_tensors["operand_v"],
        torch.ones_like(block._last_range_tensors["operand_v"]),
    )
    rho, alpha = block.residual_coefficients()
    assert alpha.item() == pytest.approx(0.05, abs=1e-6)
    assert 0.0 < alpha.item() < 0.2
    assert rho.square().add(alpha.square()).item() == pytest.approx(
        1.0, abs=1e-6)
    assert torch.isfinite(outputs).all()


def test_token_mixer_is_nonexpansive_in_infinity_norm():
    block = NormFreeGatedBlock(dim=8)
    inputs = torch.randn(2, 8, 9, 9)
    mixed = block.token_mixer(inputs)
    assert mixed.abs().max() <= inputs.abs().max() + 1e-6


def test_range_penalty_is_finite_and_differentiable():
    model = _tiny_model()
    model.set_nf_range_tracking(True)
    embeddings = model(torch.randn(2, 3, 112, 112))
    penalty = model.nf_range_penalty()
    loss = embeddings.square().mean() + 0.01 * penalty
    loss.backward()

    assert torch.isfinite(penalty)
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_range_penalty_has_finite_gradient_for_extreme_finite_outlier():
    block = NormFreeGatedBlock(dim=8)
    extreme = torch.tensor([1e30], requires_grad=True)
    block._last_range_tensors = {
        "operand_u": extreme,
        "operand_v": extreme,
        "product": extreme,
        "output": extreme,
    }

    penalty = block.range_penalty()
    penalty.backward()

    assert torch.isfinite(penalty)
    assert torch.isfinite(extreme.grad)


def test_deploy_conversion_folds_sws_and_scalar_parameterizations():
    torch.manual_seed(7)
    model = _tiny_model().eval()
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, ScaledWSConv2d) and module.bias is not None:
                module.bias.uniform_(-0.02, 0.02)
        for block in model.nf_blocks():
            block.modulator_scale.raw.fill_(0.4)
    inputs = torch.randn(2, 3, 112, 112)
    with torch.no_grad():
        expected = model(inputs)
        deployed = model.switch_to_deploy(inplace=False)
        actual = deployed(inputs)

    assert torch.allclose(actual, expected, rtol=2e-5, atol=2e-5)
    assert not any(isinstance(module, ScaledWSConv2d)
                   for module in deployed.modules())
    assert not any(isinstance(module, (BoundedScalar, SymmetricBoundedScalar))
                   for module in deployed.modules())
    assert not any(isinstance(module, nn.modules.batchnorm._BatchNorm)
                   for module in deployed.modules())
    assert any(isinstance(module, ScaledWSConv2d)
               for module in model.modules())


def test_nf12_training_config_matches_stable_recipe():
    cfg = importlib.import_module(
        "configs.ms1mv3_poolformer_nf12_fp32").config
    assert cfg.network == "poolformer_nf12"
    assert cfg.resume is False
    assert cfg.fp16 is False
    assert cfg.embedding_distill_weight == pytest.approx(1.0)
    assert cfg.nf_range_loss_weight > 0.0
    assert cfg.nf_alpha_init < cfg.nf_alpha_max
    assert cfg.nf_alpha_max == pytest.approx(0.1)
    assert cfg.nf_input_gain_max == pytest.approx(1.5)
    assert cfg.nf_modulator_scale_max == pytest.approx(0.1)
    assert cfg.gradient_clip_scope == "backbone"
