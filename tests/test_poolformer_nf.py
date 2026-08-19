import importlib

import pytest
import torch
import torch.nn as nn

from backbones import get_model
from backbones.poolformer_nf import (
    BoundedScalar,
    FixedScalar,
    NormFreeGatedBlock,
    NormFreePoolFormer,
    ScaledWSConv2d,
    SymmetricBoundedScalar,
)


def _tiny_model(num_classes=16, **kwargs):
    return NormFreePoolFormer(
        layers=(1, 1, 1, 1),
        embed_dims=(8, 16, 32, 64),
        num_classes=num_classes,
        fp16=False,
        **kwargs,
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
    sws_convs = [
        module for module in model.modules()
        if isinstance(module, ScaledWSConv2d)
    ]
    assert sws_convs
    assert all(module.bias is None for module in sws_convs)
    assert all(not module.gain.requires_grad for module in sws_convs)


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
    assert rho.add(alpha).item() == pytest.approx(
        1.0, abs=1e-6)
    assert torch.isfinite(outputs).all()


def test_rezero_fixed_gate_block_starts_as_identity_with_bounded_residual():
    torch.manual_seed(5)
    block = NormFreeGatedBlock(
        dim=8,
        residual_mode="rezero",
        alpha_init=0.0,
        alpha_max=1.0 / 12.0,
        fixed_modulator_scale=0.02,
        initial_modulation_progress=1.0,
    )
    inputs = torch.randn(2, 8, 7, 7)
    outputs = block(inputs)
    rho, alpha = block.residual_coefficients()

    assert torch.equal(outputs, inputs)
    assert rho.item() == pytest.approx(1.0)
    assert alpha.item() == pytest.approx(0.0)
    assert isinstance(block.modulator_scale, FixedScalar)
    assert block.modulation_coefficient().item() == pytest.approx(0.02)
    assert list(block.modulator_scale.parameters()) == []

    with torch.no_grad():
        block.alpha.raw.fill_(100.0)
    assert block.alpha().item() <= 1.0 / 12.0


def test_registered_rezero_fixed_gate_nf12_uses_stable_scalars():
    model = get_model(
        "poolformer_nf12_rezero_fixed_gate",
        num_features=32,
        fp16=False,
    )
    blocks = model.nf_blocks()

    assert len(blocks) == 12
    assert all(block.residual_mode == "rezero" for block in blocks)
    assert all(block.alpha().item() == pytest.approx(0.0)
               for block in blocks)
    assert all(isinstance(block.modulator_scale, FixedScalar)
               for block in blocks)
    assert all(block.modulator_scale().item() == pytest.approx(0.02)
               for block in blocks)


def test_token_mixer_is_nonexpansive_in_infinity_norm():
    block = NormFreeGatedBlock(dim=8)
    inputs = torch.randn(2, 8, 9, 9)
    mixed = block.token_mixer(inputs)
    assert mixed.abs().max() <= inputs.abs().max() + 1e-6


def test_reverse_modulation_schedule_maps_first_group_to_last_block():
    model = _tiny_model()
    model.set_nf_modulation_progresses(
        (0.25, 0.0, 0.0, 0.0), order="reverse")
    blocks = model.nf_blocks()

    assert blocks[-1].modulation_progress.item() == pytest.approx(0.25)
    assert all(block.modulation_progress.item() == pytest.approx(0.0)
               for block in blocks[:-1])


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


def test_rezero_fixed_gate_deploy_conversion_preserves_output():
    torch.manual_seed(9)
    model = _tiny_model(
        residual_mode="rezero",
        alpha_init=0.0,
        alpha_max=0.1,
        fixed_modulator_scale=0.02,
        initial_modulation_progress=1.0,
    ).eval()
    with torch.no_grad():
        for block in model.nf_blocks():
            block.alpha.raw.fill_(0.25)
    inputs = torch.randn(2, 3, 112, 112)

    with torch.no_grad():
        expected = model(inputs)
        deployed = model.switch_to_deploy(inplace=False)
        actual = deployed(inputs)

    assert torch.allclose(actual, expected, rtol=2e-5, atol=2e-5)
    assert not any(isinstance(module, FixedScalar)
                   for module in deployed.modules())


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
    assert cfg.nf_input_gain_max == pytest.approx(1.25)
    assert cfg.nf_modulator_scale_max == pytest.approx(0.02)
    assert cfg.nf_initial_modulation_progress == pytest.approx(0.0)
    assert cfg.nf_learnable_ws_gain is False
    assert len(cfg.nf_modulation_group_epochs) == 12
    assert cfg.nf_modulation_order == "reverse"
    assert cfg.gradient_clip_scope == "backbone"


def test_rezero_fixed_gate_config_is_fresh_bounded_student():
    cfg = importlib.import_module(
        "configs.ms1mv3_poolformer_nf12_rezero_fixed_gate_fp32").config

    assert cfg.network == "poolformer_nf12_rezero_fixed_gate"
    assert cfg.resume is False
    assert "backbone_init" not in cfg
    assert cfg.nf_residual_mode == "rezero"
    assert cfg.nf_alpha_init == pytest.approx(0.0)
    assert cfg.nf_alpha_max == pytest.approx(1.0 / 12.0)
    assert cfg.nf_fixed_modulator_scale == pytest.approx(0.02)
    assert cfg.nf_learnable_ws_gain is False
    assert cfg.embedding_distill_weight > 0.0
    assert cfg.nf_range_loss_weight > 0.0
