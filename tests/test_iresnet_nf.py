import importlib

import pytest
import torch
import torch.nn as nn

from backbones import get_model
from backbones.iresnet_nf import (
    NormFreeIResNet,
    NormFreeResidualBlock,
)
from backbones.poolformer_nf import (
    BoundedScalar,
    ScaledWSConv2d,
    SymmetricBoundedScalar,
)


def _tiny_model(num_classes=16):
    return NormFreeIResNet(
        layers=(1, 1, 1, 1),
        channels=(8, 16, 32, 64),
        num_classes=num_classes,
        fp16=False,
    )


def test_nf12_factory_has_twelve_residual_products_and_no_backbone_norm():
    model = get_model("iresnet_nf12", num_features=32, fp16=False)
    assert len(model.nf_blocks()) == 12
    assert all(
        not module.gain.requires_grad
        for module in model.modules()
        if isinstance(module, ScaledWSConv2d)
    )

    forbidden = (nn.LayerNorm, nn.GroupNorm, nn.modules.batchnorm._BatchNorm)
    backbone_norms = [
        name for name, module in model.named_modules()
        if not name.startswith("head") and isinstance(module, forbidden)
    ]
    assert backbone_norms == []


def test_quadratic_is_exactly_linear_at_initialization_and_scales_are_bounded():
    block = NormFreeResidualBlock(8, 8)
    block.set_range_tracking(True)
    inputs = torch.randn(2, 8, 7, 7)
    outputs = block(inputs)

    operand_u = block._last_range_tensors["operand_u"]
    modulator = block._last_range_tensors["modulator"]
    product = block._last_range_tensors["product"]
    assert torch.equal(modulator, torch.ones_like(modulator))
    assert torch.equal(product, operand_u)
    assert block.quadratic_scale().item() == pytest.approx(0.0, abs=1e-8)
    rho, alpha = block.residual_coefficients()
    assert alpha.item() == pytest.approx(0.02, abs=1e-6)
    assert 0.0 < alpha.item() < 0.1
    assert rho.add(alpha).item() == pytest.approx(
        1.0, abs=1e-6)
    assert torch.isfinite(outputs).all()


def test_quadratic_modulation_uses_explicit_interval_and_progress():
    block = NormFreeResidualBlock(
        8, 8, quadratic_scale_max=0.1,
        modulation_input_bound=5.0,
        initial_modulation_progress=0.0,
    )
    with torch.no_grad():
        block.quadratic_scale.raw.fill_(0.4)
    assert block.modulation_coefficient().item() == pytest.approx(0.0)

    block.set_modulation_progress(0.5)
    expected = block.quadratic_scale().item() * 0.5 / 5.0
    assert block.modulation_coefficient().item() == pytest.approx(expected)


def test_model_can_enable_quadratic_blocks_in_reverse_order():
    model = _tiny_model()
    model.set_nf_modulation_progresses((0.1, 0.2, 0.3, 0.4), order="reverse")
    assert model.nf_modulation_group_count() == 4
    assert [block.modulation_progress.item() for block in model.nf_blocks()] == (
        pytest.approx([0.4, 0.3, 0.2, 0.1]))


def test_stage_downsampling_matches_iresnet_112_to_7_topology():
    model = _tiny_model().eval()
    with torch.no_grad():
        features = model.forward_features(torch.randn(2, 3, 112, 112))
        embeddings = model(torch.randn(2, 3, 112, 112))
    assert features.shape == (2, 64, 7, 7)
    assert embeddings.shape == (2, 16)


def test_range_penalty_is_finite_and_reaches_quadratic_parameters():
    model = _tiny_model()
    model.set_nf_range_tracking(True)
    embeddings = model(torch.randn(2, 3, 112, 112))
    penalty = model.nf_range_penalty()
    summaries = model.nf_range_summary()
    loss = embeddings.square().mean() + 0.01 * penalty
    loss.backward()

    assert torch.isfinite(penalty)
    assert all("product_absmax" in summary and "output_rms" in summary
               for summary in summaries.values())
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
    assert all(
        block.quadratic_scale.raw.grad is not None
        for block in model.nf_blocks()
    )


def test_deploy_conversion_is_equivalent_and_removes_training_parameterizations():
    torch.manual_seed(11)
    model = _tiny_model().eval()
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, ScaledWSConv2d) and module.bias is not None:
                module.bias.uniform_(-0.02, 0.02)
        for block in model.nf_blocks():
            block.quadratic_scale.raw.fill_(0.35)
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


def test_nf12_training_config_uses_r50_teacher_and_stable_recipe():
    cfg = importlib.import_module(
        "configs.ms1mv3_iresnet_nf12_fp32").config
    assert cfg.network == "iresnet_nf12"
    assert cfg.resume is False
    assert cfg.fp16 is False
    assert cfg.embedding_teacher_network == "r50"
    assert cfg.embedding_distill_weight == pytest.approx(1.0)
    assert cfg.nf_range_loss_weight > 0.0
    assert cfg.nf_alpha_init < cfg.nf_alpha_max
    assert cfg.nf_quadratic_scale_max <= 0.1
    assert cfg.nf_modulation_input_bound == pytest.approx(cfg.nf_range_limit)
    assert cfg.nf_initial_modulation_progress == 0.0
    assert len(cfg.nf_modulation_group_epochs) == 12
    assert not cfg.nf_learnable_ws_gain
    assert cfg.gradient_clip_scope == "backbone"
    assert cfg.stable_gradient_clip
