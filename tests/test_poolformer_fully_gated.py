import pytest
import torch
import torch.nn as nn

from backbones import get_model
from backbones.poolformer_fully_gated import (
    FullyGatedPoolFormer,
    GatedMlp,
    LayerNorm2d,
    SimpleGate,
    poolformer_fully_gated_s24,
)


def test_simple_gate_is_split_product():
    inputs = torch.tensor([[[[2.0]], [[-3.0]], [[4.0]], [[5.0]]]])
    expected = torch.tensor([[[[8.0]], [[-15.0]]]])
    assert torch.equal(SimpleGate()(inputs), expected)


def test_layer_norm_is_channelwise_at_each_spatial_position():
    inputs = torch.randn(2, 8, 3, 5)
    output = LayerNorm2d(8)(inputs)

    assert torch.allclose(
        output.mean(dim=1), torch.zeros_like(output[:, 0]), atol=1e-6)
    assert torch.allclose(
        output.square().mean(dim=1),
        torch.ones_like(output[:, 0]),
        atol=5e-4,
    )


def test_gated_mlp_uses_exact_nafnet_expansion():
    mlp = GatedMlp(64, pre_gate_features=128)
    assert mlp.fc1.in_channels == 64
    assert mlp.fc1.out_channels == 128
    assert mlp.fc2.in_channels == 64
    assert mlp.fc2.out_channels == 64


def test_s24_has_all_gates_active_and_zero_residual_scales():
    model = poolformer_fully_gated_s24(
        face_embedding=False, num_classes=16, fp16=False)
    gates = [module for module in model.modules()
             if isinstance(module, SimpleGate)]

    assert len(gates) == 24
    assert not any(isinstance(module, nn.GELU) for module in model.modules())
    first_block = model.network[0][0]
    assert torch.count_nonzero(first_block.layer_scale_1) == 0
    assert torch.count_nonzero(first_block.layer_scale_2) == 0

    model.eval()
    with torch.no_grad():
        output = model(torch.randn(1, 3, 112, 112))
    assert output.shape == (1, 16)
    assert output.dtype == torch.float32
    assert torch.isfinite(output).all()


def test_registered_face_backbone_produces_fp32_embeddings():
    model = get_model(
        "poolformer_fully_gated_s24", num_features=32, fp16=False)
    model.eval()
    with torch.no_grad():
        output = model(torch.randn(2, 3, 112, 112))
    assert output.shape == (2, 32)
    assert output.dtype == torch.float32
    assert torch.isfinite(output).all()


def test_backbone_rejects_accidental_fp16_mode():
    with pytest.raises(ValueError, match="FP32 stability experiment"):
        poolformer_fully_gated_s24(
            face_embedding=False, num_classes=16, fp16=True)


def test_small_fully_gated_model_has_finite_backward_pass():
    model = FullyGatedPoolFormer(
        layers=[1],
        embed_dims=[8],
        ffn_expands=[2.0],
        downsamples=[False],
        num_classes=4,
        face_embedding=False,
        fp16=False,
    )
    output = model(torch.randn(2, 3, 16, 16))
    output.square().mean().backward()

    gradients = [
        parameter.grad for parameter in model.parameters()
        if parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
