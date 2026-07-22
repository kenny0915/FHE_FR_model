import torch

from backbones.poolformer_no_ln_x2_act import (
    SimpleGate,
    poolformer_s24_mlp2,
)


def test_simple_gate_matches_split_product_and_captures_ranges_and_gradients():
    gate = SimpleGate(range_limit=1.0, sample_size=1024)
    gate.set_instrumentation(True, gradient_scale=8.0)
    inputs = torch.tensor(
        [[[[2.0]], [[-3.0]], [[4.0]], [[5.0]]]],
        requires_grad=True,
    )

    output = gate(inputs)
    expected = torch.tensor([[[[8.0]], [[-15.0]]]])
    assert torch.allclose(output, expected)
    (8.0 * output.sum()).backward()

    stats = gate.range_stats()
    assert stats["product_absmax"].item() == 15.0
    assert stats["product_outside_fraction"].item() == 1.0
    assert stats["gradient_absmax"].item() == 1.0
    assert stats["finite"].item() == 1.0
    assert stats["gradient_finite"].item() == 1.0


def test_s24_mlp2_uses_24_gates_mlp2_width_and_skip_init():
    model = poolformer_s24_mlp2(
        pretrained=False,
        face_embedding=False,
        num_classes=16,
        fp16=False,
        gate_stats_sample_size=256,
    )
    gates = [module for module in model.modules() if isinstance(module, SimpleGate)]

    assert len(gates) == 24
    assert model.network[0][0].mlp.fc1.out_channels == 4 * 64
    assert model.network[0][0].mlp.fc2.in_channels == 2 * 64
    assert torch.allclose(
        model.network[0][0].layer_scale_1,
        torch.zeros_like(model.network[0][0].layer_scale_1),
    )
    assert torch.allclose(
        model.network[0][0].layer_scale_2,
        torch.zeros_like(model.network[0][0].layer_scale_2),
    )

    model.eval()
    model.set_simple_gate_instrumentation(True)
    with torch.no_grad():
        output = model(torch.randn(1, 3, 112, 112))
    assert output.shape == (1, 16)
    assert torch.isfinite(output).all()
    stats = model.simple_gate_range_stats()
    assert len(stats) == 24
    assert all("residual_scale_absmax" in layer for layer in stats.values())
