import torch

from backbones.poolformer_no_ln_x2_act import (
    Mlp,
    SimpleGate,
    poolformer_s24_mlp2,
)


def test_simple_gate_matches_split_product_and_captures_ranges_and_gradients():
    gate = SimpleGate(range_limit=1.0, sample_size=1024)
    gate.set_blend(1.0)
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


def test_progressive_gate_starts_at_gelu_and_ends_at_product():
    gate = SimpleGate(initial_blend=0.0)
    inputs = torch.tensor(
        [[[[1.0]], [[-2.0]], [[0.5]], [[3.0]]]],
    )
    first, second = inputs.chunk(2, dim=1)
    assert torch.allclose(gate(inputs), torch.nn.functional.gelu(first))

    gate.set_blend(0.25)
    expected = torch.lerp(
        torch.nn.functional.gelu(first), first * second, 0.25)
    assert torch.allclose(gate(inputs), expected)

    gate.set_blend(1.0)
    assert torch.allclose(gate(inputs), first * second)


def test_distillation_warms_multiplier_while_main_path_is_gelu():
    gate = SimpleGate(initial_blend=0.0, sample_size=1024)
    gate.set_auxiliary_losses(True)
    inputs = torch.randn(2, 4, 3, 3, requires_grad=True)
    output = gate(inputs)
    loss = output.mean() + gate.distillation_loss()
    loss.backward()

    assert inputs.grad[:, 2:].abs().sum().item() > 0
    assert torch.isfinite(gate.range_penalty())


def test_multiplier_initialization_is_local_gelu_approximation():
    mlp = Mlp(4, hidden_features=8)
    half = mlp.fc1.out_channels // 2
    assert torch.allclose(
        mlp.fc1.weight[half:],
        0.3989422804014327 * mlp.fc1.weight[:half],
    )
    assert torch.allclose(
        mlp.fc1.bias[half:],
        0.5 + 0.3989422804014327 * mlp.fc1.bias[:half],
    )


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
    assert [len(group) for group in model.simple_gate_group_names()] == [4, 4, 4, 4, 4, 4]
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
    model.set_simple_gate_blends((1.0,) * 6)
    model.set_simple_gate_instrumentation(True)
    with torch.no_grad():
        output = model(torch.randn(1, 3, 112, 112))
    assert output.shape == (1, 16)
    assert torch.isfinite(output).all()
    stats = model.simple_gate_range_stats()
    assert len(stats) == 24
    assert all("residual_scale_absmax" in layer for layer in stats.values())
