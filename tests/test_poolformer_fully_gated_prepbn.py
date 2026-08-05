from collections import OrderedDict

import pytest
import torch

from backbones import get_model
from backbones.poolformer_fully_gated import (
    FullyGatedPoolFormer,
    SimpleGate,
)
from backbones.poolformer_fully_gated_prepbn import (
    PRepBNFullyGatedPoolFormer,
    ProgressiveRepBatchNorm2d,
)


def _small_baseline():
    return FullyGatedPoolFormer(
        layers=[1],
        embed_dims=[8],
        ffn_expands=[2.0],
        downsamples=[False],
        num_classes=4,
        face_embedding=False,
        fp16=False,
    )


def _small_prepbn():
    return PRepBNFullyGatedPoolFormer(
        layers=[1],
        embed_dims=[8],
        ffn_expands=[2.0],
        downsamples=[False],
        num_classes=4,
        face_embedding=False,
        fp16=False,
    )


def test_layernorm_checkpoint_warm_start_is_exact_at_gamma_one():
    torch.manual_seed(7)
    baseline = _small_baseline().eval()
    prepbn = _small_prepbn().eval()
    incompatible = prepbn.load_backbone_init_state_dict(
        baseline.state_dict())

    inputs = torch.randn(2, 3, 16, 16)
    with torch.no_grad():
        expected = baseline(inputs)
        actual = prepbn(inputs)

    assert incompatible.unexpected_keys == []
    assert torch.equal(actual, expected)
    assert all(module._gamma == 1.0 for module in prepbn.prepbn_modules())


def test_warm_start_accepts_ddp_prefix_and_rejects_incomplete_checkpoint():
    baseline_state = _small_baseline().state_dict()
    ddp_state = OrderedDict(
        (f"module.{name}", value) for name, value in baseline_state.items())
    _small_prepbn().load_backbone_init_state_dict(ddp_state)

    incomplete = OrderedDict(baseline_state)
    incomplete.pop("norm.bias")
    with pytest.raises(RuntimeError, match="missing=.*norm.bias"):
        _small_prepbn().load_backbone_init_state_dict(incomplete)


def test_repbn_final_graph_matches_fixed_channel_affine():
    torch.manual_seed(11)
    module = ProgressiveRepBatchNorm2d(6, eta_init=0.75).eval()
    with torch.no_grad():
        module.bn.running_mean.copy_(torch.randn(6))
        module.bn.running_var.copy_(torch.rand(6) + 0.25)
        module.bn.weight.copy_(torch.randn(6))
        module.bn.bias.copy_(torch.randn(6))
    module.set_progress(current_step=10, total_steps=10)

    inputs = torch.randn(2, 6, 3, 5)
    scale, bias = module.equivalent_affine()
    expected = (
        scale.view(1, -1, 1, 1) * inputs
        + bias.view(1, -1, 1, 1)
    )

    with torch.no_grad():
        actual = module(inputs)
        module.ln.weight.fill_(123.0)
        module.ln.bias.fill_(-456.0)
        without_teacher = module(inputs)

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)
    assert torch.equal(without_teacher, actual)


def test_mid_transition_trains_layernorm_and_repbn_branches():
    module = ProgressiveRepBatchNorm2d(8)
    module.set_progress(current_step=5, total_steps=10)
    output = module(torch.randn(2, 8, 4, 4))
    output.square().mean().backward()

    assert module._gamma == 0.5
    assert module.ln.weight.grad is not None
    assert module.bn.weight.grad is not None
    assert module.eta.grad is not None
    assert all(torch.isfinite(parameter.grad).all() for parameter in (
        module.ln.weight, module.bn.weight, module.eta))


def test_gamma_zero_keeps_teacher_in_training_graph_only():
    module = ProgressiveRepBatchNorm2d(8, eta_init=0.0)
    module.set_progress(current_step=10, total_steps=10)
    module.train()
    module(torch.randn(2, 8, 4, 4)).square().mean().backward()

    assert module.ln.weight.grad is not None
    assert torch.count_nonzero(module.ln.weight.grad) == 0


def test_registered_s24_has_49_prepbn_sites_and_unchanged_gates():
    model = get_model(
        "poolformer_fully_gated_prepbn_s24",
        num_features=32,
        fp16=False,
    )

    assert len(model.prepbn_modules()) == 49
    assert sum(isinstance(module, SimpleGate)
               for module in model.modules()) == 24
