from collections import OrderedDict

import pytest
import torch

from backbones import get_model
from backbones.poolformer_fully_gated import (
    FullyGatedPoolFormer,
    LayerNorm2d,
    SimpleGate,
)
from backbones.poolformer_fully_gated_affine import (
    AffineFullyGatedPoolFormer,
    ProgressiveAffineNorm2d,
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


def _small_affine():
    return AffineFullyGatedPoolFormer(
        layers=[1],
        embed_dims=[8],
        ffn_expands=[2.0],
        downsamples=[False],
        num_classes=4,
        face_embedding=False,
        fp16=False,
    )


def test_layernorm_checkpoint_warm_start_is_exact_at_gamma_one():
    torch.manual_seed(19)
    baseline = _small_baseline().eval()
    affine = _small_affine().eval()
    incompatible = affine.load_backbone_init_state_dict(
        baseline.state_dict())

    inputs = torch.randn(2, 3, 16, 16)
    with torch.no_grad():
        expected = baseline(inputs)
        actual = affine(inputs)

    assert incompatible.unexpected_keys == []
    assert torch.equal(actual, expected)
    assert all(module._gamma == 1.0
               for module in affine.affine_norm_modules())


def test_warm_start_accepts_ddp_prefix_and_rejects_incomplete_checkpoint():
    baseline_state = _small_baseline().state_dict()
    ddp_state = OrderedDict(
        (f"module.{name}", value) for name, value in baseline_state.items())
    _small_affine().load_backbone_init_state_dict(ddp_state)

    incomplete = OrderedDict(baseline_state)
    incomplete.pop("norm.bias")
    with pytest.raises(RuntimeError, match="missing=.*norm.bias"):
        _small_affine().load_backbone_init_state_dict(incomplete)


def test_calibration_fits_per_channel_least_squares_solution():
    torch.manual_seed(23)
    module = ProgressiveAffineNorm2d(5).eval()
    batches = [
        torch.randn(3, 5, 4, 2),
        0.75 * torch.randn(2, 5, 3, 2) + 0.4,
    ]

    module.begin_calibration()
    targets = []
    with torch.no_grad():
        for inputs in batches:
            targets.append(module(inputs))
    diagnostics = module.finish_calibration(ridge=0.0, distributed=False)

    flat_x = torch.cat([
        inputs.permute(1, 0, 2, 3).reshape(5, -1)
        for inputs in batches
    ], dim=1).double()
    flat_y = torch.cat([
        target.permute(1, 0, 2, 3).reshape(5, -1)
        for target in targets
    ], dim=1).double()
    centered_x = flat_x - flat_x.mean(dim=1, keepdim=True)
    centered_y = flat_y - flat_y.mean(dim=1, keepdim=True)
    expected_scale = (
        (centered_x * centered_y).sum(dim=1)
        / centered_x.square().sum(dim=1)
    )
    expected_bias = flat_y.mean(dim=1) - expected_scale * flat_x.mean(dim=1)

    assert diagnostics["count"] == flat_x.shape[1]
    assert torch.allclose(
        module.affine.weight.double(), expected_scale, atol=1e-7, rtol=1e-6)
    assert torch.allclose(
        module.affine.bias.double(), expected_bias, atol=1e-7, rtol=1e-6)


def test_final_eval_graph_uses_only_fixed_affine_student():
    torch.manual_seed(29)
    module = ProgressiveAffineNorm2d(6).eval()
    with torch.no_grad():
        module.affine.weight.copy_(torch.randn(6))
        module.affine.bias.copy_(torch.randn(6))
    module.set_progress(current_step=10, total_steps=10)
    inputs = torch.randn(2, 6, 3, 5)

    with torch.no_grad():
        expected = module.affine(inputs)
        actual = module(inputs)

    assert torch.equal(actual, expected)

    def fail_if_called(_):
        raise AssertionError("LayerNorm teacher ran in the final eval graph")

    module.ln.forward = fail_if_called
    with torch.no_grad():
        without_teacher = module(inputs)
    assert torch.equal(without_teacher, expected)


def test_gamma_zero_keeps_teacher_in_training_graph_only():
    module = ProgressiveAffineNorm2d(8)
    module.set_progress(current_step=10, total_steps=10)
    module.train()
    module(torch.randn(2, 8, 4, 4)).square().mean().backward()

    assert module.ln.weight.grad is not None
    assert torch.count_nonzero(module.ln.weight.grad) == 0
    assert module.affine.weight.grad is not None


def test_gamma_one_keeps_future_affine_student_in_training_graph():
    module = ProgressiveAffineNorm2d(8)
    module.train()
    module(torch.randn(2, 8, 4, 4)).square().mean().backward()

    assert module.ln.weight.grad is not None
    assert module.affine.weight.grad is not None
    assert torch.count_nonzero(module.affine.weight.grad) == 0


def test_folding_removes_all_layernorm_teachers_without_output_change():
    model = _small_affine().eval()
    for module in model.affine_norm_modules():
        module.set_progress(current_step=1, total_steps=1)
    inputs = torch.randn(2, 3, 16, 16)

    with torch.no_grad():
        expected = model(inputs)
        model.fold_affine_norms_for_inference()
        actual = model(inputs)

    assert torch.equal(actual, expected)
    assert not any(isinstance(module, ProgressiveAffineNorm2d)
                   for module in model.modules())
    assert not any(isinstance(module, LayerNorm2d)
                   for module in model.modules())


def test_registered_s24_has_49_affine_sites_and_unchanged_gates():
    model = get_model(
        "poolformer_fully_gated_affine_s24",
        num_features=32,
        fp16=False,
    )

    assert len(model.affine_norm_modules()) == 49
    assert sum(isinstance(module, SimpleGate)
               for module in model.modules()) == 24


def test_s24_affine_groups_are_forward_ordered_and_independent():
    model = get_model(
        "poolformer_fully_gated_affine_s24",
        num_features=32,
        fp16=False,
        affine_blocks_per_group=1,
    )
    groups = model.affine_group_names()

    assert len(groups) == 25
    assert groups[0] == (
        "network.0.0.norm1", "network.0.0.norm2",
    )
    assert groups[-1] == ("norm",)

    blends = (1.0, 0.5) + (0.0,) * 23
    model.set_affine_group_blends(blends)
    modules = dict(model.named_modules())
    assert modules[groups[0][0]]._gamma == 0.0
    assert modules[groups[1][0]]._gamma == 0.5
    assert modules[groups[2][0]]._gamma == 1.0


def test_token_mixer_norm_bias_is_fixed_to_zero():
    model = _small_affine()
    first_block = model.network[0][0]

    assert first_block.norm1.force_zero_bias
    assert not first_block.norm1.affine.bias.requires_grad
    assert first_block.norm2.affine.bias.requires_grad


def test_group_calibration_updates_only_requested_group():
    torch.manual_seed(31)
    model = _small_affine().eval()
    final_weight_before = model.norm.affine.weight.detach().clone()
    model.begin_affine_calibration(group_index=0)
    with torch.no_grad():
        model(torch.randn(3, 3, 16, 16))
    diagnostics = model.finish_affine_calibration(
        ridge=1e-6, distributed=False)

    assert {item["name"] for item in diagnostics} == {
        "network.0.0.norm1", "network.0.0.norm2"}
    assert torch.equal(model.network[0][0].norm1.affine.bias,
                       torch.zeros_like(model.network[0][0].norm1.affine.bias))
    assert torch.equal(model.norm.affine.weight, final_weight_before)
