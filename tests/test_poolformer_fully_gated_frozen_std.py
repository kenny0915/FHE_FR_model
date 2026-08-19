from collections import OrderedDict

import pytest
import torch

from backbones import get_model
from backbones.poolformer_fully_gated import (
    FullyGatedPoolFormer,
    SimpleGate,
)
from backbones.poolformer_fully_gated_frozen_std import (
    FrozenStdFullyGatedPoolFormer,
    FrozenStdLayerNorm2d,
    SpatialTailFrozenStdLayerNorm2d,
)


def _small_baseline(layers=(1,)):
    return FullyGatedPoolFormer(
        layers=list(layers),
        embed_dims=[8] * len(layers),
        ffn_expands=[2.0] * len(layers),
        downsamples=[False] * len(layers),
        num_classes=4,
        face_embedding=False,
        fp16=False,
    )


def _small_frozen_std(layers=(1,)):
    return FrozenStdFullyGatedPoolFormer(
        layers=list(layers),
        embed_dims=[8] * len(layers),
        ffn_expands=[2.0] * len(layers),
        downsamples=[False] * len(layers),
        num_classes=4,
        face_embedding=False,
        fp16=False,
    )


def _small_spatial_frozen_std(layers=(1,)):
    return FrozenStdFullyGatedPoolFormer(
        layers=list(layers),
        embed_dims=[8] * len(layers),
        ffn_expands=[2.0] * len(layers),
        downsamples=[False] * len(layers),
        num_classes=4,
        face_embedding=False,
        fp16=False,
        frozen_std_spatial_tail=True,
    )


def test_layernorm_checkpoint_warm_start_is_exact_before_switch():
    torch.manual_seed(41)
    baseline = _small_baseline().eval()
    converted = _small_frozen_std().eval()
    incompatible = converted.load_backbone_init_state_dict(
        baseline.state_dict())
    inputs = torch.randn(2, 3, 16, 16)

    with torch.no_grad():
        expected = baseline(inputs)
        actual = converted(inputs)

    assert incompatible.unexpected_keys == []
    assert torch.equal(actual, expected)
    assert converted.frozen_std_frozen_count() == 0


def test_warm_start_accepts_ddp_prefix_and_rejects_incomplete_state():
    baseline_state = _small_baseline().state_dict()
    ddp_state = OrderedDict(
        (f"module.{name}", value) for name, value in baseline_state.items())
    _small_frozen_std().load_backbone_init_state_dict(ddp_state)

    incomplete = OrderedDict(baseline_state)
    incomplete.pop("norm.bias")
    with pytest.raises(RuntimeError, match="missing=.*norm.bias"):
        _small_frozen_std().load_backbone_init_state_dict(incomplete)


def test_spatial_wrapper_warm_start_is_exact_before_transition():
    torch.manual_seed(42)
    baseline = _small_baseline().eval()
    converted = _small_spatial_frozen_std().eval()
    converted.load_backbone_init_state_dict(baseline.state_dict())
    inputs = torch.randn(2, 3, 16, 16)

    with torch.no_grad():
        expected = baseline(inputs)
        actual = converted(inputs)

    assert torch.equal(actual, expected)
    assert converted.frozen_std_frozen_count() == 0


def test_hard_switch_uses_tracked_scalar_and_stops_ema_updates():
    torch.manual_seed(43)
    module = FrozenStdLayerNorm2d(6, momentum=0.5)
    module.train()
    inputs = torch.randn(3, 6, 4, 5)
    module(inputs)
    tracked = module.running_std.detach().clone()
    tracked_batches = module.num_batches_tracked.item()

    frozen = module.freeze(distributed=False)
    actual = module(inputs)
    centered = inputs - inputs.mean(dim=1, keepdim=True)
    expected = (
        centered * tracked.reciprocal()
        * module.weight.view(1, -1, 1, 1)
        + module.bias.view(1, -1, 1, 1)
    )
    module(torch.randn_like(inputs))

    assert frozen == pytest.approx(tracked.item())
    assert torch.equal(actual, expected)
    assert module.num_batches_tracked.item() == tracked_batches


def test_freeze_requires_observed_data():
    with pytest.raises(RuntimeError, match="before observing training data"):
        FrozenStdLayerNorm2d(4).freeze(distributed=False)


def test_extreme_finite_inputs_do_not_overflow_tracked_std():
    module = FrozenStdLayerNorm2d(4, momentum=0.9).train()
    inputs = torch.tensor(
        [[[[1e25]], [[-1e25]], [[5e24]], [[-5e24]]]],
        dtype=torch.float32,
    )

    exact_output = module(inputs)
    assert torch.isfinite(module.running_std)
    assert module.running_std.item() > 1e24

    module.freeze(distributed=False)
    frozen_output = module(inputs)
    assert torch.isfinite(exact_output).all()
    assert torch.isfinite(frozen_output).all()


def test_finite_batch_repairs_inf_ema_from_old_checkpoint():
    module = FrozenStdLayerNorm2d(4, momentum=0.9).train()
    module.ema_initialized.fill_(True)
    module.running_std.fill_(float("inf"))

    module(torch.randn(2, 4, 3, 3))

    assert torch.isfinite(module.running_std)
    assert module.running_std.item() > 0.0


def test_spatial_wrapper_tracks_positionwise_mean_and_batch_tail():
    torch.manual_seed(45)
    module = SpatialTailFrozenStdLayerNorm2d(6, momentum=0.5).train()
    scale = torch.tensor(
        [[1.0, 2.0, 4.0], [0.5, 3.0, 8.0]]).view(1, 1, 2, 3)
    inputs = torch.randn(5, 6, 2, 3) * scale

    module(inputs)

    assert module.running_std.shape == (1, 1, 2, 3)
    assert module.running_mean_std.shape == (1, 1, 2, 3)
    assert torch.all(module.running_std >= module.running_mean_std)
    assert torch.isfinite(module.running_std).all()


def test_spatial_transition_uses_margin_and_gradual_blend():
    torch.manual_seed(46)
    module = SpatialTailFrozenStdLayerNorm2d(6, momentum=0.5).train()
    inputs = torch.randn(4, 6, 2, 3)
    module(inputs)
    tracked_tail = module.running_std.detach().clone()
    diagnostics = module.begin_transition(
        distributed=False,
        margin=1.25,
        max_tail_to_mean_ratio=100.0,
        max_frozen_std=1e6,
    )
    centered = inputs - module._stable_channel_mean(inputs)
    exact = centered / module._stable_channel_std(inputs).unsqueeze(1)
    frozen = centered * module.inverse_frozen_std

    module.set_blend(0.5)
    actual = module(inputs)
    expected = 0.5 * exact + 0.5 * frozen

    assert diagnostics["frozen_max"] == pytest.approx(
        float((tracked_tail * 1.25).amax()))
    assert torch.equal(module.frozen_std, tracked_tail * 1.25)
    assert torch.allclose(actual, expected)
    assert not module._is_frozen

    module.set_blend(1.0)
    assert module._is_frozen
    assert torch.equal(module(inputs), module._frozen_forward(inputs))


def test_spatial_transition_rejects_unsafe_tail_ratio():
    module = SpatialTailFrozenStdLayerNorm2d(4, momentum=0.0).train()
    inputs = torch.randn(4, 4, 2, 2)
    inputs[0, :, 0, 0].mul_(100.0)
    module(inputs)

    with pytest.raises(FloatingPointError, match="tail ratio is unsafe"):
        module.begin_transition(
            distributed=False,
            margin=1.0,
            max_tail_to_mean_ratio=1.1,
            max_frozen_std=1e6,
        )


def test_spatial_state_dict_restores_dynamic_map_and_active_blend():
    torch.manual_seed(49)
    module = SpatialTailFrozenStdLayerNorm2d(6).train()
    inputs = torch.randn(3, 6, 2, 3)
    module(inputs)
    module.begin_transition(
        distributed=False,
        max_tail_to_mean_ratio=100.0,
        max_frozen_std=1e6,
    )
    module.set_blend(0.5)
    expected = module(inputs)

    restored = SpatialTailFrozenStdLayerNorm2d(6).train()
    restored.load_state_dict(module.state_dict(), strict=True)

    assert restored.running_std.shape == (1, 1, 2, 3)
    assert restored._transition_started
    assert restored._blend == pytest.approx(0.5)
    assert torch.equal(restored(inputs), expected)


def test_groups_follow_norm2_then_norm1_then_final_order():
    model = _small_frozen_std(layers=(1, 1))
    assert model.frozen_std_group_names() == (
        ("network.0.0.norm2",),
        ("network.1.0.norm2",),
        ("network.0.0.norm1",),
        ("network.1.0.norm1",),
        ("norm",),
    )

    model.train()
    model(torch.randn(2, 3, 16, 16))
    model.freeze_frozen_std_group(0, distributed=False)
    assert model.frozen_std_frozen_count() == 1
    model.freeze_frozen_std_group(2, distributed=False)
    with pytest.raises(RuntimeError, match="not a prefix"):
        model.frozen_std_frozen_count()


def test_final_input_auxiliary_loss_matches_variance_concentration():
    torch.manual_seed(47)
    model = _small_frozen_std()
    model.set_frozen_std_auxiliary_loss(True)
    model.train()
    inputs = torch.randn(2, 3, 16, 16)
    model(inputs)
    hidden = model._frozen_std_final_input
    hidden_float = hidden.float()
    centered = hidden_float - hidden_float.mean(dim=1, keepdim=True)
    std = torch.sqrt(centered.square().mean(dim=1) + model.norm.eps)
    expected = (std - std.detach().mean()).square().mean()

    actual = model.frozen_std_auxiliary_loss()
    actual.backward()

    assert torch.allclose(actual.detach(), expected.detach())
    assert model._frozen_std_final_input is None
    assert model.patch_embed.proj.weight.grad is not None
    assert torch.isfinite(model.patch_embed.proj.weight.grad).all()


def test_state_dict_restores_hard_switch_and_output():
    torch.manual_seed(53)
    model = _small_frozen_std().train()
    inputs = torch.randn(2, 3, 16, 16)
    model(inputs)
    model.freeze_frozen_std_group(0, distributed=False)
    model.eval()
    with torch.no_grad():
        expected = model(inputs)

    restored = _small_frozen_std().eval()
    restored.load_state_dict(model.state_dict(), strict=True)
    with torch.no_grad():
        actual = restored(inputs)

    assert restored.frozen_std_frozen_count() == 1
    assert torch.equal(actual, expected)


def test_registered_s24_has_49_sites_and_unchanged_gates():
    model = get_model(
        "poolformer_fully_gated_frozen_std_s24",
        num_features=32,
        fp16=False,
    )

    assert len(model.frozen_std_modules()) == 49
    assert len(model.frozen_std_group_names()) == 49
    assert sum(isinstance(module, SimpleGate)
               for module in model.modules()) == 24


def test_registered_spatial_s24_has_49_progressive_sites():
    model = get_model(
        "poolformer_fully_gated_spatial_frozen_std_s24",
        num_features=32,
        fp16=False,
    )

    assert len(model.frozen_std_modules()) == 49
    assert all(isinstance(module, SpatialTailFrozenStdLayerNorm2d)
               for module in model.frozen_std_modules())
    assert sum(isinstance(module, SimpleGate)
               for module in model.modules()) == 24
