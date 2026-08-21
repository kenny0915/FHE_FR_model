from copy import deepcopy

import pytest
import torch
import torch.nn as nn

from backbones import get_model
from backbones.poolformer import (
    GroupNorm,
    Mlp,
    poolformer_s24,
    poolformer_s24_gelu12,
    poolformer_s24_gelu8,
)
from utils.utils_config import get_config


EXPECTED_GELU12_MASK = {
    "stage1": (True, False, True, False),
    "stage2": (True, False, True, False),
    "stage3": (
        True, False, True, False, True, False,
        True, False, True, False, True, False,
    ),
    "stage4": (True, False, True, False),
}

EXPECTED_GELU8_MASK = {
    "stage1": (True, False, False, False),
    "stage2": (True, False, False, False),
    "stage3": (
        True, False, False, True, False, False,
        True, False, False, True, False, False,
    ),
    "stage4": (True, False, True, False),
}


def _stage_activations(model):
    return {
        stage_name: tuple(type(block.mlp.act) for block in model.network[index])
        for stage_name, index in zip(
            ("stage1", "stage2", "stage3", "stage4"),
            (0, 2, 4, 6),
        )
    }


@pytest.mark.parametrize(
    "factory,expected_mask,expected_counts",
    [
        (poolformer_s24_gelu12, EXPECTED_GELU12_MASK, (2, 2, 6, 2)),
        (poolformer_s24_gelu8, EXPECTED_GELU8_MASK, (1, 1, 4, 2)),
    ],
)
def test_reduced_gelu_presets_use_required_distributed_masks(
        factory, expected_mask, expected_counts):
    model = factory(face_embedding=False, num_classes=8, fp16=False)

    assert model.gelu_mask == expected_mask
    assert model.gelu_depth == sum(expected_counts)
    assert tuple(model.gelu_count_by_stage.values()) == expected_counts
    assert sum(isinstance(module, nn.GELU) for module in model.modules()) \
        == sum(expected_counts)
    assert sum(isinstance(module, GroupNorm) for module in model.modules()) == 49

    activations = _stage_activations(model)
    for stage_name, stage_mask in expected_mask.items():
        assert activations[stage_name] == tuple(
            nn.GELU if enabled else nn.Identity for enabled in stage_mask)


def test_identity_mlp_preserves_shape_and_configurable_hidden_width():
    mlp = Mlp(
        in_features=8,
        hidden_features=24,
        out_features=12,
        use_activation=False,
    )

    assert isinstance(mlp.act, nn.Identity)
    assert mlp.fc1.weight.shape == (24, 8, 1, 1)
    assert mlp.fc2.weight.shape == (12, 24, 1, 1)
    assert mlp(torch.randn(2, 8, 5, 7)).shape == (2, 12, 5, 7)


def test_reduced_models_keep_baseline_widths_params_and_checkpoint_keys():
    baseline = poolformer_s24(face_embedding=False, num_classes=8, fp16=False)
    reduced = poolformer_s24_gelu12(
        face_embedding=False, num_classes=8, fp16=False)

    assert baseline.state_dict().keys() == reduced.state_dict().keys()
    assert sum(parameter.numel() for parameter in baseline.parameters()) == \
        sum(parameter.numel() for parameter in reduced.parameters())
    reduced.load_backbone_init_state_dict(deepcopy(baseline.state_dict()))

    for stage_index in (0, 2, 4, 6):
        for block in reduced.network[stage_index]:
            assert block.mlp.fc1.out_channels == 4 * block.mlp.fc1.in_channels


@pytest.mark.parametrize(
    "network,expected_arch",
    [
        ("poolformer_s24_gelu12", "gelu12"),
        ("poolformer_s24_gelu8", "gelu8"),
    ],
)
def test_registered_reduced_backbones_produce_face_embeddings(
        network, expected_arch):
    model = get_model(network, num_features=16, fp16=False).eval()
    with torch.no_grad():
        output = model(torch.randn(1, 3, 112, 112))

    assert model.arch_config == expected_arch
    assert output.shape == (1, 16)
    assert output.dtype == torch.float32
    assert torch.isfinite(output).all()


@pytest.mark.parametrize("arch_config", ["gelu12", "gelu8"])
def test_reduced_presets_reject_wrong_s24_factory_selection(arch_config):
    other = "gelu8" if arch_config == "gelu12" else "gelu12"
    factory = (
        poolformer_s24_gelu12 if arch_config == "gelu12"
        else poolformer_s24_gelu8
    )
    with pytest.raises(ValueError, match="requires arch_config"):
        factory(arch_config=other)


def test_invalid_reduced_architecture_configuration_fails_early():
    with pytest.raises(ValueError, match="stage3 GELU mask must have 12 entries"):
        poolformer_s24(gelu_mask={
            "stage1": (True,) * 4,
            "stage2": (True,) * 4,
            "stage3": (True,) * 11,
            "stage4": (True,) * 4,
        })


@pytest.mark.parametrize("depth", [12, 8])
def test_reduced_gelu_configs_inherit_fully_gated_fp32_recipe(depth):
    pytest.importorskip("easydict")
    baseline = get_config("configs/ms1mv3_poolformer_s24_fully_gated_fp32")
    cfg = get_config(f"configs/ms1mv3_poolformer_s24_gelu{depth}_fp32")

    assert cfg.network == f"poolformer_s24_gelu{depth}"
    assert cfg.arch_config == f"gelu{depth}"
    assert cfg.output == f"work_dirs/ms1mv3_poolformer_s24_gelu{depth}_fp32"
    for key in (
        "fp16", "optimizer", "lr", "weight_decay", "batch_size",
        "gradient_clip", "gradient_clip_type", "gradient_clip_scope",
        "num_epoch", "warmup_epoch", "margin_list",
    ):
        assert cfg[key] == baseline[key]
