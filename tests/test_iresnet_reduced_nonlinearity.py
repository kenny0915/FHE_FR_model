import copy

import pytest
import torch
from torch import nn

from backbones import get_model
from backbones.iresnet import IBasicBlock, iresnet18, iresnet50
from backbones.iresnet_custom import iresnet18 as reference_iresnet18
from utils.utils_config import get_config


NL13_ACTIVE_PRELUS = {
    "prelu",
    "layer1.0.prelu",
    "layer1.2.prelu",
    "layer2.0.prelu",
    "layer2.3.prelu",
    "layer3.0.prelu",
    "layer3.3.prelu",
    "layer3.6.prelu",
    "layer3.9.prelu",
    "layer3.13.prelu",
    "layer4.0.prelu",
    "layer4.1.prelu",
    "layer4.2.prelu",
}

NL9_ACTIVE_PRELUS = {
    "prelu",
    "layer1.0.prelu",
    "layer2.0.prelu",
    "layer3.0.prelu",
    "layer3.3.prelu",
    "layer3.9.prelu",
    "layer3.13.prelu",
    "layer4.0.prelu",
    "layer4.2.prelu",
}


def _custom_r18_mask():
    return {
        "stem": True,
        "stage1": (True, False),
        "stage2": (True, False),
        "stage3": (True, False),
        "stage4": (True, False),
    }


def test_default_backbone_is_numerically_identical_to_reference():
    torch.manual_seed(7)
    reference = reference_iresnet18(dropout=0, fp16=False).eval()
    configurable = iresnet18(dropout=0, fp16=False).eval()

    assert configurable.state_dict().keys() == reference.state_dict().keys()
    configurable.load_state_dict(copy.deepcopy(reference.state_dict()), strict=True)

    inputs = torch.randn(1, 3, 112, 112)
    with torch.no_grad():
        expected = reference(inputs)
        actual = configurable(inputs)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_nl13_uses_the_required_activation_mask():
    model = get_model("r50_nl13", dropout=0, fp16=False)
    active = {
        name for name, module in model.named_modules()
        if isinstance(module, nn.PReLU)
    }
    inactive = {
        name for name, module in model.named_modules()
        if isinstance(module, nn.Identity)
    }

    assert model.arch_config == "nl13"
    assert model.nonlinear_depth == 13
    assert active == NL13_ACTIVE_PRELUS
    assert len(inactive) == 12
    assert all(
        isinstance(stage[0].prelu, nn.PReLU)
        for stage in (model.layer1, model.layer2, model.layer3, model.layer4)
    )
    assert all(
        block.mid_channels == channels
        for stage, channels in zip(
            (model.layer1, model.layer2, model.layer3, model.layer4),
            (64, 128, 256, 512),
        )
        for block in stage
    )


def test_nl9_uses_a_nested_distributed_activation_mask():
    model = get_model("r50_nl9", dropout=0, fp16=False)
    active = {
        name for name, module in model.named_modules()
        if isinstance(module, nn.PReLU)
    }
    inactive = {
        name for name, module in model.named_modules()
        if isinstance(module, nn.Identity)
    }

    assert model.arch_config == "nl9"
    assert model.nonlinear_depth == 9
    assert active == NL9_ACTIVE_PRELUS
    assert active < NL13_ACTIVE_PRELUS
    assert len(inactive) == 16
    assert all(
        isinstance(stage[0].prelu, nn.PReLU)
        for stage in (model.layer1, model.layer2, model.layer3, model.layer4)
    )
    assert all(
        block.mid_channels == channels
        for stage, channels in zip(
            (model.layer1, model.layer2, model.layer3, model.layer4),
            (64, 128, 256, 512),
        )
        for block in stage
    )


@pytest.mark.parametrize("network", ["r50_nl13", "r50_nl9"])
def test_reduced_model_stage_and_embedding_shapes(network):
    model = get_model(network, dropout=0, fp16=False).eval()
    stage_shapes = {}
    handles = [
        stage.register_forward_hook(
            lambda _module, _inputs, output, name=name:
            stage_shapes.__setitem__(name, tuple(output.shape)))
        for name, stage in (
            ("stage1", model.layer1),
            ("stage2", model.layer2),
            ("stage3", model.layer3),
            ("stage4", model.layer4),
        )
    ]
    with torch.no_grad():
        embedding = model(torch.randn(1, 3, 112, 112))
    for handle in handles:
        handle.remove()

    assert stage_shapes == {
        "stage1": (1, 64, 56, 56),
        "stage2": (1, 128, 28, 28),
        "stage3": (1, 256, 14, 14),
        "stage4": (1, 512, 7, 7),
    }
    assert embedding.shape == (1, 512)


def test_block_supports_identity_and_custom_internal_width():
    downsample = nn.Sequential(
        nn.Conv2d(32, 64, kernel_size=1, stride=2, bias=False),
        nn.BatchNorm2d(64),
    )
    block = IBasicBlock(
        32,
        64,
        stride=2,
        downsample=downsample,
        mid_channels=48,
        use_activation=False,
    ).eval()

    assert isinstance(block.prelu, nn.Identity)
    assert block.conv1.weight.shape == (48, 32, 3, 3)
    assert block.conv2.weight.shape == (64, 48, 3, 3)
    with torch.no_grad():
        output = block(torch.randn(2, 32, 28, 28))
    assert output.shape == (2, 64, 14, 14)


def test_baseline_checkpoint_initializes_reduced_model_strictly():
    baseline = iresnet18(dropout=0, fp16=False)
    reduced = iresnet18(
        dropout=0,
        fp16=False,
        activation_mask=_custom_r18_mask(),
    )
    baseline_state = baseline.state_dict()

    reduced.load_backbone_init_state_dict(baseline_state)

    for name, value in reduced.state_dict().items():
        torch.testing.assert_close(value, baseline_state[name], rtol=0, atol=0)


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"arch_config": "unknown"}, "Unknown IResNet arch_config"),
        (
            {"activation_mask": {**_custom_r18_mask(), "stage1": (True,)}},
            "stage1 activation mask must have 2 entries",
        ),
        (
            {
                "mid_widths": {
                    "stage1": (64, 64),
                    "stage2": (128, 128),
                    "stage3": (256, 256),
                    "stage4": (512, 0),
                }
            },
            "stage4 mid_widths must be positive integers",
        ),
    ],
)
def test_invalid_architecture_configuration_fails_early(kwargs, error):
    with pytest.raises((TypeError, ValueError), match=error):
        iresnet18(**kwargs)


@pytest.mark.parametrize(
    "config_path, expected_init",
    [
        ("configs/ms1mv3_r50_nl13", "work_dirs/ms1mv3_r50/model.pt"),
        ("configs/casia_r50_nl13", "work_dirs/casia_r50/model.pt"),
        ("configs/ms1mv3_r50_nl9", "work_dirs/ms1mv3_r50/model.pt"),
        ("configs/casia_r50_nl9", "work_dirs/casia_r50/model.pt"),
    ],
)
def test_reduced_nonlinearity_training_configs(config_path, expected_init):
    cfg = get_config(config_path)
    expected_arch = config_path.rsplit("_", 1)[-1]
    assert cfg.network == f"r50_{expected_arch}"
    assert cfg.arch_config == expected_arch
    assert cfg.backbone_init == expected_init
    assert cfg.output.endswith(f"_r50_{expected_arch}")


@pytest.mark.parametrize("arch_config", ["nl13", "nl9"])
def test_reduced_presets_are_only_valid_for_iresnet50(arch_config):
    with pytest.raises(ValueError, match="stage1 activation mask"):
        iresnet18(arch_config=arch_config)


def test_direct_iresnet50_nl13_selection_matches_factory():
    direct = iresnet50(arch_config="nl13")
    assert direct.activation_mask["stage3"] == (
        True, False, False, True, False, False, True,
        False, False, True, False, False, False, True,
    )


def test_direct_iresnet50_nl9_selection_matches_factory():
    direct = iresnet50(arch_config="nl9")
    assert direct.activation_mask["stage3"] == (
        True, False, False, True, False, False, False,
        False, False, True, False, False, False, True,
    )
