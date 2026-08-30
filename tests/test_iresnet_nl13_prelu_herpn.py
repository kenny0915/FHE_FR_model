import copy
import importlib.util
import sys
import types

import pytest
import torch
from torch import nn

from backbones import get_model
from backbones.iresnet_nl13_prelu_herpn import NL13_ACTIVATION_NAMES
from backbones.iresnet_no_relu import FoldedHerPN
from backbones.iresnet_prelu_herpn import PReLUHerPNActivation


class _EasyDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


def _load_standalone_config(path):
    fake_easydict = types.ModuleType("easydict")
    fake_easydict.EasyDict = _EasyDict
    previous = sys.modules.get("easydict")
    sys.modules["easydict"] = fake_easydict
    try:
        spec = importlib.util.spec_from_file_location(
            "_test_nl13_prelu_herpn_config", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.config
    finally:
        if previous is None:
            del sys.modules["easydict"]
        else:
            sys.modules["easydict"] = previous


def _model(**kwargs):
    return get_model(
        "r50_nl13_prelu_herpn",
        dropout=0,
        fp16=False,
        arch_config="nl13",
        **kwargs,
    )


def test_nl13_prelu_herpn_wraps_only_thirteen_retained_activations():
    model = _model(herpn_progress=0.0)
    progressive = {
        name for name, module in model.named_modules()
        if isinstance(module, PReLUHerPNActivation)
    }
    identities = [
        module for module in model.modules() if isinstance(module, nn.Identity)
    ]
    frozen_teachers = [
        module for module in model.modules() if isinstance(module, nn.PReLU)
    ]

    assert model.arch_config == "nl13"
    assert model.nonlinear_depth == 13
    assert progressive == set(NL13_ACTIVATION_NAMES)
    assert len(identities) == 12
    assert len(frozen_teachers) == 13
    assert all(not module.weight.requires_grad for module in frozen_teachers)


def test_temporary_nl13_checkpoint_loads_strictly_at_blend_zero():
    teacher = get_model(
        "r50_nl13", dropout=0, fp16=False, arch_config="nl13").eval()
    student = _model(herpn_progress=0.0).eval()
    teacher_state = copy.deepcopy(teacher.state_dict())

    student.load_backbone_init_state_dict(teacher_state)

    student_activations = dict(student.named_progressive_activations())
    teacher_modules = dict(teacher.named_modules())
    for name in NL13_ACTIVATION_NAMES:
        torch.testing.assert_close(
            student_activations[name].prelu.weight,
            teacher_modules[name].weight,
            rtol=0,
            atol=0,
        )

    inputs = torch.randn(1, 3, 112, 112)
    with torch.no_grad():
        expected = teacher(inputs)
        actual = student(inputs)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_nl13_prelu_herpn_folds_to_thirteen_quadratics():
    model = _model(herpn_progress=5.0).eval()
    model.fold_herpn_for_inference()

    assert sum(isinstance(module, FoldedHerPN)
               for module in model.modules()) == 13
    assert not any(isinstance(module, PReLUHerPNActivation)
                   for module in model.modules())
    assert not any(isinstance(module, nn.PReLU)
                   for module in model.modules())


def test_scaled_nl13_requires_and_uses_public_intervals():
    model = _model(
        herpn_progress=0.0,
        prelu_herpn_layerwise_scale=True,
        prelu_herpn_initial_scale=1.0,
    )
    assert model.layerwise_poly_activation_names() == list(
        NL13_ACTIVATION_NAMES)
    assert model.uncalibrated_layerwise_poly_names() == list(
        NL13_ACTIVATION_NAMES)

    first = NL13_ACTIVATION_NAMES[0]
    model.set_layerwise_poly_input_scale(first, 7.5)
    assert model.uncalibrated_layerwise_poly_names() == list(
        NL13_ACTIVATION_NAMES[1:])
    model.set_herpn_blends({first: 1.0})
    activation = dict(model.named_progressive_activations())[first]
    assert activation._blend == 1.0
    assert float(activation.input_scale) == 7.5


def test_scaled_nl13_losses_can_target_one_conditioning_activation():
    model = _model(
        herpn_progress=0.0,
        prelu_herpn_layerwise_scale=True,
    )
    activations = dict(model.named_progressive_activations())
    first, second = NL13_ACTIVATION_NAMES[:2]
    model.set_layerwise_poly_input_scale(first, 2.0)
    model.set_layerwise_poly_input_scale(second, 3.0)
    activations[first]._last_range_penalty = torch.tensor(2.0)
    activations[second]._last_range_penalty = torch.tensor(6.0)
    activations[first]._last_distillation_loss = torch.tensor(3.0)
    activations[second]._last_distillation_loss = torch.tensor(9.0)

    torch.testing.assert_close(model.herpn_range_penalty(), torch.tensor(4.0))
    torch.testing.assert_close(
        model.herpn_range_penalty((second,)), torch.tensor(6.0))
    torch.testing.assert_close(
        model.herpn_distillation_loss((first,)), torch.tensor(3.0))
    with pytest.raises(ValueError, match="Unknown NL13"):
        model.herpn_range_penalty(("missing.activation",))


def test_scaled_nl13_config_is_causal_and_finishes_conversion():
    cfg = _load_standalone_config(
        "configs/ms1mv3_r50_nl13_prelu_herpn_scaled.py")
    flattened = tuple(
        name for group in cfg.herpn_conversion_groups for name in group)

    assert cfg.prelu_herpn_layerwise_scale is True
    assert cfg.layerwise_poly_causal_strict_calibration is True
    assert cfg.layerwise_poly_strict_recalibrate_before_blend is True
    assert flattened == NL13_ACTIVATION_NAMES
    assert all(len(group) == 1 for group in cfg.herpn_conversion_groups)
    assert cfg.herpn_group_epochs[0] == 0.5
    assert (cfg.herpn_group_epochs[-1] + cfg.herpn_transition_epochs
            < cfg.num_epoch)


def test_nl13_prelu_herpn_rejects_wrong_architecture():
    with pytest.raises(ValueError, match="requires arch_config='nl13'"):
        get_model(
            "r50_nl13_prelu_herpn",
            dropout=0,
            fp16=False,
            arch_config="nl9",
        )


def test_fast_nl13_config_groups_shallow_activations_and_finishes_early():
    cfg = _load_standalone_config(
        "configs/ms1mv3_r50_nl13_prelu_herpn.py")
    model = _model(
        herpn_range_limit=cfg.herpn_range_limit,
        herpn_bn_eps=cfg.herpn_bn_eps,
        herpn_progress=cfg.herpn_initial_progress,
        prelu_herpn_distill_eps=cfg.prelu_herpn_distill_eps,
    )
    flattened = tuple(
        name for group in cfg.herpn_conversion_groups for name in group)

    assert cfg.network == "r50_nl13_prelu_herpn"
    assert cfg.backbone_init == "work_dirs/ms1mv3_r50_nl13/model.pt"
    assert cfg.embedding_distill_weight == 0.0
    assert flattened == NL13_ACTIVATION_NAMES
    assert len(cfg.herpn_conversion_groups) == 8
    assert cfg.herpn_conversion_groups[0] == (
        "prelu", "layer1.0.prelu", "layer1.2.prelu")
    assert cfg.herpn_conversion_groups[1] == (
        "layer2.0.prelu", "layer2.3.prelu")
    assert all(len(group) == 1 for group in cfg.herpn_conversion_groups[-4:])
    assert set(dict(model.named_progressive_activations())) == set(
        NL13_ACTIVATION_NAMES)
    assert all(
        right >= left + cfg.herpn_transition_epochs
        for left, right in zip(
            cfg.herpn_group_epochs, cfg.herpn_group_epochs[1:])
    )
    final_conversion_epoch = (
        cfg.herpn_group_epochs[-1] + cfg.herpn_transition_epochs)
    assert final_conversion_epoch == 16
    assert cfg.num_epoch - final_conversion_epoch == 4
    assert cfg.num_epoch < 34


@pytest.mark.parametrize(
    ("path", "target"),
    (
        ("configs/ms1mv3_r50_nl13_prelu_herpn_selective7_layer41.py",
         "layer4.1.prelu"),
        ("configs/ms1mv3_r50_nl13_prelu_herpn_selective7_layer42.py",
         "layer4.2.prelu"),
    ),
)
def test_selective_seventh_nl13_config_preserves_safe_prefix(path, target):
    cfg = _load_standalone_config(path)
    order = tuple(
        name for group in cfg.herpn_conversion_groups for name in group)

    assert order[:6] == NL13_ACTIVATION_NAMES[:6]
    assert order[6] == target
    assert set(order) == set(NL13_ACTIVATION_NAMES)
    assert len(order) == len(set(order)) == 13
    assert cfg.layerwise_poly_allow_selective_order
    assert cfg.layerwise_poly_training_group_limit == 7
    assert cfg.layerwise_poly_conditioning_range_loss_weight == pytest.approx(
        1.0e-8)
    assert cfg.layerwise_poly_optimizer_lr_scale == pytest.approx(1.0e-4)
    assert cfg.herpn_group_epochs[6] == pytest.approx(1.0)
    assert cfg.herpn_group_epochs[7] == pytest.approx(100.0)
    assert cfg.layerwise_poly_max_input_scale == pytest.approx(1.0e6)
    assert not cfg.herpn_require_full_conversion
    assert cfg.num_epoch == 8


def test_selective_eighth_nl13_config_extends_terminal_checkpoint():
    cfg = _load_standalone_config(
        "configs/"
        "ms1mv3_r50_nl13_prelu_herpn_selective8_layer42_layer41.py")
    order = tuple(
        name for group in cfg.herpn_conversion_groups for name in group)

    assert order[:6] == NL13_ACTIVATION_NAMES[:6]
    assert order[6:8] == ("layer4.2.prelu", "layer4.1.prelu")
    assert set(order) == set(NL13_ACTIVATION_NAMES)
    assert len(order) == len(set(order)) == 13
    assert cfg.backbone_init.endswith(
        "selective7_layer42/model_herpn_group_07_bnrecalibrated.pt")
    assert all(
        start + cfg.herpn_transition_epochs <= 0.0
        for start in cfg.herpn_group_epochs[:7])
    assert cfg.herpn_group_epochs[7] == pytest.approx(1.0)
    assert cfg.layerwise_poly_training_group_limit == 8
    assert cfg.layerwise_poly_optimizer_lr_scale == pytest.approx(1.0e-4)
    assert cfg.num_epoch == 8
