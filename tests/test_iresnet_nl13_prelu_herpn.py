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
