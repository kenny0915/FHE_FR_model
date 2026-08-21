import copy
import importlib.util
import sys
import types

import pytest
import torch
from torch import nn

from backbones import get_model
from backbones.iresnet_nl9_prelu_herpn import NL9_ACTIVATION_NAMES
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
            "_test_nl9_prelu_herpn_config", path)
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
        "r50_nl9_prelu_herpn",
        dropout=0,
        fp16=False,
        arch_config="nl9",
        **kwargs,
    )


def test_nl9_prelu_herpn_wraps_only_the_nine_retained_activations():
    model = _model(herpn_progress=0.0)
    progressive = {
        name for name, module in model.named_modules()
        if isinstance(module, PReLUHerPNActivation)
    }
    identities = {
        name for name, module in model.named_modules()
        if isinstance(module, nn.Identity)
    }

    assert model.arch_config == "nl9"
    assert model.nonlinear_depth == 9
    assert progressive == set(NL9_ACTIVATION_NAMES)
    assert len(identities) == 16
    frozen_teachers = [
        module for module in model.modules() if isinstance(module, nn.PReLU)
    ]
    assert len(frozen_teachers) == 9
    assert all(not module.weight.requires_grad for module in frozen_teachers)


def test_temporary_nl9_checkpoint_loads_strictly_at_exact_blend_zero():
    teacher = get_model(
        "r50_nl9", dropout=0, fp16=False, arch_config="nl9").eval()
    student = _model(herpn_progress=0.0).eval()
    teacher_state = copy.deepcopy(teacher.state_dict())

    student.load_backbone_init_state_dict(teacher_state)

    student_activations = dict(student.named_progressive_activations())
    teacher_modules = dict(teacher.named_modules())
    for name in NL9_ACTIVATION_NAMES:
        torch.testing.assert_close(
            student_activations[name].prelu.weight,
            teacher_modules[name].weight,
            rtol=0,
            atol=0,
        )
        assert student_activations[name]._blend == 0.0

    inputs = torch.randn(1, 3, 112, 112)
    with torch.no_grad():
        expected = teacher(inputs)
        actual = student(inputs)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_nl9_prelu_herpn_folds_all_nine_activations_to_quadratics():
    model = _model(herpn_progress=5.0).eval()
    assert len(model.progressive_activations()) == 9

    model.fold_herpn_for_inference()

    assert sum(isinstance(module, FoldedHerPN)
               for module in model.modules()) == 9
    assert not any(isinstance(module, PReLUHerPNActivation)
                   for module in model.modules())
    assert not any(isinstance(module, nn.PReLU)
                   for module in model.modules())


def test_nl9_prelu_herpn_requires_complete_conversion_before_folding():
    model = _model(herpn_progress=0.0).eval()
    with pytest.raises(RuntimeError, match="All 9 NL9 activations"):
        model.fold_herpn_for_inference()


def test_nl9_prelu_herpn_config_schedules_singletons_in_forward_order():
    cfg = _load_standalone_config(
        "configs/ms1mv3_r50_nl9_prelu_herpn.py")
    model = _model(
        herpn_range_limit=cfg.herpn_range_limit,
        herpn_bn_eps=cfg.herpn_bn_eps,
        herpn_progress=cfg.herpn_initial_progress,
        prelu_herpn_distill_eps=cfg.prelu_herpn_distill_eps,
    )

    assert cfg.network == "r50_nl9_prelu_herpn"
    assert cfg.backbone_init == "work_dirs/ms1mv3_r50_nl9/model.pt"
    assert cfg.embedding_teacher_network == "r50_nl9"
    assert cfg.embedding_teacher_checkpoint == cfg.backbone_init
    assert tuple(group[0] for group in cfg.herpn_conversion_groups) == (
        NL9_ACTIVATION_NAMES)
    assert all(len(group) == 1 for group in cfg.herpn_conversion_groups)
    assert set(dict(model.named_progressive_activations())) == set(
        NL9_ACTIVATION_NAMES)
    assert all(
        right >= left + cfg.herpn_transition_epochs
        for left, right in zip(
            cfg.herpn_group_epochs, cfg.herpn_group_epochs[1:])
    )
    final_conversion_epoch = (
        cfg.herpn_group_epochs[-1] + cfg.herpn_transition_epochs)
    assert cfg.num_epoch - final_conversion_epoch == 6


def test_fast_grouped_nl9_config_is_isolated_and_finishes_early():
    original = _load_standalone_config(
        "configs/ms1mv3_r50_nl9_prelu_herpn.py")
    fast = _load_standalone_config(
        "configs/ms1mv3_r50_nl9_prelu_herpn_fast_grouped.py")
    flattened = tuple(
        name for group in fast.herpn_conversion_groups for name in group)

    assert fast.network == "r50_nl9_prelu_herpn"
    assert fast.backbone_init == "work_dirs/ms1mv3_r50_nl9/model.pt"
    assert fast.output == (
        "work_dirs/ms1mv3_r50_nl9_prelu_herpn_fast_grouped")
    assert fast.output != original.output
    assert fast.embedding_distill_weight == 0.0
    assert flattened == NL9_ACTIVATION_NAMES
    assert len(fast.herpn_conversion_groups) == 6
    assert fast.herpn_conversion_groups[0] == (
        "prelu", "layer1.0.prelu")
    assert fast.herpn_conversion_groups[-2:] == (
        ("layer4.0.prelu",), ("layer4.2.prelu",))
    assert fast.herpn_bn_recalibration_batches == 500
    assert all(
        right >= left + fast.herpn_transition_epochs
        for left, right in zip(
            fast.herpn_group_epochs, fast.herpn_group_epochs[1:])
    )
    final_conversion_epoch = (
        fast.herpn_group_epochs[-1] + fast.herpn_transition_epochs)
    assert final_conversion_epoch == 12
    assert fast.num_epoch - final_conversion_epoch == 4
    assert fast.num_epoch < original.num_epoch
