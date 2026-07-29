import importlib.util
import sys
import types

import torch
import torch.nn.functional as F
from torch import nn

from backbones import get_model
from backbones.iresnet_prelu_herpn import PReLUHerPNActivation
from utils.utils_optimizer import split_weight_decay_parameters


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
            "_test_prelu_herpn_config", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.config
    finally:
        if previous is None:
            del sys.modules["easydict"]
        else:
            sys.modules["easydict"] = previous


def test_baseline_prelu_state_loads_strictly_and_blend_zero_is_exact():
    activation = PReLUHerPNActivation(
        channels=3, blend=0.0).eval()
    slopes = torch.tensor([0.1, -0.2, 0.4])
    activation.load_state_dict({"weight": slopes}, strict=True)

    inputs = torch.randn(2, 3, 4, 4)
    expected = F.prelu(inputs, slopes)
    actual = activation(inputs)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(activation.prelu.weight, slopes)
    assert not activation.prelu.weight.requires_grad


def test_prelu_herpn_uses_exact_prelu_decomposition():
    torch.manual_seed(2)
    activation = PReLUHerPNActivation(
        channels=3, blend=1.0).eval()
    with torch.no_grad():
        activation.prelu.weight.copy_(
            torch.tensor([0.1, -0.2, 0.4]))
        activation.herpn.weight.uniform_(0.5, 1.3)
        activation.herpn.bias.uniform_(-0.2, 0.2)

    inputs = torch.randn(2, 3, 4, 4)
    slope = activation.prelu.weight.reshape(1, -1, 1, 1)
    expected = (
        slope * inputs
        + (1.0 - slope) * activation.herpn(inputs)
    )

    torch.testing.assert_close(
        activation(inputs), expected, rtol=1e-6, atol=1e-6)


def test_prelu_herpn_folds_to_the_same_degree_two_polynomial():
    torch.manual_seed(3)
    activation = PReLUHerPNActivation(
        channels=3, blend=1.0).eval()
    with torch.no_grad():
        activation.prelu.weight.copy_(
            torch.tensor([0.1, -0.2, 0.4]))
        activation.herpn.weight.uniform_(0.5, 1.3)
        activation.herpn.bias.uniform_(-0.2, 0.2)
        for batchnorm in (
                activation.herpn.bn0,
                activation.herpn.bn1,
                activation.herpn.bn2):
            batchnorm.running_mean.uniform_(-0.3, 0.3)
            batchnorm.running_var.uniform_(0.5, 1.5)

    folded = activation.folded().eval()
    inputs = torch.randn(2, 3, 4, 4)

    torch.testing.assert_close(
        folded(inputs), activation(inputs), rtol=1e-5, atol=1e-5)
    assert folded.coefficient2.shape == (3, 1, 1)


def test_relative_distillation_remains_active_at_full_conversion():
    activation = PReLUHerPNActivation(
        channels=2, blend=1.0).train()
    inputs = torch.randn(4, 2, 3, 3)
    output = activation(inputs)
    loss = activation.distillation_loss()

    assert output.shape == inputs.shape
    assert loss is not None
    assert torch.isfinite(loss)
    assert float(loss) > 0.0
    loss.backward()
    assert activation.herpn.weight.grad is not None
    assert torch.isfinite(activation.herpn.weight.grad).all()
    assert activation.prelu.weight.grad is None


def test_r50_config_schedules_every_prelu_herpn_activation():
    cfg = _load_standalone_config(
        "configs/ms1mv3_r50_prelu_herpn.py")
    model = get_model(
        cfg.network,
        dropout=0,
        fp16=False,
        herpn_range_limit=cfg.herpn_range_limit,
        herpn_bn_eps=cfg.herpn_bn_eps,
        herpn_progress=cfg.herpn_initial_progress,
        prelu_herpn_distill_eps=cfg.prelu_herpn_distill_eps,
    )
    expected = {
        name for name, module in model.named_modules()
        if isinstance(module, PReLUHerPNActivation)
    }
    scheduled = {
        name for group in cfg.herpn_conversion_groups for name in group
    }

    assert len(expected) == 25
    assert scheduled == expected
    assert sum(map(len, cfg.herpn_conversion_groups)) == 25
    assert len(cfg.herpn_group_epochs) == len(
        cfg.herpn_conversion_groups)
    assert all(
        right >= left + cfg.herpn_transition_epochs
        for left, right in zip(
            cfg.herpn_group_epochs, cfg.herpn_group_epochs[1:])
    )
    final_conversion_epoch = (
        cfg.herpn_group_epochs[-1] + cfg.herpn_transition_epochs)
    assert cfg.num_epoch - final_conversion_epoch == 4
    assert cfg.output == "work_dirs/ms1mv3_r50_prelu_herpn"


def test_selective_weight_decay_protects_herpn_norm_and_bias():
    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 4, 3, bias=True)
            self.bn = nn.BatchNorm2d(4)
            self.activation = PReLUHerPNActivation(4, blend=0.0)
            self.linear = nn.Linear(4, 2, bias=False)

    model = TinyModel()
    decay, no_decay = split_weight_decay_parameters(model)
    decay_ids = {id(parameter) for parameter in decay}
    no_decay_ids = {id(parameter) for parameter in no_decay}

    assert id(model.conv.weight) in decay_ids
    assert id(model.linear.weight) in decay_ids
    assert id(model.conv.bias) in no_decay_ids
    assert id(model.bn.weight) in no_decay_ids
    assert id(model.bn.bias) in no_decay_ids
    assert id(model.activation.herpn.weight) in no_decay_ids
    assert id(model.activation.herpn.bias) in no_decay_ids
    assert id(model.activation.prelu.weight) not in decay_ids | no_decay_ids
    assert decay_ids.isdisjoint(no_decay_ids)
