import copy

import pytest
import torch
from torch import nn

from backbones.iresnet import IBasicBlock
from controlled_degree2.model import replace_prelu_with_quadratic
from controlled_degree2.rescale_residual_graph import rescale_bn, rescale_graph, capture


class TinyResidualGraph(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 2, 3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(2)
        self.prelu = nn.PReLU(2)
        previous = 2
        for stage, channels in enumerate((2, 3, 4, 5), 1):
            downsample = nn.Sequential(nn.Conv2d(previous, channels, 1, bias=True), nn.BatchNorm2d(channels))
            setattr(self, f'layer{stage}', nn.Sequential(
                IBasicBlock(previous, channels, downsample=downsample),
                IBasicBlock(channels, channels)))
            previous = channels
        self.bn2 = nn.BatchNorm2d(previous)
        self.fc = nn.Linear(previous, 7)
        calibration = {name: {'lam_fit': [2.] * m.num_parameters, 'lam_reg': [1.5] * m.num_parameters}
                       for name, m in self.named_modules() if isinstance(m, nn.PReLU)}
        replace_prelu_with_quadratic(self, calibration)
        # Nontrivial running statistics and offsets catch incorrect BN scaling.
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.running_mean.uniform_(-0.3, 0.3)
                m.running_var.uniform_(0.5, 1.5)
                m.weight.data.uniform_(0.5, 1.2)
                m.bias.data.uniform_(-0.1, 0.1)

    def forward(self, x):
        x = self.prelu(self.bn1(self.conv1(x)))
        for stage in range(1, 5):
            x = getattr(self, f'layer{stage}')(x)
        return self.fc(self.bn2(x).mean((2, 3)))


def test_frozen_bn_general_input_output_scales():
    torch.manual_seed(4)
    bn = nn.BatchNorm2d(3).double().eval()
    bn.running_mean.normal_()
    bn.running_var.uniform_(0.1, 2)
    bn.weight.data.normal_()
    bn.bias.data.normal_()
    transformed = copy.deepcopy(bn)
    rescale_bn(transformed, 0.07, 0.3)
    x = torch.randn(2, 3, 4, 4, dtype=torch.float64)
    torch.testing.assert_close(transformed(x * 0.07), bn(x) * 0.3, atol=1e-13, rtol=1e-13)


def test_graph_equivalence_at_every_residual_and_serialization():
    torch.manual_seed(11)
    torch.set_num_threads(2)
    original = TinyResidualGraph().eval()
    scaled = copy.deepcopy(original)
    scales = [0.25, 0.125, 0.0625, 0.5]
    structure = [(name, type(m)) for name, m in scaled.named_modules()]
    mapping = rescale_graph(scaled, scales)
    assert structure == [(name, type(m)) for name, m in scaled.named_modules()]
    reloaded = TinyResidualGraph().eval()
    reloaded.load_state_dict(scaled.state_dict(), strict=True)
    x = torch.randn(3, 3, 8, 8)
    names = [f'layer{s}.{i}' for s in range(1, 5) for i in range(2)]
    expected, before = capture(original, x, names)
    actual, after = capture(reloaded, x, names)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    for name in names:
        torch.testing.assert_close(after[name], before[name] * scales[int(name[5]) - 1], atol=2e-6, rtol=2e-5)
    for name, scale in mapping.items():
        old, new = original.get_submodule(name), reloaded.get_submodule(name)
        torch.testing.assert_close(new.lam_fit, old.lam_fit * scale)
        torch.testing.assert_close(new.lam_reg, old.lam_reg * scale)


def test_requires_eval_and_valid_scales():
    model = TinyResidualGraph()
    with pytest.raises(ValueError, match='eval'):
        rescale_graph(model, [1] * 4)
    model.eval()
    for scales in ([1, 1, 1], [0, 1, 1, 1], [float('nan'), 1, 1, 1]):
        with pytest.raises(ValueError, match='four finite'):
            rescale_graph(model, scales)
