import numpy as np
import pytest
import torch
from torch import nn

from controlled_degree2.model import DirectQuadratic
from controlled_degree2.recipe_a_recovery import (
    affine_parameters, averaged_gradients, check_split, configure,
    finite_prefix_loss, snapshot_source, stage_passes,
)


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(1)
        self.prelu = DirectQuadratic(1, lam_fit=1, name='prelu')
        self.layer1 = nn.ModuleList([nn.Module()])
        self.layer1[0].bn2 = nn.BatchNorm2d(1)
        self.layer1[0].prelu = DirectQuadratic(1, lam_fit=1, name='layer1.0.prelu')
        self.later_called = False

    def forward(self, x):
        x = self.prelu(self.bn1(x))
        self.later_called = True
        return self.layer1[0].prelu(self.layer1[0].bn2(x))


def test_stops_before_overflow_and_restores_graph_flags():
    model = Toy()
    parameters = affine_parameters(model)
    model.train()
    flags = [(m.training, getattr(m, 'clip_eval', None)) for m in model.modules()]
    coefficients = model.prelu.coeffs.clone()
    optimizer = torch.optim.SGD(parameters, lr=.01)
    loss, site, ratios = finite_prefix_loss(model, torch.full((2, 1, 2, 2), 4.), 2)
    assert site == 'prelu' and not model.later_called
    assert ratios.min() > 1 and torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(model.bn1.weight.grad).all()
    assert model.layer1[0].bn2.weight.grad is None
    before = model.bn1.weight.detach().clone()
    averaged_gradients(parameters, 1, torch.device('cpu'))
    optimizer.step()
    assert model.bn1.weight < before
    assert torch.equal(model.prelu.coeffs, coefficients)
    assert flags == [(m.training, getattr(m, 'clip_eval', None)) for m in model.modules()]


def test_downstream_escape_only_updates_local_affine():
    model = Toy()
    affine_parameters(model)
    with torch.no_grad():
        model.layer1[0].bn2.bias.fill_(5.)
    loss, site, _ = finite_prefix_loss(model, torch.zeros(2, 1, 2, 2), 2)
    assert site == 'layer1.0.prelu'
    loss.backward()
    assert model.layer1[0].bn2.bias.grad is not None
    assert model.bn1.bias.grad is None


def test_safe_prefix_and_progressive_clipping():
    model = Toy()
    affine_parameters(model)
    configure(model, 1)
    assert not model.prelu.clip_eval and model.layer1[0].prelu.clip_eval
    loss, site, _ = finite_prefix_loss(model, torch.zeros(2, 1, 2, 2), 1)
    assert site is None and loss == 0
    configure(model, 5)
    assert not any(m.clip_eval for m in model.modules() if isinstance(m, DirectQuadratic))


def test_invalid_prefix_does_not_leave_detach_hooks():
    model = Toy()
    affine_parameters(model)
    with pytest.raises(FloatingPointError):
        finite_prefix_loss(model, torch.full((2, 1, 2, 2), float('inf')), 2)
    assert all(not m._forward_pre_hooks for m in model.modules())


def test_gate_never_advances_on_empty_or_escaping_scan():
    assert stage_passes(dict(rows=4, nonfinite=0, escaping_rows=0))
    for rows, bad, escapes in [(0, 0, 0), (4, 1, 0), (4, 0, 1)]:
        assert not stage_passes(dict(rows=rows, nonfinite=bad, escaping_rows=escapes))


def test_recovery_identity_isolation():
    split = dict(labels=np.array([0, 0, 1, 1]), train=np.array([0, 1]), dev=np.array([2, 3]))
    check_split(split, [0])
    with pytest.raises(ValueError):
        check_split(split, [0, 1])
    split['dev'] = np.array([1, 3])
    with pytest.raises(ValueError):
        check_split(split, [0])


def test_source_snapshot_is_separate_read_only_and_exact(tmp_path):
    source = tmp_path/'source'
    source.mkdir()
    for name in ('last.pt', 'prepared.pt', 'split.npz', 'provenance.json'):
        (source/name).write_bytes(b'original bytes')
    output = tmp_path/'recovery'
    record = snapshot_source(source, output)
    assert (source/'last.pt').read_bytes() == (output/'source_epoch8.pt').read_bytes()
    assert (output/'source_epoch8.pt').stat().st_mode & 0o222 == 0
    assert len(record['source_sha256']) == 64
    with pytest.raises(FileExistsError):
        snapshot_source(source, output)
    with pytest.raises(ValueError):
        snapshot_source(source, source/'nested')
