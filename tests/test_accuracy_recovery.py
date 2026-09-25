from argparse import Namespace

import pytest
import torch
from torch import nn

from controlled_degree2.accuracy_recovery import training_command
from controlled_degree2.losses import active_weighted_loss
from controlled_degree2.model import DirectQuadratic
from controlled_degree2.train import accuracy_parameter_groups, parse_args


def test_disabled_nonfinite_loss_does_not_poison_gradient():
    x = torch.tensor(2., requires_grad=True)
    unused = x * float('inf')
    loss = active_weighted_loss([(1., x.square()), (0., unused)])
    loss.backward()
    assert x.grad.item() == 4.
    with pytest.raises(ValueError):
        active_weighted_loss([(0., unused)])


def test_coefficient_training_respects_frozen_prefix_and_serializes():
    model = nn.Sequential(DirectQuadratic(2), DirectQuadratic(2), nn.Conv2d(2, 2, 1))
    model[0].requires_grad_(False)
    groups = accuracy_parameter_groups(model, ('0',), .01, .1)
    assert not model[0].coeffs.requires_grad
    assert model[1].coeffs.requires_grad
    assert groups[1]['lr'] == .001
    assert groups[1]['weight_decay'] == 0.
    before = model[1].coeffs.detach().clone()
    optimizer = torch.optim.SGD(groups)
    model(torch.full((2, 2, 2, 2), .2)).square().mean().backward()
    optimizer.step()
    assert not torch.equal(before, model[1].coeffs)
    restored = nn.Sequential(DirectQuadratic(2), DirectQuadratic(2), nn.Conv2d(2, 2, 1))
    restored.load_state_dict(model.state_dict())
    model.eval(); restored.eval()
    x = torch.randn(2, 2, 2, 2)
    torch.testing.assert_close(model(x), restored(x))
    assert not restored[1].clip_eval
    assert restored[1].degree == 2


@pytest.mark.parametrize('multiplier', [0., -1., float('nan'), float('inf')])
def test_invalid_coefficient_learning_rate(multiplier):
    with pytest.raises(ValueError):
        accuracy_parameter_groups(nn.Sequential(DirectQuadratic(2)), (), .01, multiplier)


@pytest.mark.parametrize('arm', ['control', 'fixed', 'coefficients'])
def test_recipe_parses_with_real_trainer(monkeypatch, arm):
    args = Namespace(arm=arm, gpus=4, checkpoint='original.pt', teacher='teacher.pt',
                     dataset_root='ms1m', output_root='output')
    command = training_command(args)
    monkeypatch.setattr('sys.argv', ['train'] + command[5:])
    parsed = parse_args()
    assert parsed.student_init == 'original.pt'
    assert parsed.save_every_epoch
    assert parsed.swap_epochs == 0
    assert parsed.train_coefficients == (arm == 'coefficients')
    assert parsed.freeze_batchnorm_stats == (arm != 'control')
    if arm != 'control':
        assert parsed.causal_tail_beta == 0
        assert parsed.aug_pathological == 0
