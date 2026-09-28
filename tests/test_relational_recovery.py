from argparse import Namespace

import pytest
import torch

from controlled_degree2.losses import relational_loss
from controlled_degree2.relational_recovery import command_for, WEIGHTS
from controlled_degree2.train import parse_args


def test_identity_and_permutation_invariance():
    torch.manual_seed(7)
    target = torch.randn(8, 5)
    student = target + .1 * torch.randn_like(target)
    assert abs(relational_loss(target, target).item()) < 1e-6
    permutation = torch.randperm(8)
    torch.testing.assert_close(relational_loss(student, target),
                               relational_loss(student[permutation], target[permutation]))


def test_teacher_detached_and_student_optimization_reduces_loss():
    torch.manual_seed(11)
    student = torch.randn(8, 5, requires_grad=True)
    teacher = torch.randn(8, 5, requires_grad=True)
    optimizer = torch.optim.SGD([student], lr=.01)
    before = relational_loss(student, teacher).item()
    for _ in range(10):
        optimizer.zero_grad()
        loss = relational_loss(student, teacher)
        loss.backward()
        assert torch.isfinite(student.grad).all()
        assert teacher.grad is None
        optimizer.step()
    assert relational_loss(student, teacher).item() < before


def test_mask_excludes_nonfinite_rows_before_similarity():
    s = torch.randn(5, 4, requires_grad=True)
    t = torch.randn(5, 4)
    with torch.no_grad():
        s[0] = float('nan'); t[0] = float('inf')
    mask = torch.tensor([False, True, True, True, True])
    loss = relational_loss(s, t, mask)
    torch.testing.assert_close(loss, relational_loss(s[1:], t[1:]))
    loss.backward()
    assert torch.isfinite(s.grad).all() and s.grad[0].count_nonzero() == 0


@pytest.mark.parametrize('count', [0, 1, 2])
def test_no_neighbours_or_single_neighbour_is_zero(count):
    s = torch.randn(count, 4, requires_grad=True)
    loss = relational_loss(s, torch.randn_like(s))
    assert loss.item() == 0.
    loss.backward()
    assert torch.isfinite(s.grad).all()


@pytest.mark.parametrize('temperature', [0., -1., float('nan'), float('inf')])
def test_invalid_temperature(temperature):
    with pytest.raises(ValueError):
        relational_loss(torch.ones(3, 2), torch.ones(3, 2), temperature=temperature)


@pytest.mark.parametrize('arm', WEIGHTS)
@pytest.mark.parametrize('smoke', [False, True])
def test_recipe_preserves_isolated_comparison(monkeypatch, arm, smoke):
    args = Namespace(arm=arm, smoke=smoke, checkpoint='original.pt', teacher='teacher.pt',
                     dataset_root='ms1m', canary_root='canaries', output_root='out', gpus=4)
    command = command_for(args)
    monkeypatch.setattr('sys.argv', ['train'] + command[7:])
    parsed = parse_args()
    assert parsed.w_relational == WEIGHTS[arm]
    assert parsed.relational_temperature == .05
    assert parsed.epochs == (1 if smoke else 6)
    assert parsed.limit_batches == (8 if smoke else 0)
    assert parsed.student_init == 'original.pt'
    assert parsed.freeze_batchnorm_stats and not parsed.train_coefficients
    assert parsed.save_every_epoch and parsed.aug_pathological == 0
    assert parsed.output_dir == f'out/{arm}'
