import numpy as np
import torch
from torch import nn

from controlled_degree2.recipe_a import (
    IdentityHead, identity_split, per_identity_rows, phase_at,
    recipe_loss, verification_metric,
    TrainingTailReplay,
)
from controlled_degree2.model import DirectQuadratic


def test_identity_split_and_center_rows_do_not_leak():
    labels = np.repeat(np.arange(50), 10)
    train, dev = identity_split(labels, fraction=.2)
    assert not set(labels[train]) & set(labels[dev])
    assert len(train)+len(dev) == len(labels)
    rows = per_identity_rows(labels, train, 4, 42)
    assert set(rows) <= set(train)
    assert np.unique(labels[rows], return_counts=True)[1].tolist() == [4]*40
    assert np.array_equal(identity_split(labels, fraction=.2)[0], train)


def test_progressive_schedule_finishes_before_unclipped_phase():
    assert phase_at(0) == ([0.]*5, True)
    assert phase_at(9) == ([1.]*5, False)
    values, clipped = phase_at(5)
    assert values == sorted(values, reverse=True)
    assert clipped and 0 < sum(values) < 5


def test_arcface_mask_and_gradients():
    torch.manual_seed(1)
    head = IdentityHead(torch.randn(8, 4))
    features = torch.randn(3, 4, requires_grad=True)
    labels = torch.tensor([0, 2, 4])
    loss = head(features, labels, torch.tensor([True, True, False]))
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(features.grad).all()
    assert features.grad[:2].abs().sum() > 0
    assert features.grad[2].abs().sum() == 0
    assert head.weight.grad.abs().sum() > 0


def test_full_quadratic_unclipped_training_loss_backward():
    torch.manual_seed(9)
    q = DirectQuadratic(3, lam_fit=2., lam_reg=1., name='prelu')
    q.clip = False
    prefix = nn.Conv2d(3, 3, 1)
    x = torch.randn(4, 3, 4, 4)*2
    target = x.mean((2, 3))
    y = q(prefix(x))
    features = y.mean((2, 3))
    mask = torch.ones(4, dtype=torch.bool)
    kd, boundary = recipe_loss(features, target, {1:y}, {1:x}, mask, [q], {'prelu':48}, .1)
    head = IdentityHead(torch.randn(4, 3))
    optimizer = torch.optim.SGD(list(prefix.parameters())+list(head.parameters()), lr=.001)
    loss = head(features, torch.arange(4), mask)+kd+boundary
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(prefix.weight.grad).all()
    assert boundary > 0
    optimizer.step()
    assert q.alpha == 1 and not q.clip and q.coeffs.grad is None


def test_verification_uses_identity_pairs():
    features = torch.eye(4).repeat_interleave(3, 0)
    labels = np.repeat(np.arange(4), 3)
    result = verification_metric(features, labels, 42, negative_pairs=1000)
    assert result['tar_1e4'] == 1.
    assert result['positive_pairs'] == 12
    assert result['empirical_far'] == 0.


def test_tail_replay_excludes_pathological_rows_and_retains_labels():
    replay = TrainingTailReplay(2)
    q = DirectQuadratic(1)
    q.last_sample_ratio = torch.tensor([1., 100., 5.])
    images = torch.arange(3.).reshape(3, 1, 1, 1)
    labels = torch.tensor([10, 11, 12])
    replay.update(images, labels, torch.tensor([True, False, True]), [q])
    x, y, mask = replay.inject(torch.zeros_like(images), labels.clone(), torch.zeros(3, dtype=torch.bool))
    assert x[0].item() == 2 and y[0] == 12 and mask[0]
    assert 11 not in replay.labels.tolist()
