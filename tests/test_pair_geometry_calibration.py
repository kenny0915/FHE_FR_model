import pytest
import torch
from controlled_degree2.calibrate_pair_geometry import pair_geometry_loss


def test_pair_loss_excludes_same_source_views_and_is_rotation_invariant():
    torch.manual_seed(4)
    x = torch.randn(8, 4, dtype=torch.float64)
    rotation, _ = torch.linalg.qr(torch.randn(4, 4, dtype=torch.float64))
    ids = torch.arange(8)//2
    loss, stats = pair_geometry_loss(x @ rotation.T, x, x, ids)
    assert loss < 1e-20 and stats['pairs'] == 24
    with pytest.raises(ValueError, match='different source'):
        pair_geometry_loss(x, x, x, torch.zeros(8))


def test_high_similarity_error_has_finite_gradient_and_detached_targets():
    teacher = torch.tensor([[1.,0.],[.8,.6]], requires_grad=True)
    source = torch.eye(2, requires_grad=True)
    output = source.detach().clone().requires_grad_(True)
    loss, stats = pair_geometry_loss(output,teacher,source,torch.arange(2))
    torch.testing.assert_close(loss,torch.tensor(1.28))
    assert stats['pairs'] == stats['tail_pairs'] == 1
    loss.backward()
    assert torch.isfinite(output.grad).all() and output.grad.abs().sum() > 0
    assert teacher.grad is None and source.grad is None
    # The output cannot remove its own errors by moving the tail-selection mask.
    _, changed = pair_geometry_loss(torch.ones_like(output),teacher,source,torch.arange(2))
    assert changed['tail_pairs'] == stats['tail_pairs']


def test_empty_tail_remains_connected_with_zero_loss_for_exact_teacher():
    x = torch.eye(3,requires_grad=True)
    loss, stats = pair_geometry_loss(x,x.detach(),x.detach(),torch.arange(3))
    assert stats['tail_pairs'] == 0 and stats['pairs'] == 3 and loss == 0
    loss.backward()
    assert torch.isfinite(x.grad).all()
