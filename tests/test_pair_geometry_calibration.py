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


def test_validation_threshold_includes_near_boundary_pairs():
    from controlled_degree2.calibrate_pair_geometry import evaluate
    source = torch.eye(2)
    teacher = torch.tensor([[1., 0.], [.25, (1-.25**2)**.5]])
    ids = torch.arange(2)
    high = evaluate(torch.eye(2), torch.zeros(2), source, teacher, ids, .3)
    near = evaluate(torch.eye(2), torch.zeros(2), source, teacher, ids, .2)
    assert high['tail_mse'] == 0.
    assert abs(near['tail_mse'] - .0625) < 1e-6
    assert abs(high['all_mse'] - near['all_mse']) < 1e-8


def test_population_loss_and_affine_gradients_match_explicit_pairs():
    torch.manual_seed(28)
    source = torch.randn(7, 4, dtype=torch.float64)
    teacher = torch.randn_like(source)
    matrix = torch.eye(4, dtype=torch.float64).requires_grad_()
    bias = torch.zeros(4, dtype=torch.float64, requires_grad=True)
    output = source @ matrix.T + bias
    loss, stats = pair_geometry_loss(output, teacher, source, torch.arange(7), .2)
    cosine = torch.nn.functional.cosine_similarity
    errors, tail = [], []
    for i in range(7):
        for j in range(i+1, 7):
            target = cosine(teacher[i], teacher[j], dim=0)
            baseline = cosine(source[i], source[j], dim=0)
            error = (cosine(output[i], output[j], dim=0)-target).square()
            errors.append(error)
            if target >= .2 or baseline >= .2:
                tail.append(error)
    reference = torch.stack(errors).mean() + torch.stack(tail).mean()
    assert stats['pairs'] == 21 and stats['tail_pairs'] == len(tail)
    torch.testing.assert_close(loss, reference)
    actual_grad = torch.autograd.grad(loss, (matrix, bias), retain_graph=True)
    expected_grad = torch.autograd.grad(reference, (matrix, bias))
    for actual, expected in zip(actual_grad, expected_grad):
        torch.testing.assert_close(actual, expected)
