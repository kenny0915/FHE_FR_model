import numpy as np
import pytest
import torch
from controlled_degree2.supervised_template_pairs import restrict_pairs, supervised_margin_loss


def test_partition_filter_excludes_any_outside_endpoint_and_deduplicates():
    pairs = np.array([[10, 20, 1], [20, 10, 1], [20, 30, 0], [40, 10, 0], [30, 40, 1]])
    result = restrict_pairs(pairs, np.array([20, 10]))
    np.testing.assert_array_equal(result, [[0, 1, 1]])
    assert restrict_pairs(pairs, np.array([99])).shape == (0, 3)
    with pytest.raises(ValueError, match='conflicting'):
        restrict_pairs(np.array([[10,20,1], [20,10,0]]), np.array([10,20]))


def test_margin_loss_value_and_gradient_push_scores_in_correct_directions():
    # Independent positive and negative pairs isolate gradient directions.
    x = torch.tensor([[1.,0.], [0.,1.], [1.,0.], [.8,.6]], dtype=torch.float64, requires_grad=True)
    pairs = torch.tensor([[0,1,1], [2,3,0]])
    loss = supervised_margin_loss(x, pairs)
    assert float(loss) == pytest.approx(.4**2+.6**2)
    grad, = torch.autograd.grad(loss, x)
    new = x.detach()-.01*grad
    cosine = torch.nn.functional.cosine_similarity
    assert cosine(new[0],new[1],dim=0) > 0
    assert cosine(new[2],new[3],dim=0) < .8
    assert torch.isfinite(grad).all()


def test_balanced_loss_is_not_diluted_by_duplicating_negative_pairs():
    x = torch.tensor([[1.,0.], [0.,1.], [.8,.6]])
    pairs = torch.tensor([[0,1,1], [0,2,0]])
    duplicated = torch.cat([pairs[:1], pairs[1:].repeat(10,1)])
    torch.testing.assert_close(supervised_margin_loss(x,pairs), supervised_margin_loss(x,duplicated))
    with pytest.raises(ValueError, match='both positive'):
        supervised_margin_loss(x,pairs[:1])
