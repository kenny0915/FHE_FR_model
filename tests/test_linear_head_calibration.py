import copy

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from controlled_degree2.calibrate_linear_head import fit_alignment, fold_alignment, source_split


def test_ridge_recovers_known_rotation_on_unseen_embeddings():
    torch.manual_seed(9)
    x = torch.randn(256, 6, dtype=torch.float64)
    rotation, _ = torch.linalg.qr(torch.randn(6, 6, dtype=torch.float64))
    matrix, bias = fit_alignment(x, x @ rotation.T, 1e-7)
    heldout = torch.randn(32, 6, dtype=torch.float64)
    prediction = heldout @ matrix.T + bias
    assert (1-F.cosine_similarity(prediction, heldout @ rotation.T)).mean() < 1e-10
    identity, zero = fit_alignment(x, x, .1)
    torch.testing.assert_close(identity, torch.eye(6, dtype=x.dtype))
    torch.testing.assert_close(zero, torch.zeros(6, dtype=x.dtype))


def test_affine_folding_preserves_output_and_all_bn_state():
    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(7, 4)
            self.features = nn.BatchNorm1d(4)
        def forward(self, x):
            return self.features(self.fc(x))
    torch.manual_seed(10)
    model = Head().eval()
    with torch.no_grad():
        model.features.running_mean.copy_(torch.randn(4))
        model.features.running_var.copy_(torch.rand(4)+.5)
        model.features.weight.copy_(torch.tensor([.3, -.7, 1.2, -.4]))
        model.features.bias.copy_(torch.randn(4))
    old = copy.deepcopy(model)
    matrix, bias = torch.randn(4, 4), torch.randn(4)
    x = torch.randn(20, 7)
    fold_alignment(model, matrix, bias)
    torch.testing.assert_close(model(x), old(x) @ matrix.T+bias, atol=2e-6, rtol=2e-6)
    for n, t in model.features.state_dict().items():
        assert torch.equal(t, old.features.state_dict()[n])
    with torch.no_grad():
        model.features.weight[0] = 0
    with pytest.raises(ValueError, match='singular'):
        fold_alignment(model, matrix, bias)


def test_split_is_deterministic_and_disjoint_by_source_image():
    a, b = source_split(100, 60, 20, 4)
    c, d = source_split(100, 60, 20, 4)
    assert np.array_equal(a, c) and np.array_equal(b, d)
    assert len(set(a) | set(b)) == 80
    with pytest.raises(ValueError, match='disjoint'):
        source_split(100, 90, 20, 4)


def test_invalid_fit_rejected():
    x = torch.ones(8, 3)
    with pytest.raises(ValueError, match='positive ridge'):
        fit_alignment(x, x, 0)
    with pytest.raises(ValueError, match='nonzero'):
        fit_alignment(x, x*0, 1)
    with pytest.raises(ValueError, match='finite matching'):
        fit_alignment(x*float('inf'), x, 1)
