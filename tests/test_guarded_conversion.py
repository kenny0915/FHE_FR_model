import torch
from torch import nn

from controlled_degree2.guarded_conversion import GuardedConversion
from controlled_degree2.model import DirectQuadratic


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(1)
        self.prelu = DirectQuadratic(1, lam_fit=1., name='prelu')
        self.prelu.alpha = 1.
        self.prelu.clip = self.prelu.clip_eval = False
        self.bn1.eval()

    def forward(self, x):
        return self.prelu(self.bn1(x)).flatten(1)


def test_routing_repairs_overflow_input_and_preserves_good_feature_gradient():
    backbone = Toy()
    backbone.prelu.coeffs.requires_grad_(True)
    model = GuardedConversion(backbone, feature_dim=4)
    images = torch.stack([torch.full((1, 2, 2), .25), torch.full((1, 2, 2), 1e20)])
    features, rows, repair, scores = model(images)
    assert rows.tolist() == [0] and scores[1] > 4
    assert torch.isfinite(features).all() and torch.isfinite(repair)
    (features.square().mean()+repair).backward()
    assert torch.isfinite(backbone.bn1.weight.grad).all()
    assert backbone.prelu.coeffs.grad is not None
    assert torch.isfinite(backbone.prelu.coeffs.grad).all()
    assert not backbone.prelu.clip and not backbone.prelu.clip_eval
    assert not backbone.bn1.training and backbone.prelu.training


def test_all_escaping_rows_still_produce_repair_gradients():
    backbone = Toy()
    model = GuardedConversion(backbone, feature_dim=4)
    features, rows, repair, scores = model(torch.full((2, 1, 2, 2), 1e20))
    assert not len(rows) and features.shape == (1, 4)
    assert features.eq(0).all() and scores.gt(4).all()
    repair.backward()
    assert backbone.bn1.weight.grad is not None
    assert torch.isfinite(backbone.bn1.weight.grad).all()


def test_good_rows_are_identical_to_unclipped_backbone():
    backbone = Toy()
    model = GuardedConversion(backbone, feature_dim=4)
    images = torch.randn(3, 1, 2, 2)*.1
    expected = backbone(images)
    features, rows, repair, _ = model(images)
    torch.testing.assert_close(features, expected, rtol=0, atol=0)
    assert rows.tolist() == [0, 1, 2] and repair == 0
