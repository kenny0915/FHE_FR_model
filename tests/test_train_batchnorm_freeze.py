import torch
from torch import nn

from train_v2 import (
    freeze_batchnorm_before_activation,
    freeze_batchnorm_for_training,
)


def test_freeze_batchnorm_for_training_preserves_stats_and_affine():
    model = nn.Sequential(
        nn.Conv2d(3, 4, 1),
        nn.BatchNorm2d(4),
        nn.Sequential(nn.SyncBatchNorm(4), nn.ReLU()),
    ).train()
    batchnorms = (model[1], model[2][0])
    initial_means = [module.running_mean.clone() for module in batchnorms]

    count = freeze_batchnorm_for_training(model, affine=True)
    model(torch.randn(8, 3, 5, 5)).sum().backward()

    assert count == 2
    assert all(not module.training for module in batchnorms)
    assert all(not module.weight.requires_grad for module in batchnorms)
    assert all(not module.bias.requires_grad for module in batchnorms)
    for module, expected in zip(batchnorms, initial_means):
        torch.testing.assert_close(module.running_mean, expected)


def test_freeze_batchnorm_before_activation_leaves_downstream_adaptive():
    model = nn.Sequential(
        nn.Conv2d(3, 4, 1),
        nn.BatchNorm2d(4),
        nn.Conv2d(4, 4, 1),
        nn.PReLU(4),
        nn.BatchNorm2d(4),
    ).train()
    upstream = model[1]
    downstream = model[4]
    upstream_mean = upstream.running_mean.clone()
    downstream_mean = downstream.running_mean.clone()

    count = freeze_batchnorm_before_activation(model, "3")
    model(torch.randn(8, 3, 5, 5))

    assert count == 1
    assert not upstream.training
    assert downstream.training
    torch.testing.assert_close(upstream.running_mean, upstream_mean)
    assert not torch.equal(downstream.running_mean, downstream_mean)
