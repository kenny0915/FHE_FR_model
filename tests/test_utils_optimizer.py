import torch
import pytest

from utils.utils_optimizer import (
    select_gradient_clip_parameters,
    temporary_optimizer_lr_scale,
)


def test_temporary_optimizer_lr_scale_restores_scheduler_lr():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=0.2)

    parameter.grad = torch.tensor(1.0)
    with temporary_optimizer_lr_scale(optimizer, 0.1):
        assert optimizer.param_groups[0]["lr"] == pytest.approx(0.02)
        optimizer.step()

    assert torch.allclose(parameter, torch.tensor(0.98))
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.2)


def test_gradient_clip_scope_can_isolate_backbone_from_classifier():
    backbone = torch.nn.Linear(3, 2)
    classifier = torch.nn.Linear(2, 5)
    optimizer = torch.optim.AdamW([
        {"params": backbone.parameters()},
        {"params": classifier.parameters()},
    ])

    backbone_parameters = select_gradient_clip_parameters(
        optimizer, backbone, scope="backbone")
    all_parameters = select_gradient_clip_parameters(
        optimizer, backbone, scope="all")

    assert {id(parameter) for parameter in backbone_parameters} == {
        id(parameter) for parameter in backbone.parameters()}
    assert len(all_parameters) == len(list(backbone.parameters())) + len(
        list(classifier.parameters()))


def test_gradient_clip_scope_rejects_unknown_value():
    backbone = torch.nn.Linear(3, 2)
    optimizer = torch.optim.SGD(backbone.parameters(), lr=0.1)
    with pytest.raises(ValueError, match="gradient_clip_scope"):
        select_gradient_clip_parameters(optimizer, backbone, scope="classifier")
