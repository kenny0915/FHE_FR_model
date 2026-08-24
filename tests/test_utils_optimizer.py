import torch
import pytest

from utils.utils_optimizer import (
    clip_grad_norm_stable,
    nonfinite_gradient_diagnostics,
    nonfinite_gradient_tensor_count,
    select_gradient_clip_parameters,
    temporary_optimizer_lr_scale,
)


def test_nonfinite_gradient_diagnostics_reports_parameter_and_counts():
    backbone = torch.nn.Linear(3, 2)
    classifier = torch.nn.Linear(2, 2)
    backbone.weight.grad = torch.tensor([
        [1.0, float("inf"), 2.0],
        [float("nan"), -3.0, 4.0],
    ])
    classifier.bias.grad = torch.tensor([float("-inf"), 5.0])

    count, diagnostics = nonfinite_gradient_diagnostics((
        ("backbone", backbone),
        ("partial_fc", classifier),
    ))

    assert count == 2
    assert [item["name"] for item in diagnostics] == [
        "backbone.weight", "partial_fc.bias"]
    assert diagnostics[0]["nonfinite_elements"] == 2
    assert diagnostics[0]["finite_absmax"] == pytest.approx(4.0)
    assert diagnostics[1]["nonfinite_elements"] == 1
    assert diagnostics[1]["finite_absmax"] == pytest.approx(5.0)

    device_count = nonfinite_gradient_tensor_count((
        ("backbone", backbone),
        ("partial_fc", classifier),
    ))
    assert device_count.device == backbone.weight.device
    assert int(device_count) == 2


def test_nonfinite_gradient_diagnostics_limits_details_not_count():
    module = torch.nn.Linear(2, 2)
    module.weight.grad = torch.full_like(module.weight, float("nan"))
    module.bias.grad = torch.full_like(module.bias, float("inf"))

    count, diagnostics = nonfinite_gradient_diagnostics(
        (("model", module),), max_diagnostics=1)

    assert count == 2
    assert len(diagnostics) == 1


def test_stable_gradient_clip_handles_finite_fp32_norm_overflow():
    parameter = torch.nn.Parameter(torch.zeros(2))
    parameter.grad = torch.tensor([1e30, -1e30])

    total_norm = clip_grad_norm_stable(
        [parameter], max_norm=1.0, error_if_nonfinite=True)

    assert torch.isfinite(total_norm)
    assert total_norm.dtype == torch.float64
    assert total_norm.item() == pytest.approx(2 ** 0.5 * 1e30, rel=1e-6)
    assert torch.linalg.vector_norm(parameter.grad).item() == pytest.approx(
        1.0, rel=1e-6)


def test_temporary_optimizer_lr_scale_restores_scheduler_lr():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=0.2)

    parameter.grad = torch.tensor(1.0)
    with temporary_optimizer_lr_scale(optimizer, 0.1):
        assert optimizer.param_groups[0]["lr"] == pytest.approx(0.02)
        optimizer.step()

    assert torch.allclose(parameter, torch.tensor(0.98))
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.2)


def test_temporary_optimizer_lr_scale_can_target_one_scope():
    backbone = torch.nn.Parameter(torch.tensor(1.0))
    polynomial = torch.nn.Parameter(torch.tensor(1.0))
    classifier = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([
        {"params": [backbone], "lr": 0.2, "scope": "backbone"},
        {"params": [polynomial], "lr": 0.2, "scope": "layerwise_poly"},
        {"params": [classifier], "lr": 0.2, "scope": "classifier"},
    ])

    with temporary_optimizer_lr_scale(
            optimizer, 0.1, scope="backbone"):
        assert optimizer.param_groups[0]["lr"] == pytest.approx(0.02)
        assert optimizer.param_groups[1]["lr"] == pytest.approx(0.2)
        assert optimizer.param_groups[2]["lr"] == pytest.approx(0.2)

    assert [group["lr"] for group in optimizer.param_groups] == pytest.approx(
        [0.2, 0.2, 0.2])


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
