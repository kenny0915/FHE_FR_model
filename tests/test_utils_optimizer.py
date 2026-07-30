import torch
import pytest

from utils.utils_optimizer import temporary_optimizer_lr_scale


def test_temporary_optimizer_lr_scale_restores_scheduler_lr():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=0.2)

    parameter.grad = torch.tensor(1.0)
    with temporary_optimizer_lr_scale(optimizer, 0.1):
        assert optimizer.param_groups[0]["lr"] == pytest.approx(0.02)
        optimizer.step()

    assert torch.allclose(parameter, torch.tensor(0.98))
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.2)
