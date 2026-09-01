import torch
from torch import nn

from refresh_bn_stats import (
    bn_buffers_are_finite,
    restore_bn_buffers,
    snapshot_bn_buffers,
    synchronize_bad_forward,
)


def test_bn_buffer_snapshot_restores_rejected_forward_state():
    layer = nn.BatchNorm1d(3)
    snapshot = snapshot_bn_buffers([layer])

    layer.running_mean.fill_(float("nan"))
    layer.running_var.fill_(9.0)
    layer.num_batches_tracked.fill_(7)
    assert not bn_buffers_are_finite([layer])

    restore_bn_buffers(snapshot)

    assert bn_buffers_are_finite([layer])
    assert torch.equal(layer.running_mean, torch.zeros(3))
    assert torch.equal(layer.running_var, torch.ones(3))
    assert layer.num_batches_tracked.item() == 0


def test_synchronize_bad_forward_is_local_without_process_group():
    assert not synchronize_bad_forward(False, torch.device("cpu"))
    assert synchronize_bad_forward(True, torch.device("cpu"))
