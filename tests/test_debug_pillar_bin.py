import torch

from eval.debug_pillar_bin import first_nonfinite_activation, tensor_summary


def test_tensor_summary_and_first_nonfinite_activation():
    summary = tensor_summary(torch.tensor([1.0, float("inf"), float("nan")]))
    assert summary["numel"] == 3
    assert summary["finite"] == 1
    assert summary["posinf"] == 1
    assert summary["nan"] == 1
    assert summary["finite_absmax"] == 1.0

    trace = [
        {"output": {"finite": 4, "numel": 4}},
        {"output": {"finite": 3, "numel": 4}},
        {"output": {"finite": 0, "numel": 4}},
    ]
    assert first_nonfinite_activation(trace) == 1
