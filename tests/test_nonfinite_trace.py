import torch
import pytest
from torch import nn

from eval.nonfinite_trace import trace_first_nonfinite


class _Square(nn.Module):
    def forward(self, inputs):
        return inputs.square()


def test_trace_first_nonfinite_finds_first_bad_leaf_and_previous_output():
    model = nn.Sequential(
        nn.Identity(),
        _Square(),
        nn.Identity(),
    ).eval()
    inputs = torch.tensor([[2.0], [1.0e30]], dtype=torch.float32)

    records = trace_first_nonfinite(model, inputs)

    assert records[0]["first_nonfinite_module"] == ""
    assert records[0]["final_nonfinite_values"] == 0
    assert records[1]["first_nonfinite_module"] == "1"
    assert records[1]["previous_module"] == "0"
    assert records[1]["previous_finite_absmax"] == pytest.approx(1.0e30)
    assert records[1]["first_nonfinite_values"] == 1
    assert records[1]["final_nonfinite_values"] == 1
