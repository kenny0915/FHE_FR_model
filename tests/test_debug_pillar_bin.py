import torch

from eval.debug_pillar_bin import (
    build_parser,
    compact_activation_trace,
    first_nonfinite_activation,
    tensor_summary,
)


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


def test_compact_activation_trace():
    trace = [{
        "name": "layer1.0.prelu",
        "input": {"finite_absmax": 6.0},
        "output": {"finite_absmax": 8.0, "finite": 3, "numel": 4},
    }]
    assert compact_activation_trace(trace) == [{
        "name": "layer1.0.prelu",
        "input_absmax": 6.0,
        "output_absmax": 8.0,
        "output_nonfinite": 1,
    }]


def test_debug_parser_accepts_diagnostic_clip_and_forced_rows():
    args = build_parser().parse_args([
        "--checkpoint", "model.pt",
        "--bin", "lfw.bin",
        "--diagnostic-clip",
        "--trace-rows", "3,7",
    ])
    assert args.diagnostic_clip is True
    assert args.trace_rows == "3,7"
