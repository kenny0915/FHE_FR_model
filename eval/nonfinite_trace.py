"""Trace the first module producing a non-finite value for selected rows."""

from contextlib import ExitStack

import torch


def _tensor_output(value):
    if torch.is_tensor(value):
        return value
    if isinstance(value, (tuple, list)):
        return next((item for item in value if torch.is_tensor(item)), None)
    return None


def _row_summary(value):
    tensor = _tensor_output(value)
    if tensor is None or tensor.ndim == 0:
        return None
    flattened = tensor.detach().float().reshape(tensor.shape[0], -1)
    finite = torch.isfinite(flattened)
    safe_absolute = torch.where(finite, flattened.abs(), 0.0)
    return {
        "finite_absmax": safe_absolute.amax(dim=1).cpu(),
        "nonfinite_values": (~finite).sum(dim=1).cpu(),
    }


@torch.no_grad()
def trace_first_nonfinite(model, inputs):
    """Return an execution-order first-failure record for every input row.

    This is intended for a small deterministic replay of rows already known
    to produce non-finite final embeddings. Hooks summarize outputs only; they
    do not alter the model or its inference graph.
    """
    events = []

    input_summary = _row_summary(inputs)
    if input_summary is None:
        raise ValueError("trace inputs must have a batch dimension")
    events.append(("__input__", input_summary))

    def record(name):
        def hook(_module, _inputs, output):
            summary = _row_summary(output)
            if summary is not None:
                events.append((name, summary))
        return hook

    with ExitStack() as stack:
        for name, module in model.named_modules():
            if not name:
                continue
            is_leaf = not any(module.children())
            is_polynomial_boundary = module.__class__.__name__ in {
                "HerPN", "FoldedHerPN", "ProgressiveHerPNActivation",
            }
            if not (is_leaf or is_polynomial_boundary):
                continue
            handle = module.register_forward_hook(record(name))
            stack.callback(handle.remove)
        output = model(inputs)

    output_summary = _row_summary(output)
    if output_summary is not None:
        events.append(("__embedding__", output_summary))

    records = []
    for row in range(inputs.shape[0]):
        previous_name = ""
        previous_absmax = 0.0
        first_name = ""
        first_absmax = 0.0
        first_nonfinite_values = 0
        for name, summary in events:
            nonfinite_values = int(summary["nonfinite_values"][row].item())
            finite_absmax = float(summary["finite_absmax"][row].item())
            if nonfinite_values:
                first_name = name
                first_absmax = finite_absmax
                first_nonfinite_values = nonfinite_values
                break
            previous_name = name
            previous_absmax = finite_absmax
        final_nonfinite_values = int(
            output_summary["nonfinite_values"][row].item()
            if output_summary is not None else 0
        )
        records.append({
            "first_nonfinite_module": first_name,
            "first_nonfinite_values": first_nonfinite_values,
            "first_finite_absmax": first_absmax,
            "previous_module": previous_name,
            "previous_finite_absmax": previous_absmax,
            "final_nonfinite_values": final_nonfinite_values,
        })
    return records
