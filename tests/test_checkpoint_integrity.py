import torch

from evaluate_checkpoint_integrity import compare


def make_state(offset=0.0):
    state = {
        "conv.weight": torch.tensor([1.0 + offset]),
        "bn.running_mean": torch.tensor([0.0]),
        "bn.running_var": torch.tensor([1.0]),
        "bn.num_batches_tracked": torch.tensor(1),
        "fc.weight": torch.tensor([2.0]),
        "features.running_mean": torch.tensor([0.0]),
    }
    for index in range(25):
        prefix = "prelu" if index == 0 else f"layer.{index}.prelu"
        state[f"{prefix}.prelu.weight"] = torch.tensor([0.25])
        state[f"{prefix}.herpn.weight"] = torch.tensor([1.0 + offset])
        state[f"{prefix}.herpn.bn2.running_var"] = torch.tensor([1.0])
    return state


def test_integrity_comparison_counts_frozen_state_and_quadratics():
    reference = make_state()
    checkpoint = make_state(offset=0.01)
    # Immutable tensors must remain exact; only trainable weights move.
    checkpoint["fc.weight"] = reference["fc.weight"].clone()
    for name in checkpoint:
        if name.endswith(".prelu.weight"):
            checkpoint[name] = reference[name].clone()
    result = compare(reference, checkpoint, equivalent=checkpoint)
    assert result["all_tensors_finite"]
    assert result["bn_buffers_bitwise_equal"] == result["bn_buffer_count"]
    assert result["frozen_tensors_bitwise_equal"] == result["frozen_tensor_count"]
    assert result["quadratic_coefficient_count"] == 25
    assert result["quadratic_coefficient_l2_ratio_min"] > 1.0
    assert result["equivalent_tensors_bitwise_equal"] == len(checkpoint)
