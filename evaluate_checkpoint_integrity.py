"""Compare a calibrated HerPN checkpoint with its immutable reference."""

import argparse
import json
import math

import torch


def load_state(path):
    checkpoint = torch.load(path, map_location="cpu")
    state = checkpoint
    if isinstance(checkpoint, dict):
        for key in ("state_dict_backbone", "state_dict", "model"):
            if isinstance(checkpoint.get(key), dict):
                state = checkpoint[key]
                break
    if not isinstance(state, dict):
        raise TypeError(f"Checkpoint does not contain a state dict: {path}")
    return {
        (name[7:] if name.startswith("module.") else name): tensor
        for name, tensor in state.items()
    }


def frozen_tensor_name(name):
    return (
        name.startswith(("fc.", "features."))
        or name.endswith(".prelu.weight")
    )


def quadratic_coefficient2(state, weight_name, eps):
    prefix = weight_name[:-len("weight")]
    weight = state[weight_name].double().squeeze()
    scale_name = prefix + "basis_scale"
    if scale_name in state:
        scale = state[scale_name].double().squeeze()[2]
    else:
        scale = torch.ones_like(weight)
    variance = state[prefix + "bn2.running_var"].double()
    return weight * scale / torch.sqrt(8.0 * math.pi * (variance + eps))


def compare(reference, checkpoint, equivalent=None, herpn_bn_eps=1e-4):
    reference_names = set(reference)
    checkpoint_names = set(checkpoint)
    if reference_names != checkpoint_names:
        raise ValueError(
            "State keys differ: "
            f"missing={sorted(reference_names - checkpoint_names)[:5]}, "
            f"extra={sorted(checkpoint_names - reference_names)[:5]}")

    bn_names = tuple(name for name in reference if name.endswith(
        ("running_mean", "running_var", "num_batches_tracked")))
    frozen_names = tuple(name for name in reference if frozen_tensor_name(name))
    trainable_names = tuple(
        name for name in reference
        if name.endswith(("weight", "bias")) and name not in frozen_names)

    relative_changes = []
    for name in trainable_names:
        delta = (checkpoint[name].double() - reference[name].double()).norm()
        scale = max(reference[name].double().norm().item(), 1.0)
        relative_changes.append((delta.item() / scale, name))

    ratios = []
    for name in reference:
        if not name.endswith("prelu.herpn.weight"):
            continue
        before = quadratic_coefficient2(reference, name, herpn_bn_eps)
        after = quadratic_coefficient2(checkpoint, name, herpn_bn_eps)
        ratios.append(after.norm().item() / before.norm().item())
    if len(ratios) != 25:
        raise ValueError(f"Expected 25 HerPN quadratic terms, found {len(ratios)}")

    finite_count = sum(
        bool(torch.isfinite(tensor).all()) for tensor in checkpoint.values())
    result = {
        "tensor_count": len(checkpoint),
        "finite_tensor_count": finite_count,
        "all_tensors_finite": finite_count == len(checkpoint),
        "bn_buffer_count": len(bn_names),
        "bn_buffers_bitwise_equal": sum(
            torch.equal(reference[name], checkpoint[name]) for name in bn_names),
        "frozen_tensor_count": len(frozen_names),
        "frozen_tensors_bitwise_equal": sum(
            torch.equal(reference[name], checkpoint[name])
            for name in frozen_names),
        "worst_relative_parameter_change": max(relative_changes)[0],
        "worst_relative_parameter_name": max(relative_changes)[1],
        "quadratic_coefficient_l2_ratio_min": min(ratios),
        "quadratic_coefficient_l2_ratio_max": max(ratios),
        "quadratic_coefficient_l2_ratio_mean": sum(ratios) / len(ratios),
        "quadratic_coefficient_count": len(ratios),
    }
    if equivalent is not None:
        if set(equivalent) != checkpoint_names:
            raise ValueError("Equivalent-checkpoint state keys differ")
        equal_count = sum(
            torch.equal(equivalent[name], checkpoint[name])
            for name in checkpoint)
        result.update({
            "equivalent_tensor_count": len(checkpoint),
            "equivalent_tensors_bitwise_equal": equal_count,
        })
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("reference")
    parser.add_argument("checkpoint")
    parser.add_argument("--equivalent-checkpoint")
    parser.add_argument("--herpn-bn-eps", type=float, default=1e-4)
    parser.add_argument("--output")
    args = parser.parse_args()
    result = compare(
        load_state(args.reference), load_state(args.checkpoint),
        (load_state(args.equivalent_checkpoint)
         if args.equivalent_checkpoint else None),
        herpn_bn_eps=args.herpn_bn_eps)
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(text)
    print(text, end="")


if __name__ == "__main__":
    main()
