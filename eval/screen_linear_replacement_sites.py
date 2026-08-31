"""Select the least disruptive site for a tail-safe affine replacement."""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backbones import get_model
from backbones.iresnet_prelu_herpn import PReLULinearActivation
from eval import verification
from eval.compare_activation_distributions import extract_state_dict
from eval.screen_channelwise_linear_replacement import load_recordio_orientations
from eval.screen_linear_replacement import MODEL_KWARGS


def replace_submodule(model, name, replacement):
    parent_name, attribute = name.rsplit(".", 1)
    parent = model.get_submodule(parent_name)
    original = getattr(parent, attribute)
    setattr(parent, attribute, replacement)
    return original


def tail_safe_activation(source):
    channels = source.prelu.weight.numel()
    replacement = PReLULinearActivation(
        channels=channels,
        distill_eps=float(source.distill_eps.item()),
        stage_index=source.stage_index,
        blend=1.0,
        trainable=False,
    ).to(device=source.prelu.weight.device, dtype=source.prelu.weight.dtype)
    with torch.no_grad():
        replacement.prelu.weight.copy_(source.prelu.weight)
        replacement.weight.copy_(source.prelu.weight.reshape(-1, 1, 1))
        replacement.bias.zero_()
    return replacement.eval()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", default="ms1m-retinaface-t1")
    parser.add_argument("--targets", nargs="+", default=[
        "lfw", "cfp_fp", "agedb_30"])
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--legacy-prefix", type=int, default=8)
    parser.add_argument("--recordio-index", type=int, default=86052)
    parser.add_argument("--max-accuracy-drop", type=float, default=0.005)
    parser.add_argument("--max-tail-growth", type=float, default=10.0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint-output", required=True)
    args = parser.parse_args()

    device = torch.device("cuda")
    model_kwargs = dict(MODEL_KWARGS)
    model_kwargs["prelu_herpn_linear_indices"] = (24,)
    model = get_model("r50_prelu_herpn", **model_kwargs)
    raw = torch.load(args.checkpoint, map_location="cpu")
    state = extract_state_dict(raw)
    model.load_backbone_init_state_dict(state)
    model.float().to(device).eval()

    activation_names = [
        name for name, _ in model.named_progressive_activations()
    ]
    candidate_names = activation_names[args.legacy_prefix:]
    stress = load_recordio_orientations(args.data_dir, args.recordio_index)
    datasets = {
        target: verification.load_bin(
            str(Path(args.data_dir) / f"{target}.bin"), (112, 112))
        for target in args.targets
    }
    diagnostics = {"absmax": 0.0, "nonfinite": 0}

    def capture_output(_, __, output):
        values = output.detach().float()
        finite = torch.isfinite(values)
        diagnostics["nonfinite"] += int((~finite).sum().item())
        if bool(finite.any()):
            diagnostics["absmax"] = max(
                diagnostics["absmax"],
                float(values[finite].abs().amax().item()))

    output_handle = model.register_forward_hook(capture_output)

    def evaluate_current():
        result = {"targets": {}}
        diagnostics.update(absmax=0.0, nonfinite=0)
        with torch.no_grad():
            stress_embeddings = model(stress.to(device))
        result["recordio"] = {
            "embedding_finite": bool(torch.isfinite(stress_embeddings).all()),
            "embedding_absmax": diagnostics["absmax"],
            "nonfinite_elements": diagnostics["nonfinite"],
        }
        del stress_embeddings
        complete = result["recordio"]["embedding_finite"]
        for target, data in datasets.items():
            diagnostics.update(absmax=0.0, nonfinite=0)
            try:
                _, _, accuracy, std, xnorm, _ = verification.test(
                    data, model, args.batch_size, 10,
                    fail_on_nonfinite=True)
                metrics = {
                    "accuracy_flip": float(accuracy),
                    "accuracy_std": float(std),
                    "xnorm": float(xnorm),
                    "embedding_absmax": diagnostics["absmax"],
                    "nonfinite_elements": diagnostics["nonfinite"],
                }
            except FloatingPointError as error:
                complete = False
                metrics = {
                    "error": str(error),
                    "embedding_absmax": diagnostics["absmax"],
                    "nonfinite_elements": diagnostics["nonfinite"],
                }
            result["targets"][target] = metrics
        result["complete"] = complete
        return result

    try:
        source = evaluate_current()
        candidates = []
        for activation_index, name in enumerate(activation_names):
            if name not in candidate_names:
                continue
            source_activation = model.get_submodule(name)
            replacement = tail_safe_activation(source_activation)
            original = replace_submodule(model, name, replacement)
            try:
                candidate = evaluate_current()
            finally:
                replace_submodule(model, name, original)
            candidate.update(
                activation_index=activation_index,
                activation_name=name,
                slope_min=float(replacement.weight.min().item()),
                slope_mean=float(replacement.weight.mean().item()),
                slope_max=float(replacement.weight.max().item()),
            )
            candidates.append(candidate)
    finally:
        output_handle.remove()

    accepted = []
    for candidate in candidates:
        if not candidate["complete"]:
            continue
        drops = {
            target: (
                source["targets"][target]["accuracy_flip"]
                - candidate["targets"][target]["accuracy_flip"])
            for target in args.targets
        }
        candidate["accuracy_drops"] = drops
        target_tail_safe = all(
            candidate["targets"][target]["embedding_absmax"]
            <= source["targets"][target]["embedding_absmax"] * args.max_tail_growth
            for target in args.targets
        )
        recordio_tail_safe = (
            candidate["recordio"]["embedding_absmax"]
            <= source["recordio"]["embedding_absmax"] * args.max_tail_growth
        )
        candidate["tail_growth_gate"] = {
            "targets": target_tail_safe,
            "recordio": recordio_tail_safe,
        }
        if (max(drops.values()) <= args.max_accuracy_drop
                and target_tail_safe and recordio_tail_safe):
            accepted.append(candidate)

    selected = max(
        accepted,
        key=lambda item: sum(
            item["targets"][target]["accuracy_flip"]
            for target in args.targets),
        default=None,
    )
    checkpoint_output = None
    if selected is not None:
        selected_index = selected["activation_index"]
        selected_name = selected["activation_name"]
        selected_kwargs = dict(MODEL_KWARGS)
        selected_kwargs["prelu_herpn_linear_indices"] = (selected_index,)
        selected_kwargs["prelu_herpn_linear_trainable"] = False
        selected_model = get_model("r50_prelu_herpn", **selected_kwargs)
        selected_model.load_backbone_init_state_dict(state)
        selected_activation = selected_model.get_submodule(selected_name)
        with torch.no_grad():
            selected_activation.weight.copy_(
                selected_activation.prelu.weight.reshape(-1, 1, 1))
            selected_activation.bias.zero_()
            selected_activation.set_blend(1.0)
        checkpoint_output = Path(args.checkpoint_output)
        checkpoint_output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {key: value.detach().cpu()
             for key, value in selected_model.state_dict().items()},
            checkpoint_output,
        )

    result = {
        "source": source,
        "candidates": candidates,
        "selected": selected,
        "checkpoint_output": (
            str(checkpoint_output) if checkpoint_output is not None else None),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    if selected is None:
        raise RuntimeError(
            f"No replacement site passed the gates; diagnostics saved to {output}")


if __name__ == "__main__":
    main()
