"""Screen square-free full-replacement variants on face verification bins."""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backbones import get_model
from eval import verification
from eval.compare_activation_distributions import extract_state_dict


MODEL_KWARGS = {
    "dropout": 0.0,
    "fp16": False,
    "num_features": 512,
    "herpn_progress": 0.0,
    "herpn_bn_eps": 1e-4,
    "herpn_range_limit": 1.0,
    "prelu_herpn_layerwise_scale": True,
    "prelu_herpn_initial_scale": 1.0,
    "prelu_herpn_distill_eps": 1e-4,
    "prelu_herpn_legacy_prefix": 8,
    "prelu_herpn_linear_indices": (24,),
}


def apply_variant(activation, variant):
    with torch.no_grad():
        fitted_bias = activation.bias.detach().clone()
        teacher_slope = activation.prelu.weight.detach().reshape(-1, 1, 1)
        if variant == "fitted":
            pass
        elif variant == "constant":
            activation.weight.zero_()
        elif variant == "zero":
            activation.weight.zero_()
            activation.bias.zero_()
        elif variant == "prelu_slope":
            activation.weight.copy_(teacher_slope)
            activation.bias.zero_()
        elif variant == "prelu_slope_bias":
            activation.weight.copy_(teacher_slope)
            activation.bias.copy_(fitted_bias)
        elif variant == "small_positive_bias":
            activation.weight.copy_(teacher_slope.abs())
            activation.bias.copy_(fitted_bias)
        else:
            raise ValueError(f"Unknown variant: {variant}")
        activation.set_blend(1.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--data-dir", default="ms1m-retinaface-t1")
    parser.add_argument("--targets", nargs="+", default=[
        "lfw", "cfp_fp", "agedb_30"])
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    device = torch.device("cuda")
    model = get_model("r50_prelu_herpn", **MODEL_KWARGS)
    raw = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(extract_state_dict(raw), strict=True)
    activation = dict(model.named_modules())["layer4.2.prelu"]
    apply_variant(activation, args.variant)
    model.float().to(device).eval()

    diagnostics = {"absmax": 0.0, "nonfinite": 0}

    def capture_output(_, __, output):
        values = output.detach().float()
        finite = torch.isfinite(values)
        diagnostics["nonfinite"] += int((~finite).sum().item())
        if bool(finite.any()):
            diagnostics["absmax"] = max(
                diagnostics["absmax"],
                float(values[finite].abs().amax().item()),
            )

    handle = model.register_forward_hook(capture_output)
    result = {"variant": args.variant, "targets": {}}
    try:
        for target in args.targets:
            diagnostics.update(absmax=0.0, nonfinite=0)
            data = verification.load_bin(
                str(Path(args.data_dir) / f"{target}.bin"), (112, 112))
            _, _, accuracy, std, xnorm, _ = verification.test(
                data, model, args.batch_size, 10, fail_on_nonfinite=True)
            result["targets"][target] = {
                "accuracy_flip": float(accuracy),
                "accuracy_std": float(std),
                "xnorm": float(xnorm),
                "embedding_absmax": diagnostics["absmax"],
                "nonfinite_elements": diagnostics["nonfinite"],
            }
            del data
    finally:
        handle.remove()

    activation = dict(model.named_modules())["layer4.2.prelu"]
    result["coefficient"] = {
        "weight_min": float(activation.weight.min().item()),
        "weight_mean": float(activation.weight.mean().item()),
        "weight_max": float(activation.weight.max().item()),
        "bias_min": float(activation.bias.min().item()),
        "bias_mean": float(activation.bias.mean().item()),
        "bias_max": float(activation.bias.max().item()),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
