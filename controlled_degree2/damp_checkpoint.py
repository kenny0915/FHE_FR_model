"""Export a degree-2 checkpoint with selected quadratic terms damped."""

from __future__ import annotations

import argparse
import os

import torch

from controlled_degree2.model import (
    damp_quadratic_terms,
    load_controlled_checkpoint,
    parse_lam_scale,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--scales",
        required=True,
        help="prefix factors, for example 'prelu:0.95+layer1:0.95'",
    )
    args = parser.parse_args()

    scales = parse_lam_scale(args.scales)
    if not scales:
        raise ValueError("at least one quadratic damping scale is required")
    model, payload = load_controlled_checkpoint(args.checkpoint, device="cpu")
    calibration = payload["poly_calib"]
    touched = damp_quadratic_terms(model, calibration, scales)
    if not touched:
        raise ValueError(f"quadratic damping matched no activations: {scales}")

    payload["state_dict_backbone"] = {
        name: value.detach().cpu() for name, value in model.state_dict().items()
    }
    payload["poly_calib"] = calibration
    history = list(payload.get("quadratic_tail_damping", ()))
    history.append({
        "source_checkpoint": os.path.abspath(args.checkpoint),
        "requested_scales": scales,
        "touched": touched,
        "selection_data": "MS1MV3 deployment-tail audit only",
        "inference_graph_changed": False,
    })
    payload["quadratic_tail_damping"] = history
    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    temporary = output + ".tmp"
    torch.save(payload, temporary)
    os.replace(temporary, output)
    print(
        f"saved {output}: degree={payload['degree']} damped={len(touched)} "
        f"scales={scales}",
        flush=True,
    )


if __name__ == "__main__":
    main()
