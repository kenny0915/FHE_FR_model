"""Export a degree-2 checkpoint with selected BN preactivations contracted."""

from __future__ import annotations

import argparse
import os

import torch

from controlled_degree2.model import (
    contract_preactivation_affines,
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
        help="activation prefix factors, e.g. 'layer1.1.prelu:0.9'",
    )
    parser.add_argument(
        "--selection-data",
        default="non-IJB deployment-tail audit only",
        help="checkpoint provenance for the source-only contraction choice",
    )
    args = parser.parse_args()

    scales = parse_lam_scale(args.scales)
    if not scales:
        raise ValueError("at least one preactivation contraction is required")
    model, payload = load_controlled_checkpoint(args.checkpoint, device="cpu")
    touched = contract_preactivation_affines(model, scales)
    if not touched:
        raise ValueError(f"contraction matched no activations: {scales}")

    payload["state_dict_backbone"] = {
        name: value.detach().cpu() for name, value in model.state_dict().items()
    }
    history = list(payload.get("preactivation_contractions", ()))
    history.append({
        "source_checkpoint": os.path.abspath(args.checkpoint),
        "requested_scales": scales,
        "touched": touched,
        "selection_data": args.selection_data,
        "inference_graph_changed": False,
        "fhe_cost": "none; factors folded into existing BatchNorm affine",
    })
    payload["preactivation_contractions"] = history
    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    temporary = output + ".tmp"
    torch.save(payload, temporary)
    os.replace(temporary, output)
    print(
        f"saved {output}: degree={payload['degree']} contracted={len(touched)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
