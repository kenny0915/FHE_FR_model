"""Contract only source-audited BN channels in a controlled checkpoint."""

from __future__ import annotations

import argparse
import json
import os

import torch

from controlled_degree2.model import (
    load_controlled_checkpoint,
    preactivation_batchnorm_name,
)


def select_tail_channels(entry, topk, min_ratio):
    eligible = [
        row for row in entry["channels"]
        if float(row["max_ratio"]) >= float(min_ratio)
    ]
    return tuple(int(row["channel"]) for row in eligible[:int(topk)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--scope", action="append", required=True)
    parser.add_argument("--topk", type=int, required=True)
    parser.add_argument("--min-ratio", type=float, default=1.0)
    parser.add_argument("--factor", type=float, required=True)
    parser.add_argument(
        "--selection-data",
        default="non-IJB source-tail channel audit only",
    )
    args = parser.parse_args()
    if args.topk <= 0 or args.min_ratio <= 0:
        raise ValueError("topk and minimum ratio must be positive")
    if not 0.0 < args.factor <= 1.0:
        raise ValueError("factor must be in (0, 1]")

    with open(args.audit, encoding="utf-8") as handle:
        audit = json.load(handle)
    model, payload = load_controlled_checkpoint(args.checkpoint, device="cpu")
    touched = {}
    with torch.no_grad():
        for scope in args.scope:
            if scope not in audit["activations"]:
                raise ValueError(f"audit has no activation {scope!r}")
            channels = select_tail_channels(
                audit["activations"][scope], args.topk, args.min_ratio
            )
            if not channels:
                raise ValueError(f"no audited tail channels selected for {scope}")
            batchnorm_name = preactivation_batchnorm_name(scope)
            batchnorm = model.get_submodule(batchnorm_name)
            index = torch.tensor(channels, dtype=torch.long)
            batchnorm.weight.index_copy_(
                0, index, batchnorm.weight.index_select(0, index) * args.factor
            )
            batchnorm.bias.index_copy_(
                0, index, batchnorm.bias.index_select(0, index) * args.factor
            )
            touched[scope] = {
                "batchnorm": batchnorm_name,
                "channels": list(channels),
                "factor": float(args.factor),
            }

    payload["state_dict_backbone"] = {
        name: value.detach().cpu() for name, value in model.state_dict().items()
    }
    history = list(payload.get("preactivation_channel_contractions", ()))
    history.append({
        "source_checkpoint": os.path.abspath(args.checkpoint),
        "audit": os.path.abspath(args.audit),
        "topk": int(args.topk),
        "min_ratio": float(args.min_ratio),
        "factor": float(args.factor),
        "touched": touched,
        "selection_data": args.selection_data,
        "inference_graph_changed": False,
        "fhe_cost": "none; factors folded into existing BatchNorm affine",
    })
    payload["preactivation_channel_contractions"] = history
    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    temporary = output + ".tmp"
    torch.save(payload, temporary)
    os.replace(temporary, output)
    print(
        f"saved {output}: degree={payload['degree']} scopes={len(touched)} "
        f"channels={sum(len(row['channels']) for row in touched.values())}",
        flush=True,
    )


if __name__ == "__main__":
    main()
