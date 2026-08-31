"""Materialize a selected PReLU site as the fixed affine ``a_c*x``."""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backbones import get_model
from eval.compare_activation_distributions import extract_state_dict
from eval.screen_linear_replacement import MODEL_KWARGS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--activation-index", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    model_kwargs = dict(MODEL_KWARGS)
    model_kwargs["prelu_herpn_linear_indices"] = (args.activation_index,)
    model_kwargs["prelu_herpn_linear_trainable"] = False
    model = get_model("r50_prelu_herpn", **model_kwargs)
    raw = torch.load(args.checkpoint, map_location="cpu")
    model.load_backbone_init_state_dict(extract_state_dict(raw))
    activations = model.named_progressive_activations()
    if not 0 <= args.activation_index < len(activations):
        raise ValueError(
            f"activation-index must be in [0, {len(activations) - 1}]")
    name, activation = activations[args.activation_index]
    with torch.no_grad():
        activation.weight.copy_(activation.prelu.weight.reshape(-1, 1, 1))
        activation.bias.zero_()
        activation.set_blend(1.0)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {key: value.detach().cpu() for key, value in model.state_dict().items()},
        output,
    )
    print(
        f"saved {output} with activation {args.activation_index}={name}, "
        f"slope=[{float(activation.weight.min()):.7g}, "
        f"{float(activation.weight.max()):.7g}]",
        flush=True,
    )


if __name__ == "__main__":
    main()
