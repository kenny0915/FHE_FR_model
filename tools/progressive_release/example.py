"""Smoke inference or inference on one already aligned 112x112 RGB face."""
import argparse
import json
from pathlib import Path

import torch
from loader import load_model
from controlled_degree2.rescale_residual_graph import BOUNDARIES, capture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', help='already aligned 112x112 image; otherwise use a zero tensor smoke test')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--output', help='optional .pt file containing embedding and failure flag')
    parser.add_argument('--zero-failed', action='store_true', help='zero entire embedding if any BTS boundary exceeds [-1,1] or output is nonfinite')
    args = parser.parse_args()
    torch.set_num_threads(4)
    model, _ = load_model(args.checkpoint, args.device)
    if args.image:
        from PIL import Image
        with Image.open(args.image) as source:
            image = source.convert('RGB')
            if image.size != (112, 112):
                raise ValueError('Provide a face already aligned to 112x112; resizing is not alignment')
            pixels = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
        inputs = pixels.reshape(112, 112, 3).permute(2, 0, 1).float().unsqueeze(0) / 127.5 - 1
    else:
        inputs = torch.zeros(1, 3, 112, 112)
    embedding, maps = capture(model, inputs.to(args.device), BOUNDARIES)
    ranges = {name: [float(value.min()), float(value.max())] for name, value in maps.items()}
    failed = not bool(torch.isfinite(embedding).all()) or any(
        not bool(torch.isfinite(value).all()) or bool((value.abs() > 1).any()) for value in maps.values())
    if args.zero_failed and failed:
        embedding.zero_()
    # The model returns raw embeddings. Normalize only for a cosine-comparison experiment.
    if args.output:
        torch.save({'embedding': embedding.cpu(), 'failed': failed}, Path(args.output))
    print(json.dumps(dict(input='aligned face' if args.image else 'synthetic zero smoke test',
                         embedding_shape=list(embedding.shape), finite=bool(torch.isfinite(embedding).all()),
                         bts_failed=failed, zeroed=args.zero_failed and failed, boundary_ranges=ranges), indent=2))


if __name__ == '__main__':
    main()
