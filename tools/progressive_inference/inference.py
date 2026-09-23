"""Standalone inference for the scaled, unfused degree-2 iResNet50 checkpoint."""
import argparse
from pathlib import Path

import torch
from torch import nn
from backbone import iresnet50


class Quadratic(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.coeffs = nn.Parameter(torch.zeros(channels, 3), requires_grad=False)
        self.register_buffer('lam_fit', torch.ones(channels))
        self.register_buffer('lam_reg', torch.ones(channels))
        self.register_buffer('slope', torch.zeros(channels))

    def forward(self, inputs):
        work = inputs.float()
        shape = (1, self.coeffs.shape[0]) + (1,) * (inputs.ndim - 2)
        c0, c1, c2 = self.coeffs.t().reshape((3,) + shape)
        return (c0 + c1 * work + c2 * (work * work)).to(inputs.dtype)


def load_model(checkpoint=None, device='cpu'):
    checkpoint = checkpoint or Path(__file__).resolve().with_name('model.pt')
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if payload.get('format') != 'fhe-fr/controlled-direct-degree2-v1' or payload.get('degree') != 2:
        raise ValueError('Expected the supplied unfused degree-2 checkpoint')
    model = iresnet50(dropout=0, fp16=False)
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.PReLU):
            parent_name, _, leaf = name.rpartition('.')
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, leaf, Quadratic(module.num_parameters))
    model.load_state_dict(payload['state_dict_backbone'], strict=True)
    return model.to(device).eval()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--image', help='already aligned 112x112 face; omission runs a zero-input smoke test')
    parser.add_argument('--output', default='embedding.txt', help='512 raw values, one per line')
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.image:
        from PIL import Image
        with Image.open(args.image) as source:
            image = source.convert('RGB')
            if image.size != (112, 112):
                raise ValueError('Input must already be face-aligned to 112x112')
            pixels = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
        inputs = pixels.reshape(112, 112, 3).permute(2, 0, 1).float().unsqueeze(0) / 127.5 - 1
    else:
        inputs = torch.zeros(1, 3, 112, 112)
    model = load_model(args.checkpoint, args.device)
    with torch.inference_mode():
        embedding = model(inputs.to(args.device))[0].cpu()
    if not torch.isfinite(embedding).all():
        raise FloatingPointError('Nonfinite embedding; not writing an output file')
    Path(args.output).write_text(''.join(format(value, '.17g') + '\n' for value in embedding.tolist()))
    print(f'Saved {embedding.numel()} raw float32 values to {args.output}')


if __name__ == '__main__':
    main()
