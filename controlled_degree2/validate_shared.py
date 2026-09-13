"""Final MS1MV3 holdout report for a fixed diagnostic shared checkpoint.

This does not select or modify the checkpoint. It reuses recipe A validation
when training ended before the usual post-curriculum validation stage.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from controlled_degree2.recipe_a import digest, validate
from controlled_degree2.shared import build_shared_iresnet50


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    checkpoint, output = Path(args.checkpoint), Path(args.output)
    checksum = digest(checkpoint)
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if payload.get('network') != 'r50_shared_d2' or payload.get('inference_input_bounds') is not False:
        raise ValueError('final diagnostic validation requires an explicitly unclipped shared checkpoint')
    split_path = checkpoint.parent/'split.npz'
    if digest(split_path) != payload['provenance']['split_sha256']:
        raise ValueError('development split checksum differs from training provenance')
    split = np.load(split_path)
    config = dict(payload['config'])
    config.update(output=str(output), shared=True, inference_bound=False, batch_size=128, workers=4)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda', 0)
    model = build_shared_iresnet50(dropout=0, fp16=False).to(device).eval()
    model.load_state_dict(payload['state_dict_backbone'], strict=True)
    result = validate(model, SimpleNamespace(**config), 0, 1, device, split)
    if digest(checkpoint) != checksum:
        raise ValueError('checkpoint changed during fixed validation')
    result.update(checkpoint_sha256=checksum, epoch=payload['epoch'],
                  source_images=len(split['dev']), forwards=5*len(split['dev']),
                  selection='report only; checkpoint already fixed before final evaluations')
    (output/'result.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
