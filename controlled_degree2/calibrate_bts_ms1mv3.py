"""Select residual coordinate scales exclusively from sampled MS1MV3 images."""
import argparse
import copy
import json
import random
from pathlib import Path

import torch

from controlled_degree2.model import load_controlled_checkpoint, save_checkpoint
from controlled_degree2.rescale_residual_graph import (
    BOUNDARIES, capture, rescale_graph, sha256_file, update_ranges,
)


def sample_indices(length, count, seed):
    if not 0 < count <= length:
        raise ValueError('Calibration count must be positive and fit the dataset')
    return sorted(random.Random(seed).sample(range(length), count))


def choose_scales(ranges, target):
    if not 0 < target < 1:
        raise ValueError('Target must be in (0,1)')
    scales = []
    for stage in range(1, 5):
        peak = max(max(abs(ranges[name]['min']), abs(ranges[name]['max']))
                   for name in BOUNDARIES if name.startswith(f'layer{stage}.'))
        if not torch.isfinite(torch.tensor(peak)):
            raise ValueError('Nonfinite calibration range')
        scales.append(min(1., target / peak) if peak else 1.)
    return scales


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--dataset-root', default='ms1m-retinaface-t1')
    parser.add_argument('--images', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--target-absmax', type=float, default=0.8)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('batch-size must be positive')
    from dataset import MXFaceDataset
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(args.seed)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    destination = output / 'student_scaled.pt'
    if destination.exists() or (output / 'calibration.json').exists():
        raise FileExistsError('Use a fresh output directory')
    data = MXFaceDataset(args.dataset_root, local_rank=0)
    indices = sample_indices(len(data), args.images, args.seed)
    manifest = [{'dataset_index': i, 'recordio_index': int(data.imgidx[i]), 'orientation': 0} for i in indices]
    (output / 'calibration_images.json').write_text(json.dumps(manifest, indent=2) + '\n')

    def batches():
        for start in range(0, len(indices), args.batch_size):
            yield torch.stack([data.get_oriented(i, 0)[0] for i in indices[start:start + args.batch_size]]).to(args.device)

    model, payload = load_controlled_checkpoint(args.checkpoint, device=args.device)
    model.eval()
    ranges = {}
    with torch.inference_mode():
        for batch in batches():
            embedding, values = capture(model, batch, BOUNDARIES)
            if not torch.isfinite(embedding).all():
                raise FloatingPointError('Nonfinite source MS1MV3 embedding; calibration aborted')
            update_ranges(ranges, values)
    scales = choose_scales(ranges, args.target_absmax)
    print('MS1MV3 stage scales:', scales, flush=True)
    transformed = copy.deepcopy(model)
    mapping = rescale_graph(transformed, scales)
    calibration = {}
    for name, scale in mapping.items():
        module = transformed.get_submodule(name)
        calibration[name] = dict(lam_fit=module.lam_fit.flatten().tolist(),
                                 lam_reg=module.lam_reg.flatten().tolist(), coordinate_scale=scale)
    provenance = dict(source=str(Path(args.checkpoint).resolve()), source_sha256=sha256_file(args.checkpoint),
                      stage_scales=scales, eval_only=True, calibration_dataset='MS1MV3',
                      calibration_images=args.images, seed=args.seed,
                      sampling='random without replacement, sorted for IO, original orientation only',
                      image_manifest_sha256=sha256_file(output / 'calibration_images.json'),
                      recordio_index_sha256=sha256_file(Path(args.dataset_root) / 'train.idx'))
    save_checkpoint(str(destination), transformed, calibration,
                    teacher_weights=payload.get('teacher_weights', 'work_dirs/ms1mv3_r50/model.pt'),
                    extra={'residual_graph_scaling': provenance})
    del transformed
    transformed, _ = load_controlled_checkpoint(str(destination), device=args.device)
    transformed.eval()
    after = {}
    metrics = dict(images=0, embedding_max_abs_error=0., embedding_relative_l2_max=0., embedding_cosine_min=1.)
    with torch.inference_mode():
        for batch in batches():
            reference, before = capture(model, batch, BOUNDARIES)
            actual, values = capture(transformed, batch, BOUNDARIES)
            if not torch.isfinite(actual).all():
                raise FloatingPointError('Nonfinite rescaled calibration embedding')
            torch.testing.assert_close(actual, reference, atol=1e-4, rtol=1e-4)
            for name in BOUNDARIES:
                torch.testing.assert_close(values[name], before[name] * scales[int(name[5]) - 1], atol=1e-4, rtol=1e-4)
            update_ranges(after, values)
            error = actual - reference
            metrics['images'] += len(batch)
            metrics['embedding_max_abs_error'] = max(metrics['embedding_max_abs_error'], error.abs().max().item())
            metrics['embedding_relative_l2_max'] = max(metrics['embedding_relative_l2_max'], (error.norm(dim=1) / reference.norm(dim=1).clamp_min(1e-12)).max().item())
            metrics['embedding_cosine_min'] = min(metrics['embedding_cosine_min'], torch.nn.functional.cosine_similarity(actual.double(), reference.double(), dim=1).min().item())
    if any(r['min'] < -1 or r['max'] > 1 for r in after.values()):
        raise ValueError('Scaled calibration boundary outside [-1,1]')
    report = dict(settings=vars(args), provenance=provenance, before=ranges, after=after,
                  equivalence=metrics, equivalence_passed=True,
                  approximation_target='per-channel PReLU; q_new(s*x)=s*q_old(x)',
                  approximation_interval='per-channel [-s*lam_fit, s*lam_fit]',
                  degree=2, activation_multiplicative_depth=1,
                  checkpoint_sha256=sha256_file(destination))
    (output / 'calibration.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
