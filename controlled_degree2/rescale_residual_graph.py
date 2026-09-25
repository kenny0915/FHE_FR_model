"""Offline, eval-equivalent stage scaling; no new inference operations.

Each stage has one scale because its identity shortcuts tie the block outputs.
Quadratics still approximate per-channel PReLU, on [-s*lam_fit, s*lam_fit].
Degree and multiplicative depth are unchanged. Frozen-BN inference only: this
is not an optimizer-state migration or a guarantee for train-mode BatchNorm.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import torch
from torch import nn

from controlled_degree2.model import DirectQuadratic, load_controlled_checkpoint

BOUNDARIES = ('layer1.2', 'layer2.3', 'layer3.3', 'layer3.7', 'layer3.11', 'layer4.1')


def validate_boundaries(names):
    """Canonical residual-block OUTPUT names, using zero-based iResNet50 indices."""
    valid = {f'layer{s}.{i}' for s, length in enumerate((3, 4, 14, 3), 1)
             for i in range(length)}
    names = tuple(name if name.startswith('layer') else 'layer' + name for name in names)
    if not names or len(set(names)) != len(names) or any(name not in valid for name in names):
        raise ValueError('Provide unique, valid iResNet50 residual block boundaries')
    return names


BTS_LAYOUTS = {
    'bts6': BOUNDARIES,
    'bts14': validate_boundaries('1.0 1.2 2.1 2.3 3.0 3.2 3.4 3.6 3.8 3.10 3.12 3.13 4.0 4.2'.split()),
    'bts9': validate_boundaries('1.1 2.1 2.3 3.1 3.4 3.7 3.10 3.13 4.1'.split()),
    'bts7': validate_boundaries('1.2 2.3 3.2 3.6 3.10 3.13 4.2'.split()),
    'bts5': validate_boundaries('2.0 3.0 3.5 3.10 4.0'.split()),
}


@torch.no_grad()
def rescale_bn(bn, input_scale, output_scale):
    """Frozen BN': BN'(a*x) = b*BN(x), with original running buffers."""
    if not isinstance(bn, nn.BatchNorm2d) or not bn.affine or not bn.track_running_stats:
        raise ValueError('Expected affine BatchNorm2d with running statistics')
    # Compute the affine reparameterization in FP64 before storing it.
    gamma = bn.weight.double()
    offset = gamma * bn.running_mean.double() / (bn.running_var.double() + bn.eps).sqrt()
    bn.bias.copy_(output_scale * bn.bias.double()
                  + output_scale * (1 / input_scale - 1) * offset)
    bn.weight.copy_(gamma * (output_scale / input_scale))


@torch.no_grad()
def rescale_graph(model, stage_scales):
    """Mutate an unfused controlled iResNet; return activation scale mapping."""
    if model.training or any(m.training for m in model.modules()):
        raise ValueError('Call eval() first; equivalence requires frozen BatchNorm')
    if len(stage_scales) != 4 or any(not 0 < s <= 1 for s in stage_scales):
        raise ValueError('Provide four finite scales in (0, 1]')
    polynomial_scales = {}

    def conv(module, a, b):
        module.weight.mul_(b / a)
        if module.bias is not None:
            module.bias.mul_(b)

    def quadratic(module, scale, name):
        if not isinstance(module, DirectQuadratic):
            raise ValueError('Expected DirectQuadratic: ' + name)
        module.coeffs[:, 0].mul_(scale)
        module.coeffs[:, 2].div_(scale)
        module.lam_fit.mul_(scale)
        module.lam_reg.mul_(scale)
        polynomial_scales[name] = scale

    previous = stage_scales[0]
    conv(model.conv1, 1, previous)
    rescale_bn(model.bn1, previous, previous)
    quadratic(model.prelu, previous, 'prelu')
    for index, scale in enumerate(stage_scales, 1):
        for block_index, block in enumerate(getattr(model, f'layer{index}')):
            name = f'layer{index}.{block_index}'
            if block.downsample is None and previous != scale:
                raise ValueError('Cannot change scale across an identity shortcut')
            rescale_bn(block.bn1, previous, previous)
            conv(block.conv1, previous, scale)
            rescale_bn(block.bn2, scale, scale)
            quadratic(block.prelu, scale, name + '.prelu')
            conv(block.conv2, scale, scale)
            rescale_bn(block.bn3, scale, scale)
            if block.downsample is not None:
                conv(block.downsample[0], previous, scale)
                rescale_bn(block.downsample[1], scale, scale)
            previous = scale
    # Restore the original coordinate system before flatten/fc/features.
    rescale_bn(model.bn2, previous, 1)
    return polynomial_scales


def capture(model, batch, names):
    outputs = {}
    handles = [model.get_submodule(name).register_forward_hook(
        lambda m, i, o, name=name: outputs.__setitem__(name, o.detach().clone()))
        for name in names]
    try:
        with torch.inference_mode():
            embedding = model(batch)
    finally:
        for handle in handles:
            handle.remove()
    return embedding, outputs


def update_ranges(ranges, outputs):
    for name, value in outputs.items():
        if not torch.isfinite(value).all():
            raise ValueError('Nonfinite feature map: ' + name)
        row = ranges.setdefault(name, {'min': float('inf'), 'max': -float('inf')})
        row['min'] = min(row['min'], value.min().item())
        row['max'] = max(row['max'], value.max().item())


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    from eval.layer_statistics import iter_ijb_batches
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--ijbc-root', default='ijb/IJBC')
    parser.add_argument('--calibration-images', type=int, default=100)
    parser.add_argument('--validation-images', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--target-absmax', type=float, default=0.8)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if min(args.calibration_images, args.validation_images, args.batch_size, args.threads) < 1:
        parser.error('Counts must be positive')
    if not 0 < args.target_absmax < 1:
        parser.error('target-absmax must be in (0, 1)')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / 'student_scaled.pt'
    if checkpoint_path.exists() or (output_dir / 'report.json').exists():
        raise FileExistsError('Use a fresh output directory')
    torch.set_num_threads(args.threads)
    model, payload = load_controlled_checkpoint(args.checkpoint)
    model.eval()
    calibration = {}
    count = 0
    for batch, _ in iter_ijb_batches(args.ijbc_root, 'IJBC', args.batch_size, args.calibration_images):
        _, maps = capture(model, batch, BOUNDARIES)
        update_ranges(calibration, maps)
        count += len(batch)
    if count != args.calibration_images:
        raise ValueError('Insufficient calibration images')
    scales = []
    for stage in range(1, 5):
        peak = max(max(abs(r['min']), abs(r['max'])) for name, r in calibration.items()
                   if name.startswith(f'layer{stage}.'))
        scales.append(min(1.0, args.target_absmax / peak) if peak else 1.0)
    scaled = copy.deepcopy(model)
    polynomial_scales = rescale_graph(scaled, scales)
    # Preserve only inference metadata: no stale optimizer/calibration history.
    calibration_export = {}
    for name, scale in polynomial_scales.items():
        module = scaled.get_submodule(name)
        calibration_export[name] = {
            'lam_fit': module.lam_fit.flatten().tolist(),
            'lam_reg': module.lam_reg.flatten().tolist(),
            'coordinate_scale': scale,
        }
    provenance = {'source': str(Path(args.checkpoint).resolve()),
                  'source_sha256': sha256_file(args.checkpoint),
                  'stage_scales': scales, 'eval_only': True}
    from controlled_degree2.model import save_checkpoint
    save_checkpoint(str(checkpoint_path), scaled, calibration_export,
                    teacher_weights=payload.get('teacher_weights', 'work_dirs/ms1mv3_r50/model.pt'),
                    extra={'residual_graph_scaling': provenance})
    # Verify the exported/reloaded artifact, not just the in-memory model.
    del scaled
    scaled, _ = load_controlled_checkpoint(str(checkpoint_path))
    scaled.eval()
    names = [f'layer{s}.{i}' for s in range(1, 5)
             for i in range(len(getattr(model, f'layer{s}')))]
    report = {'settings': vars(args), 'provenance': provenance,
              'calibration_before': calibration, 'splits': {}}
    total = args.calibration_images + args.validation_images
    seen = 0
    for batch, _ in iter_ijb_batches(args.ijbc_root, 'IJBC', args.batch_size, total):
        n = len(batch)
        # Split a batch crossing the calibration/held-out boundary exactly.
        cut = max(0, min(n, args.calibration_images - seen))
        for label, part in [('calibration', batch[:cut]), ('held_out', batch[cut:])]:
            if not len(part):
                continue
            record = report['splits'].setdefault(label, {'images': 0, 'before': {}, 'after': {},
                'embedding_max_abs_error': 0, 'embedding_relative_l2_max': 0,
                'embedding_cosine_min': 1, 'block_rescaled_max_abs_error': 0})
            original, before = capture(model, part, names)
            actual, after = capture(scaled, part, names)
            torch.testing.assert_close(actual, original, atol=2e-5, rtol=2e-4)
            for name in names:
                scale = scales[int(name[5]) - 1]
                torch.testing.assert_close(after[name], before[name] * scale, atol=2e-5, rtol=2e-4)
                err = (after[name] / scale - before[name]).abs().max().item()
                record['block_rescaled_max_abs_error'] = max(record['block_rescaled_max_abs_error'], err)
            update_ranges(record['before'], {k: before[k] for k in BOUNDARIES})
            update_ranges(record['after'], {k: after[k] for k in BOUNDARIES})
            error = actual - original
            record['images'] += len(part)
            record['embedding_max_abs_error'] = max(record['embedding_max_abs_error'], error.abs().max().item())
            relative = error.norm(dim=1) / original.norm(dim=1).clamp_min(1e-12)
            record['embedding_relative_l2_max'] = max(record['embedding_relative_l2_max'], relative.max().item())
            cosine = torch.nn.functional.cosine_similarity(actual.double(), original.double(), dim=1)
            record['embedding_cosine_min'] = min(record['embedding_cosine_min'], cosine.min().item())
        seen += n
    if seen != total:
        raise ValueError('Insufficient validation images')
    for record in report['splits'].values():
        record['all_boundaries_in_unit_interval'] = all(
            r['min'] >= -1 and r['max'] <= 1 for r in record['after'].values())
    report['equivalence_passed'] = True
    (output_dir / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
