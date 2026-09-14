"""Teacher-distilled IJB-C numerical calibration for fresh channelwise students.

IJB-C images and optional numerical-failure manifests are calibration data.
No identity or verification-pair labels are read. Temporary auxiliary clamps
are used only for gradient training; the saved inference backbone is unclipped.
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from controlled_degree2.model import (
    load_controlled_checkpoint, load_teacher, quadratic_modules, save_checkpoint,
)
from controlled_degree2.recipe_a import digest
from controlled_degree2.recipe_a_recovery import configure, finite_prefix_batch_loss
from controlled_degree2.recipe_a_recovery_v2 import sitewise
from utils.utils_optimizer import clip_grad_norm_stable


def parameters_for_calibration(model, scope='spatial'):
    if scope not in ('spatial', 'all', 'head'):
        raise ValueError('unknown calibration parameter scope')
    model.requires_grad_(False)
    for module in model.modules():
        spatial = scope != 'head' and isinstance(module, (nn.Conv2d, nn.BatchNorm2d))
        head = scope != 'spatial' and isinstance(module, (nn.Linear, nn.BatchNorm1d))
        if spatial or head:
            for parameter in module.parameters(recurse=False):
                parameter.requires_grad_(True)
    if scope != 'head':
        for module in quadratic_modules(model):
            module.coeffs.requires_grad_(True)
    return [p for p in model.parameters() if p.requires_grad]


def exact_teacher_loss(model, images, target, guard=4.):
    """Distill the actual unclipped graph on numerically safe rows only.

    Probe without autograd, then rerun the selected images. Masking a loss
    after an overflowing forward would still propagate NaN gradients.
    """
    sites = list(quadratic_modules(model))
    if any(module.training for module in model.modules()) or any(q.clip_eval for q in sites):
        raise ValueError('exact KD requires evaluation mode and unclipped activations')
    safe = torch.ones(len(images), dtype=torch.bool, device=images.device)
    handles = []

    def observe(module, inputs):
        relative = inputs[0].detach().abs()/module.lam_fit.reshape(1, -1, 1, 1)
        safe.logical_and_(relative.flatten(1).amax(1) <= guard)

    try:
        for module in sites:
            handles.append(module.register_forward_pre_hook(observe))
        with torch.no_grad():
            probe = model(images)
            norms = probe.norm(dim=1)
            safe.logical_and_(torch.isfinite(probe).all(1) & torch.isfinite(norms) & (norms > 0))
    finally:
        for handle in handles:
            handle.remove()
    rows = int(safe.sum())
    if not rows:
        return next(quadratic_modules(model)).coeffs.sum()*0, 0
    output = model(images[safe])
    norms = output.norm(dim=1)
    if not torch.isfinite(output).all() or not torch.isfinite(norms).all() or (norms <= 0).any():
        raise FloatingPointError('non-finite exact-KD selected-row rerun')
    loss = (1-F.cosine_similarity(output, target[safe], dim=1)).mean()
    return loss*rows/len(images), rows


def calibration_loss(model, teacher, images, guard=.9, range_weight=1.,
                     exact_kd_weight=0., exact_kd_guard=4.):
    """Bounded auxiliary KD plus exact-unclipped-prefix repair gradients."""
    sites = list(quadratic_modules(model))
    penalties, student_hints, teacher_hints, handles = [], {}, {}, []

    def observe(module, inputs):
        relative = inputs[0].double().abs()/module.lam_fit.double().reshape(1, -1, 1, 1)
        excess = (relative-guard).clamp_min(0).square().flatten(1)
        penalties.append(excess.amax(1).mean()+.01*excess.mean())

    try:
        configure(model, 0)
        for module in sites:
            handles.append(module.register_forward_pre_hook(observe))
        for name in ('layer1', 'layer2', 'layer3', 'layer4'):
            if not hasattr(model, name):
                continue
            handles.append(model.get_submodule(name).register_forward_hook(
                lambda m, i, o, n=name: student_hints.__setitem__(n, o)))
            handles.append(teacher.get_submodule(name).register_forward_hook(
                lambda m, i, o, n=name: teacher_hints.__setitem__(n, o)))
        with torch.no_grad():
            target = teacher(images)
        output = model(images)
        if not torch.isfinite(target).all() or not torch.isfinite(output).all():
            raise FloatingPointError('non-finite teacher or auxiliary embedding')
        kd = (1-F.cosine_similarity(output, target, dim=1)).mean()
        hints = [((s-teacher_hints[n]).square().flatten(1).mean(1)
                  /teacher_hints[n].square().flatten(1).mean(1).clamp_min(1e-6)).mean()
                 for n, s in student_hints.items()]
        auxiliary = kd + (.3*torch.stack(hints).mean() if hints else 0)
        auxiliary = auxiliary + range_weight*torch.stack(penalties).mean()
    finally:
        for handle in handles:
            handle.remove()
        configure(model, len(sites))
    prefix, _, _ = finite_prefix_batch_loss(model, images, len(sites), guard=1.,
                                           target=guard, detach_bn=False)
    exact, exact_rows = (exact_teacher_loss(model, images, target, exact_kd_guard)
                         if exact_kd_weight else (images.new_zeros(()), 0))
    loss = auxiliary + range_weight*prefix + exact_kd_weight*exact
    return loss, dict(kd=float(kd.detach()), prefix=float(prefix.detach()), loss=float(loss.detach()),
                      exact_kd=float(exact.detach()), exact_rows=exact_rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--teacher', default='work_dirs/ms1mv3_r50/model.pt')
    parser.add_argument('--ijb-root', default='ijb/IJBC')
    parser.add_argument('--output', required=True)
    parser.add_argument('--manifest', action='append', default=[])
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--range-weight', type=float, default=1.)
    parser.add_argument('--exact-kd-weight', type=float, default=0.)
    parser.add_argument('--exact-kd-guard', type=float, default=4.)
    parser.add_argument('--parameter-scope', choices=('spatial', 'all', 'head'), default='spatial',
                        help='train spatial layers/quadratics, all backbone parameters, or final linear/BN head')
    parser.add_argument('--seed', type=int, default=20260914)
    parser.add_argument('--deadline', type=float, required=True)
    args = parser.parse_args()
    if args.steps <= 0 or args.batch_size <= 0 or args.lr <= 0 or args.range_weight < 0:
        parser.error('positive sizes/LR and nonnegative range weight required')
    if args.exact_kd_weight < 0 or args.exact_kd_guard <= 0:
        parser.error('nonnegative exact KD weight and positive guard required')
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        parser.error('this calibration adapter runs on one GPU')
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    device = torch.device('cuda')
    model, source = load_controlled_checkpoint(args.checkpoint, device)
    provenance = source.get('provenance', {})
    if source.get('pure_quadratic') is not True or provenance.get('teacher_sha256') != digest(args.teacher):
        raise ValueError('need completed teacher-derived channelwise source with matching provenance')
    teacher = load_teacher(args.teacher, device).eval().requires_grad_(False)
    sitewise(model)
    if len(model._recovery_sites) != 25:
        raise ValueError('expected all 25 activation sites')
    parameters = parameters_for_calibration(model, args.parameter_scope)
    optimizer = torch.optim.SGD(parameters, lr=args.lr)
    fixed_buffers = {n: t.clone() for n, t in model.named_buffers()}
    from utils.utils_ijbc_replay import (
        IJBCSourceDataset, IJBCOrientationDataset, load_ijbc_replay_orientations,
    )
    count = len(IJBCSourceDataset(args.ijb_root))
    rng = np.random.default_rng(args.seed)
    replay = load_ijbc_replay_orientations(args.manifest) if args.manifest else ()
    if any(i >= count for i, _ in replay):
        raise ValueError('replay index outside IJB-C metadata')
    records = []
    for _ in range(args.steps):
        batch = [(int(i), int(o)) for i, o in zip(rng.integers(count, size=args.batch_size),
                                                  rng.integers(2, size=args.batch_size))]
        if replay:
            for k in range(args.batch_size//2):
                batch[k] = replay[int(rng.integers(len(replay)))]
        records.extend(batch)
    dataset = IJBCOrientationDataset(args.ijb_root, records)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers,
                        multiprocessing_context='spawn' if args.workers else None)
    record = dict(config=vars(args), source_sha256=digest(args.checkpoint),
                  teacher_sha256=digest(args.teacher), calibration_dataset='IJBC',
                  uses_ijbc_pair_labels=False, inference_clipping=False)
    (root/'calibration_config.json').write_text(json.dumps(record, indent=2))
    for step, (images, _, _) in enumerate(loader, 1):
        if time.time() >= args.deadline:
            raise TimeoutError('calibration deadline reached')
        optimizer.zero_grad(set_to_none=True)
        loss, metrics = calibration_loss(model, teacher, images.to(device), range_weight=args.range_weight,
                                         exact_kd_weight=args.exact_kd_weight,
                                         exact_kd_guard=args.exact_kd_guard)
        if not torch.isfinite(loss):
            raise FloatingPointError('non-finite calibration loss')
        loss.backward()
        clip_grad_norm_stable(parameters, 1., error_if_nonfinite=True)
        optimizer.step()
        if any(not torch.isfinite(p).all() for p in parameters):
            raise FloatingPointError('non-finite updated parameter')
        if step % 25 == 0 or step == args.steps:
            if any(not torch.equal(t, fixed_buffers[n]) for n, t in model.named_buffers()):
                raise RuntimeError('calibration changed fixed BN/range buffers')
            configure(model, 25)
            save_checkpoint(str(root/'last.tmp.pt'), model, source['poly_calib'],
                            teacher_weights=args.teacher,
                            extra=dict(origin='channelwise_ijbc_calibration', pure_quadratic=True,
                                       provenance=provenance, calibration=record, step=step,
                                       diagnostic_only=True, optimizer=optimizer.state_dict()))
            os.replace(root/'last.tmp.pt', root/'last.pt')
            with (root/'metrics.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(step=step, **metrics))+'\n')
            print(dict(step=step, **metrics), flush=True)
    print('CALIBRATION_COMPLETE: candidate requires full exported IJB-C audit', flush=True)


if __name__ == '__main__':
    main()
