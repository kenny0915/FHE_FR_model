"""Train-source-only finite-prefix recovery for the recipe-A epoch-8 checkpoint.

The original channelwise PReLU fits on [-lam_fit, lam_fit] stay unchanged.
Only existing pre-polynomial BN affines are repaired; inference adds no ops.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import itertools
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from controlled_degree2.model import (
    build_controlled_iresnet50, load_teacher, preactivation_batchnorm_name,
    quadratic_modules, save_checkpoint,
)
from controlled_degree2.recipe_a import (
    IdentityHead, barrier, context, digest, loader, reduce,
    require_finite, train as resume_accuracy,
)
from controlled_degree2.augment import prepare_range_batch
from utils.utils_optimizer import clip_grad_norm_stable


def group_number(name):
    return 0 if name == 'prelu' else int(name[5])


def configure(model, open_groups):
    model.eval()  # fixed running statistics; autograd remains enabled
    for module in quadratic_modules(model):
        module.alpha = 1.
        module.clip = False
        module.clip_eval = group_number(module.name) >= open_groups


def affine_parameters(model):
    model.requires_grad_(False)
    parameters = []
    for module in quadratic_modules(model):
        bn = model.get_submodule(preactivation_batchnorm_name(module.name))
        if not isinstance(bn, nn.modules.batchnorm._BatchNorm):
            raise TypeError('recovery requires the unfused recipe-A BN checkpoint')
        bn.weight.requires_grad_(True)
        bn.bias.requires_grad_(True)
        parameters.extend((bn.weight, bn.bias))
    return parameters


class PrefixEscape(Exception):
    def __init__(self, name, penalty, ratios):
        self.name, self.penalty, self.ratios = name, penalty, ratios


def finite_prefix_loss(model, images, open_groups, guard=1., target=.9):
    """Stop *before* evaluating the first escaping quadratic.

    BN inputs are detached in this probe only, so the penalty can update the
    immediately preceding affine without backward through the polynomial
    recurrence. Completed safe rows of this batch are retried in later draws.
    All hooks/flags are restored, including on exceptions.
    """
    if not 0 < target < guard:
        raise ValueError('require 0 < target < guard')
    flags = [(m, m.training) for m in model.modules()]
    qflags = [(m, m.alpha, m.clip, m.clip_eval) for m in quadratic_modules(model)]
    handles = []

    def check(module, inputs):
        raw = inputs[0].float()
        if not bool(torch.isfinite(raw).all()):
            raise FloatingPointError(f'non-finite input before finite prefix guard: {module.name}')
        relative = raw / module.lam_fit.reshape(1, -1, 1, 1)
        ratios = relative.detach().abs().flatten(1).amax(1)
        if bool((ratios > guard).any()):
            # FP64 avoids overflow of the loss even for a large but finite
            # first escape. No NaN replacement or downstream graph is used.
            excess = (relative.double().abs()-target).clamp_min(0)
            penalty = excess.flatten(1).amax(1).square().mean()
            raise PrefixEscape(module.name, penalty, ratios)

    try:
        configure(model, open_groups)
        for module in model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                handles.append(module.register_forward_pre_hook(
                    lambda m, inputs: (inputs[0].detach(),)))
        for module in quadratic_modules(model):
            if group_number(module.name) < open_groups:
                handles.append(module.register_forward_pre_hook(check))
        try:
            result = model(images)
            if not bool(torch.isfinite(result).all()):
                raise FloatingPointError('non-finite suffix despite finite guarded prefix')
            return images.new_zeros(()), None, images.new_zeros(len(images))
        except PrefixEscape as escape:
            return escape.penalty, escape.name, escape.ratios
    finally:
        for handle in handles:
            handle.remove()
        for module, training in flags:
            module.training = training
        for module, alpha, clip, clip_eval in qflags:
            module.alpha, module.clip, module.clip_eval = alpha, clip, clip_eval


def stage_passes(report):
    # Approximation interval escapes are a repair objective, not a numerical
    # failure. The actual unclipped prefix must produce finite, nonzero norms.
    return report['rows'] > 0 and report['nonfinite'] == 0


def finite_prefix_batch_loss(model, images, open_groups, guard=1., target=.9):
    """Retry safe rows so one escaping row cannot starve deeper affines."""
    total = images.new_zeros((), dtype=torch.float64)
    remaining = images
    first_site, first_ratios = None, images.new_zeros(len(images))
    while len(remaining):
        loss, site, ratios = finite_prefix_loss(model, remaining, open_groups, guard, target)
        total = total + loss * (len(remaining) / len(images))
        if first_site is None:
            first_site, first_ratios = site, ratios
        if site is None:
            break
        # Every unsuccessful pass removes at least one row. Hooks are restored
        # by finite_prefix_loss before retrying the smaller original-image batch.
        remaining = remaining[ratios <= guard]
    return total, first_site, first_ratios


def validate_resume(state, payload, source_record):
    recovery = state.get('recovery', {})
    if (state.get('provenance') != payload['provenance'] or
            recovery.get('source', {}).get('source_sha256') != source_record['source_sha256']):
        raise ValueError('recovery resume provenance/source mismatch')
    if not 1 <= recovery.get('group', 0) <= 5 or recovery.get('step', -1) < 0:
        raise ValueError('invalid recovery resume position')
    # Recovery is allowed to alter only the pre-polynomial BN affine tensors.
    allowed = set()
    for name in payload['state_dict_backbone']:
        if name.endswith('.lam_fit'):
            bn = preactivation_batchnorm_name(name[:-len('.lam_fit')])
            allowed.update((bn+'.weight', bn+'.bias'))
    current = state['state_dict_backbone']
    if current.keys() != payload['state_dict_backbone'].keys():
        raise ValueError('recovery state keys changed')
    for name, value in current.items():
        if not torch.isfinite(value).all():
            raise ValueError(f'non-finite recovery tensor: {name}')
        if name not in allowed and not torch.equal(value, payload['state_dict_backbone'][name]):
            raise ValueError(f'recovery modified frozen tensor: {name}')
    return recovery


def failure_training_rows(root, label, world, train_rows):
    rows = np.unique([item['source_index'] for rank in range(world)
                      for item in json.loads((Path(root)/f'{label}.rank{rank}.json').read_text())['failures']])
    if not np.isin(rows, train_rows).all():
        raise ValueError('gate replay contains a non-training row')
    return rows.astype(np.int64)


def check_split(split, active_ids):
    train_ids = np.unique(split['labels'][split['train']])
    dev_ids = np.unique(split['labels'][split['dev']])
    if np.intersect1d(train_ids, dev_ids).size or set(train_ids) != set(active_ids):
        raise ValueError('training split identity isolation/active head IDs do not match')


def snapshot_source(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError('recovery output must be separate from the source directory')
    output.mkdir(parents=True, exist_ok=False)
    original = source/'last.pt'
    checksum = digest(original)
    shutil.copy2(original, output/'source_epoch8.pt')
    if digest(output/'source_epoch8.pt') != checksum or digest(original) != checksum:
        raise RuntimeError('source changed while taking recovery snapshot')
    (output/'source_epoch8.pt').chmod(0o440)
    for name in ('prepared.pt', 'split.npz', 'provenance.json'):
        shutil.copy2(source/name, output/name)
    record = dict(source=str(original), source_sha256=checksum,
                  snapshot='source_epoch8.pt', code_commit=os.environ.get('RECIPE_COMMIT', 'local-test'))
    (output/'recovery_source.json').write_text(json.dumps(record, indent=2))
    return record


def variants(images, stress):
    yield 'clean', images
    yield 'flip', images.flip(-1)
    if stress:
        yield 'lowres', torch.nn.functional.interpolate(
            torch.nn.functional.interpolate(images, size=(20, 20), mode='bilinear', align_corners=False),
            size=(112, 112), mode='bilinear', align_corners=False)
        shifted = torch.full_like(images, -1)
        shifted[:, :, 4:, 4:] = images[:, :, :-4, :-4]
        yield 'shift', shifted
        yield 'dark', (images+1)*.3-1


@torch.no_grad()
def gate(model, args, rows, labels, rank, world, device, groups, stress, label):
    """Exact finite check, no prefix abort and no output sanitization."""
    configure(model, groups)
    bad, escapes, count = 0, 0, 0
    peak = torch.zeros((), device=device)
    row_ratios = None
    failures = []

    def observe(module, inputs):
        nonlocal row_ratios
        ratio = (inputs[0].float().abs()/module.lam_fit.reshape(1, -1, 1, 1)).flatten(1).amax(1)
        # Nonfinite diagnostics stay nonfinite; never enter a loss or model.
        ratio = torch.where(torch.isfinite(ratio), ratio, torch.full_like(ratio, float('inf')))
        row_ratios = torch.maximum(row_ratios, ratio)

    def observe_output(module, inputs, output):
        nonlocal row_ratios
        finite = torch.isfinite(output).flatten(1).all(1)
        row_ratios = torch.where(finite, row_ratios, torch.full_like(row_ratios, float('inf')))

    handles = [m.register_forward_pre_hook(observe) for m in quadratic_modules(model)
               if group_number(m.name) < groups]
    handles += [m.register_forward_hook(observe_output) for m in quadratic_modules(model)
                if group_number(m.name) < groups]
    try:
        for images, _, indices in loader(args, rows[rank::world], labels, True):
            images = images.to(device)
            for variant, x in variants(images, stress):
                row_ratios = torch.zeros(len(x), device=device)
                z = model(x).float()
                norms = z.norm(dim=1)
                finite = (torch.isfinite(z).all(1) & torch.isfinite(norms) & (norms > 0)
                          & torch.isfinite(row_ratios))
                bad += int((~finite).sum())
                escapes += int((row_ratios > args.guard).sum())
                peak = torch.maximum(peak, row_ratios.max())
                count += len(x)
                for index in indices[(~finite).cpu()].tolist():
                    if len(failures) < 4096:
                        failures.append(dict(source_index=index, variant=variant))
    finally:
        for handle in handles:
            handle.remove()
    totals = reduce(torch.tensor([count, bad, escapes], device=device, dtype=torch.long)).tolist()
    reduce(peak, torch.distributed.ReduceOp.MAX)
    report = dict(rows=totals[0], nonfinite=totals[1], escaping_rows=totals[2], max_ratio=float(peak),
                  open_groups=groups, train_only=True, stress=stress)
    path = Path(args.output)
    (path/f'{label}.rank{rank}.json').write_text(json.dumps(dict(failures=failures, truncated=len(failures)==4096)))
    if rank == 0:
        (path/f'{label}.json').write_text(json.dumps(report, indent=2))
        print(f'GATE {label}: {report}', flush=True)
    return report


def averaged_gradients(parameters, world, device):
    flat = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1) for p in parameters])
    require_finite(flat, device, 'recovery local gradients')
    reduce(flat)
    flat /= world
    require_finite(flat, device, 'recovery reduced gradients')
    offset = 0
    for p in parameters:
        p.grad = flat[offset:offset+p.numel()].view_as(p).clone()
        offset += p.numel()


def run(args, rank, world, device):
    root = Path(args.output)
    if rank == 0:
        if not args.reuse_output:
            snapshot_source(args.source, root)
        (root/'recovery_config.json').write_text(json.dumps(vars(args), indent=2))
    barrier()
    payload = torch.load(root/'source_epoch8.pt', map_location='cpu', weights_only=False)
    prepared = torch.load(root/'prepared.pt', map_location='cpu', weights_only=False)
    if payload.get('epoch') != 8 or payload.get('origin') != 'teacher_only_recipe_a':
        raise ValueError('expected original recipe-A epoch-8 checkpoint')
    if payload['provenance'] != prepared['metadata']:
        raise ValueError('checkpoint/preparation provenance mismatch')
    if digest(root/'split.npz') != prepared['metadata']['split_sha256']:
        raise ValueError('split checksum mismatch')
    if prepared['metadata']['config'].get('smoke'):
        raise ValueError('recovery must start from the real epoch-8 checkpoint, including smoke')
    original_config = payload['config']
    continuation_config = dict(original_config)
    continuation_config['lr'] *= args.continuation_lr_factor
    continuation_config['head_lr'] *= args.continuation_lr_factor
    args.dataset_root = original_config['dataset_root']
    if digest(original_config['teacher']) != prepared['metadata']['teacher_sha256']:
        raise ValueError('teacher checksum mismatch')
    split = np.load(root/'split.npz')
    check_split(split, prepared['active_ids'])
    torch.manual_seed(args.seed)
    model = build_controlled_iresnet50(dropout=0, fp16=False).to(device)
    model.load_state_dict(payload['state_dict_backbone'], strict=True)
    require_finite(torch.cat([v.flatten() for v in model.state_dict().values()]), device, 'source checkpoint')
    parameters = affine_parameters(model)
    teacher = load_teacher(original_config['teacher'], device).eval().requires_grad_(False)
    head = IdentityHead(payload['head']['weight']).to(device).eval().requires_grad_(False)
    mapping = torch.full((len(prepared['centers']),), -1, dtype=torch.long, device=device)
    mapping[torch.tensor(prepared['active_ids'], device=device)] = torch.arange(len(prepared['active_ids']), device=device)
    optimizer = torch.optim.SGD(parameters, lr=args.lr, momentum=.9)  # no decay of BN controllers
    rng = np.random.default_rng(args.seed)
    gate_rows = rng.permutation(split['train'])[:args.gate_images]
    sampler = torch.utils.data.DistributedSampler(split['train'], num_replicas=world, rank=rank,
                                                  shuffle=True, seed=args.seed, drop_last=True)
    batches = loader(args, split['train'], split['labels'], sampler=sampler)
    source_record = json.loads((root/'recovery_source.json').read_text())
    start_group, start_step = 1, 0
    ready_resume = False
    resume_path = root/'recovery_last.pt' if args.reuse_output else None
    if resume_path is None or not resume_path.exists():
        resume_path = Path(args.resume) if args.resume else None
    if resume_path is not None:
        resumed = torch.load(resume_path, map_location='cpu', weights_only=False)
        position = validate_resume(resumed, payload, source_record)
        model.load_state_dict(resumed['state_dict_backbone'], strict=True)
        optimizer.load_state_dict(resumed['recovery_optimizer'])
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr
        start_group, start_step = position['group'], position['step']
        ready_resume = position['ready']
        if rank == 0:
            print(f'RECOVERY_RESUME path={resume_path} group={start_group} step={start_step} ready={ready_resume} optimizer_states={len(optimizer.state)}', flush=True)
        del resumed

    reports = {}

    def save(name, group, step, report=None, ready=False):
        if rank != 0:
            return
        if report is not None:
            reports[group] = dict(report, checked_at_step=step)
        extra = dict(epoch=8, best=-1., config=continuation_config, provenance=prepared['metadata'],
                     head=payload['head'], origin='teacher_only_recipe_a',
                     recovery=dict(group=group, step=step, gate=reports.get(group), ready=ready,
                                   source=source_record, config=vars(args)),
                     recovery_optimizer=optimizer.state_dict(), pure_quadratic=ready)
        # Clear old conversion momentum for the optional accuracy continuation.
        state = copy.deepcopy(payload['optimizer'])
        state['state'] = {}
        extra['optimizer'] = state
        save_checkpoint(str(root/(name+'.tmp')), model, prepared['calibration'],
                        teacher_weights=original_config['teacher'], extra=extra)
        os.replace(root/(name+'.tmp'), root/name)

    initial = [p.detach().clone() for p in parameters] if args.smoke else None
    # Diagnose the actual failed full-unclipped path even in the smoke test.
    gate(model, args, gate_rows[:32] if args.smoke else gate_rows, split['labels'], rank, world,
         device, 5, True, 'initial_unclipped')
    for group in range(start_group, 6) if not ready_resume else ():
        offset = start_step if group == start_group else 0
        sampler.set_epoch(group*100000+offset)
        iterator = iter(batches)
        cache = None
        failure_batches = failure_iterator = None
        passed = False
        for step in itertools.count(offset+1):
            try:
                images, labels, indices = next(iterator)
            except StopIteration:
                sampler.set_epoch(group*100000+step)
                iterator = iter(batches)
                images, labels, indices = next(iterator)
            if failure_batches is not None and step % 4 == 0:
                try:
                    images, labels, indices = next(failure_iterator)
                except StopIteration:
                    failure_iterator = iter(failure_batches)
                    images, labels, indices = next(failure_iterator)
            images, labels = images.to(device), mapping[labels.to(device)]
            if bool((labels < 0).any()):
                raise ValueError('non-training identity reached recovery')
            images, mask = prepare_range_batch(images, pathological_fraction=.02,
                                               crop_probability=.1, lowres_probability=.2,
                                               photo_probability=.2, stress_probability=.1)
            if cache is not None:
                n = min(len(cache[0]), max(1, len(images)//16))
                images[:n], labels[:n], mask[:n] = cache[0][:n], cache[1][:n], cache[2][:n]
            optimizer.zero_grad(set_to_none=True)
            penalty, site, ratios = finite_prefix_batch_loss(model, images, group, args.guard, args.target)
            require_finite(penalty, device, 'finite-prefix penalty')
            if penalty.requires_grad:
                (args.range_weight*penalty).backward()
            # Conservative identity anchor; this temporary clipped path is
            # never claimed to validate or replace the unclipped candidate.
            configure(model, 0)
            with torch.no_grad():
                target = teacher(images)
            z = model(images)
            require_finite(z, device, 'clipped identity anchor')
            weights = mask.float()
            kd = ((1-torch.nn.functional.cosine_similarity(z, target, dim=1))*weights).sum()/weights.sum().clamp_min(1)
            anchor = kd + .1*head(z, labels, mask)
            require_finite(anchor, device, 'identity anchor loss')
            anchor.backward()
            averaged_gradients(parameters, world, device)
            clip_grad_norm_stable(parameters, 1., error_if_nonfinite=True)
            optimizer.step()
            if site is not None:
                chosen = ratios.topk(min(32, len(ratios))).indices
                cache = (images[chosen].detach().clone(), labels[chosen].clone(), mask[chosen].clone())
            if rank == 0 and (step == offset+1 or step % 25 == 0):
                print(f'RECOVERY group={group} step={step} first_escape={site} range={float(penalty):.6g} anchor={float(anchor):.6g}', flush=True)
            if args.smoke and step == offset+2:
                delta = max(float((p-p0).abs().max()) for p, p0 in zip(parameters, initial))
                if not delta > 0:
                    raise RuntimeError('smoke did not update any BN affine')
                # Test the stopped-prefix backward with *all* groups open too.
                optimizer.zero_grad(set_to_none=True)
                full_loss, full_site, _ = finite_prefix_batch_loss(model, images, 5, args.guard, args.target)
                if full_loss.requires_grad:
                    full_loss.backward()
                averaged_gradients(parameters, world, device)
                save('smoke.pt', group, step)
                barrier()
                if rank == 0:
                    if digest(Path(args.source)/'last.pt') != source_record['source_sha256']:
                        raise RuntimeError('original checkpoint changed during smoke test')
                    print(f'RECOVERY_SMOKE_OK world={world} batch={args.batch_size} affine_delta={delta} full_prefix_site={full_site}', flush=True)
                return
            if step % args.save_every == 0:
                save('recovery_last.pt', group, step)
            if step % args.steps_per_group == 0 and rank == 0:
                print(f'RECOVERY_WINDOW group={group} step={step}; continuing until finite gates pass', flush=True)
            if step % args.check_every == 0:
                report = gate(model, args, gate_rows, split['labels'], rank, world, device, group, True,
                              f'group{group}_step{step}')
                save('recovery_last.pt', group, step, report)
                passed = stage_passes(report)
                if passed and group == 5:
                    final = gate(model, args, split['train'], split['labels'], rank, world,
                                 device, 5, False, f'full_training_step{step}')
                    passed = final['rows'] == 2*len(split['train']) and stage_passes(final)
                    save('recovery_last.pt', group, step, final, passed)
                    if not passed:
                        barrier()  # All ranks' diagnostic indices must be visible.
                        failed_rows = failure_training_rows(root, f'full_training_step{step}', world, split['train'])
                        if len(failed_rows):
                            # One in four repair batches targets actual failed
                            # training rows; ordinary train coverage is retained.
                            failed_rows = np.tile(failed_rows, max(1, (world+len(failed_rows)-1)//len(failed_rows)))
                            failure_batches = loader(args, failed_rows[rank::world], split['labels'], True)
                            failure_iterator = iter(failure_batches)
                    if not passed and rank == 0:
                        print('FULL_SCAN_RETRY: staying in group 5; no accuracy handoff', flush=True)
                if passed:
                    save(f'group{group}.pt', group, step, report)
                    barrier()
                    break
    if ready_resume:
        step, final = start_step, position['gate']
        if final['rows'] != 2*len(split['train']) or not stage_passes(final):
            raise ValueError('ready resume lacks a passing full training gate')
    save('recovered.pt', 5, step, final, True)
    barrier()
    if rank == 0:
        if digest(Path(args.source)/'last.pt') != source_record['source_sha256']:
            raise RuntimeError('original checkpoint changed during recovery')
        print('RECOVERY_READY: full training clean/flip finite gate passed', flush=True)
    barrier()
    if args.continue_training:
        continuation = SimpleNamespace(**continuation_config)
        continuation.output = str(root)
        continuation.resume = str(root/'last.pt' if (root/'last.pt').exists() else root/'recovered.pt')
        continuation.smoke = False
        # Do not retain recovery models/gradients during the full trainer.
        del model, teacher, head, parameters, optimizer, payload, prepared
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        resume_accuracy(continuation, rank, world, device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default='work_dirs/recipe_a_376833')
    parser.add_argument('--output', required=True)
    parser.add_argument('--resume', help='repair checkpoint to import into a new output directory')
    parser.add_argument('--reuse-output', action='store_true', help='restart own output after Slurm requeue')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=20260911)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--range-weight', type=float, default=1.)
    parser.add_argument('--continuation-lr-factor', type=float, default=.1)
    parser.add_argument('--guard', type=float, default=1.)
    parser.add_argument('--target', type=float, default=.9)
    parser.add_argument('--gate-images', type=int, default=8192)
    parser.add_argument('--steps-per-group', type=int, default=2000)
    parser.add_argument('--check-every', type=int, default=250)
    parser.add_argument('--save-every', type=int, default=25)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--continue-training', action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if not 0 < args.target < args.guard or args.lr <= 0 or min(args.batch_size, args.gate_images, args.check_every, args.save_every) < 1:
        parser.error('invalid range boundaries, learning rate, or sizes')
    if args.steps_per_group < max(2, args.check_every):
        parser.error('steps-per-group must include a checkpoint/gate interval and two smoke steps')
    if not 0 < args.continuation_lr_factor <= 1 or args.range_weight <= 0:
        parser.error('invalid continuation learning-rate factor or range loss weight')
    rank, world, device = context()
    torch.set_num_threads(2)
    torch.manual_seed(args.seed+rank)
    run(args, rank, world, device)
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
