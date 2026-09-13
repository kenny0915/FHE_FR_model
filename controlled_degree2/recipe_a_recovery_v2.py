"""Recipe A revision: sitewise joint recovery, training split only.

Fixed PReLU fits on [-lam_fit[c], lam_fit[c]], fixed BN moments, 25 squares.
This changes A's optimization protocol, not its deployed function family.
"""
from __future__ import annotations

import argparse
import copy
import itertools
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from controlled_degree2.augment import prepare_range_batch
from controlled_degree2.model import build_controlled_iresnet50, load_teacher, quadratic_modules, save_checkpoint
from controlled_degree2.recipe_a import IdentityHead, SourceRows, barrier, context, digest, loader, require_finite, train
from controlled_degree2.recipe_a_recovery import (
    averaged_gradients, check_split, configure, finite_prefix_batch_loss, gate, snapshot_source, variants,
)
from utils.utils_multi_objective import combine_conflict_aware_gradients

POLICY = 'recipe_a_sitewise_joint_v2'
FIXED_POLICY = ('seed', 'lr', 'target', 'guard', 'gate_ratio', 'gate_images',
                'hint_weight', 'range_weight', 'max_step_ratio', 'batch_size',
                'replay_batch_size', 'check_every', 'continuation_lr_factor')


def joint_parameters(model, train_coefficients=False):
    model.requires_grad_(False)
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.BatchNorm2d)):
            for parameter in module.parameters(recurse=False):
                parameter.requires_grad_(True)
    if train_coefficients:
        from controlled_degree2.shared import SharedQuadratic
        for module in quadratic_modules(model):
            if not isinstance(module, SharedQuadratic) or module.coeffs.shape != (1, 3):
                raise ValueError('trainable recovery coefficients must be layer-shared')
            module.coeffs.requires_grad_(True)
    return [(name, p) for name, p in model.named_parameters() if p.requires_grad]


def sitewise(model):
    model._recovery_sites = [m.name for m in quadratic_modules(model)]
    return model._recovery_sites


def bounded_gate(report, ratio, expected_rows):
    return (report['rows'] == expected_rows and expected_rows > 0
            and report['nonfinite'] == 0 and report['max_ratio'] <= ratio)


def full_gate(report, train_count):
    return report['rows'] == 2*train_count and train_count > 0 and report['nonfinite'] == 0


def validate_state(state, original, allowed):
    if state.keys() != original.keys():
        raise ValueError('state keys differ from epoch-8 source')
    for name, value in state.items():
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f'non-finite checkpoint tensor: {name}')
        if name not in allowed and not torch.equal(value.cpu(), original[name].cpu()):
            raise ValueError(f'frozen tensor changed: {name}')


def validate_resume(state, source, args, allowed, site_count):
    recovery = state.get('recovery', {})
    from controlled_degree2.shared_recovery_support import POLICY as SHARED_POLICY
    expected_policy = SHARED_POLICY if getattr(args, 'shared', False) else POLICY
    if recovery.get('policy') != expected_policy or state.get('provenance') != source['provenance']:
        raise ValueError('not a compatible A-v2 recovery checkpoint')
    if recovery.get('source_sha256') != args.source_sha256:
        raise ValueError('recovery source checksum mismatch')
    policy_keys = FIXED_POLICY + (('accuracy_epochs', 'deadline') if getattr(args, 'shared', False) else ())
    for key in policy_keys:
        if recovery['config'][key] != getattr(args, key):
            raise ValueError(f'resume changes policy: {key}')
    if not 1 <= recovery['site'] <= site_count or recovery['step'] < 0:
        raise ValueError('invalid recovery cursor')
    validate_state(state['state_dict_backbone'], source['state_dict_backbone'], allowed)
    return recovery


def replay_records(root, label, world, train_rows):
    records = set()
    valid_variants = {'clean', 'flip', 'lowres', 'shift', 'dark'}
    for rank in range(world):
        payload = json.loads((Path(root)/f'{label}.rank{rank}.json').read_text())
        for item in payload['failures']:
            index, variant = item['source_index'], item['variant']
            if type(index) is not int or variant not in valid_variants:
                raise ValueError('invalid exact replay record')
            records.add((index, variant))
    records = sorted(records)
    if records and not np.isin([r[0] for r in records], train_rows).all():
        raise ValueError('non-training source in replay')
    return records


def merge_replay(current, previous, capacity=65536):
    # Newly observed failures must not be displaced by lower-index historical
    # rows when a full scan discovers many new counterexamples.
    latest = {tuple(r) for r in current}
    return (sorted(latest) + sorted({tuple(r) for r in previous}-latest))[:capacity]


class ExactReplay(Dataset):
    def __init__(self, args, records, labels):
        self.records = records
        self.source = SourceRows(args.dataset_root, [r[0] for r in records], labels, True)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        image, label, row = self.source[index]
        requested = self.records[index][1]
        for name, transformed in variants(image.unsqueeze(0), True):
            if name == requested:
                return transformed.squeeze(0), label, row
        raise ValueError(f'unknown replay variant: {requested}')


def replay_loader(args, records, labels, rank, world, epoch):
    if not records:
        return None
    dataset = ExactReplay(args, records, labels)
    sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, seed=args.seed, drop_last=False)
    sampler.set_epoch(epoch)
    return DataLoader(dataset, batch_size=args.replay_batch_size, sampler=sampler,
                      num_workers=args.workers, pin_memory=True,
                      multiprocessing_context='spawn' if args.workers else None)


def clipped_objectives(model, teacher, head, images, labels, mask, args):
    """All-block hints and all-site range loss; only this auxiliary path clips."""
    student_hints, teacher_hints, penalties = {}, {}, []
    handles = []
    names = [name.rsplit('.', 1)[0] for name in model._recovery_sites if name != 'prelu']

    def range_hook(module, inputs):
        relative = inputs[0].double().abs()/module.lam_fit.double().reshape(1, -1, 1, 1)
        excess = (relative-args.target).clamp_min(0).square().flatten(1)
        penalties.append(excess.amax(1).mean() + .01*excess.mean())

    try:
        configure(model, 0)
        for name in names:
            handles.append(model.get_submodule(name).register_forward_hook(
                lambda m, i, o, name=name: student_hints.__setitem__(name, o)))
            handles.append(teacher.get_submodule(name).register_forward_hook(
                lambda m, i, o, name=name: teacher_hints.__setitem__(name, o)))
        for module in quadratic_modules(model):
            handles.append(module.register_forward_pre_hook(range_hook))
        with torch.no_grad():
            target = teacher(images)
        output = model(images)
        require_finite(output, images.device, 'v2 clipped auxiliary embeddings')
        weights = mask.float()
        average = lambda value: (value*weights).sum()/weights.sum().clamp_min(1)
        kd = average(1-torch.nn.functional.cosine_similarity(output, target, dim=1))
        hints = []
        for name in names:
            s, t = student_hints[name].float(), teacher_hints[name].float()
            hints.append(average((s-t).square().flatten(1).mean(1)
                                 / t.square().flatten(1).mean(1).clamp_min(1e-6)))
        hint = torch.stack(hints).mean()
        clean = kd + args.hint_weight*hint + .1*head(output, labels, mask)
        return clean, torch.stack(penalties).mean(), dict(kd=float(kd.detach()), hint=float(hint.detach()))
    finally:
        for handle in handles:
            handle.remove()


def global_gradients(loss, parameters, world, device, retain_graph=False):
    if loss.requires_grad:
        gradients = torch.autograd.grad(loss, parameters, allow_unused=True, retain_graph=retain_graph)
    else:
        gradients = [None]*len(parameters)
    for parameter, gradient in zip(parameters, gradients):
        parameter.grad = gradient
    averaged_gradients(parameters, world, device)
    result = tuple(p.grad for p in parameters)
    for parameter in parameters:
        parameter.grad = None
    return result


def update(model, teacher, head, images, labels, mask, replay, site, parameters, optimizer, args, world, device):
    optimizer.zero_grad(set_to_none=True)
    clean, broad_range, metrics = clipped_objectives(model, teacher, head, images, labels, mask, args)
    require_finite(clean, device, 'v2 identity/hint loss')
    clean_gradients = global_gradients(clean, parameters, world, device, retain_graph=True)
    # Backprop through the finite upstream Conv/BN/polynomial prefix, not
    # detached BN-only controllers. Stop before any out-of-interval square.
    probe = images[:args.replay_batch_size] if replay is None else replay
    shadow, first_site, _ = finite_prefix_batch_loss(
        model, probe, site, args.guard, args.target, detach_bn=False)
    tail = args.range_weight*(broad_range + shadow)
    require_finite(tail, device, 'v2 broad/prefix range loss')
    tail_gradients = global_gradients(tail, parameters, world, device)
    # Project globally averaged objectives, so every rank makes the same
    # conflict decision. No momentum: the per-tensor cap bounds actual SGD.
    combined, stats = combine_conflict_aware_gradients(
        parameters, clean_gradients, tail_gradients, learning_rate=args.lr,
        tail_to_clean_ratio=10000., max_step_update_ratio=args.max_step_ratio)
    require_finite(torch.cat([g.flatten() for g in combined]), device, 'v2 combined gradients')
    for parameter, gradient in zip(parameters, combined):
        parameter.grad = gradient
    optimizer.step()
    return dict(clean=float(clean.detach()), range=float(tail.detach()), first_escape=first_site, **metrics, **stats)


def run(args, rank, world, device):
    root = Path(args.output)
    shared = bool(getattr(args, 'shared', False))
    if shared:
        from controlled_degree2 import shared_recovery_support as support
    policy = support.POLICY if shared else POLICY
    if rank == 0:
        if not args.reuse_output:
            if shared:
                support.snapshot_source(args.source, root, args.source_checkpoint)
            else:
                snapshot_source(args.source, root)
    barrier()
    source_record = json.loads((root/'recovery_source.json').read_text())
    args.source_sha256 = source_record['source_sha256']
    source_path = root/source_record.get('snapshot', 'source_epoch8.pt')
    if digest(source_path) != args.source_sha256:
        raise ValueError('source snapshot checksum mismatch')
    payload = torch.load(source_path, map_location='cpu', weights_only=False)
    prepared = torch.load(root/'prepared.pt', map_location='cpu', weights_only=False)
    if shared:
        support.validate_source(payload)
    elif payload.get('epoch') != 8 or payload.get('origin') != 'teacher_only_recipe_a':
        raise ValueError('v2 requires preserved teacher-only epoch 8')
    if payload['provenance'] != prepared['metadata'] or prepared['metadata']['config'].get('smoke'):
        raise ValueError('invalid preparation provenance')
    config = payload['config']
    if args.batch_size*world != config['global_batch']:
        raise ValueError('A-v2 preserves the original global batch size')
    args.dataset_root = config['dataset_root']
    if digest(root/'split.npz') != prepared['metadata']['split_sha256'] or digest(config['teacher']) != prepared['metadata']['teacher_sha256']:
        raise ValueError('teacher/split checksum mismatch')
    split = np.load(root/'split.npz')
    check_split(split, prepared['active_ids'])
    if shared:
        from controlled_degree2.shared import build_shared_iresnet50
        model = build_shared_iresnet50(dropout=0, fp16=False).to(device)
    else:
        model = build_controlled_iresnet50(dropout=0, fp16=False).to(device)
    model.load_state_dict(payload['state_dict_backbone'], strict=True)
    sites = sitewise(model)
    named = joint_parameters(model, train_coefficients=shared)
    allowed, parameters = set(n for n, p in named), tuple(p for n, p in named)
    validate_state(model.state_dict(), payload['state_dict_backbone'], allowed)
    teacher = load_teacher(config['teacher'], device).eval().requires_grad_(False)
    head = IdentityHead(payload['head']['weight']).to(device).eval().requires_grad_(False)
    mapping = torch.full((len(prepared['centers']),), -1, dtype=torch.long, device=device)
    mapping[torch.tensor(prepared['active_ids'], device=device)] = torch.arange(len(prepared['active_ids']), device=device)
    optimizer = torch.optim.SGD(parameters, lr=args.lr, momentum=0.)
    continuation_config = dict(config)
    for key in ('lr', 'head_lr'):
        continuation_config[key] *= args.continuation_lr_factor
    if shared:
        continuation_config = support.continuation_config(continuation_config, args)
    start_site, start_step, ready, last_report = 1, 0, False, None
    cursor_replay = []
    resume = root/'recovery_last.pt' if args.reuse_output else Path(args.resume) if args.resume else None
    if resume is not None and resume.exists():
        state = torch.load(resume, map_location='cpu', weights_only=False)
        position = validate_resume(state, payload, args, allowed, len(sites))
        model.load_state_dict(state['state_dict_backbone'], strict=True)
        optimizer.load_state_dict(state['recovery_optimizer'])
        start_site, start_step, ready = position['site'], position['step'], position['ready']
        last_report, cursor_replay = position['gate'], position['replay']
        if rank == 0:
            print(f'JOINT_RESUME site={start_site} step={start_step} ready={ready}', flush=True)
        del state
    elif args.resume:
        raise FileNotFoundError(args.resume)
    gate_rows = np.random.default_rng(args.seed).permutation(split['train'])[:args.gate_images]
    sampler = DistributedSampler(split['train'], num_replicas=world, rank=rank, seed=args.seed, drop_last=True)
    batches = loader(args, split['train'], split['labels'], sampler=sampler)
    if rank == 0:
        (root/'recovery_v2_config.json').write_text(json.dumps(dict(vars(args), policy=policy, sites=sites, trainable=sorted(allowed)), indent=2))

    def save(name, site, step, report, records, is_ready=False):
        if rank != 0:
            return
        validate_state(model.state_dict(), payload['state_dict_backbone'], allowed)
        extra = dict(epoch=-1 if shared else 8, best=-1., config=continuation_config, provenance=prepared['metadata'],
                     head=payload['head'], origin='teacher_only_recipe_a', pure_quadratic=is_ready,
                     recovery_optimizer=optimizer.state_dict(),
                     recovery=dict(policy=policy, site=site, step=step, ready=is_ready,
                                   source_sha256=args.source_sha256, config=vars(args), gate=report, replay=records))
        if shared:
            extra.update(network='r50_shared_d2', format='fhe-fr/shared-degree2-v1',
                         origin='shared_sitewise_recovery', inference_input_bounds=False,
                         fhe_requires_comparisons=False, all_activations_quadratic=True,
                         approximation_target='pooled PReLU on inherited MS1MV3 per-layer intervals',
                         interval='per-layer [-radius, radius]', development={'phase': 'training-only repair'})
        extra['optimizer'] = copy.deepcopy(payload['optimizer'])
        extra['optimizer']['state'] = {}
        save_checkpoint(str(root/(name+'.tmp')), model, prepared['calibration'], teacher_weights=config['teacher'], extra=extra)
        os.replace(root/(name+'.tmp'), root/name)

    step = start_step
    for site in range(start_site, len(sites)+1) if not ready else ():
        offset = start_step if site == start_site else 0
        sampler.set_epoch(site*100000+offset)
        iterator = iter(batches)
        audit_label = f'site{site}_entry{offset}'
        last_report = gate(model, args, gate_rows, split['labels'], rank, world, device, site, True, audit_label)
        barrier()
        records = replay_records(root, audit_label, world, split['train'])
        # Keep known full-corpus counterexamples across wall-time restarts.
        records = merge_replay(records, cursor_replay)
        if records and not np.isin([r[0] for r in records], split['train']).all():
            raise ValueError('resume replay includes non-training sources')
        cursor_replay = []
        replay_batches = replay_loader(args, records, split['labels'], rank, world, offset)
        replay_iterator = iter(replay_batches) if replay_batches is not None else None
        initial = model.conv1.weight.detach().clone() if args.smoke else None
        for step in itertools.count(offset+1):
            if getattr(args, 'deadline', 0.) and time.time() >= args.deadline:
                save('recovery_last.pt', site, step-1, last_report, records)
                barrier()
                raise TimeoutError('shared recovery campaign deadline reached')
            try:
                images, labels, _ = next(iterator)
            except StopIteration:
                sampler.set_epoch(site*100000+step)
                iterator = iter(batches)
                images, labels, _ = next(iterator)
            images, labels = images.to(device), mapping[labels.to(device)]
            if bool((labels < 0).any()):
                raise ValueError('non-training label in repair batch')
            images, mask = prepare_range_batch(images, pathological_fraction=.02, crop_probability=.1,
                                               lowres_probability=.2, photo_probability=.2, stress_probability=.1)
            replay = None
            if replay_iterator is not None:
                try:
                    replay, _, _ = next(replay_iterator)
                except StopIteration:
                    replay_batches.sampler.set_epoch(step)
                    replay_iterator = iter(replay_batches)
                    replay, _, _ = next(replay_iterator)
                replay = replay.to(device)
            metrics = update(model, teacher, head, images, labels, mask, replay, site, parameters, optimizer, args, world, device)
            if rank == 0 and (step == offset+1 or step % 25 == 0):
                print(f'JOINT site={site}/25 name={sites[site-1]} step={step} replay={len(records)} {metrics}', flush=True)
            if args.smoke and step == offset+2:
                delta = float((model.conv1.weight-initial).abs().max())
                if not delta > 0:
                    raise RuntimeError('smoke did not update convolution weights')
                # Exercise a real all-site guarded Conv backward too.
                smoke_loss, _, _ = finite_prefix_batch_loss(model, images[:2], len(sites), detach_bn=False)
                global_gradients(smoke_loss, parameters, world, device)
                save('smoke.pt', site, step, last_report, records)
                barrier()
                if rank == 0:
                    print(f'JOINT_SMOKE_OK world={world} batch={args.batch_size} conv_delta={delta} trainable={len(parameters)}', flush=True)
                return
            if step % args.save_every == 0:
                save('recovery_last.pt', site, step, last_report, records)
            if step % args.check_every:
                continue
            audit_label = f'site{site}_step{step}'
            last_report = gate(model, args, gate_rows, split['labels'], rank, world, device, site, True, audit_label)
            passed = bounded_gate(last_report, args.gate_ratio, 5*len(gate_rows))
            barrier()
            mined = replay_records(root, audit_label, world, split['train'])
            if passed and site == len(sites):
                audit_label = f'full_training_step{step}'
                last_report = gate(model, args, split['train'], split['labels'], rank, world, device, site, False, audit_label)
                passed = full_gate(last_report, len(split['train']))
                ready = passed
                barrier()
                mined = replay_records(root, audit_label, world, split['train'])
            # Retain counterexamples, preserving exact variant and train-only
            # provenance, but bound replay memory to 65536 source/variant pairs.
            records = merge_replay(mined, records)
            save('recovery_last.pt', site, step, last_report, records, ready)
            barrier()
            if passed:
                save(f'site{site}.pt', site, step, last_report, records, ready)
                barrier()
                break
            replay_iterator = None
            replay_batches = replay_loader(args, records, split['labels'], rank, world, step)
            replay_iterator = iter(replay_batches) if replay_batches is not None else None
    if not ready or not full_gate(last_report, len(split['train'])):
        raise ValueError('accuracy handoff requires full training finite gate')
    configure(model, len(sites))
    save('recovered.pt', len(sites), step, last_report, [], True)
    barrier()
    if rank == 0:
        print('JOINT_RECOVERY_READY: all 25 sites and full train clean/flip gate passed', flush=True)
    if args.continue_training:
        continuation = SimpleNamespace(**continuation_config)
        continuation.output = str(root)
        continuation.resume = str(root/'last.pt' if (root/'last.pt').exists() else root/'recovered.pt')
        continuation.smoke = False
        del model, teacher, head, named, parameters, optimizer, payload, prepared
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        train(continuation, rank, world, device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shared', action='store_true')
    parser.add_argument('--source-checkpoint', default='continuation_best.pt')
    parser.add_argument('--deadline', type=float, default=0.)
    parser.add_argument('--accuracy-epochs', type=int, default=24)
    parser.add_argument('--source', default='work_dirs/recipe_a_376833')
    parser.add_argument('--output', required=True)
    parser.add_argument('--resume')
    parser.add_argument('--reuse-output', action='store_true')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--replay-batch-size', type=int, default=32)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=20260911)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--range-weight', type=float, default=1.)
    parser.add_argument('--hint-weight', type=float, default=.3)
    parser.add_argument('--max-step-ratio', type=float, default=1e-4)
    parser.add_argument('--guard', type=float, default=1.)
    parser.add_argument('--target', type=float, default=.9)
    parser.add_argument('--gate-ratio', type=float, default=2.)
    parser.add_argument('--gate-images', type=int, default=8192)
    parser.add_argument('--check-every', type=int, default=250)
    parser.add_argument('--save-every', type=int, default=25)
    parser.add_argument('--continuation-lr-factor', type=float, default=.1)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--continue-training', action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.shared and (args.deadline <= time.time() or args.accuracy_epochs <= 0):
        parser.error('shared repair requires a future campaign deadline and positive accuracy epochs')
    if not 0 < args.target < args.guard <= args.gate_ratio or not 0 < args.continuation_lr_factor <= 1:
        parser.error('invalid boundaries/continuation rate')
    if min(args.batch_size, args.replay_batch_size, args.gate_images, args.check_every, args.save_every,
           args.lr, args.range_weight, args.hint_weight, args.max_step_ratio) <= 0 or args.workers < 0:
        parser.error('invalid size or optimization policy')
    args.replay_ratio = args.gate_ratio
    rank, world, device = context()
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    run(args, rank, world, device)
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
