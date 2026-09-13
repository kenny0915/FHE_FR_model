"""Teacher-only recipe A: fresh calibration, ArcFace, stage KD, range control.

No prior polynomial checkpoint or evaluation-dataset statistics are accepted.
Run prepare and train with torchrun; see recipe_a.slurm and recipe_a.md.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler

from controlled_degree2.calibrate import ChannelAbsHistogram, weighted_quadratic_abs_fit
from controlled_degree2.augment import prepare_range_batch
from controlled_degree2.model import (
    load_teacher, prelu_names, replace_prelu_with_quadratic,
    quadratic_modules, save_checkpoint, set_quadratic_schedule,
)
from utils.utils_optimizer import clip_grad_norm_stable


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def identity_split(labels, seed=20260911, fraction=0.02):
    """Split identities, never individual images, before calibration/mining."""
    ids = np.unique(labels)
    if len(ids) < 2 or not 0 < fraction < 1:
        raise ValueError('need at least two identities and 0 < fraction < 1')
    rng = np.random.default_rng(seed)
    held = rng.permutation(ids)[:max(1, min(len(ids)-1, round(len(ids)*fraction)))]
    dev = np.isin(labels, held)
    return np.flatnonzero(~dev), np.flatnonzero(dev)


def per_identity_rows(labels, rows, count, seed):
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(rows)
    order = np.argsort(labels[shuffled], kind='stable')
    grouped = shuffled[order]
    _, starts = np.unique(labels[grouped], return_index=True)
    ends = np.r_[starts[1:], len(grouped)]
    return np.concatenate([grouped[a:min(a+count, b)] for a, b in zip(starts, ends)])


class SourceRows(Dataset):
    """Reopen RecordIO in each worker; deterministic validation orientation."""
    def __init__(self, root, rows, labels, deterministic=False):
        self.root, self.rows, self.labels = root, rows, labels
        self.deterministic = deterministic
        self.source = None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        if self.source is None:
            from dataset import MXFaceDataset
            self.source = MXFaceDataset(self.root, local_rank=0)
        row = int(self.rows[index])
        image, label = (self.source.get_oriented(row, 0) if self.deterministic
                        else self.source[row])
        if int(label) != int(self.labels[row]):
            raise ValueError(f'train.lst/RecordIO label mismatch at row {row}')
        return image, label, row


def context():
    world = int(os.environ.get('WORLD_SIZE', '1'))
    rank = int(os.environ.get('RANK', '0'))
    local = int(os.environ.get('LOCAL_RANK', '0'))
    device = torch.device('cuda', local) if torch.cuda.is_available() else torch.device('cpu')
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group('nccl' if device.type == 'cuda' else 'gloo')
    return rank, world, device


def barrier():
    if dist.is_initialized():
        dist.barrier()


def reduce(tensor, op=dist.ReduceOp.SUM):
    if dist.is_initialized():
        dist.all_reduce(tensor, op=op)
    return tensor


def require_finite(value, device, where):
    bad = torch.tensor(int(not bool(torch.isfinite(value).all())), device=device)
    reduce(bad, dist.ReduceOp.MAX)
    if bad.item():
        raise FloatingPointError(f'{where}: non-finite on at least one rank; aborting, no skipped update')


def loader(args, rows, labels, deterministic=False, sampler=None):
    return DataLoader(SourceRows(args.dataset_root, rows, labels, deterministic),
                      batch_size=args.batch_size, sampler=sampler,
                      num_workers=args.workers, pin_memory=True,
                      drop_last=sampler is not None,
                      multiprocessing_context='spawn' if args.workers else None,
                      persistent_workers=args.workers > 0)


def prepare(args, rank, world, device):
    root = Path(args.output)
    if rank == 0:
        root.mkdir(parents=True, exist_ok=True)
        if (root / 'prepared.pt').exists():
            raise FileExistsError('prepared.pt already exists; use a fresh output directory')
        with open(Path(args.dataset_root) / 'train.lst') as stream:
            labels = np.fromiter((int(line.rstrip().rsplit('\t', 1)[1]) for line in stream), dtype=np.int64)
        train, dev = identity_split(labels, args.seed, args.dev_fraction)
        if args.smoke:
            ids = np.random.default_rng(args.seed).permutation(np.unique(labels[train]))[:64]
            train = train[np.isin(labels[train], ids)]
            # Full production microbatch smoke: repeat only these training rows
            # if necessary to guarantee two distributed optimizer steps.
            needed = args.batch_size*world*2
            if len(train) < needed:
                train = np.tile(train, math.ceil(needed/len(train)))
            dev_ids = np.unique(labels[dev])[:16]
            dev = dev[np.isin(labels[dev], dev_ids)]
        dev = per_identity_rows(labels, dev, args.dev_per_identity, args.seed)
        np.savez(root / 'split.npz', labels=labels, train=train, dev=dev)
    barrier()
    split = np.load(root / 'split.npz')
    labels, train = split['labels'], split['train']
    teacher = load_teacher(args.teacher, device).eval().requires_grad_(False)
    centers_rows = per_identity_rows(labels, train, args.center_images, args.seed)
    calibration_rows = np.random.default_rng(args.seed).permutation(train)[:args.calibration_images]
    rows = np.unique(np.r_[centers_rows, calibration_rows])
    calibration_set = set(calibration_rows.tolist())
    center_set = set(centers_rows.tolist())
    names = prelu_names(teacher)
    if len(names) != 25:
        raise ValueError('recipe A requires original 25-PReLU iResNet50 teacher')
    hist = {n: ChannelAbsHistogram(teacher.get_submodule(n).weight.numel(), 1e-5, 1e3, 512, device) for n in names}
    selected = None
    def observe(name, inputs):
        if selected is not None and bool(selected.any()):
            # Fixed spatial subsampling reduces calibration cost, never uses IJB.
            hist[name].update(inputs[0][selected, :, ::4, ::4])
    handles = [teacher.get_submodule(n).register_forward_pre_hook(
        lambda m, inputs, n=n: observe(n, inputs)) for n in names]
    centers = torch.zeros(int(labels.max())+1, 512, device=device)
    counts = torch.zeros(centers.shape[0], device=device)
    with torch.no_grad():
        for step, (images, targets, indices) in enumerate(loader(args, rows[rank::world], labels, True)):
            images, targets = images.to(device), targets.to(device)
            selected = torch.tensor([int(i) in calibration_set for i in indices], device=device)
            features = F.normalize(teacher(images).float(), dim=1)
            if not torch.isfinite(features).all():
                raise FloatingPointError('non-finite teacher during preparation')
            chosen = torch.tensor([int(i) in center_set for i in indices], device=device)
            centers.index_add_(0, targets[chosen], features[chosen])
            counts.index_add_(0, targets[chosen], torch.ones_like(targets[chosen], dtype=torch.float))
            if rank == 0 and step % 50 == 0:
                print(f'prepare batch={step} local_rows={len(rows[rank::world])}', flush=True)
    for handle in handles:
        handle.remove()
    reduce(centers)
    reduce(counts)
    calibration = {}
    for name in names:
        h = hist[name]
        reduce(h.counts)
        reduce(h.maximum, dist.ReduceOp.MAX)
        probabilities = h.counts / h.counts.sum(1, keepdim=True).clamp_min(1)
        bin_centers, widths = h.bin_geometry()
        quantile_indices = (probabilities.cumsum(1) < args.quantile).sum(1).clamp_max(h.bins-1)
        # Upper bin edge; one global widening policy, no layer-specific tuning.
        upper_edges = bin_centers * 10**((h.log_max-h.log_min)/(2*h.bins))
        lam = (upper_edges[quantile_indices] * args.guard_band).clamp_min(0.05)
        even, error = weighted_quadratic_abs_fit(probabilities, bin_centers, widths, lam,
                                                slope=teacher.get_submodule(name).weight)
        if getattr(args, 'shared', False):
            from controlled_degree2.shared import fit_shared
            pooled = h.counts.sum(0)
            index = int((pooled.cumsum(0)/pooled.sum() < args.quantile).sum().clamp_max(h.bins-1))
            radius = max(.05, float(upper_edges[index])*args.guard_band)
            coefficients = fit_shared(h.counts, bin_centers, teacher.get_submodule(name).weight,
                                      radius, args.initialization)
            calibration[name] = dict(radius=radius, reg_ratio=args.reg_ratio,
                                     coefficients=coefficients.cpu().tolist(),
                                     teacher_max=float(h.maximum.max()),
                                     initialization=args.initialization)
            continue
        calibration[name] = dict(lam_fit=lam.cpu().tolist(), lam_reg=(lam*args.reg_ratio).cpu().tolist(),
                                 even_coeffs=even.cpu().tolist(), teacher_max=h.maximum.cpu().tolist(),
                                 fit_relative_error=error.cpu().tolist())
    active = np.unique(labels[train])
    if not bool((counts[torch.as_tensor(active, device=device)] > 0).all()):
        raise ValueError('missing training identity centers')
    if rank == 0:
        metadata = dict(config=vars(args), teacher_sha256=digest(args.teacher),
                        split_sha256=digest(root/'split.npz'),
                        code_commit=os.environ.get('RECIPE_COMMIT', 'uncommitted-smoke'),
                        teacher_may_have_seen_development_identities=True,
                        approximation_target='channelwise PReLU', interval='[-lam_fit, lam_fit]',
                        source='MS1MV3 training identities only; fresh teacher statistics')
        if getattr(args, 'shared', False):
            metadata.update(approximation_target='joint channel PReLU fit with one coefficient triplet per layer', interval='[-radius, radius] per layer')
        torch.save(dict(calibration=calibration, centers=F.normalize(centers, dim=1).cpu(),
                        active_ids=active.tolist(), metadata=metadata), root/'prepared.pt')
        (root/'provenance.json').write_text(json.dumps(metadata, indent=2))
        print('PREPARE_OK', flush=True)
    barrier()


class IdentityHead(nn.Module):
    """Replicated full ArcFace head; ~190 MB weights for MS1MV3, suitable for H200."""
    def __init__(self, centers):
        super().__init__()
        self.weight = nn.Parameter(centers.clone())

    def forward(self, features, labels, mask, margin=0.5):
        cosine = F.linear(F.normalize(features.float(), dim=1), F.normalize(self.weight.float(), dim=1))
        cosine = cosine.clamp(-1+1e-6, 1-1e-6)
        target = cosine.gather(1, labels[:, None]).squeeze(1)
        phi = target * math.cos(margin) - (1-target.square()).sqrt() * math.sin(margin)
        phi = torch.where(target > math.cos(math.pi-margin), phi, target-math.sin(math.pi-margin)*margin)
        logits = cosine.scatter(1, labels[:, None], phi[:, None]) * 64
        values = F.cross_entropy(logits, labels, reduction='none')
        return (values * mask).sum() / mask.sum().clamp_min(1)


class TrainingTailReplay:
    """Small rank-local cache of actual augmented training tails, never dev rows."""
    def __init__(self, capacity=32):
        self.capacity = capacity
        self.images = self.labels = self.scores = None

    def inject(self, images, labels, mask):
        if self.images is not None:
            count = min(max(1, len(images)//16), len(self.images))
            images[:count] = self.images[:count]
            labels[:count] = self.labels[:count]
            mask[:count] = True
        return images, labels, mask

    @torch.no_grad()
    def update(self, images, labels, mask, modules):
        scores = torch.stack([m.last_sample_ratio for m in modules]).amax(0)
        images, labels, scores = images[mask].detach(), labels[mask].detach(), scores[mask]
        if self.images is not None:
            images, labels, scores = (torch.cat((old, new)) for old, new in
                                      ((self.images, images), (self.labels, labels), (self.scores, scores)))
        chosen = scores.topk(min(self.capacity, len(scores))).indices
        self.images, self.labels, self.scores = images[chosen].clone(), labels[chosen].clone(), scores[chosen].clone()


def phase_at(epoch, warmup=1.0, conversion=8.0):
    # Five groups: stem and four stages. Every group's ramp is two epochs.
    progress = epoch-warmup
    ramp = conversion/4
    starts = [i*(conversion-ramp)/4 for i in range(5)]
    alphas = [max(0., min(1., (progress-start)/ramp)) for start in starts]
    return alphas, epoch < warmup+conversion


def apply_phase(student, epoch, args):
    if getattr(args, 'all_quadratic_start', False):
        alphas, clipped = [1.]*5, False
    else:
        alphas, clipped = phase_at(epoch, args.head_warmup, args.conversion_epochs)
    inference_bound = bool(getattr(args, 'inference_bound', False))
    clipped = clipped or inference_bound
    for module in quadratic_modules(student):
        group = 0 if module.name == 'prelu' else int(module.name[5])
        module.alpha, module.clip, module.clip_eval = alphas[group], clipped, inference_bound
    if getattr(args, 'unclipped_continuation', False):
        from controlled_degree2.unclipped_policy import apply_unclipping
        clipped = apply_unclipping(student, epoch, args.unclip_epochs)
    # Use pretrained running statistics throughout; affine weights can adapt.
    for module in student.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm) and getattr(args, 'batchnorm_mode', 'frozen') == 'frozen':
            module.eval()
    return alphas, clipped


def hook_stages(model, storage):
    return [model.get_submodule(f'layer{i}').register_forward_hook(
        lambda m, inp, out, i=i: storage.__setitem__(i, out)) for i in range(1, 5)]


def recipe_loss(features, target, student_hints, teacher_hints, mask, modules, elements, hint_weight):
    weights = mask.float()
    cosine = ((1-F.cosine_similarity(features.float(), target.float(), dim=1))*weights).sum()/weights.sum().clamp_min(1)
    hints = []
    for stage in student_hints:
        s, t = student_hints[stage].float(), teacher_hints[stage].float()
        error = (s-t).square().flatten(1).mean(1) / t.square().flatten(1).mean(1).clamp_min(1e-6)
        hints.append((error*weights).sum()/weights.sum().clamp_min(1))
    hint = torch.stack(hints).mean()
    boundary = torch.stack([m.last_penalty/elements[m.name] + 0.01*m.last_sample_penalty.mean() for m in modules]).mean()
    return cosine + hint_weight*hint, boundary


def verification_metric(features, labels, seed, negative_pairs=2_000_000):
    """Fixed sampled impostor protocol; seed never depends on measured scores."""
    features = F.normalize(features.float().cpu(), dim=1)
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    positive = []
    for identity in np.unique(labels):
        rows = np.flatnonzero(labels == identity)
        i, j = np.triu_indices(len(rows), 1)
        if len(i):
            positive.append((features[rows[i]]*features[rows[j]]).sum(1))
    if not positive or len(np.unique(labels)) < 2:
        raise ValueError('development verification requires positive and negative pairs')
    negative = []
    remaining = negative_pairs
    while remaining:
        i, j = rng.integers(len(labels), size=(2, min(65536, remaining*2)))
        keep = labels[i] != labels[j]
        i, j = i[keep][:remaining], j[keep][:remaining]
        negative.append((features[i]*features[j]).sum(1))
        remaining -= len(i)
    scores = torch.cat(negative)
    threshold = torch.quantile(scores, 0.9999, interpolation='higher')
    return dict(tar_1e4=float((torch.cat(positive)>threshold).float().mean()),
                threshold=float(threshold), empirical_far=float((scores>threshold).float().mean()),
                negative_pairs=len(scores), positive_pairs=sum(len(x) for x in positive))


@torch.no_grad()
def validate(student, args, rank, world, device, split):
    student.eval()
    set_quadratic_schedule(student, alpha=1., clip_eval=bool(getattr(args, 'inference_bound', False)))
    from eval.finite_audit import FiniteAudit
    audit = FiniteAudit(student) if getattr(args, 'shared', False) else None
    features, degraded, labels = [], [], []
    bad, peak = 0, 0.
    activation_ratio = torch.zeros((), device=device)
    def observe(module, inputs):
        ratio = (inputs[0].float().abs()/module.lam_fit.reshape(1, -1, 1, 1)).amax()
        activation_ratio.copy_(torch.maximum(activation_ratio, torch.nan_to_num(ratio, nan=float('inf'))))
    handles = [m.register_forward_pre_hook(observe) for m in quadratic_modules(student)]
    for images, targets, _ in loader(args, split['dev'][rank::world], split['labels'], True):
        images = images.to(device)
        outputs = {}
        for variant in ('clean', 'flip', 'lowres', 'shift', 'dark'):
            x = images if variant == 'clean' else images.flip(-1)
            if variant == 'lowres':
                x = F.interpolate(F.interpolate(images, size=(20, 20), mode='bilinear', align_corners=False),
                                  size=(112, 112), mode='bilinear', align_corners=False)
            elif variant == 'shift':
                x = torch.full_like(images, -1.)
                x[:, :, 4:, 4:] = images[:, :, :-4, :-4]
            elif variant == 'dark':
                x = (images+1)*.3-1
            z = student(x).float()
            finite = torch.isfinite(z).all(1) & torch.isfinite(z.norm(dim=1)) & (z.norm(dim=1) > 0)
            bad += int((~finite).sum())
            peak = max(peak, float(z.abs().max()))
            outputs[variant] = z.cpu()
        features.append(outputs['clean']+outputs['flip'])
        degraded.append(outputs['lowres'])
        labels.extend(targets.tolist())
    for handle in handles:
        handle.remove()
    reduce(activation_ratio, dist.ReduceOp.MAX)
    if audit is not None:
        audit_result = audit.result()
        audit.close()
        boundary_bad = torch.tensor(audit_result['nonfinite_values'], device=device, dtype=torch.long)
        reduce(boundary_bad)
        if rank == 0:
            Path(args.output, 'last_validation_ranges.json').write_text(json.dumps(audit_result, indent=2))
    # Bounded development embeddings, no images in the collective.
    local = (torch.cat(features), labels, bad, peak, torch.cat(degraded))
    gathered = [None]*world
    if world > 1:
        dist.all_gather_object(gathered, local)
    else:
        gathered = [local]
    result = None
    if rank == 0:
        bad = sum(x[2] for x in gathered)
        result = dict(nonfinite=bad + (int(boundary_bad) if audit is not None else 0), embedding_absmax=max(x[3] for x in gathered), activation_max_ratio=float(activation_ratio))
        if result['nonfinite'] == 0:
            all_labels = sum([x[1] for x in gathered], [])
            clean = verification_metric(torch.cat([x[0] for x in gathered]), all_labels, args.seed, args.negative_pairs)
            lowres = verification_metric(torch.cat([x[4] for x in gathered]), all_labels, args.seed, args.negative_pairs)
            result.update(clean=clean, lowres=lowres, tar_1e4=min(clean['tar_1e4'], lowres['tar_1e4']))
    message = [result]
    if world > 1:
        dist.broadcast_object_list(message, src=0)
    return message[0]


def load_shared_warm_start(student, head, path, provenance):
    """New optimization policy, identical teacher/split and identity-head mapping."""
    source = torch.load(path, map_location='cpu', weights_only=False)
    if source.get('network') not in ('r50_shared_d2', 'r50_shared_d2_bounded'):
        raise ValueError('warm start requires a layer-shared degree-two checkpoint')
    if source.get('provenance') != provenance:
        raise ValueError('warm start teacher/split/calibration provenance differs')
    student.load_state_dict(source['state_dict_backbone'], strict=True)
    head.load_state_dict(source['head'], strict=True)
    return dict(path=str(Path(path).resolve()), sha256=digest(path), epoch=source['epoch'])


def train(args, rank, world, device):
    root = Path(args.output)
    prepared = torch.load(root/'prepared.pt', map_location='cpu', weights_only=False)
    if bool(prepared['metadata']['config']['smoke']) != args.smoke:
        raise ValueError('smoke preparation cannot initialize production training')
    if digest(root/'split.npz') != prepared['metadata']['split_sha256'] or digest(args.teacher) != prepared['metadata']['teacher_sha256']:
        raise ValueError('teacher or split changed since preparation')
    split = np.load(root/'split.npz')
    torch.manual_seed(args.seed)
    student = load_teacher(args.teacher, device)
    teacher = copy.deepcopy(student).eval().requires_grad_(False)
    if getattr(args, 'shared', False):
        from controlled_degree2.shared import replace_shared
        replace_shared(student, prepared['calibration'])
    else:
        replace_prelu_with_quadratic(student, prepared['calibration'])
    student.to(device)
    if getattr(args, 'shared', False) and args.batchnorm_mode == 'train' and world > 1:
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
    active = torch.tensor(prepared['active_ids'], dtype=torch.long)
    head = IdentityHead(prepared['centers'][active]).to(device)
    warm_start_info = {}
    if getattr(args, 'warm_start', None):
        warm_start_info = load_shared_warm_start(student, head, args.warm_start, prepared['metadata'])
    if getattr(args, 'unclipped_continuation', False):
        from controlled_degree2.unclipped_policy import project_coefficients
        project_coefficients(student, args.curvature_cap)
    mapping = torch.full((len(prepared['centers']),), -1, dtype=torch.long, device=device)
    mapping[active.to(device)] = torch.arange(len(active), device=device)
    student_hints, teacher_hints, elements = {}, {}, {}
    handles = hook_stages(student, student_hints)+hook_stages(teacher, teacher_hints)
    modules = list(quadratic_modules(student))
    for module in modules:
        handles.append(module.register_forward_pre_hook(
            lambda m, inputs: elements.__setitem__(m.name, inputs[0][0].numel())))
    optimizer = torch.optim.SGD([dict(params=[p for p in student.parameters() if p.requires_grad], lr=args.lr),
                                 dict(params=head.parameters(), lr=args.head_lr)], momentum=.9,
                                weight_decay=5e-4, nesterov=True)
    if getattr(args, 'shared', False):
        coefficient_params = [m.coeffs for m in modules]
        coefficient_ids = {id(p) for p in coefficient_params}
        optimizer.param_groups[0]['params'] = [p for p in optimizer.param_groups[0]['params'] if id(p) not in coefficient_ids]
        optimizer.add_param_group(dict(params=coefficient_params, lr=args.coefficient_lr, weight_decay=0.))
    start, best = 0, -1.
    if args.resume:
        state = torch.load(args.resume, map_location='cpu', weights_only=False)
        if state['provenance'] != prepared['metadata']:
            raise ValueError('resume checkpoint has different preparation provenance')
        for key in ('seed', 'head_warmup', 'conversion_epochs', 'epochs', 'lr', 'head_lr', 'range_weight', 'global_batch', 'shared', 'initialization', 'coefficient_lr', 'batchnorm_mode', 'inference_bound', 'all_quadratic_start', 'unclipped_continuation', 'unclip_epochs', 'curvature_cap'):
            if state['config'].get(key, vars(args)[key]) != vars(args)[key]:
                raise ValueError(f'resume changes fixed training policy: {key}')
        student.load_state_dict(state['state_dict_backbone'], strict=True)
        head.load_state_dict(state['head'])
        optimizer.load_state_dict(state['optimizer'])
        start, best = state['epoch']+1, state['best']
    model = DDP(student, device_ids=[device.index] if device.type == 'cuda' else None, broadcast_buffers=False) if world > 1 else student
    classifier = DDP(head, device_ids=[device.index] if device.type == 'cuda' else None) if world > 1 else head
    train_rows = split['train']
    if args.smoke and len(train_rows) < 2*args.global_batch:
        train_rows = np.tile(train_rows, math.ceil(2*args.global_batch/len(train_rows)))
    sampler = DistributedSampler(train_rows, num_replicas=world, rank=rank, shuffle=True, seed=args.seed, drop_last=True)
    batches = loader(args, train_rows, split['labels'], sampler=sampler)
    if args.global_batch != args.batch_size*world:
        raise ValueError('recipe A requires global_batch = batch_size * world_size (no accumulation)')
    if rank == 0:
        (root/'train_config.json').write_text(json.dumps(vars(args), indent=2))
    if start == 0 and not args.smoke:
        baseline = validate(teacher, args, rank, world, device, split)
        teacher_hints.clear()
        if rank == 0:
            (root/'teacher_development.json').write_text(json.dumps(baseline, indent=2))
            print(f'TEACHER_DEVELOPMENT {baseline}', flush=True)
    replay = TrainingTailReplay()
    smoke_initial = student.conv1.weight.detach().clone() if args.smoke else None
    for epoch in range(start, args.epochs):
        torch.manual_seed(args.seed+epoch*world+rank)
        np.random.seed(args.seed+epoch*world+rank)
        sampler.set_epoch(epoch)
        student.train()
        steps = min(len(batches), args.limit_batches) if args.limit_batches else len(batches)
        for step, (images, labels, _) in enumerate(batches):
            if step >= steps:
                break
            if args.deadline and time.time() >= args.deadline:
                raise TimeoutError('experiment wall-clock deadline reached')
            progress = epoch+step/steps
            if args.smoke:
                progress = args.head_warmup+.1+step*.05
            alphas, clipped = apply_phase(student, progress, args)
            factor = .05+.95*.5*(1+math.cos(math.pi*progress/args.epochs))
            optimizer.param_groups[0]['lr'] = (0. if progress < args.head_warmup else args.lr*factor)
            optimizer.param_groups[1]['lr'] = args.head_lr*factor
            if getattr(args, 'shared', False):
                optimizer.param_groups[2]['lr'] = args.coefficient_lr*factor
            images, labels = images.to(device), mapping[labels.to(device)]
            if bool((labels < 0).any()):
                raise ValueError('development identity reached training')
            images, mask = prepare_range_batch(images, pathological_fraction=.02,
                                               crop_probability=.1, lowres_probability=.2,
                                               photo_probability=.2, stress_probability=.1)
            images, labels, mask = replay.inject(images, labels, mask)
            optimizer.zero_grad(set_to_none=True)
            # FP32 is intentional for both optimization and deployment consistency.
            with torch.no_grad():
                target = teacher(images)
            features = model(images)
            require_finite(features, device, 'training embeddings')
            classification = classifier(features, labels, mask, margin=.5*min(1., progress+0.1))
            kd, boundary = recipe_loss(features, target, student_hints, teacher_hints, mask, modules,
                                       elements, hint_weight=.3*(1-progress/args.epochs))
            beta = args.range_weight*min(1., (progress+.05)/2.)
            loss = classification+kd+beta*boundary
            require_finite(loss, device, 'training loss')
            loss.backward()
            parameters = [p for group in optimizer.param_groups for p in group['params']]
            bad = torch.tensor(int(any(p.grad is not None and not torch.isfinite(p.grad).all() for p in parameters)), device=device)
            reduce(bad, dist.ReduceOp.MAX)
            if bad.item():
                raise FloatingPointError('non-finite gradients; aborting without optimizer update')
            clip_grad_norm_stable(parameters, 5., error_if_nonfinite=True)
            optimizer.step()
            if getattr(args, 'unclipped_continuation', False):
                project_coefficients(student, args.curvature_cap)
            replay.update(images, labels, mask, modules)
            if rank == 0 and step % 25 == 0:
                print(f'epoch={epoch} step={step}/{steps} loss={loss.item():.4f} arc={classification.item():.4f} kd={kd.item():.4f} range={boundary.item():.4f} alpha={alphas} clipped={clipped}', flush=True)
            student_hints.clear()
            teacher_hints.clear()
            if args.smoke and step == 1:
                delta = (student.conv1.weight.detach()-smoke_initial).abs().max()
                if not bool(delta > 0):
                    raise RuntimeError('smoke did not update backbone weights')
                if rank == 0:
                    save_checkpoint(str(root/'smoke.pt'), student, prepared['calibration'], teacher_weights=args.teacher,
                                    extra=dict(pure_quadratic=False, diagnostic_only=True))
                    print(f'SMOKE_OK: world={world} batch={args.batch_size} two optimizer updates; backbone_delta={delta.item():.3g}; checkpoint exported', flush=True)
                barrier()
                return
        full = getattr(args, 'all_quadratic_start', False) or epoch >= args.head_warmup+args.conversion_epochs
        if getattr(args, 'unclipped_continuation', False):
            full = epoch >= args.unclip_epochs
        result = validate(student, args, rank, world, device, split) if full else {'phase': 'conversion'}
        improved = full and result['nonfinite'] == 0 and result['tar_1e4'] > best
        if improved:
            best = result['tar_1e4']
        if rank == 0:
            extra = dict(epoch=epoch, best=best, development=result, pure_quadratic=full,
                         origin='shared_degree2_warm_start' if warm_start_info else 'teacher_only_recipe_a', config=vars(args), provenance=prepared['metadata'],
                         head=head.state_dict(), optimizer=optimizer.state_dict())
            if getattr(args, 'shared', False):
                bounded = bool(getattr(args, 'inference_bound', False))
                extra.update(network='r50_shared_d2_bounded' if bounded else 'r50_shared_d2',
                             format='fhe-fr/shared-degree2-v1', approximation_target='layer-shared fit to PReLU',
                             interval='per-layer [-radius, radius]', warm_start=warm_start_info,
                             all_activations_quadratic=full, pure_quadratic=full and not bounded,
                             inference_input_bounds=bounded, fhe_requires_comparisons=bounded)
            save_checkpoint(str(root/'last.tmp.pt'), student, prepared['calibration'], teacher_weights=args.teacher, extra=extra)
            os.replace(root/'last.tmp.pt', root/'last.pt')
            if improved:
                save_checkpoint(str(root/'student_best.tmp.pt'), student, prepared['calibration'], teacher_weights=args.teacher,
                                extra={k:v for k,v in extra.items() if k not in ('head', 'optimizer')})
                os.replace(root/'student_best.tmp.pt', root/'student_best.pt')
            with open(root/'metrics.jsonl', 'a') as stream:
                stream.write(json.dumps(dict(epoch=epoch, **result))+'\n')
            print(f'EPOCH_COMPLETE {epoch}: {result}', flush=True)
        barrier()
    for handle in handles:
        handle.remove()
    if rank == 0:
        print('TRAIN_COMPLETE', flush=True)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['prepare', 'train'])
    p.add_argument('--teacher', default='work_dirs/ms1mv3_r50/model.pt')
    p.add_argument('--dataset-root', default='ms1m-retinaface-t1')
    p.add_argument('--output', required=True)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--global-batch', type=int, default=2048)
    p.add_argument('--seed', type=int, default=20260911)
    p.add_argument('--dev-fraction', type=float, default=.02)
    p.add_argument('--dev-per-identity', type=int, default=6)
    p.add_argument('--center-images', type=int, default=4)
    p.add_argument('--calibration-images', type=int, default=100000)
    p.add_argument('--quantile', type=float, default=.9995)
    p.add_argument('--guard-band', type=float, default=1.5)
    p.add_argument('--reg-ratio', type=float, default=.6)
    p.add_argument('--head-warmup', type=float, default=1.)
    p.add_argument('--conversion-epochs', type=float, default=8.)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--lr', type=float, default=.004)
    p.add_argument('--head-lr', type=float, default=.02)
    p.add_argument('--range-weight', type=float, default=1.)
    p.add_argument('--negative-pairs', type=int, default=2000000)
    p.add_argument('--limit-batches', type=int, default=0, help='smoke test only')
    p.add_argument('--batchnorm-mode', choices=['frozen', 'train'], default='frozen')
    p.add_argument('--shared', action='store_true')
    p.add_argument('--initialization', choices=['fit', 'near_linear'], default='fit')
    p.add_argument('--coefficient-lr', type=float, default=.0001)
    p.add_argument('--deadline', type=float, default=0., help='absolute UTC Unix time; all ranks stop before updates')
    p.add_argument('--inference-bound', action='store_true', help='keep calibrated input clamps in inference; requires comparisons')
    p.add_argument('--all-quadratic-start', action='store_true')
    p.add_argument('--warm-start', help='reuse shared backbone/head with a new optimizer and policy')
    p.add_argument('--unclipped-continuation', action='store_true')
    p.add_argument('--unclip-epochs', type=float, default=0.)
    p.add_argument('--curvature-cap', type=float, default=0.)
    p.add_argument('--resume')
    p.add_argument('--smoke', action='store_true', help='isolated 64-identity preparation and two optimizer steps')
    args = p.parse_args()
    if args.unclipped_continuation:
        if not args.shared or not args.all_quadratic_start or args.inference_bound:
            p.error('unclipped continuation requires shared, all-quadratic, unbounded inference')
        if not 0 <= args.unclip_epochs < args.epochs or not math.isfinite(args.curvature_cap) or args.curvature_cap < 0:
            p.error('invalid unclipping duration or coefficient cap')
    if args.all_quadratic_start and args.head_warmup != 0:
        p.error('all-quadratic start requires zero head warmup')
    if (args.inference_bound or args.warm_start or args.all_quadratic_start) and not args.shared:
        p.error('bounded inference and warm starts require the shared model')
    if args.warm_start and args.resume:
        p.error('warm start and optimizer resume are mutually exclusive')
    if args.epochs <= 0 or args.conversion_epochs <= 0 or args.head_warmup < 0 or (not args.all_quadratic_start and args.epochs <= args.head_warmup+args.conversion_epochs):
        p.error('need positive conversion duration and a final pure-quadratic phase')
    if not 0 < args.quantile < 1 or args.guard_band <= 0 or not 0 < args.reg_ratio <= 1:
        p.error('invalid calibration quantile, guard band or regularization ratio')
    if min(args.batch_size, args.global_batch, args.center_images, args.calibration_images, args.negative_pairs) <= 0 or args.dev_per_identity < 2:
        p.error('positive sizes and at least two development images per identity are required')
    return args


def main():
    args = parse_args()
    rank, world, device = context()
    torch.set_num_threads(2)
    torch.manual_seed(args.seed+rank)
    (prepare if args.stage == 'prepare' else train)(args, rank, world, device)
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
