"""Paired MS1MV3-only conversion experiment. Run on one Slurm GPU.

This is a bounded mechanism test, not an IJB-C accuracy claim. Both arms
start from the original teacher. Only the adaptive arm refreshes upcoming
site statistics and waits for consecutive held-out stability/quality gates.
"""
import argparse
import copy
import json
import subprocess
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from controlled_degree2.adaptive_conversion import ConversionGate, ConversionQuadratic, tail_penalty
from controlled_degree2.calibrate import ChannelAbsHistogram, weighted_quadratic_abs_fit
from controlled_degree2.model import load_teacher, prelu_names, prelu_to_quadratic_coefficients
from controlled_degree2.recipe_a import SourceRows, IdentityHead, digest
from controlled_degree2.augment import prepare_range_batch
from eval.finite_audit import FiniteAudit


def write(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')


def batches(args, rows, labels, deterministic=True):
    generator = torch.Generator().manual_seed(args.seed)
    return DataLoader(SourceRows(args.dataset_root, rows, labels, deterministic),
                      batch_size=args.batch_size, num_workers=args.workers,
                      multiprocessing_context='spawn' if args.workers else None,
                      generator=generator)


@torch.no_grad()
def refit(args, model, site, rows, labels):
    hist = ChannelAbsHistogram(site.channels, 1e-5, 1e3, 512, site.coeffs.device)
    def observe(module, inputs):
        if not torch.isfinite(inputs[0]).all():
            raise FloatingPointError('nonfinite input while profiling student')
        hist.update(inputs[0][:, :, ::4, ::4])
    handle = site.register_forward_pre_hook(observe)
    try:
        for images, _, _ in batches(args, rows, labels):
            model(images.cuda())
    finally:
        handle.remove()
    probabilities = hist.counts/hist.counts.sum(1, keepdim=True).clamp_min(1)
    centers, widths = hist.bin_geometry()
    index = (probabilities.cumsum(1) < .9995).sum(1).clamp_max(hist.bins-1)
    edges = centers*10**((hist.log_max-hist.log_min)/(2*hist.bins))
    radius = (edges[index]*1.5).clamp_min(.05).float()
    even, error = weighted_quadratic_abs_fit(probabilities, centers, widths, radius, slope=site.slope)
    site.coeffs.copy_(prelu_to_quadratic_coefficients(site.slope, radius, even))
    site.lam_fit.copy_(radius)
    site.lam_reg.copy_(.6*radius)
    return dict(site=site.name, images=len(rows), interval='[-R_c,R_c]',
                target='original channelwise PReLU on current student inputs',
                radius=radius.cpu().tolist(), coefficients=site.coeffs.cpu().tolist(),
                median_fit_error=float(error.median()))


@torch.no_grad()
def probe(args, model, teacher, sites, rows, labels, current, cached_images=None):
    audit = FiniteAudit(model)
    peak = 0.
    def observe(module, inputs):
        nonlocal peak
        ratio = inputs[0].abs()/module.lam_fit.reshape(1, -1, 1, 1)
        peak = max(peak, float(ratio.max())) if torch.isfinite(ratio).all() else float('inf')
    handles = [q.register_forward_pre_hook(observe) for q in sites[:current+1]]
    total, count = 0., 0
    try:
        image_batches = (cached_images if cached_images is not None else
                         (images for images, _, _ in batches(args, rows, labels)))
        for images in image_batches:
            images = images.cuda()
            images = torch.cat((images, images.flip(-1)))
            output, target = model(images), teacher(images)
            kd = 1-F.cosine_similarity(output, target)
            total += float(kd.sum()); count += len(images)
    finally:
        for handle in handles:
            handle.remove()
        result = audit.result(); audit.close()
    return dict(kd=total/count, max_ratio=peak, nonfinite=result['nonfinite_values'],
                augmented_rows=count, audited_rows=result['boundaries']['.output']['observed_rows'])


def run_arm(args, arm, prepared, split, root):
    torch.manual_seed(args.seed)
    teacher = load_teacher(args.teacher, torch.device('cuda')).eval().requires_grad_(False)
    model = copy.deepcopy(teacher)
    model.requires_grad_(True)
    names = prelu_names(teacher)
    if len(names) != 25:
        raise ValueError('expected original 25-PReLU teacher')
    sites = []
    for name in names:
        original = teacher.get_submodule(name)
        c = prepared['calibration'][name]
        coefficients = prelu_to_quadratic_coefficients(original.weight, c['lam_fit'], c['even_coeffs'])
        q = ConversionQuadratic(original.weight.numel(), c['lam_fit'], c['lam_reg'],
                                coefficients, original.weight.detach(), name).cuda()
        q.alpha = 0.; q.clip = q.clip_eval = False
        q.coeffs.requires_grad_(True)
        parent, _, key = name.rpartition('.')
        setattr(model.get_submodule(parent), key, q); sites.append(q)
    model.eval()  # Frozen BN moments and no dropout; hybrid remains active in eval.
    active = torch.tensor(prepared['active_ids'], device='cuda')
    head = IdentityHead(prepared['centers'][prepared['active_ids']]).cuda()
    mapping = torch.full((len(prepared['centers']),), -1, device='cuda', dtype=torch.long)
    mapping[active] = torch.arange(len(active), device='cuda')
    coefficient_ids = {id(q.coeffs) for q in sites}
    optimizer = torch.optim.SGD([
        dict(params=[p for p in model.parameters() if id(p) not in coefficient_ids], lr=.001),
        dict(params=head.parameters(), lr=.005),
        dict(params=[q.coeffs for q in sites], lr=.00002, weight_decay=0.)],
        momentum=.9, nesterov=True, weight_decay=.0005)
    train_rows, labels = split['train'], split['labels']
    rng = np.random.default_rng(args.seed)
    calibration_rows = rng.permutation(train_rows)[:args.calibration_images]
    gate_rows = rng.permutation(split['dev'])[:args.gate_images]
    # Gate images are deterministic. Reuse their CPU tensors rather than
    # launching RecordIO workers at every check; this does not change probes.
    gate_images = [images for images, _, _ in batches(args, gate_rows, labels)]
    # Fixed sequence shared by both arms, regardless of conversion speed.
    sample_rows = rng.choice(train_rows, args.sites*4*args.max_phase_steps*args.batch_size)
    iterator = iter(batches(args, sample_rows, labels, deterministic=False))
    student_hints, teacher_hints = {}, {}
    hooks = []
    for i in range(1, 5):
        hooks.append(model.get_submodule(f'layer{i}').register_forward_hook(
            lambda m, x, y, i=i: student_hints.__setitem__(i, y)))
        hooks.append(teacher.get_submodule(f'layer{i}').register_forward_hook(
            lambda m, x, y, i=i: teacher_hints.__setitem__(i, y)))
    records, profiles = [], []
    step = 0; converted = 0; status = 'pilot_completed'
    replay_images = replay_labels = replay_scores = None
    for index in range(args.sites):
        if arm == 'adaptive':
            profiles.append(refit(args, model, sites[index], calibration_rows, labels))
            optimizer.state.pop(sites[index].coeffs, None)
        baseline = probe(args, model, teacher, sites, gate_rows, labels, index, gate_images)
        if baseline['nonfinite'] or not np.isfinite([baseline['kd'], baseline['max_ratio']]).all():
            status = 'unsafe_baseline'; break
        for alpha in (.25, .5, .75, 1.):
            gate = ConversionGate(baseline['kd'], baseline['max_ratio'])
            sites[index].alpha = alpha
            safe_state = copy.deepcopy(model.state_dict())
            passed = False
            phase_steps = args.min_phase_steps if arm == 'fixed' else args.max_phase_steps
            for local_step in range(1, phase_steps+1):
                images, targets, _ = next(iterator)
                images, targets = images.cuda(), mapping[targets.cuda()]
                assert (targets >= 0).all()
                torch.manual_seed(args.seed+step)
                images, mask = prepare_range_batch(images, pathological_fraction=0.,
                    crop_probability=.1, lowres_probability=.2, photo_probability=.2, stress_probability=0.)
                if replay_images is not None:
                    n = min(len(images)//16, len(replay_images))
                    images[:n], targets[:n] = replay_images[:n], replay_labels[:n]
                penalties, scores = [], []
                def observe(module, inputs):
                    loss, peak = tail_penalty(inputs[0], module.lam_fit)
                    penalties.append(loss); scores.append(peak)
                handles = [q.register_forward_pre_hook(observe) for q in sites[:index+1]]
                optimizer.zero_grad(set_to_none=True)
                try:
                    with torch.no_grad():
                        target = teacher(images)
                    output = model(images)
                    if not torch.isfinite(output).all():
                        raise FloatingPointError('nonfinite embedding')
                    arc = head(output, targets, mask)
                    kd = (1-F.cosine_similarity(output, target)).mean()
                    hint = torch.stack([((student_hints[i]-teacher_hints[i]).square().flatten(1).mean(1)
                        /teacher_hints[i].square().flatten(1).mean(1).clamp_min(1e-6)).mean() for i in range(1, 5)]).mean()
                    tail = torch.stack(penalties).mean()
                    loss = arc+kd+.3*hint+tail
                    if not torch.isfinite(loss):
                        raise FloatingPointError('nonfinite loss')
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(list(model.parameters())+list(head.parameters()), 5., error_if_nonfinite=True)
                    optimizer.step()
                    if any(not torch.isfinite(p).all() for p in model.parameters()):
                        raise FloatingPointError('nonfinite parameter update')
                except (FloatingPointError, RuntimeError) as error:
                    model.load_state_dict(safe_state)
                    sites[index].alpha = alpha-.25
                    status = 'stopped_on_failure'
                    records.append(dict(step=step, site=index, alpha=alpha, failure=str(error)))
                    break
                finally:
                    for handle in handles:
                        handle.remove()
                score = torch.stack(scores).amax(0)
                candidates = (images.detach(), targets.detach(), score)
                if replay_images is not None:
                    candidates = tuple(torch.cat((old, new)) for old, new in
                        zip((replay_images, replay_labels, replay_scores), candidates))
                chosen = candidates[2].topk(min(32, len(candidates[2]))).indices
                replay_images, replay_labels, replay_scores = [x[chosen].clone() for x in candidates]
                step += 1
                student_hints.clear(); teacher_hints.clear()
                if local_step % args.check_every == 0:
                    report = probe(args, model, teacher, sites, gate_rows, labels, index, gate_images)
                    finite = np.isfinite([report['kd'], report['max_ratio']]).all()
                    ready = gate.observe(report)
                    record = dict(step=step, site=index, alpha=alpha, phase_step=local_step,
                                  gate_ready=ready, arc=float(arc), train_kd=float(kd), tail=float(tail),
                                  probe=report if finite else dict(nonfinite=report['nonfinite'], invalid_metrics=True))
                    records.append(record); print(arm, json.dumps(record), flush=True)
                    write(root/f'{arm}_progress.json', records)
                    if not finite or report['nonfinite']:
                        model.load_state_dict(safe_state); sites[index].alpha = alpha-.25
                        status = 'stopped_on_failure'; break
                    if ready and local_step >= args.min_phase_steps:
                        passed = True
                        if arm == 'adaptive': break
            if status != 'pilot_completed': break
            if arm == 'adaptive' and not passed:
                status = 'gate_stalled'; break
        if status != 'pilot_completed': break
        converted += 1
    for hook in hooks: hook.remove()
    final = probe(args, model, teacher, sites, gate_rows, labels, index, gate_images)
    for key in ('kd', 'max_ratio'):
        if not np.isfinite(final[key]):
            final[key] = None
    # A stopped hybrid is diagnostic only; never masquerade as a pure checkpoint.
    torch.save(dict(state_dict_backbone=model.state_dict(), alphas=[q.alpha for q in sites],
                    pure_quadratic=False, diagnostic_only=True, config=vars(args),
                    teacher_sha256=digest(args.teacher)), root/f'{arm}_diagnostic.pt')
    report = dict(arm=arm, status=status, converted_sites=converted, steps=step,
                  pure_quadratic=False, ijbc_evaluated=False, final_probe=final,
                  profiles=profiles, records=records)
    write(root/f'{arm}_result.json', report)
    return {k:v for k,v in report.items() if k not in ('records', 'profiles')}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    p.add_argument('--teacher', default='work_dirs/ms1mv3_r50/model.pt')
    p.add_argument('--prepared-root', default='work_dirs/channelwise_ijbc96_20260914')
    p.add_argument('--dataset-root', default='ms1m-retinaface-t1')
    p.add_argument('--sites', type=int, default=2)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--calibration-images', type=int, default=2048)
    p.add_argument('--gate-images', type=int, default=512)
    p.add_argument('--min-phase-steps', type=int, default=100)
    p.add_argument('--max-phase-steps', type=int, default=300)
    p.add_argument('--check-every', type=int, default=25)
    p.add_argument('--seed', type=int, default=20260928)
    args = p.parse_args()
    if not 1 <= args.sites <= 24 or args.max_phase_steps < args.min_phase_steps:
        p.error('pilot requires 1..24 sites and max phase steps >= min')
    if min(args.batch_size, args.calibration_images, args.gate_images, args.min_phase_steps, args.check_every) <= 0:
        p.error('sizes and steps must be positive')
    if args.min_phase_steps % args.check_every or args.max_phase_steps % args.check_every:
        p.error('phase limits must be multiples of check interval')
    root = Path(args.output); root.mkdir(parents=True, exist_ok=False)
    prepared_root = Path(args.prepared_root)
    prepared = torch.load(prepared_root/'prepared.pt', map_location='cpu', weights_only=False)
    assert prepared['metadata']['teacher_sha256'] == digest(args.teacher)
    assert prepared['metadata']['split_sha256'] == digest(prepared_root/'split.npz')
    with np.load(prepared_root/'split.npz') as z:
        split = {k:z[k] for k in z.files}
    write(root/'config.json', dict(config=vars(args), code_commit=subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], text=True).strip(), teacher_sha256=digest(args.teacher),
        prepared_sha256=digest(prepared_root/'prepared.pt'), split_sha256=digest(prepared_root/'split.npz'),
        uses_ijbc_images=False, uses_ijbc_pair_labels=False,
        comparison='fixed teacher fit/time schedule versus refreshed student fit/consecutive quality gates'))
    results = []
    for arm in ('fixed', 'adaptive'):
        results.append(run_arm(args, arm, prepared, split, root))
        torch.cuda.empty_cache()
    write(root/'comparison.json', results)
    print(json.dumps(results, indent=2), flush=True)


if __name__ == '__main__':
    main()
