"""Read-only, single-GPU replay of an exact captured training failure.

Compare captured loss arithmetic with active-row FP64 diagnostics. An optional
backward check on identity rows computes gradients without updating weights.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.nn import functional as F


def scalar(value):
    value = float(value)
    return value if torch.isfinite(torch.tensor(value, dtype=torch.float64)) else str(value)


def hint_arithmetic(student, teacher, mask):
    reports = {}
    for dtype, name in ((torch.float32, 'fp32'), (torch.float64, 'fp64')):
        s, t = student.to(dtype), teacher.to(dtype)
        per_row = (s-t).square().flatten(1).mean(1)/t.square().flatten(1).mean(1).clamp_min(1e-6)
        reports[name] = dict(
            nonfinite_rows=int((~torch.isfinite(per_row)).sum()),
            nonfinite_active_rows=int((~torch.isfinite(per_row[mask])).sum()),
            nonfinite_row_indices=(~torch.isfinite(per_row)).nonzero().flatten().tolist(),
            legacy_masked_mean=scalar((per_row*mask.to(dtype)).sum()/mask.sum().clamp_min(1)),
            active_mean=scalar(per_row[mask].mean()) if bool(mask.any()) else 0.,
        )
    return reports


def legacy_recipe_loss(features, target, student_hints, teacher_hints, mask, modules, elements, hint_weight):
    weights = mask.float()
    cosine = ((1-F.cosine_similarity(features.float(), target.float(), dim=1))*weights).sum()/weights.sum().clamp_min(1)
    hints = []
    for stage in student_hints:
        s, t = student_hints[stage].float(), teacher_hints[stage].float()
        error = (s-t).square().flatten(1).mean(1)/t.square().flatten(1).mean(1).clamp_min(1e-6)
        hints.append((error*weights).sum()/weights.sum().clamp_min(1))
    boundary = torch.stack([m.last_penalty/elements[m.name]+.01*m.last_sample_penalty.mean() for m in modules]).mean()
    return cosine+hint_weight*torch.stack(hints).mean(), boundary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state', required=True)
    parser.add_argument('--batch', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--check-active-backward', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    device = torch.device('cuda')
    from controlled_degree2.model import load_controlled_checkpoint, load_teacher, quadratic_modules
    from controlled_degree2.recipe_a import IdentityHead, apply_phase, hook_stages, recipe_loss, digest
    from eval.finite_audit import FiniteAudit
    model, source = load_controlled_checkpoint(args.state, device)
    batch = torch.load(args.batch, map_location='cpu', weights_only=False)
    if source['step'] != batch['report']['step'] or source['epoch'] != batch['report']['epoch']:
        raise ValueError('batch and checkpoint belong to different failure steps')
    config = SimpleNamespace(**source['config'])
    if digest(config.teacher) != source['provenance']['teacher_sha256']:
        raise ValueError('teacher checkpoint changed')
    teacher = load_teacher(config.teacher, device).eval()
    head = IdentityHead(source['head']['weight']).to(device)
    images, labels, mask = [batch[n].to(device) for n in ('images', 'labels', 'mask')]
    student_hints, teacher_hints, elements, ranges = {}, {}, {}, {}
    modules = list(quadratic_modules(model))
    handles = hook_stages(model, student_hints)+hook_stages(teacher, teacher_hints)
    def observe(module, inputs):
        x = inputs[0]
        elements[module.name] = x[0].numel()
        values = {}
        for dtype, name in ((torch.float32, 'fp32'), (torch.float64, 'fp64')):
            excess = (x.to(dtype).abs()/module.lam_reg.to(dtype).reshape(1, -1, 1, 1)-1).clamp_min(0)
            values[name] = dict(mean_square=scalar(excess.square().mean()),
                                legacy_sum_then_scale=scalar(excess.square().sum()/x.shape[0]/x[0].numel()))
        ratios = (x.double().abs()/module.lam_fit.double().reshape(1, -1, 1, 1)).flatten(1).amax(1)
        row = int(ratios.argmax())
        values['largest_fit_ratio'] = dict(row=row, ratio=scalar(ratios[row]), identity_loss_enabled=bool(mask[row]))
        ranges[module.name] = values
    handles += [m.register_forward_pre_hook(observe) for m in modules]
    model.train()
    apply_phase(model, source['progress'], config)
    audit = FiniteAudit(model)
    try:
        with torch.no_grad():
            target = teacher(images)
            features = model(images)
            classification = head(features, labels, mask, margin=.5*min(1., source['progress']+.1))
            captured_loss = recipe_loss if source['config'].get('loss_masking') == 'active_rows' else legacy_recipe_loss
            kd, boundary = captured_loss(features, target, student_hints, teacher_hints,
                                       mask, modules, elements, .3*(1-source['progress']/config.epochs))
            beta = config.range_weight*min(1., (source['progress']+.05)/2.)
            report = dict(
                state_sha256=digest(args.state), batch_sha256=digest(args.batch),
                captured_report=batch['report'],
                replay=dict(loss=scalar(classification+kd+beta*boundary), classification=scalar(classification),
                            kd=scalar(kd), boundary=scalar(boundary)),
                inference_audit=audit.result(),
                embedding_fp32_nonfinite_norms=int((~torch.isfinite(features.norm(dim=1))).sum()),
                embedding_fp64_nonfinite_norms=int((~torch.isfinite(features.double().norm(dim=1))).sum()),
                hints={str(n): hint_arithmetic(s, teacher_hints[n], mask) for n, s in student_hints.items()},
                ranges=ranges,
                interpretation='arithmetic diagnosis only; no update and no inference certification',
            )
    finally:
        for handle in handles:
            handle.remove()
        audit.close()
    if args.check_active_backward:
        for module in modules:
            module.coeffs.requires_grad_(bool(getattr(config, 'train_channel_coefficients', False)))
        student_hints.clear()
        teacher_hints.clear()
        handles = hook_stages(model, student_hints)+hook_stages(teacher, teacher_hints)
        try:
            active_images, active_labels = images[mask], labels[mask]
            if not len(active_images):
                raise ValueError('captured batch contains no identity rows')
            with torch.no_grad():
                active_target = teacher(active_images)
            active_features = model(active_images)
            active_mask = torch.ones(len(active_images), dtype=torch.bool, device=device)
            ce = head(active_features, active_labels, active_mask, margin=.5*min(1., source['progress']+.1))
            kd, boundary = recipe_loss(active_features, active_target, student_hints, teacher_hints,
                                        active_mask, modules, elements, .3*(1-source['progress']/config.epochs))
            loss = ce+kd+beta*boundary
            result = dict(rows=len(active_images), loss=scalar(loss.detach()), kd=scalar(kd.detach()),
                          boundary=scalar(boundary.detach()), optimizer_updates=0)
            if bool(torch.isfinite(loss)):
                loss.backward()
                named = list(model.named_parameters())+[('head.'+n,p) for n,p in head.named_parameters()]
                result['nonfinite_gradients'] = [n for n,p in named if p.grad is not None and not torch.isfinite(p.grad).all()]
                result['gradient_tensors'] = sum(p.grad is not None for _,p in named)
            report['active_only_backward'] = result
        finally:
            for handle in handles:
                handle.remove()
    Path(args.output).write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(report['replay'], indent=2))


if __name__ == '__main__':
    main()
