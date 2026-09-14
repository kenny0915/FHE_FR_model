"""Read-only, single-GPU replay of an exact captured training failure.

Compare legacy FP32 loss arithmetic with active-row FP64 diagnostics; no
parameters, optimizer state, gradients or training policy are changed.
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state', required=True)
    parser.add_argument('--batch', required=True)
    parser.add_argument('--output', required=True)
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
            kd, boundary = recipe_loss(features, target, student_hints, teacher_hints,
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
    Path(args.output).write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(report['replay'], indent=2))


if __name__ == '__main__':
    main()
