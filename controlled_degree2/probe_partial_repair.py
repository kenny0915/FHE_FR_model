"""Bounded GPU experiment repairing a captured partial-conversion input.

Only existing preactivation BN affines change. The exact conversion phase
is preserved; no activation is clipped. This produces diagnostic evidence,
not a completed student or a resumable training checkpoint.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from controlled_degree2.model import load_controlled_checkpoint, load_teacher, quadratic_modules
from controlled_degree2.recipe_a import apply_phase, digest
from controlled_degree2.recipe_a_recovery import affine_parameters, finite_prefix_loss
from utils.utils_optimizer import clip_grad_norm_stable


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--state', required=True)
    p.add_argument('--batch', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--row', type=int, required=True)
    p.add_argument('--steps', type=int, default=200)
    p.add_argument('--guard', type=float, default=4.)
    p.add_argument('--target', type=float, default=2.)
    p.add_argument('--lr', type=float, default=.01)
    args = p.parse_args()
    if args.steps <= 0 or args.lr <= 0 or not 0 < args.target < args.guard:
        p.error('require positive steps/LR and 0 < target < guard')
    root = Path(args.output)
    root.mkdir(exist_ok=False)
    torch.set_num_threads(2)
    model, source = load_controlled_checkpoint(args.state, torch.device('cuda'))
    batch = torch.load(args.batch, map_location='cpu', weights_only=False)
    if source['step'] != batch['report']['step'] or source['epoch'] != batch['report']['epoch']:
        raise ValueError('batch/state mismatch')
    images = batch['images'].to('cuda')
    row = images[args.row:args.row+1]
    if len(row) != 1:
        raise ValueError('invalid row')
    params = affine_parameters(model)
    model.train()
    apply_phase(model, source['progress'], SimpleNamespace(**source['config']))
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.eval()
    frozen = {n: t.detach().clone() for n, t in model.state_dict().items()
              if n not in {n for n, p in model.named_parameters() if p.requires_grad}}
    optimizer = torch.optim.SGD(params, lr=args.lr)
    teacher_path = source['config']['teacher']
    if digest(teacher_path) != source['provenance']['teacher_sha256']:
        raise ValueError('teacher provenance mismatch')
    teacher = load_teacher(teacher_path, torch.device('cuda')).eval().requires_grad_(False)
    with torch.no_grad():
        target = teacher(images)
        original = model(images)
        original_valid = torch.isfinite(original).all(1) & torch.isfinite(original.norm(dim=1))
    del teacher
    def audit():
        with torch.no_grad():
            out = model(images)
            valid = torch.isfinite(out).all(1) & torch.isfinite(out.norm(dim=1))
            common = valid & original_valid
            cosine = torch.nn.functional.cosine_similarity
            return dict(nonfinite_rows=int((~torch.isfinite(out).all(1)).sum()),
                        nonfinite_norm_rows=int((~torch.isfinite(out.norm(dim=1))).sum()),
                        teacher_comparison_rows=int(valid.sum()),
                        teacher_cosine=float(cosine(out[valid], target[valid]).mean()) if valid.any() else None,
                        original_valid_teacher_cosine=float(cosine(out[common], target[common]).mean()) if common.any() else None,
                        original_valid_embedding_cosine=float(cosine(out[common], original[common]).mean()) if common.any() else None)
    report = dict(config=vars(args), source_sha256=digest(args.state), before=audit(), steps=[])
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        loss, site, ratios = finite_prefix_loss(model, row, 25, args.guard, args.target,
                                                preserve_phase=True)
        if site is None:
            report['guard_passed'] = True
            break
        if not torch.isfinite(loss):
            raise FloatingPointError('nonfinite repair loss')
        loss.backward()
        clip_grad_norm_stable(params, 1., error_if_nonfinite=True)
        optimizer.step()
        if step % 10 == 0 or step == args.steps-1:
            report['steps'].append(dict(step=step, site=site, loss=float(loss.detach()),
                                       max_ratio=float(ratios.max())))
    report['after'] = audit()
    report['frozen_tensors_unchanged'] = all(torch.equal(model.state_dict()[n], t) for n,t in frozen.items())
    report['inference_clipping'] = any(m.clip or m.clip_eval for m in quadratic_modules(model))
    if not report['frozen_tensors_unchanged'] or report['inference_clipping']:
        raise RuntimeError('repair changed forbidden state')
    (root/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
