"""Fit a regularized teacher alignment and fold it into the existing FC.

IJB-C images only; no identity or pair labels. The source backbone is frozen.
Source-image-disjoint validation selects ridge strength by teacher cosine.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from controlled_degree2.model import load_controlled_checkpoint, load_teacher, save_checkpoint
from controlled_degree2.recipe_a import digest
from controlled_degree2.recipe_a_recovery import configure
from controlled_degree2.recipe_a_recovery_v2 import sitewise


def fit_alignment(source, teacher, ridge):
    """Identity-anchored weighted least squares, including an affine offset."""
    if ridge <= 0:
        raise ValueError('positive ridge required')
    x, y = source.double(), teacher.double()
    if x.ndim != 2 or x.shape != y.shape or not torch.isfinite(x).all() or not torch.isfinite(y).all():
        raise ValueError('finite matching embedding matrices required')
    norm = x.norm(dim=1, keepdim=True)
    if (norm <= 0).any() or (y.norm(dim=1) <= 0).any():
        raise ValueError('nonzero embeddings required')
    design = torch.cat((x, torch.ones_like(norm)), dim=1)/norm
    residual = F.normalize(y, dim=1) - x/norm
    gram = design.T @ design / len(x)
    cross = design.T @ residual / len(x)
    scale = gram.diagonal().mean()
    delta = torch.linalg.solve(gram + ridge*scale*torch.eye(gram.shape[0], device=x.device), cross)
    return torch.eye(x.shape[1], device=x.device, dtype=x.dtype) + delta[:-1].T, delta[-1]


@torch.no_grad()
def fold_alignment(model, matrix, bias):
    """Compose output affine with FC and fixed BN, preserving every BN buffer."""
    fc, bn = model.fc, model.features
    if fc.bias is None or bn.training:
        raise ValueError('biased FC and evaluation-mode output BN required')
    scale = bn.weight.double()/torch.sqrt(bn.running_var.double()+bn.eps)
    if (scale.abs() < 1e-12).any():
        raise ValueError('output BN has a singular scale')
    offset = bn.bias.double()-bn.running_mean.double()*scale
    weight = scale[:, None]*fc.weight.double()
    intercept = scale*fc.bias.double()+offset
    new_weight = (matrix.double() @ weight)/scale[:, None]
    new_bias = (matrix.double() @ intercept+bias.double()-offset)/scale
    new_weight, new_bias = new_weight.to(fc.weight), new_bias.to(fc.bias)
    if not torch.isfinite(new_weight).all() or not torch.isfinite(new_bias).all():
        raise FloatingPointError('non-finite folded FC')
    fc.weight.copy_(new_weight)
    fc.bias.copy_(new_bias)


def source_split(count, fit_images, validation_images, seed):
    if min(fit_images, validation_images) <= 0 or fit_images+validation_images > count:
        raise ValueError('positive disjoint source-image split required')
    ids = np.random.default_rng(seed).choice(count, fit_images+validation_images, replace=False)
    return ids[:fit_images], ids[fit_images:]


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--teacher', default='work_dirs/ms1mv3_r50/model.pt')
    p.add_argument('--ijb-root', default='ijb/IJBC')
    p.add_argument('--fit-images', type=int, default=32768)
    p.add_argument('--validation-images', type=int, default=8192)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--seed', type=int, default=20260925)
    args = p.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    root = Path(args.output)
    root.mkdir(exist_ok=False, parents=True)
    device = torch.device('cuda')
    model, state = load_controlled_checkpoint(args.checkpoint, device)
    if not state.get('pure_quadratic') or state['provenance']['teacher_sha256'] != digest(args.teacher):
        raise ValueError('pure quadratic original-teacher-derived source required')
    sitewise(model)
    if len(model._recovery_sites) != 25:
        raise ValueError('expected 25 quadratic sites')
    configure(model, 25)
    model.requires_grad_(False)
    teacher = load_teacher(args.teacher, device).eval().requires_grad_(False)
    from utils.utils_ijbc_replay import IJBCSourceDataset, IJBCOrientationDataset
    fit_ids, val_ids = source_split(len(IJBCSourceDataset(args.ijb_root)),
                                   args.fit_images, args.validation_images, args.seed)
    np.savez(root/'split.npz', fit_source_images=fit_ids, validation_source_images=val_ids)
    config = dict(config=vars(args), source_sha256=digest(args.checkpoint),
                  teacher_sha256=digest(args.teacher), split_sha256=digest(root/'split.npz'),
                  uses_ijbc_pair_labels=False, inference_clipping=False,
                  selection='minimum held-out source-image teacher cosine loss; identity is a candidate',
                  approximation='unchanged per-channel PReLU fit and lam_fit intervals')
    (root/'calibration_config.json').write_text(json.dumps(config, indent=2))
    cache = {}
    for name, ids in [('fit', fit_ids), ('validation', val_ids)]:
        records = [(int(i), o) for i in ids for o in (0, 1)]
        loader = DataLoader(IJBCOrientationDataset(args.ijb_root, records), batch_size=args.batch_size,
                            num_workers=4, multiprocessing_context='spawn')
        xs, ys = [], []
        for step, (images, _, _) in enumerate(loader):
            images = images.to(device)
            x, y = model(images), teacher(images)
            if not torch.isfinite(x).all() or not torch.isfinite(y).all():
                raise FloatingPointError('non-finite calibration embedding')
            xs.append(x.cpu()); ys.append(y.cpu())
            if step % 25 == 0:
                print(name, step, 'batches', flush=True)
        cache[name] = (torch.cat(xs), torch.cat(ys))
    torch.save(cache, root/'embedding_cache.pt')
    x, y = (a.to(device) for a in cache['fit'])
    vx, vy = (a.to(device).double() for a in cache['validation'])
    baseline = float((1-F.cosine_similarity(vx, vy)).mean())
    best_loss, best = baseline, None
    rows = []
    for ridge in (1e-4, 1e-3, .01, .1, 1., 10., 100.):
        matrix, bias = fit_alignment(x, y, ridge)
        prediction = vx @ matrix.T + bias
        loss = float((1-F.cosine_similarity(prediction, vy)).mean())
        rows.append(dict(ridge=ridge, validation_cosine_loss=loss))
        if loss < best_loss:
            best_loss, best = loss, (ridge, matrix, bias)
    report = dict(baseline_validation_cosine_loss=baseline, candidates=rows,
                  improved=best is not None, selected_ridge=None if best is None else best[0])
    (root/'selection.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    if best is None:
        return
    fold_alignment(model, best[1], best[2])
    changed = [n for n, t in model.state_dict().items()
               if not torch.equal(t.cpu(), state['state_dict_backbone'][n])]
    if not changed or not set(changed) <= {'fc.weight', 'fc.bias'}:
        raise RuntimeError(f'unexpected state changes: {changed}')
    check_records = [(int(i), o) for i in val_ids[:32] for o in (0, 1)]
    check_images = next(iter(DataLoader(IJBCOrientationDataset(args.ijb_root, check_records),
                                       batch_size=len(check_records))))[0].to(device)
    actual = model(check_images).double()
    expected = vx[:len(check_records)] @ best[1].T + best[2]
    torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-4)
    report['fold_check_rows'] = len(check_records)
    report['fold_max_abs_error'] = float((actual-expected).abs().max())
    (root/'selection.json').write_text(json.dumps(report, indent=2))
    torch.save(dict(matrix=best[1].cpu(), bias=best[2].cpu()), root/'alignment.pt')
    save_checkpoint(str(root/'candidate.pt'), model, state['poly_calib'], teacher_weights=args.teacher,
                    extra=dict(origin='channelwise_linear_calibration', pure_quadratic=True,
                               provenance=state['provenance'], calibration=config,
                               selection=report, diagnostic_only=True))
    print('CANDIDATE_SAVED: requires full exported IJB-C evaluation', flush=True)


if __name__ == '__main__':
    main()
