"""Unlabeled teacher pair-geometry alignment using a verified embedding cache."""
import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from controlled_degree2.calibrate_linear_head import fold_alignment
from controlled_degree2.model import load_controlled_checkpoint, save_checkpoint
from controlled_degree2.recipe_a import digest
from controlled_degree2.recipe_a_recovery import configure
from controlled_degree2.recipe_a_recovery_v2 import sitewise


def pair_geometry_loss(output, teacher, source, source_ids, threshold=.3):
    """Balance all cross-image pairs and fixed teacher/source high-similarity pairs.

    Source and teacher masks are detached and fixed with respect to the learned
    mapping. Same-image views and diagonals are excluded. No labels are used.
    """
    out = F.normalize(output, dim=1)
    with torch.no_grad():
        target = F.normalize(teacher, dim=1)
        baseline = F.normalize(source, dim=1)
        target_sim = target @ target.T
        source_sim = baseline @ baseline.T
        valid = torch.triu(source_ids[:, None] != source_ids[None, :], diagonal=1)
        tail = valid & ((target_sim >= threshold) | (source_sim >= threshold))
    if not valid.any():
        raise ValueError('need pairs from different source images')
    error = (out @ out.T-target_sim).square()
    overall = error[valid].mean()
    high = error[tail].mean() if tail.any() else overall*0
    return overall+high, dict(all_mse=overall.detach(), tail_mse=high.detach(),
                              pairs=int(valid.sum()), tail_pairs=int(tail.sum()))


@torch.no_grad()
def evaluate(matrix, bias, source, teacher, source_ids):
    totals = dict(all_mse=0., tail_mse=0.)
    batches = 0
    for start in range(0, len(source), 1024):
        x, y, ids = source[start:start+1024], teacher[start:start+1024], source_ids[start:start+1024]
        _, stats = pair_geometry_loss(x @ matrix.T+bias, y, x, ids)
        for k in totals:
            totals[k] += float(stats[k])
        batches += 1
    return {k:v/batches for k,v in totals.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache-root', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--steps', type=int, default=2000)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--seed', type=int, default=20260926)
    args = p.parse_args()
    if args.steps <= 0 or args.lr <= 0:
        p.error('positive steps and learning rate required')
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    device = torch.device('cuda')
    root, cache_root = Path(args.output), Path(args.cache_root)
    root.mkdir(parents=True, exist_ok=False)
    config = json.loads((cache_root/'calibration_config.json').read_text())
    source_path, teacher_path = config['config']['checkpoint'], config['config']['teacher']
    if digest(source_path) != config['source_sha256'] or digest(teacher_path) != config['teacher_sha256']:
        raise ValueError('cache source/teacher hash mismatch')
    if digest(cache_root/'split.npz') != config['split_sha256']:
        raise ValueError('cache split hash mismatch')
    cache = torch.load(cache_root/'embedding_cache.pt', map_location='cpu', weights_only=False)
    x, y = (t.to(device) for t in cache['fit'])
    vx, vy = (t.to(device) for t in cache['validation'])
    if len(x) != 2*config['config']['fit_images'] or len(vx) != 2*config['config']['validation_images']:
        raise ValueError('cache does not match recorded paired-orientation split')
    if any(not torch.isfinite(t).all() for t in (x,y,vx,vy)):
        raise ValueError('non-finite cache')
    ids = torch.arange(len(x),device=device)//2
    vids = torch.arange(len(vx),device=device)//2
    dim = x.shape[1]
    identity = torch.eye(dim,device=device)
    matrix = torch.nn.Parameter(identity.clone())
    bias = torch.nn.Parameter(torch.zeros(dim,device=device))
    optimizer = torch.optim.Adam([matrix,bias],lr=args.lr)
    baseline = evaluate(matrix,bias,vx,vy,vids)
    best_score, best = sum(baseline.values()), None
    records = []
    for step in range(1,args.steps+1):
        # Sample 256 source images, retaining both views. Duplicate sampled
        # images are correctly excluded by original source ids in the mask.
        image_ids = torch.randint(len(x)//2,(256,),device=device)
        rows = torch.stack((2*image_ids,2*image_ids+1),dim=1).flatten()
        sx, sy = x[rows], y[rows]
        output = sx @ matrix.T+bias
        geometry, _ = pair_geometry_loss(output,sy,sx,ids[rows])
        anchor = (1-F.cosine_similarity(output,sy)).mean()
        penalty = ((matrix-identity).square().sum()+bias.square().sum())/dim
        loss = geometry+.01*anchor+.001*penalty
        if not torch.isfinite(loss):
            raise FloatingPointError('non-finite geometry loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([matrix,bias],1.,error_if_nonfinite=True)
        optimizer.step()
        if step % 100 == 0 or step == args.steps:
            if not torch.isfinite(matrix).all() or not torch.isfinite(bias).all():
                raise FloatingPointError('non-finite affine parameters')
            metrics = evaluate(matrix,bias,vx,vy,vids)
            records.append(dict(step=step,**metrics))
            score = sum(metrics.values())
            if score < best_score:
                best_score = score
                best = (step,matrix.detach().clone(),bias.detach().clone())
            print(records[-1],flush=True)
    report = dict(baseline=baseline,validation=records,improved=best is not None,
                  selected_step=None if best is None else best[0],
                  selection='held-out cross-image all-pair MSE plus fixed high-similarity-pair MSE')
    (root/'selection.json').write_text(json.dumps(report,indent=2))
    record = dict(config=vars(args),source_sha256=config['source_sha256'],
                  teacher_sha256=config['teacher_sha256'],split_sha256=config['split_sha256'],
                  cache_sha256=digest(cache_root/'embedding_cache.pt'),uses_ijbc_pair_labels=False,
                  inference_clipping=False,tail_threshold=.3,point_anchor_weight=.01,identity_penalty=.001)
    (root/'calibration_config.json').write_text(json.dumps(record,indent=2))
    if best is None:
        print('NO_IMPROVEMENT',flush=True)
        return
    model, state = load_controlled_checkpoint(source_path,device)
    if not state.get('pure_quadratic') or state['provenance']['teacher_sha256'] != config['teacher_sha256']:
        raise ValueError('invalid source provenance')
    if len(sitewise(model)) != 25:
        raise ValueError('expected 25 quadratic sites')
    configure(model,25)
    model.requires_grad_(False)
    probe = torch.randn(16,model.fc.in_features,device=device)*.01
    expected = model.features(model.fc(probe)) @ best[1].T + best[2]
    fold_alignment(model,best[1],best[2])
    actual = model.features(model.fc(probe))
    torch.testing.assert_close(actual,expected,atol=2e-4,rtol=2e-4)
    report['head_fold_probe_rows'] = len(probe)
    report['head_fold_max_abs_error'] = float((actual-expected).abs().max())
    (root/'selection.json').write_text(json.dumps(report,indent=2))
    changed = [n for n,t in model.state_dict().items() if not torch.equal(t.cpu(),state['state_dict_backbone'][n])]
    if not changed or not set(changed) <= {'fc.weight','fc.bias'}:
        raise RuntimeError(f'unexpected state changes: {changed}')
    torch.save(dict(matrix=best[1].cpu(),bias=best[2].cpu()),root/'alignment.pt')
    save_checkpoint(str(root/'candidate.pt'),model,state['poly_calib'],teacher_weights=teacher_path,
                    extra=dict(origin='channelwise_pair_geometry_calibration',pure_quadratic=True,
                               provenance=state['provenance'],calibration=record,selection=report,
                               diagnostic_only=True))
    print('CANDIDATE_SAVED: requires full exported IJB-C evaluation',flush=True)


if __name__ == '__main__':
    main()
