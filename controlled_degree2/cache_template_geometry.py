"""Extract complete, disjoint IJB-C template caches on a Slurm GPU node."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from controlled_degree2.model import load_controlled_checkpoint, load_teacher
from controlled_degree2.recipe_a import digest
from controlled_degree2.recipe_a_recovery import configure
from controlled_degree2.recipe_a_recovery_v2 import sitewise
from controlled_degree2.template_geometry import aggregate_templates, template_split


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--teacher', default='work_dirs/ms1mv3_r50/model.pt')
    p.add_argument('--ijb-root', default='ijb/IJBC')
    p.add_argument('--fit-templates', type=int, default=2048)
    p.add_argument('--validation-templates', type=int, default=512)
    p.add_argument('--seed', type=int, default=20260927)
    p.add_argument('--batch-size', type=int, default=256)
    args = p.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    meta_root = Path(args.ijb_root)/'meta'
    template_path = meta_root/'ijbc_face_tid_mid.txt'
    face_path = meta_root/'ijbc_name_5pts_score.txt'
    template_rows = [line.split() for line in template_path.read_text().splitlines()]
    face_rows = [line.split() for line in face_path.read_text().splitlines()]
    if [r[0] for r in template_rows] != [r[0] for r in face_rows]:
        raise ValueError('template metadata and image metadata order mismatch')
    tids = np.array([int(r[1]) for r in template_rows])
    mids = np.array([int(r[2]) for r in template_rows])
    detector = np.array([float(r[-1]) for r in face_rows], dtype=np.float64)
    split = template_split(tids, args.fit_templates, args.validation_templates, args.seed)
    np.savez(root/'split.npz', **split)
    device = torch.device('cuda')
    model, state = load_controlled_checkpoint(args.checkpoint, device)
    teacher_hash = digest(args.teacher)
    if not state.get('pure_quadratic') or state['provenance']['teacher_sha256'] != teacher_hash:
        raise ValueError('pure quadratic original-teacher-derived source required')
    if len(sitewise(model)) != 25:
        raise ValueError('expected 25 quadratic sites')
    configure(model, 25)
    model.eval().requires_grad_(False)
    teacher = load_teacher(args.teacher, device).eval().requires_grad_(False)
    config = dict(config=vars(args), source_sha256=digest(args.checkpoint),
                  teacher_sha256=teacher_hash, split_sha256=digest(root/'split.npz'),
                  metadata_sha256={str(p): digest(p) for p in (template_path, face_path)},
                  cache_layout='complete_template_barycenters',
                  uses_ijbc_pair_labels=False, inference_clipping=False,
                  approximation='unchanged per-channel PReLU target and lam_fit interval',
                  interpretation='IJB-C calibration-set performance, not untouched test accuracy')
    (root/'calibration_config.json').write_text(json.dumps(config, indent=2)+'\n')
    from utils.utils_ijbc_replay import IJBCOrientationDataset
    cache = {}
    for name in ('fit', 'validation'):
        indices = split[name+'_source_images']
        records = [(int(i), o) for i in indices for o in (0, 1)]
        loader = DataLoader(IJBCOrientationDataset(args.ijb_root, records),
                            batch_size=args.batch_size, num_workers=4,
                            multiprocessing_context='spawn')
        xs, ys = [], []
        for step, (images, _, _) in enumerate(loader):
            images = images.to(device)
            x, y = model(images), teacher(images)
            if not torch.isfinite(x).all() or not torch.isfinite(y).all():
                raise FloatingPointError('nonfinite extracted embedding')
            xs.append(x.cpu()); ys.append(y.cpu())
            if step % 25 == 0:
                print(name, step, 'batches', flush=True)
        x, y = (torch.cat(v).reshape(len(indices), 2, -1).double() for v in (xs, ys))
        tx, weight, ids = aggregate_templates(x, tids[indices], mids[indices], detector[indices])
        ty, teacher_weight, teacher_ids = aggregate_templates(y, tids[indices], mids[indices], detector[indices])
        torch.testing.assert_close(weight, teacher_weight)
        torch.testing.assert_close(ids, teacher_ids)
        np.testing.assert_array_equal(ids.numpy(), split[name+'_templates'])
        if (weight <= 0).any():
            raise ValueError('zero total detector weight in complete template')
        # Positive division preserves normalized template direction and makes
        # the per-view affine act as x@A.T+b on each cached barycenter.
        cache[name] = dict(source=tx/weight[:, None], teacher=ty/weight[:, None],
                           bias_weight=weight, template_ids=ids)
        print(name, len(indices), 'images', len(ids), 'complete templates', flush=True)
    torch.save(cache, root/'template_cache.pt')
    config['cache_sha256'] = digest(root/'template_cache.pt')
    (root/'calibration_config.json').write_text(json.dumps(config, indent=2)+'\n')
    print('COMPLETE_TEMPLATE_CACHE_SAVED', flush=True)


if __name__ == '__main__':
    main()
