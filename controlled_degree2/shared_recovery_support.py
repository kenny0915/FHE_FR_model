"""Admission and handoff for a conditional training-data-gated shared repair.

This permits the previous shared checkpoint; the original per-channel v2
recovery admission remains unchanged. No IJB-derived repair input is read.
"""
import json
import os
from pathlib import Path
import shutil
import torch
from controlled_degree2.recipe_a import digest

POLICY = 'shared_degree2_sitewise_joint_v1'


def validate_source(payload):
    if payload.get('network') not in ('r50_shared_d2', 'r50_shared_d2_bounded'):
        raise ValueError('shared recovery requires a shared degree-two source')
    tensors = [v for k, v in payload['state_dict_backbone'].items() if k.endswith('.coeffs')]
    if len(tensors) != 25 or any(v.shape != (1, 3) for v in tensors):
        raise ValueError('shared source must have exactly 75 polynomial coefficients')
    if 'head' not in payload or 'optimizer' not in payload:
        raise ValueError('shared recovery needs the matching classification head and optimizer layout')


def snapshot_source(source, output, filename):
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError('recovery output must be separate from source')
    if Path(filename).name != filename:
        raise ValueError('source checkpoint must be a filename within its source directory')
    original = source/filename
    checksum = digest(original)
    payload = torch.load(original, map_location='cpu', weights_only=False)
    validate_source(payload)
    output.mkdir(parents=True, exist_ok=False)
    saved = output/'shared_source.pt'
    shutil.copy2(original, saved)
    if digest(saved) != checksum or digest(original) != checksum:
        raise RuntimeError('shared source changed during snapshot')
    saved.chmod(0o440)
    for name in ('prepared.pt', 'split.npz'):
        shutil.copy2(source/name, output/name)
    record = dict(source=str(original), source_sha256=checksum, snapshot=saved.name,
                  code_commit=os.environ.get('RECIPE_COMMIT', 'local-test'))
    (output/'recovery_source.json').write_text(json.dumps(record, indent=2))
    return record


def continuation_config(config, args):
    config = dict(config)
    config.update(shared=True, inference_bound=False, all_quadratic_start=True,
                  head_warmup=0., warm_start=None, unclipped_continuation=True,
                  unclip_epochs=0., curvature_cap=0., batchnorm_mode='frozen',
                  epochs=args.accuracy_epochs, deadline=args.deadline)
    return config
