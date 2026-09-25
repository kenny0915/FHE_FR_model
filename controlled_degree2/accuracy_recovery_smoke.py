"""Run all recovery arms for two real updates, verify checkpoint invariants."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--teacher', required=True)
    parser.add_argument('--dataset-root', required=True)
    parser.add_argument('--canary-root', required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    initial = torch.load(args.checkpoint, map_location='cpu', weights_only=False)['state_dict_backbone']
    reports = {}
    for arm in ('control', 'fixed', 'coefficients'):
        command = [sys.executable, '-m', 'controlled_degree2.accuracy_recovery',
                   '--arm', arm, '--smoke', '--run', '--gpus', '4']
        for name, value in vars(args).items():
            command += ['--' + name.replace('_', '-'), value]
        subprocess.run(command, check=True)
        path = Path(args.output_root) / arm / 'epoch1.pt'
        payload = torch.load(path, map_location='cpu', weights_only=False)
        state = payload['state_dict_backbone']
        if payload['step'] != 2 or not payload.get('diagnostic_only'):
            raise RuntimeError(f'{arm}: expected two completed optimizer updates and epoch export')
        if not all(torch.isfinite(value).all() for value in state.values()):
            raise RuntimeError(f'{arm}: nonfinite checkpoint')
        changed = [name for name in state if not torch.equal(state[name], initial[name])]
        coefficient_changes = [name for name in changed if name.endswith('.coeffs')]
        bn_changes = [name for name in changed if name.endswith(('running_mean', 'running_var', 'num_batches_tracked'))]
        if not changed or bool(coefficient_changes) != (arm == 'coefficients'):
            raise RuntimeError(f'{arm}: unexpected coefficient/backbone update')
        if arm != 'control' and bn_changes:
            raise RuntimeError(f'{arm}: frozen BN statistics changed')
        reports[arm] = dict(steps=payload['step'], finite_checkpoint=True,
                            changed_tensors=len(changed), changed_coefficients=len(coefficient_changes),
                            changed_bn_buffers=len(bn_changes), checkpoint=str(path))
        del payload, state
    report = dict(job_id=os.environ.get('SLURM_JOB_ID'), arms=reports,
                  microbatch=128, global_batch=2048, gpus=4,
                  scope='Two-update execution and full LFW/CPLFW evaluation; not convergence or zero-failure certification.')
    (Path(args.output_root) / 'completed.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
