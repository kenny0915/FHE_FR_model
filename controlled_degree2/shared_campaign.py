"""Bounded sequential Slurm orchestration; IJBC can only trigger success stop.

Runs the predeclared A/B/C policies, without concurrent GPU allocations.
Training checkpoint selection uses only its saved development metrics. A
failed run without any full-conversion checkpoint is recorded as unevaluable.
The initial comparison finishing is not equivalent to exhausting the budget.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

POLICIES = {
    'A': dict(SHARED_INIT='fit', SHARED_BN='train', SHARED_CONVERSION='3', SHARED_EPOCHS='12', SHARED_COEFF_LR='.0001'),
    'B': dict(SHARED_INIT='near_linear', SHARED_BN='train', SHARED_CONVERSION='3', SHARED_EPOCHS='12', SHARED_COEFF_LR='.0005'),
    'C': dict(SHARED_INIT='fit', SHARED_BN='frozen', SHARED_CONVERSION='6', SHARED_EPOCHS='16', SHARED_COEFF_LR='.00005'),
}


def command(*args):
    return subprocess.check_output(args, text=True).strip()


def snapshot(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def remaining_limit(deadline, cap=4*3600):
    seconds = min(cap, int(deadline-time.time()))
    if seconds < 120:
        raise TimeoutError('no budget remains for another job')
    return f'{seconds//3600:02}:{seconds//60%60:02}:{seconds%60:02}'


def submit(script, name, variables, deadline):
    export = ','.join(['ALL']+[f'{key}={value}' for key, value in variables.items()])
    return command('sbatch', '--parsable', '--job-name='+name, '--time='+remaining_limit(deadline),
                   '--export='+export, script).split(';')[0]


def wait_job(job, deadline):
    while True:
        if time.time() >= deadline:
            subprocess.run(['scancel', str(job)], check=True)
            raise TimeoutError(f'cancelled own job {job} at campaign deadline')
        if not command('squeue', '-h', '-j', str(job), '-o', '%T'):
            return command('sacct', '-X', '-n', '-j', str(job), '--format=State', '--parsable2')
        time.sleep(30)


def candidate(root):
    import torch
    for filename in ('student_best.pt', 'last.pt'):
        path = root/filename
        if not path.exists():
            continue
        state = torch.load(path, map_location='cpu', weights_only=False)
        if not state.get('pure_quadratic') or state.get('network') != 'r50_shared_d2':
            continue
        coefficients = [v for k,v in state['state_dict_backbone'].items() if k.endswith('.coeffs')]
        if len(coefficients) != 25 or any(v.shape != (1, 3) for v in coefficients):
            raise ValueError('checkpoint violates shared-coefficient contract')
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(8*1024*1024), b''):
                digest.update(block)
        return path, dict(checkpoint_sha256=digest.hexdigest(), epoch=state['epoch'],
                          development=state['development'], selection=filename)
    return None, dict(reason='no full-conversion layer-shared checkpoint; IJBC unavailable')


def evaluation_result(root):
    audit_path = root/'finite_audit.json'
    table_path = root/'shared_d2'/'ijbc_tar_at_far.csv'
    result = dict(ijbc_tar_at_far_01=None, inference_nonfinite_values=None)
    if audit_path.exists():
        result['inference_nonfinite_values'] = json.loads(audit_path.read_text())['nonfinite_values']
    if table_path.exists():
        with table_path.open() as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != 1:
            raise ValueError('expected exactly one final IJBC result')
        result['ijbc_tar_at_far_01'] = float(rows[0]['0.1'])
    result['target_met'] = (result['inference_nonfinite_values'] == 0
                            and result['ijbc_tar_at_far_01'] is not None
                            and result['ijbc_tar_at_far_01'] >= 96.56)
    result['metric_valid'] = result['inference_nonfinite_values'] == 0
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--first-job', required=True)
    parser.add_argument('--deadline', type=float, required=True)
    parser.add_argument('--output', default='work_dirs/shared_d2_campaign_20260913')
    args = parser.parse_args()
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    ledger_path = root/'campaign.json'
    if ledger_path.exists():
        raise FileExistsError('refusing to overwrite an existing campaign')
    state = dict(deadline=args.deadline, started=time.time(), status='running', attempts=[])
    snapshot(ledger_path, state)
    try:
        for name, policy in POLICIES.items():
            output = Path(f'work_dirs/shared_d2_{name}_20260913').resolve()
            run = dict(name=name, policy=policy, output=str(output), started=time.time())
            state['attempts'].append(run)
            if name == 'A':
                job = args.first_job
            else:
                if output.exists():
                    raise FileExistsError(output)
                job = submit('controlled_degree2/recipe_a.slurm', 'shared-d2-'+name,
                             dict(RECIPE_SHARED=1, RECIPE_SMOKE=0, SHARED_DEADLINE=args.deadline,
                                  RECIPE_OUTPUT=output, **policy), args.deadline)
            run['training_job'] = job
            snapshot(ledger_path, state)
            run['training_state'] = wait_job(job, args.deadline)
            checkpoint, info = candidate(output)
            run.update(info)
            snapshot(ledger_path, state)
            if checkpoint is not None:
                result_dir = output/'final_ijbc'
                result_dir.mkdir(exist_ok=False)
                eval_job = submit('controlled_degree2/job_ijbc.slurm', 'shared-ijbc-'+name,
                                  dict(CHECKPOINT=checkpoint, RESULT_DIR=result_dir,
                                       NETWORK='r50_shared_d2', FINITE_AUDIT=1, EVAL_JOB_NAME='shared_d2'), args.deadline)
                run['evaluation_job'] = eval_job
                snapshot(ledger_path, state)
                run['evaluation_state'] = wait_job(eval_job, args.deadline)
                run.update(evaluation_result(result_dir))
            else:
                run.update(ijbc_tar_at_far_01=None, inference_nonfinite_values=None, target_met=False)
            run['finished'] = time.time()
            snapshot(ledger_path, state)
            if run['target_met']:
                state['status'] = 'target_met'
                break
        else:
            state['status'] = 'initial_comparison_complete'
    except TimeoutError as error:
        state.update(status='budget_exhausted', error=str(error))
    except Exception as error:
        state.update(status='orchestration_error', error=repr(error))
        raise
    finally:
        snapshot(ledger_path, state)
        print(json.dumps(state, indent=2), flush=True)


if __name__ == '__main__':
    main()
