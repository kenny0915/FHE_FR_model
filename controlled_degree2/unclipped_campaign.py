"""Fixed no-clipping campaign. IJBC results only decide the success stop."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from controlled_degree2.shared_campaign import command, snapshot, wait_job, remaining_limit
from controlled_degree2.recipe_a import digest

POLICIES = {
    'direct': dict(unclip=0, cap=0, epochs=12, lr=.0002, coeff=.00002, range_weight=1),
    'gradual': dict(unclip=6, cap=0, epochs=18, lr=.0005, coeff=.00005, range_weight=5),
    'projected': dict(unclip=6, cap=.15, epochs=18, lr=.0005, coeff=.00005, range_weight=5),
    'slow_projected': dict(unclip=12, cap=.3, epochs=28, lr=.0002, coeff=.00002, range_weight=10),
    'small_curvature': dict(unclip=6, cap=.05, epochs=24, lr=.001, coeff=.0001, range_weight=5),
    'slow_free': dict(unclip=12, cap=0, epochs=28, lr=.0002, coeff=.00001, range_weight=10),
}


def result(root):
    raw = root/'unclipped_d2'/'ijbc_tar_at_far_raw.json'
    audit = root/'finite_audit.json'
    certificate = root/'polynomial_certificate.json'
    r = dict(ijbc_tar_at_far_1e4=None, nonfinite_values=None, pure_polynomial=False, target_met=False)
    if raw.exists():
        rows = json.loads(raw.read_text())
        if len(rows) != 1:
            raise ValueError('expected one final ROC')
        point = rows[0]['points']['0.0001']
        r.update(ijbc_tar_at_far_1e4=point['tar_percent'], actual_far=point['actual_far'])
    if audit.exists():
        a = json.loads(audit.read_text())
        r.update(nonfinite_values=a['nonfinite_values'], embedding_nonfinite_rows=a['embedding_nonfinite_rows'])
    if certificate.exists():
        c = json.loads(certificate.read_text())
        r['pure_polynomial'] = c.get('pure_polynomial') is True and c.get('quadratic_sites') == 25 and c.get('inference_clipping') is False
    r['target_met'] = (r['ijbc_tar_at_far_1e4'] is not None and r['ijbc_tar_at_far_1e4'] >= 96.56
                       and r['nonfinite_values'] == 0 and r.get('embedding_nonfinite_rows') == 0 and r['pure_polynomial'])
    return r


def submit(script, name, variables, deadline, cap):
    env = dict(os.environ, **{k:str(v) for k,v in variables.items()})
    args = ['sbatch', '--parsable', '--job-name='+name, '--time='+remaining_limit(deadline, cap),
            '--exclude=25a-hgpn143,25a-hgpn144', script]
    response = subprocess.run(args, env=env, text=True, capture_output=True)
    if response.returncode:
        raise RuntimeError(f'sbatch rejected {name}: {response.stderr}; stdout={response.stdout}')
    job = response.stdout.strip().split(';')[0]
    if not job.isdigit():
        raise RuntimeError(f'ambiguous sbatch response: {response.stdout}')
    return job


def run(args):
    root = Path(args.root).resolve()
    path = root/'campaign.json'
    state = json.loads(path.read_text())
    if state['attempts']:
        raise ValueError('refusing to restart controller over existing jobs')
    source = Path('work_dirs/shared_d2_D_20260913').resolve()
    state.update(status='running', policies=POLICIES, source_sha256=digest(source/'continuation_best.pt'))
    snapshot(path, state)
    for name, policy in POLICIES.items():
        if time.time() >= state['deadline']-120:
            state['status'] = 'budget_exhausted'
            break
        out = root/name
        out.mkdir(exist_ok=True)
        for filename in ('prepared.pt', 'split.npz'):
            if not (out/filename).exists():
                shutil.copy2(source/filename, out/filename)
        run = dict(name=name, policy=policy, started=time.time(), output=str(out))
        state['attempts'].append(run)
        snapshot(path, state)
        variables = dict(RECIPE_SHARED=1, RECIPE_PREPARED=1, RECIPE_SMOKE=0, RECIPE_OUTPUT=out,
                         SHARED_DEADLINE=state['deadline'], SHARED_BN='frozen', SHARED_WARMUP=0,
                         SHARED_ALL_QUADRATIC_START=1, SHARED_INFERENCE_BOUND=0,
                         SHARED_WARM_START=source/'continuation_best.pt', SHARED_EPOCHS=policy['epochs'],
                         SHARED_LR=policy['lr'], SHARED_COEFF_LR=policy['coeff'], SHARED_HEAD_LR=.0025,
                         UNCLIPPED_CONTINUATION=1, UNCLIP_EPOCHS=policy['unclip'],
                         CURVATURE_CAP=policy['cap'], RANGE_WEIGHT=policy['range_weight'])
        try:
            job = submit('controlled_degree2/recipe_a.slurm', 'unclip-'+name, variables, state['deadline'], 6*3600)
            run['training_job'] = job
            snapshot(path, state)
            run['training_state'] = wait_job(job, state['deadline'])
            metrics = out/'metrics.jsonl'
            run['validation_history'] = [json.loads(line) for line in metrics.read_text().splitlines()] if metrics.exists() else []
            selected = out/'student_best.pt'
            if not selected.exists():
                selected = out/'last.pt'
                run['selection'] = 'diagnostic last checkpoint; no valid holdout candidate'
            else:
                run['selection'] = 'maximum finite MS1MV3 min(clean,lowres) TAR@1e-4'
            if selected.exists():
                import torch
                checkpoint = torch.load(selected, map_location='cpu', weights_only=False)
                if checkpoint.get('network') != 'r50_shared_d2' or checkpoint.get('inference_input_bounds') is not False:
                    raise ValueError('candidate must explicitly disable inference bounds')
                run.update(checkpoint=str(selected), checkpoint_sha256=digest(selected), development=checkpoint['development'])
                del checkpoint
                final = out/'final_ijbc'
                final.mkdir(exist_ok=False)
                job = submit('controlled_degree2/job_ijbc.slurm', 'unclip-final-'+name,
                             dict(CHECKPOINT=selected, RESULT_DIR=final, NETWORK='r50_shared_d2',
                                  FINITE_AUDIT=1, POLYNOMIAL_EXPORT=1, EVAL_JOB_NAME='unclipped_d2'),
                             state['deadline'], 4*3600)
                run['evaluation_job'] = job
                snapshot(path, state)
                run['evaluation_state'] = wait_job(job, state['deadline'])
                if digest(selected) != run['checkpoint_sha256']:
                    raise ValueError('candidate changed during final evaluation')
                run.update(result(final))
            else:
                run.update(ijbc_tar_at_far_1e4=None, target_met=False,
                           unavailable_reason='training failed before any completed checkpoint; no candidate to evaluate')
        except TimeoutError as error:
            run['error'] = str(error)
            state['status'] = 'budget_exhausted'
            snapshot(path, state)
            break
        run['finished'] = time.time()
        snapshot(path, state)
        if run.get('target_met'):
            state['status'] = 'target_met'
            break
    else:
        state['status'] = 'fixed_comparison_complete'
    snapshot(path, state)
    print(json.dumps(state, indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', default='work_dirs/unclipped_round_20260913')
    run(p.parse_args())
