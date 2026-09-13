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
    'adaptive_prefix': dict(mode='gated', accuracy_epochs=24, training_cap_hours=24),
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


def qualifies(evaluation, development):
    return evaluation.get('target_met') is True and development.get('nonfinite') == 0


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


def candidate_path(out):
    for filename in ('student_best.pt', 'last.pt', 'recovery_last.pt'):
        path = out/filename
        if path.exists():
            return path
    return None


def run(args):
    root = Path(args.root).resolve()
    path = root/'campaign.json'
    state = json.loads(path.read_text())
    if state['attempts'] and not getattr(args, 'resume', False):
        raise ValueError('use --resume to monitor existing jobs without resubmission')
    if state.get('status') == 'target_met':
        return
    source = Path('work_dirs/shared_d2_D_20260913').resolve()
    source_sha = digest(source/'continuation_best.pt')
    if state.get('source_sha256', source_sha) != source_sha:
        raise ValueError('warm-start source changed since campaign began')
    if state.get('policies') and state['policies'] != POLICIES:
        state.setdefault('policy_history', []).append(dict(at=time.time(), policies=state['policies'],
            reason='MS1MV3 direct embedding failure and gradual epoch-2 loss failure: prioritize gated repair after projected arm; no new IJBC metric available at decision'))
    state.update(status='running', policies=POLICIES, source_sha256=source_sha, controller_pid=os.getpid())
    snapshot(path, state)
    for name, policy in POLICIES.items():
        existing = next((entry for entry in state['attempts'] if entry['name'] == name), None)
        if existing is not None and 'finished' in existing:
            if existing.get('target_met'):
                state['status'] = 'target_met'
                break
            continue
        if time.time() >= state['deadline']-120 and existing is None:
            state['status'] = 'budget_exhausted'
            break
        out = root/name
        gated = policy.get('mode') == 'gated'
        run = existing
        if run is None:
            if not gated:
                out.mkdir(exist_ok=True)
                for filename in ('prepared.pt', 'split.npz'):
                    if not (out/filename).exists():
                        shutil.copy2(source/filename, out/filename)
            run = dict(name=name, policy=policy, started=time.time(), output=str(out))
            state['attempts'].append(run)
            snapshot(path, state)
        try:
            if 'training_job' not in run:
                if gated:
                    smoke_output = root/(name+'_smoke')
                    if 'smoke_job' not in run:
                        run['smoke_job'] = submit('controlled_degree2/shared_recovery.slurm', 'unclip-gated-smoke',
                            dict(RECIPE_OUTPUT=smoke_output, SHARED_DEADLINE=state['deadline'], RECIPE_SMOKE=1,
                                 RECOVERY_SOURCE=source, RECOVERY_ACCURACY_EPOCHS=policy['accuracy_epochs']),
                            state['deadline'], 20*60)
                        snapshot(path, state)
                    if 'smoke_state' not in run:
                        run['smoke_state'] = wait_job(run['smoke_job'], state['deadline'])
                        snapshot(path, state)
                    if run['smoke_state'].strip() != 'COMPLETED' or not (smoke_output/'smoke.pt').exists():
                        raise RuntimeError('shared gated GPU smoke failed; do not launch full repair')
                    variables = dict(RECIPE_OUTPUT=out, SHARED_DEADLINE=state['deadline'], RECIPE_SMOKE=0,
                                     RECOVERY_SOURCE=source, RECOVERY_ACCURACY_EPOCHS=policy['accuracy_epochs'])
                    script, cap = 'controlled_degree2/shared_recovery.slurm', policy['training_cap_hours']*3600
                else:
                    variables = dict(RECIPE_SHARED=1, RECIPE_PREPARED=1, RECIPE_SMOKE=0, RECIPE_OUTPUT=out,
                                     SHARED_DEADLINE=state['deadline'], SHARED_BN='frozen', SHARED_WARMUP=0,
                                     SHARED_ALL_QUADRATIC_START=1, SHARED_INFERENCE_BOUND=0,
                                     SHARED_WARM_START=source/'continuation_best.pt', SHARED_EPOCHS=policy['epochs'],
                                     SHARED_LR=policy['lr'], SHARED_COEFF_LR=policy['coeff'], SHARED_HEAD_LR=.0025,
                                     UNCLIPPED_CONTINUATION=1, UNCLIP_EPOCHS=policy['unclip'],
                                     CURVATURE_CAP=policy['cap'], RANGE_WEIGHT=policy['range_weight'])
                    script, cap = 'controlled_degree2/recipe_a.slurm', 6*3600
                run['training_job'] = submit(script, 'unclip-'+name, variables, state['deadline'], cap)
                snapshot(path, state)
            if 'training_state' not in run:
                run['training_state'] = wait_job(run['training_job'], state['deadline'])
                snapshot(path, state)
            metrics = out/'metrics.jsonl'
            run['validation_history'] = [json.loads(line) for line in metrics.read_text().splitlines()] if metrics.exists() else []
            selected = Path(run['checkpoint']) if 'checkpoint' in run else candidate_path(out)
            if selected is not None:
                import torch
                checkpoint = torch.load(selected, map_location='cpu', weights_only=False)
                if checkpoint.get('network') != 'r50_shared_d2' or checkpoint.get('inference_input_bounds') is not False:
                    raise ValueError('candidate must explicitly disable inference bounds')
                checksum = digest(selected)
                if run.get('checkpoint_sha256', checksum) != checksum:
                    raise ValueError('selected checkpoint changed since selection')
                run.update(checkpoint=str(selected), checkpoint_sha256=checksum, development=checkpoint['development'],
                           selection=('maximum finite MS1MV3 min(clean,lowres) TAR@1e-4' if selected.name == 'student_best.pt'
                                      else 'diagnostic last checkpoint; no valid holdout candidate'))
                del checkpoint
                final = out/'final_ijbc'
                if 'evaluation_job' not in run:
                    final.mkdir(exist_ok=False)
                    run['evaluation_job'] = submit('controlled_degree2/job_ijbc.slurm', 'unclip-final-'+name,
                                     dict(CHECKPOINT=selected, RESULT_DIR=final, NETWORK='r50_shared_d2',
                                          FINITE_AUDIT=1, POLYNOMIAL_EXPORT=1, EVAL_JOB_NAME='unclipped_d2'),
                                     state['deadline'], 4*3600)
                    snapshot(path, state)
                if 'evaluation_state' not in run:
                    run['evaluation_state'] = wait_job(run['evaluation_job'], state['deadline'])
                if digest(selected) != run['checkpoint_sha256']:
                    raise ValueError('candidate changed during final evaluation')
                if 'phase' in run['development'] and 'final_development' not in run:
                    # Report fixed diagnostic candidates on the same holdout.
                    # Wait for IJBC allocation to end before another allocation,
                    # but do not consume its metric until this report is fixed.
                    validation_dir = out/'final_ms1mv3'
                    if 'development_job' not in run:
                        validation_dir.mkdir(exist_ok=False)
                        run['development_job'] = submit('controlled_degree2/shared_validate.slurm', 'unclip-holdout-'+name,
                            dict(CHECKPOINT=selected, VALIDATION_OUTPUT=validation_dir), state['deadline'], 2*3600)
                        snapshot(path, state)
                    run['development_state'] = wait_job(run['development_job'], state['deadline'])
                    validation_result = validation_dir/'result.json'
                    run['final_development'] = (json.loads(validation_result.read_text()) if validation_result.exists()
                                                else {'unavailable': 'fixed holdout evaluation failed; see job log'})
                run.update(result(final))
                run['target_met'] = qualifies(run, run.get('final_development', run['development']))
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
        from controlled_degree2.campaign_accounting import collect
        collect(root)
        if run.get('target_met'):
            state['status'] = 'target_met'
            break
    else:
        state['status'] = 'fixed_comparison_complete'
    state['controller_pid'] = None
    snapshot(path, state)
    print(json.dumps(state, indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--root', default='work_dirs/unclipped_round_20260913')
    import fcntl
    args = p.parse_args()
    with (Path(args.root)/'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            run(args)
        except BaseException as error:
            ledger = Path(args.root)/'campaign.json'
            status = json.loads(ledger.read_text())
            status.update(status='controller_interrupted', controller_pid=None, controller_error=repr(error))
            snapshot(ledger, status)
            raise
