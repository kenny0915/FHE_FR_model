import json
import subprocess
import pytest
from controlled_degree2.shared_campaign import evaluation_result, remaining_limit, submit


def test_success_requires_both_metric_and_intermediate_finiteness(tmp_path):
    (tmp_path/'shared_d2').mkdir()
    (tmp_path/'shared_d2'/'ijbc_tar_at_far.csv').write_text('Methods,0.1\nshared-IJBC,96.56\n')
    assert not evaluation_result(tmp_path)['target_met']
    (tmp_path/'finite_audit.json').write_text(json.dumps(dict(nonfinite_values=1)))
    assert not evaluation_result(tmp_path)['target_met']
    (tmp_path/'finite_audit.json').write_text(json.dumps(dict(nonfinite_values=0)))
    assert not evaluation_result(tmp_path)['target_met']
    raw_path = tmp_path/'shared_d2'/'ijbc_tar_at_far_raw.json'
    raw_path.write_text(json.dumps([dict(points={'0.1': dict(tar_percent=96.559)})]))
    assert not evaluation_result(tmp_path)['target_met']
    raw_path.write_text(json.dumps([dict(points={'0.1': dict(tar_percent=96.56)})]))
    assert evaluation_result(tmp_path)['target_met']


def test_expired_budget_never_submits():
    with pytest.raises(TimeoutError):
        remaining_limit(0)


def test_submit_retries_explicit_token_service_rejection(monkeypatch):
    replies = iter([
        subprocess.CompletedProcess([], 1, '', 'get_api_token: Batch job submission failed: Unspecified error'),
        subprocess.CompletedProcess([], 0, '12345;cluster\n', ''),
    ])
    calls = []
    def run(args, **kwargs):
        calls.append(args)
        return next(replies)
    monkeypatch.setattr(subprocess, 'run', run)
    monkeypatch.setattr('controlled_degree2.shared_campaign.time.sleep', lambda _: None)
    assert submit('job.slurm', 'test', {}, float('1e10')) == '12345'
    assert len(calls) == 2


@pytest.mark.parametrize('stdout,stderr', [
    ('', 'socket timeout'),
    ('12345', 'get_api_token: Batch job submission failed: Unspecified error'),
])
def test_submit_does_not_retry_ambiguous_failure(monkeypatch, stdout, stderr):
    calls = []
    def run(args, **kwargs):
        calls.append(args)
        return subprocess.CompletedProcess(args, 1, stdout, stderr)
    monkeypatch.setattr(subprocess, 'run', run)
    with pytest.raises(subprocess.CalledProcessError):
        submit('job.slurm', 'test', {}, float('1e10'))
    assert len(calls) == 1


def test_submit_token_retries_are_bounded(monkeypatch):
    calls = []
    def run(args, **kwargs):
        calls.append(args)
        return subprocess.CompletedProcess(args, 1, '',
            'get_api_token: Batch job submission failed: Unspecified error')
    monkeypatch.setattr(subprocess, 'run', run)
    monkeypatch.setattr('controlled_degree2.shared_campaign.time.sleep', lambda _: None)
    with pytest.raises(subprocess.CalledProcessError):
        submit('job.slurm', 'test', {}, float('1e10'))
    assert len(calls) == 3
