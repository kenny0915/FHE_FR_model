import numpy as np
import pytest
import torch
from torch import nn

from controlled_degree2.model import DirectQuadratic
from controlled_degree2.recipe_a_recovery import (
    affine_parameters, averaged_gradients, check_split, configure,
    finite_prefix_loss, finite_prefix_batch_loss, snapshot_source, stage_passes, validate_resume,
    failure_training_rows, gate,
)


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(1)
        self.prelu = DirectQuadratic(1, lam_fit=1, name='prelu')
        self.layer1 = nn.ModuleList([nn.Module()])
        self.layer1[0].bn2 = nn.BatchNorm2d(1)
        self.layer1[0].prelu = DirectQuadratic(1, lam_fit=1, name='layer1.0.prelu')
        self.later_called = False

    def forward(self, x):
        x = self.prelu(self.bn1(x))
        self.later_called = True
        return self.layer1[0].prelu(self.layer1[0].bn2(x))


def test_stops_before_overflow_and_restores_graph_flags():
    model = Toy()
    parameters = affine_parameters(model)
    model.train()
    flags = [(m.training, getattr(m, 'clip_eval', None)) for m in model.modules()]
    coefficients = model.prelu.coeffs.clone()
    optimizer = torch.optim.SGD(parameters, lr=.01)
    loss, site, ratios = finite_prefix_loss(model, torch.full((2, 1, 2, 2), 4.), 2)
    assert site == 'prelu' and not model.later_called
    assert ratios.min() > 1 and torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(model.bn1.weight.grad).all()
    assert model.layer1[0].bn2.weight.grad is None
    before = model.bn1.weight.detach().clone()
    averaged_gradients(parameters, 1, torch.device('cpu'))
    optimizer.step()
    assert model.bn1.weight < before
    assert torch.equal(model.prelu.coeffs, coefficients)
    assert flags == [(m.training, getattr(m, 'clip_eval', None)) for m in model.modules()]


def test_downstream_escape_only_updates_local_affine():
    model = Toy()
    affine_parameters(model)
    with torch.no_grad():
        model.layer1[0].bn2.bias.fill_(5.)
    loss, site, _ = finite_prefix_loss(model, torch.zeros(2, 1, 2, 2), 2)
    assert site == 'layer1.0.prelu'
    loss.backward()
    assert model.layer1[0].bn2.bias.grad is not None
    assert model.bn1.bias.grad is None


def test_safe_prefix_and_progressive_clipping():
    model = Toy()
    affine_parameters(model)
    configure(model, 1)
    assert not model.prelu.clip_eval and model.layer1[0].prelu.clip_eval
    loss, site, _ = finite_prefix_loss(model, torch.zeros(2, 1, 2, 2), 1)
    assert site is None and loss == 0
    configure(model, 5)
    assert not any(m.clip_eval for m in model.modules() if isinstance(m, DirectQuadratic))


def test_invalid_prefix_does_not_leave_detach_hooks():
    model = Toy()
    affine_parameters(model)
    with pytest.raises(FloatingPointError):
        finite_prefix_loss(model, torch.full((2, 1, 2, 2), float('inf')), 2)
    assert all(not m._forward_pre_hooks for m in model.modules())


def test_gate_requires_finite_embeddings_not_interval_containment():
    assert stage_passes(dict(rows=4, nonfinite=0, escaping_rows=0))
    assert stage_passes(dict(rows=4, nonfinite=0, escaping_rows=1))
    for rows, bad, escapes in [(0, 0, 0), (4, 1, 0)]:
        assert not stage_passes(dict(rows=rows, nonfinite=bad, escaping_rows=escapes))


def test_safe_rows_reach_deeper_affines_despite_stem_escape():
    model = Toy()
    affine_parameters(model)
    with torch.no_grad():
        model.layer1[0].bn2.bias.fill_(5.)
    images = torch.tensor([4., 0.]).reshape(2, 1, 1, 1)
    loss, site, ratios = finite_prefix_batch_loss(model, images, 2)
    assert site == 'prelu' and len(ratios) == 2
    loss.backward()
    assert model.bn1.weight.grad is not None
    assert model.layer1[0].bn2.bias.grad is not None
    assert all(not m._forward_pre_hooks for m in model.modules())


def test_resume_allows_only_affine_updates_and_matching_source():
    import copy
    model = Toy()
    original = dict(provenance={'split': 'train'}, state_dict_backbone=model.state_dict())
    state = copy.deepcopy(original)
    state['recovery'] = dict(group=1, step=2000, source=dict(source_sha256='abc'))
    state['state_dict_backbone']['bn1.weight'].add_(.01)
    assert validate_resume(state, original, dict(source_sha256='abc'))['step'] == 2000
    with pytest.raises(ValueError, match='source mismatch'):
        validate_resume(state, original, dict(source_sha256='other'))
    state['state_dict_backbone']['prelu.lam_fit'].mul_(2)
    with pytest.raises(ValueError, match='frozen tensor'):
        validate_resume(state, original, dict(source_sha256='abc'))


def test_failure_replay_is_training_only(tmp_path):
    import json
    (tmp_path/'scan.rank0.json').write_text(json.dumps({'failures': [
        {'source_index': 2, 'variant': 'clean'}, {'source_index': 2, 'variant': 'flip'}]}))
    assert failure_training_rows(tmp_path, 'scan', 1, np.array([1, 2])).tolist() == [2]
    with pytest.raises(ValueError, match='non-training'):
        failure_training_rows(tmp_path, 'scan', 1, np.array([1, 3]))


def test_gate_detects_overflow_hidden_by_clipped_suffix(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import controlled_degree2.recipe_a_recovery as recovery

    class FlatToy(Toy):
        def forward(self, x):
            return super().forward(x).flatten(1)

    model = FlatToy()
    with torch.no_grad():
        model.prelu.lam_fit.fill_(1e30)
        model.prelu.coeffs.fill_(1.)
    images = torch.full((2, 1, 2, 2), 1e25)
    monkeypatch.setattr(recovery, 'loader', lambda *a, **kw: [(images, torch.zeros(2), torch.arange(2))])
    args = SimpleNamespace(output=str(tmp_path), guard=1.)
    report = gate(model, args, np.array([0, 1]), np.array([0, 0]), 0, 1,
                  torch.device('cpu'), 1, False, 'scan')
    assert report['rows'] == 4 and report['nonfinite'] == 4
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules())


@pytest.mark.parametrize('script', ['recipe_a_recovery.slurm', 'recipe_a_recovery_v2.slurm'])
def test_slurm_walltime_signal_requests_requeue(tmp_path, script):
    import os
    import subprocess
    from pathlib import Path
    repo = Path(__file__).resolve().parents[1]
    marker = tmp_path/'requeue.txt'
    commands = {
        'scontrol': '#!/bin/bash\nif [[ "$1" == show ]]; then echo localhost; else printf "%s\\n" "$*" > "$REQUEUE_MARKER"; fi\n',
        'srun': '#!/bin/bash\nkill -USR1 "$PPID"\n',
    }
    for name, command_text in commands.items():
        executable = tmp_path/name
        executable.write_text(command_text)
        executable.chmod(0o755)
    env = dict(os.environ, PATH=str(tmp_path)+os.pathsep+os.environ['PATH'],
               SLURM_SUBMIT_DIR=str(repo), SLURM_JOB_NODELIST='localhost', SLURM_JOB_ID='123',
               REQUEUE_MARKER=str(marker))
    result = subprocess.run(['bash', 'controlled_degree2/'+script],
                            cwd=repo, env=env, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert marker.read_text().strip() == 'requeue 123'


def test_recovery_identity_isolation():
    split = dict(labels=np.array([0, 0, 1, 1]), train=np.array([0, 1]), dev=np.array([2, 3]))
    check_split(split, [0])
    with pytest.raises(ValueError):
        check_split(split, [0, 1])
    split['dev'] = np.array([1, 3])
    with pytest.raises(ValueError):
        check_split(split, [0])


def test_source_snapshot_is_separate_read_only_and_exact(tmp_path):
    source = tmp_path/'source'
    source.mkdir()
    for name in ('last.pt', 'prepared.pt', 'split.npz', 'provenance.json'):
        (source/name).write_bytes(b'original bytes')
    output = tmp_path/'recovery'
    record = snapshot_source(source, output)
    assert (source/'last.pt').read_bytes() == (output/'source_epoch8.pt').read_bytes()
    assert (output/'source_epoch8.pt').stat().st_mode & 0o222 == 0
    assert len(record['source_sha256']) == 64
    with pytest.raises(FileExistsError):
        snapshot_source(source, output)
    with pytest.raises(ValueError):
        snapshot_source(source, source/'nested')
