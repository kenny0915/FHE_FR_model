import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from controlled_degree2.model import DirectQuadratic
from controlled_degree2.recipe_a_recovery import configure, finite_prefix_batch_loss
from controlled_degree2 import recipe_a_recovery_v2 as recovery


class Block(nn.Module):
    def __init__(self, polynomial=True):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(1)
        self.prelu = DirectQuadratic(1, lam_fit=1., name='layer1.0.prelu') if polynomial else nn.PReLU(1)

    def forward(self, x):
        return self.prelu(self.bn2(self.conv(x)))


class Mini(nn.Module):
    def __init__(self, polynomial=True):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(1)
        self.prelu = DirectQuadratic(1, lam_fit=1., name='prelu') if polynomial else nn.PReLU(1)
        self.layer1 = nn.ModuleList([Block(polynomial)])
        self.fc = nn.Linear(4, 3)
        self.features = nn.BatchNorm1d(3)

    def forward(self, x):
        x = self.prelu(self.bn1(self.conv1(x)))
        x = self.layer1[0](x)
        return self.features(self.fc(x.flatten(1)))


def arguments(**kwargs):
    defaults = dict(seed=1, lr=1e-3, target=.9, guard=1., gate_ratio=2., gate_images=2,
                    hint_weight=.3, range_weight=1., max_step_ratio=1e-4, batch_size=2,
                    replay_batch_size=2, check_every=1, continuation_lr_factor=.1,
                    workers=0, save_every=1, smoke=False, continue_training=False,
                    resume=None, reuse_output=False, replay_ratio=2.)
    return SimpleNamespace(**(defaults | kwargs))


def test_only_conv_and_spatial_bn_affines_train():
    model = Mini()
    original = copy.deepcopy(model.state_dict())
    named = recovery.joint_parameters(model)
    allowed = {n for n, _ in named}
    assert 'conv1.weight' in allowed and 'layer1.0.conv.weight' in allowed
    assert 'bn1.weight' in allowed and 'layer1.0.bn2.bias' in allowed
    assert 'fc.weight' not in allowed and 'features.weight' not in allowed
    with torch.no_grad():
        model.conv1.weight.add_(.01)
    recovery.validate_state(model.state_dict(), original, allowed)
    for name in ('prelu.lam_fit', 'prelu.coeffs', 'bn1.running_mean', 'fc.weight'):
        changed = copy.deepcopy(model.state_dict())
        changed[name].add_(.1)
        with pytest.raises(ValueError, match='frozen tensor'):
            recovery.validate_state(changed, original, allowed)


def test_sitewise_prefix_gradient_reaches_conv():
    model = Mini()
    recovery.joint_parameters(model)
    assert recovery.sitewise(model) == ['prelu', 'layer1.0.prelu']
    configure(model, 1)
    assert not model.prelu.clip_eval and model.layer1[0].prelu.clip_eval
    with torch.no_grad():
        model.conv1.weight.fill_(1.)
    loss, name, _ = finite_prefix_batch_loss(model, torch.full((2, 1, 2, 2), 4.), 1, detach_bn=False)
    loss.backward()
    assert name == 'prelu' and model.conv1.weight.grad.abs().sum() > 0
    assert all(not m._forward_pre_hooks for m in model.modules())


def test_bounded_gate_rejects_huge_but_finite_prefix():
    report = dict(rows=10, nonfinite=0, max_ratio=1.5)
    assert recovery.bounded_gate(report, 2., 10)
    assert not recovery.bounded_gate(report, 2., 11)
    assert not recovery.bounded_gate(report | dict(max_ratio=44870643712.), 2., 10)
    assert not recovery.bounded_gate(report | dict(nonfinite=1), 2., 10)
    assert recovery.full_gate(report, 5)
    assert not recovery.full_gate(report, 6)


def test_exact_replay_retains_variant_and_rejects_dev(tmp_path, monkeypatch):
    (tmp_path/'scan.rank0.json').write_text(json.dumps({'failures': [
        {'source_index': 1, 'variant': 'flip'}, {'source_index': 1, 'variant': 'dark'}]}))
    records = recovery.replay_records(tmp_path, 'scan', 1, np.array([0, 1]))
    assert records == [(1, 'dark'), (1, 'flip')]
    with pytest.raises(ValueError, match='non-training'):
        recovery.replay_records(tmp_path, 'scan', 1, np.array([0, 2]))
    image = torch.arange(12).reshape(3, 2, 2).float()/12
    monkeypatch.setattr(recovery, 'SourceRows', lambda *a: [(image, 0, 1)]*2)
    dataset = recovery.ExactReplay(SimpleNamespace(dataset_root='unused'), records, [0, 0])
    assert torch.equal(dataset[0][0], (image+1)*.3-1)
    assert torch.equal(dataset[1][0], image.flip(-1))


def test_replay_capacity_prioritizes_new_failures():
    assert recovery.merge_replay([(100, 'flip')], [(0, 'clean'), (1, 'dark')], 2) == [
        (100, 'flip'), (0, 'clean')]


def test_node_routes_v2_and_requeue_resume(tmp_path):
    import os
    import subprocess
    from pathlib import Path
    repo = Path(__file__).resolve().parents[1]
    marker = tmp_path/'arguments.txt'
    output = tmp_path/'run'
    output.mkdir()
    (output/'source_epoch8.pt').write_bytes(b'snapshot marker')
    commands = {
        'getent': '#!/bin/bash\necho "172.21.100.150 STREAM node"\n',
        'ip': '#!/bin/bash\necho "1: vlan1721 inet 172.21.100.150/24"\n',
        'python': '#!/bin/bash\nprintf "%s\\n" "$@" > "$ARGUMENT_MARKER"\n',
    }
    for name, text in commands.items():
        executable = tmp_path/name
        executable.write_text(text)
        executable.chmod(0o755)
    env = dict(os.environ, PATH=str(tmp_path)+os.pathsep+os.environ['PATH'],
               RECIPE_PYTHON=str(tmp_path/'python'), ARGUMENT_MARKER=str(marker),
               SLURM_PROCID='0', SLURM_NNODES='2', SLURM_RESTART_COUNT='1',
               RECIPE_MASTER_ADDR='node', RECIPE_MASTER_PORT='22000', RECIPE_OUTPUT=str(output),
               RECOVERY_RESUME='', RECIPE_SMOKE='1')
    result = subprocess.run(['bash', 'controlled_degree2/recipe_a_node.sh', 'recovery-v2'],
                            cwd=repo, env=env, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    arguments = marker.read_text().splitlines()
    assert 'controlled_degree2.recipe_a_recovery_v2' in arguments
    assert '--reuse-output' in arguments and '--smoke' in arguments


def test_joint_update_is_finite_bounded_and_preserves_buffers():
    torch.manual_seed(2)
    model, teacher = Mini(), Mini(False).eval().requires_grad_(False)
    recovery.sitewise(model)
    named = recovery.joint_parameters(model)
    parameters = tuple(p for _, p in named)
    before = copy.deepcopy(model.state_dict())
    args = arguments()
    optimizer = torch.optim.SGD(parameters, lr=args.lr)
    images = torch.randn(2, 1, 2, 2)*3
    head = lambda z, labels, mask: z.square().mean()
    metrics = recovery.update(model, teacher, head, images, torch.zeros(2, dtype=torch.long),
                              torch.ones(2, dtype=torch.bool), None, 1, parameters, optimizer,
                              args, 1, torch.device('cpu'))
    assert np.isfinite(metrics['range']) and np.isfinite(metrics['clean'])
    assert not torch.equal(before['conv1.weight'], model.conv1.weight)
    recovery.validate_state(model.state_dict(), before, {n for n, _ in named})
    for name, p in named:
        limit = args.max_step_ratio*max(before[name].double().norm().item(), 1.)
        assert (p-before[name]).double().norm() <= limit+1e-7
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules())


def test_resume_rejects_legacy_and_policy_change():
    model = Mini()
    allowed = {n for n, _ in recovery.joint_parameters(model)}
    args = arguments(source_sha256='abc')
    source = dict(provenance={'train': 1}, state_dict_backbone=model.state_dict())
    state = source | dict(recovery=dict(policy=recovery.POLICY, source_sha256='abc',
                                        config=vars(args), site=1, step=2))
    assert recovery.validate_resume(state, source, args, allowed, 2)['step'] == 2
    with pytest.raises(ValueError, match='compatible'):
        recovery.validate_resume(source | dict(recovery={}), source, args, allowed, 2)
    changed = arguments(source_sha256='abc', gate_ratio=3.)
    with pytest.raises(ValueError, match='policy'):
        recovery.validate_resume(state, source, changed, allowed, 2)


def test_runner_retries_failed_full_gate_and_resumes_ready(tmp_path, monkeypatch):
    source = tmp_path/'source'
    source.mkdir()
    teacher_path = source/'teacher.pt'
    teacher_path.write_bytes(b'teacher')
    np.savez(source/'split.npz', train=np.array([0, 1]), dev=np.array([2]), labels=np.array([0, 0, 1]))
    metadata = dict(config={'smoke': False}, split_sha256=recovery.digest(source/'split.npz'),
                    teacher_sha256=recovery.digest(teacher_path))
    original = Mini().state_dict()
    payload = dict(epoch=8, origin='teacher_only_recipe_a', provenance=metadata,
                   state_dict_backbone=original, config=dict(global_batch=2, dataset_root='unused',
                   teacher=str(teacher_path), lr=.004, head_lr=.02),
                   head={'weight': torch.ones(1, 3)}, optimizer={'state': {}, 'param_groups': []})
    torch.save(payload, source/'last.pt')
    torch.save(dict(metadata=metadata, active_ids=[0], centers=torch.ones(2, 3), calibration={}), source/'prepared.pt')
    (source/'provenance.json').write_text('{}')
    monkeypatch.setattr(recovery, 'build_controlled_iresnet50', lambda **kw: Mini())
    monkeypatch.setattr(recovery, 'load_teacher', lambda *a: Mini(False))
    monkeypatch.setattr(recovery, 'loader', lambda *a, **kw: [(torch.zeros(2, 1, 2, 2), torch.zeros(2, dtype=torch.long), torch.arange(2))])
    monkeypatch.setattr(recovery, 'prepare_range_batch', lambda images, **kw: (images, torch.ones(len(images), dtype=torch.bool)))
    monkeypatch.setattr(recovery, 'replay_loader', lambda *a, **kw: None)
    updates, scans, handoffs = [], [], []

    def fake_update(model, teacher, head, images, labels, mask, replay, site, *a):
        updates.append(site)
        with torch.no_grad():
            model.conv1.weight.add_(1e-5)
        return {}

    def fake_gate(model, args, rows, labels, rank, world, device, groups, stress, label):
        bad = int(not stress and not scans)
        if not stress:
            scans.append(label)
        failures = [{'source_index': 0, 'variant': 'flip'}] if bad else []
        (Path(args.output)/f'{label}.rank0.json').write_text(json.dumps({'failures': failures}))
        return dict(rows=len(rows)*(5 if stress else 2), nonfinite=bad, max_ratio=1.)

    from pathlib import Path
    monkeypatch.setattr(recovery, 'update', fake_update)
    monkeypatch.setattr(recovery, 'gate', fake_gate)
    monkeypatch.setattr(recovery, 'train', lambda *a: handoffs.append(True))
    args = arguments(source=str(source), output=str(tmp_path/'run'), continue_training=True)
    recovery.run(args, 0, 1, torch.device('cpu'))
    assert updates == [1, 2, 2] and len(scans) == 2 and len(handoffs) == 1
    final = torch.load(tmp_path/'run'/'recovered.pt', weights_only=False)
    assert final['recovery']['ready'] and final['recovery']['step'] == 2
    # Wall-time resume of a completed recovery must not retrain/re-audit or
    # crash because an update-loop variable is undefined.
    args.reuse_output = True
    recovery.run(args, 0, 1, torch.device('cpu'))
    assert updates == [1, 2, 2] and len(scans) == 2 and len(handoffs) == 2
