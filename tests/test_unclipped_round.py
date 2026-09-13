from types import SimpleNamespace
import json
import pytest
import torch
from torch import nn
from controlled_degree2.shared import SharedQuadratic
from controlled_degree2.recipe_a import apply_phase
from controlled_degree2.unclipped_policy import project_coefficients
from controlled_degree2.polynomial_export import export_graph


def model():
    return nn.Sequential(nn.Conv2d(3, 3, 1), nn.BatchNorm2d(3),
                         SharedQuadratic(torch.ones(3)*.25, radius=2., coefficients=[[.2, .6, .1]], name='prelu'))


def test_curriculum_never_enables_eval_clipping():
    q = model()
    args = SimpleNamespace(all_quadratic_start=True, inference_bound=False,
                           batchnorm_mode='frozen', unclipped_continuation=True, unclip_epochs=6.)
    assert apply_phase(q.train(), 0., args)[1]
    assert q[2].clip and not q[2].clip_eval
    assert not apply_phase(q.train(), 6., args)[1]
    assert not q[2].clip and not q[2].clip_eval
    x = torch.full((1, 3, 2, 2), 20.)
    q[2].eval()
    torch.testing.assert_close(q[2](x), .2+.6*x+.1*x*x)


def test_projection_and_export_preserve_scalar_contract():
    q = model()
    with torch.no_grad():
        q[2].coeffs.copy_(torch.tensor([[20., 3., -2.]]))
    project_coefficients(q, .15)
    torch.testing.assert_close(q[2].coeffs, torch.tensor([[1., 1.5, -.075]]))
    q.eval()
    graph, report = export_graph(q, expected_sites=1)
    x = torch.randn(2, 3, 4, 4)*5
    torch.testing.assert_close(q(x), graph(x), atol=2e-5, rtol=1e-5)
    assert report['pure_polynomial'] and report['coefficients'] == 3
    q[2].clip_eval = True
    with pytest.raises(ValueError, match='unclipped'):
        export_graph(q, expected_sites=1)


def test_export_rejects_hidden_nonlinearity():
    q = nn.Sequential(model(), nn.ReLU()).eval()
    with pytest.raises(ValueError, match='non-polynomial'):
        export_graph(q, expected_sites=1)


def test_full_backbone_export():
    from controlled_degree2.shared import build_shared_iresnet50
    graph, report = export_graph(build_shared_iresnet50(dropout=0, fp16=False).eval())
    assert report['quadratic_sites'] == 25
    assert not any(isinstance(m, (nn.PReLU, nn.BatchNorm2d, nn.BatchNorm1d)) for m in graph.modules())


def test_success_uses_strict_far_and_export_audit(tmp_path):
    from controlled_degree2.unclipped_campaign import result
    out = tmp_path/'unclipped_d2'
    out.mkdir()
    raw = out/'ijbc_tar_at_far_raw.json'
    raw.write_text(json.dumps([dict(points={'0.0001': dict(tar_percent=96.559, actual_far=.0001),
                                              '0.1': dict(tar_percent=99.9, actual_far=.1)})]))
    (tmp_path/'finite_audit.json').write_text(json.dumps(dict(nonfinite_values=0, embedding_nonfinite_rows=0)))
    (tmp_path/'polynomial_certificate.json').write_text(json.dumps(dict(pure_polynomial=True, quadratic_sites=25, inference_clipping=False)))
    assert not result(tmp_path)['target_met']
    raw.write_text(json.dumps([dict(points={'0.0001': dict(tar_percent=96.56, actual_far=.0001)})]))
    assert result(tmp_path)['target_met']
    (tmp_path/'finite_audit.json').write_text(json.dumps(dict(nonfinite_values=1, embedding_nonfinite_rows=0)))
    assert not result(tmp_path)['target_met']


def test_allocation_counts_failed_jobs_and_actual_elapsed():
    from controlled_degree2.campaign_accounting import summarize
    report = summarize(['123|FAILED|120|s|e|node|billing=128,gres/gpu=16',
                        '124|RUNNING|60|s|Unknown|node|gres/gpu=1'])
    assert report['allocated_gpu_hours'] == pytest.approx((120*16+60)/3600)
    assert len(report['allocations']) == 2


def test_shared_recovery_admission_and_unbounded_handoff():
    from controlled_degree2.shared_recovery_support import validate_source, continuation_config
    payload = dict(network='r50_shared_d2_bounded', head={}, optimizer={},
                   state_dict_backbone={f'site{i}.coeffs': torch.ones(1, 3) for i in range(25)})
    validate_source(payload)
    payload['state_dict_backbone']['site0.coeffs'] = torch.ones(3, 3)
    with pytest.raises(ValueError, match='75'):
        validate_source(payload)
    config = continuation_config(dict(inference_bound=True, warm_start='old.pt'),
                                 SimpleNamespace(accuracy_epochs=24, deadline=1234))
    assert not config['inference_bound'] and config['unclipped_continuation']
    assert config['unclip_epochs'] == 0 and config['warm_start'] is None
    assert config['deadline'] == 1234 and config['epochs'] == 24


def test_shared_recovery_opens_only_shared_coefficients():
    from controlled_degree2.recipe_a_recovery_v2 import joint_parameters
    q = model()
    named = dict(joint_parameters(q, train_coefficients=True))
    assert '2.coeffs' in named and named['2.coeffs'].shape == (1, 3)
    assert q[1].running_mean.requires_grad is False


def test_shared_repair_resume_cannot_extend_campaign_deadline():
    from controlled_degree2 import recipe_a_recovery_v2 as recovery
    from controlled_degree2.shared_recovery_support import POLICY
    args = SimpleNamespace(**dict.fromkeys(recovery.FIXED_POLICY, 1),
                           shared=True, source_sha256='source', accuracy_epochs=24, deadline=1234)
    source = dict(provenance={}, state_dict_backbone={'weight': torch.ones(1)})
    state = dict(source, recovery=dict(policy=POLICY, source_sha256='source',
                                      config=vars(args).copy(), site=1, step=0))
    recovery.validate_resume(state, source, args, set(), 25)
    args.deadline = 1235
    with pytest.raises(ValueError, match='deadline'):
        recovery.validate_resume(state, source, args, set(), 25)


def test_controller_resume_monitors_existing_evaluation_without_resubmission(tmp_path, monkeypatch):
    import time
    from controlled_degree2 import unclipped_campaign as campaign
    checkpoint = tmp_path/'gradual'/'last.pt'
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b'fixture')
    run = dict(name='gradual', training_job='123', training_state='FAILED',
               checkpoint=str(checkpoint), checkpoint_sha256='fixed', evaluation_job='124')
    (tmp_path/'campaign.json').write_text(json.dumps(dict(status='running', started=time.time()-100,
        deadline=time.time()+1000, source_sha256='fixed', attempts=[run])))
    monkeypatch.setattr(campaign, 'POLICIES', {'gradual': campaign.POLICIES['gradual']})
    monkeypatch.setattr(campaign, 'digest', lambda _: 'fixed')
    monkeypatch.setattr(torch, 'load', lambda *a, **k: dict(network='r50_shared_d2', inference_input_bounds=False,
                                                         development={'nonfinite':0, 'tar_1e4':.5}))
    waited = []
    monkeypatch.setattr(campaign, 'wait_job', lambda job, deadline: waited.append(job) or 'COMPLETED')
    monkeypatch.setattr(campaign, 'submit', lambda *a, **k: pytest.fail('must not resubmit a recorded job'))
    monkeypatch.setattr(campaign, 'result', lambda _: dict(target_met=True))
    monkeypatch.setattr('controlled_degree2.campaign_accounting.collect', lambda _: {})
    campaign.run(SimpleNamespace(root=str(tmp_path), resume=True))
    state = json.loads((tmp_path/'campaign.json').read_text())
    assert waited == ['124'] and state['status'] == 'target_met'
    assert len(state['attempts']) == 1 and state['attempts'][0]['evaluation_job'] == '124'


def test_failed_gated_smoke_blocks_full_allocation(tmp_path, monkeypatch):
    import time
    from controlled_degree2 import unclipped_campaign as campaign
    (tmp_path/'campaign.json').write_text(json.dumps(dict(status='running', started=time.time()-100,
        deadline=time.time()+1000, source_sha256='fixed', attempts=[dict(name='adaptive_prefix',
            smoke_job='123', smoke_state='FAILED')])))
    monkeypatch.setattr(campaign, 'POLICIES', {'adaptive_prefix': campaign.POLICIES['adaptive_prefix']})
    monkeypatch.setattr(campaign, 'digest', lambda _: 'fixed')
    monkeypatch.setattr(campaign, 'submit', lambda *a, **k: pytest.fail('failed smoke must prevent allocation'))
    with pytest.raises(RuntimeError, match='smoke failed'):
        campaign.run(SimpleNamespace(root=str(tmp_path), resume=True))


def test_test_success_cannot_override_internal_nonfinite_outputs():
    from controlled_degree2.unclipped_campaign import qualifies
    assert not qualifies({'target_met':True}, {'nonfinite':1})
    assert not qualifies({'target_met':True}, {'unavailable':'failed validation'})
    assert qualifies({'target_met':True}, {'nonfinite':0})
    assert not qualifies({'target_met':False}, {'nonfinite':0})


def test_selected_training_state_survives_later_checkpoint_replacement(tmp_path):
    from controlled_degree2.recipe_a import preserve_training_selection
    (tmp_path/'last.pt').write_bytes(b'selected backbone plus matching head and optimizer')
    preserve_training_selection(tmp_path)
    (tmp_path/'next.pt').write_bytes(b'later lower-scoring epoch')
    (tmp_path/'next.pt').replace(tmp_path/'last.pt')
    assert (tmp_path/'continuation_best.pt').read_bytes() == b'selected backbone plus matching head and optimizer'
    # Even an external in-place write of last.pt cannot alter the independent snapshot.
    (tmp_path/'last.pt').write_bytes(b'rewritten later epoch')
    assert (tmp_path/'continuation_best.pt').read_bytes() == b'selected backbone plus matching head and optimizer'


def test_legacy_resume_namespace_may_omit_new_optional_flags():
    from controlled_degree2.recipe_a import check_resume_policy
    old_config = dict(seed=1, epochs=20, lr=.001)
    check_resume_policy(old_config, SimpleNamespace(**old_config))
    with pytest.raises(ValueError, match='lr'):
        check_resume_policy(old_config, SimpleNamespace(**(old_config | {'lr':.1})))


def test_range_priority_preserves_range_descent_under_conflict():
    from controlled_degree2.recipe_a_recovery_v2 import repair_gradients
    p = nn.Parameter(torch.tensor([1., 0.]))
    clean, tail = torch.tensor([-2., 1.]), torch.tensor([1., 0.])
    args = SimpleNamespace(lr=.001, max_step_ratio=.001, gradient_priority='range')
    gradients, stats = repair_gradients([p], [clean], [tail], args)
    assert torch.dot(gradients[0], tail) > 0
    assert torch.linalg.vector_norm(args.lr*gradients[0]) <= .00100001
    assert stats['conflicts'] == 1
    args.gradient_priority = 'clean'
    gradients, _ = repair_gradients([p], [clean], [tail], args)
    assert torch.dot(gradients[0], clean) > 0


def test_repair_environment_declares_same_policy_for_smoke_and_training():
    from controlled_degree2.unclipped_campaign import POLICIES, repair_variables
    legacy = repair_variables(POLICIES['adaptive_prefix'])
    assert legacy['RECOVERY_LR'] == .0001 and legacy['RECOVERY_GRADIENT_PRIORITY'] == 'clean'
    proposed = repair_variables(POLICIES['range_first'])
    assert proposed['RECOVERY_LR'] == .001 and proposed['RECOVERY_MAX_STEP_RATIO'] == .001
    assert proposed['RECOVERY_GRADIENT_PRIORITY'] == 'range' and proposed['RECOVERY_MAX_SITE_UPDATES'] == 3000
