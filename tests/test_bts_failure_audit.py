import json

import pytest
import torch
from torch import nn

from controlled_degree2.calibrate_bts_ms1mv3 import sample_indices, choose_scales
from controlled_degree2.rescale_residual_graph import BOUNDARIES
from eval.bts_failure_audit import BTSFailureAudit


def populate(audit, rows):
    audit.start_batch()
    for name in BOUNDARIES:
        audit.observe(name, torch.zeros(rows, 2))


def test_failures_zero_both_views_preserve_safe_rows_and_endpoints(tmp_path):
    audit = BTSFailureAudit(tmp_path / 'bts')
    populate(audit, 10)
    values = torch.zeros(10, 2)
    values[0] = torch.tensor([-1., 1.])  # inclusive endpoints pass
    values[3, 0] = 1.00001  # flip only, finite embedding
    values[4, 0] = float('nan')  # boundary NaN despite finite embedding
    values[7, 0] = -float('inf')
    audit.observe(BOUNDARIES[2], values)
    embeddings = torch.ones(10, 3)
    embeddings[8, 1] = float('inf')  # finite boundaries, nonfinite embedding
    filtered = audit.filter_embeddings(embeddings, range(5), [str(i) for i in range(5)])
    assert torch.equal(filtered[:2], embeddings[:2])
    assert torch.count_nonzero(filtered[2:]) == 0
    summary = audit.finish(5)
    assert summary['failed_source_images'] == 4
    assert summary['failed_augmented_rows'] == 4
    assert summary['zeroed_augmented_rows'] == 8
    records = [json.loads(line) for line in (tmp_path / 'bts.failures.jsonl').read_text().splitlines()]
    assert [r['source_index'] for r in records] == [1, 2, 3, 4]
    assert records[0]['views'][0]['failed'] is False
    assert records[0]['views'][1]['failed'] is True
    assert records[1]['views'][0]['boundaries'][BOUNDARIES[2]]['min'] is None


def test_remainder_accumulates_and_missing_hooks_fail_closed(tmp_path):
    audit = BTSFailureAudit(tmp_path / 'bts')
    populate(audit, 4)
    audit.filter_embeddings(torch.ones(4, 2), [0, 1], ['a', 'b'])
    populate(audit, 2)
    audit.filter_embeddings(torch.ones(2, 2), [2], ['c'])
    assert audit.finish(3)['source_images'] == 3
    audit = BTSFailureAudit(tmp_path / 'missing')
    audit.start_batch()
    with pytest.raises(RuntimeError, match='coverage'):
        audit.filter_embeddings(torch.ones(2, 2), [0], ['a'])
    audit.finish(0)


def test_new_model_attachment_removes_old_hooks(tmp_path):
    def model():
        m = nn.Module()
        for stage, length in [(1, 3), (2, 4), (3, 14), (4, 3)]:
            setattr(m, f'layer{stage}', nn.Sequential(*[nn.Identity() for _ in range(length)]))
        return m
    first, last = model(), model()
    audit = BTSFailureAudit(tmp_path / 'hooks')
    audit.attach(first)
    audit.attach(last)
    audit.start_batch()
    for name in BOUNDARIES:
        first.get_submodule(name)(torch.ones(2, 1))
    assert not audit.current
    for name in BOUNDARIES:
        last.get_submodule(name)(torch.ones(2, 1))
    assert set(audit.current) == set(BOUNDARIES)
    audit.filter_embeddings(torch.ones(2, 3), [0], ['a'])
    audit.finish(1)


def test_ms1_sampling_and_shared_stage_scale():
    indices = sample_indices(10000, 1000, 42)
    assert len(set(indices)) == 1000
    assert indices == sample_indices(10000, 1000, 42)
    assert indices != sample_indices(10000, 1000, 43)
    ranges = {name: {'min': -2., 'max': 3.} for name in BOUNDARIES}
    ranges['layer3.11']['max'] = 10.
    assert choose_scales(ranges, 0.8) == [0.8 / 3, 0.8 / 3, 0.08, 0.8 / 3]
    with pytest.raises(ValueError):
        sample_indices(5, 1000, 42)


def test_actual_ijbc_forward_zeroes_before_original_flip_reshape(tmp_path, monkeypatch):
    # eval_ijbc.py is a command-line script with top-level dataset evaluation.
    # Extract its actual inference method without executing that top-level work.
    import ast
    from pathlib import Path
    from types import SimpleNamespace
    tree = ast.parse(Path('eval_ijbc.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Embedding')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'forward_db')
    audit = BTSFailureAudit(tmp_path / 'integration')

    class Inference:
        def __call__(self, images):
            for name in BOUNDARIES:
                audit.observe(name, torch.zeros(len(images), 2))
            audit.observe(BOUNDARIES[0], torch.tensor([[0., 0.], [0., 0.], [0., 0.], [0., 1.01], [0., 0.], [0., 0.]]))
            return torch.tensor([[1., 2.], [3., 4.], [5., 6.], [7., 8.], [float('nan'), 1.], [9., 10.]])

    monkeypatch.setattr(torch.Tensor, 'cuda', lambda tensor: tensor)
    scope = {'torch': torch, 'args': SimpleNamespace(fail_on_nonfinite=False), 'bts_failure_audit': audit}
    exec(compile(ast.Module(body=[method], type_ignores=[]), 'eval_ijbc.py', 'exec'), scope)
    context = SimpleNamespace(model=Inference(), nonfinite_rows=0, nonfinite_records=[], trace_model=None, batch_size=3)
    result = scope['forward_db'](context, torch.zeros(6, 3, 2, 2), [0, 1, 2], ['safe', 'range', 'nan'])
    assert result.shape == (3, 4)
    assert result[0].tolist() == [1., 2., 3., 4.]
    assert not result[1:].any()
    assert context.nonfinite_rows == 1
    assert audit.finish(3)['failed_source_images'] == 2
