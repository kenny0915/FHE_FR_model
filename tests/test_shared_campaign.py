import json
import pytest
from controlled_degree2.shared_campaign import evaluation_result, remaining_limit


def test_success_requires_both_metric_and_intermediate_finiteness(tmp_path):
    (tmp_path/'shared_d2').mkdir()
    (tmp_path/'shared_d2'/'ijbc_tar_at_far.csv').write_text('Methods,0.1\nshared-IJBC,96.56\n')
    assert not evaluation_result(tmp_path)['target_met']
    (tmp_path/'finite_audit.json').write_text(json.dumps(dict(nonfinite_values=1)))
    assert not evaluation_result(tmp_path)['target_met']
    (tmp_path/'finite_audit.json').write_text(json.dumps(dict(nonfinite_values=0)))
    assert evaluation_result(tmp_path)['target_met']


def test_expired_budget_never_submits():
    with pytest.raises(TimeoutError):
        remaining_limit(0)
