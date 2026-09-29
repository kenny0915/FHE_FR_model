import ast
from pathlib import Path

import numpy as np
import sklearn.preprocessing

from eval.ijb_similarity import compare_tar_rows


def evaluator_functions():
    # The legacy evaluator runs inference at module scope. Load its real pure
    # functions without parsing CLI arguments or accessing the full dataset.
    path = Path(__file__).resolve().parents[1] / 'eval_ijbc.py'
    tree = ast.parse(path.read_text())
    names = {'replace_nonfinite', 'image2template_feature', 'verification'}
    tree.body = [node for node in tree.body
                 if isinstance(node, ast.FunctionDef) and node.name in names]
    scope = {'np': np, 'sklearn': sklearn}
    exec(compile(tree, str(path), 'exec'), scope)
    return scope


def test_raw_templates_keep_media_means_and_template_sums():
    functions = evaluator_functions()
    aggregate = functions['image2template_feature']
    images = np.array([[2., 0.], [4., 0.], [0., 4.], [6., 8.], [0., 0.]])
    tids = np.array([10, 10, 10, 20, 30])
    mids = np.array([1, 1, 2, 3, 4])
    raw, ids = aggregate(images, tids, mids, normalize=False)
    cosine, cosine_ids = aggregate(images, tids, mids)
    np.testing.assert_array_equal(ids, cosine_ids)
    np.testing.assert_allclose(raw, [[3, 4], [6, 8], [0, 0]])
    np.testing.assert_allclose(cosine, [[.6, .8], [.6, .8], [0, 0]])
    p1, p2 = np.array([10, 10, 20]), np.array([20, 30, 30])
    verify = functions['verification']
    np.testing.assert_allclose(verify(raw, ids, p1, p2), [50, 0, 0])
    np.testing.assert_allclose(verify(cosine, ids, p1, p2), [1, 0, 0])
    np.testing.assert_allclose(
        verify(sklearn.preprocessing.normalize(raw), ids, p1, p2),
        verify(cosine, ids, p1, p2))


def test_comparison_uses_far_constraint_and_signed_percentage_points():
    def row(tar):
        return {'points': {'0.0001': {
            'tar_percent': 99.,
            'at_or_below_requested_far': {'tar_percent': tar, 'actual_far': .00009}}}}
    point = compare_tar_rows(row(95.), row(71.))['0.0001']
    assert point['drop_percentage_points'] == 24.
    assert point['cosine_actual_far'] <= .0001
    assert compare_tar_rows(row(71.), row(95.))['0.0001']['drop_percentage_points'] == -24.
