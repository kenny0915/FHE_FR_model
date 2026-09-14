import numpy as np
import pytest
import torch

from controlled_degree2.template_geometry import aggregate_templates, template_split


def test_split_keeps_every_image_and_templates_disjoint():
    templates = np.array([9, 1, 9, 4, 1, 8, 4, 9, 7, 8])
    a = template_split(templates, 3, 2, 123)
    b = template_split(templates, 3, 2, 123)
    for key in a:
        np.testing.assert_array_equal(a[key], b[key])
    assert not set(a['fit_templates']) & set(a['validation_templates'])
    for split in ('fit', 'validation'):
        np.testing.assert_array_equal(a[split+'_source_images'],
                                      np.flatnonzero(np.isin(templates, a[split+'_templates'])))
    assert len(a['fit_source_images']) + len(a['validation_source_images']) == len(templates)
    with pytest.raises(ValueError):
        template_split(templates, 4, 2, 1)


def test_matches_explicit_media_protocol_and_affine_commutation():
    torch.manual_seed(123)
    x = torch.randn(6, 2, 4, dtype=torch.float64)
    # Media ID 3 appears in two templates and must not be merged across them.
    t = torch.tensor([20, 10, 20, 10, 20, 10])
    m = torch.tensor([3, 3, 4, 3, 3, 9])
    d = torch.tensor([.2, .9, .5, .4, .8, 0.], dtype=x.dtype)
    result, weight, ids = aggregate_templates(x, t, m, d)
    expected = []
    for tid in (10, 20):
        terms = []
        for mid in torch.unique(m[t == tid]):
            rows = (t == tid) & (m == mid)
            terms.append((x[rows].sum(1)*d[rows, None]).mean(0))
        expected.append(torch.stack(terms).sum(0))
    torch.testing.assert_close(result, torch.stack(expected))
    torch.testing.assert_close(weight, torch.tensor([1.3, 2.0], dtype=x.dtype))
    assert ids.tolist() == [10, 20]
    matrix, bias = torch.randn(4, 4, dtype=x.dtype), torch.randn(4, dtype=x.dtype)
    mapped, _, _ = aggregate_templates(x@matrix.T+bias, t, m, d)
    torch.testing.assert_close(mapped, result@matrix.T+weight[:, None]*bias)


def test_aggregation_preserves_gradients_and_rejects_bad_inputs():
    x = torch.randn(3, 2, 4, requires_grad=True)
    out, _, _ = aggregate_templates(x, [1, 1, 2], [3, 3, 3], [1., 1., 1.])
    out.sum().backward()
    torch.testing.assert_close(x.grad[:2], torch.full_like(x.grad[:2], .5))
    torch.testing.assert_close(x.grad[2], torch.ones_like(x.grad[2]))
    with pytest.raises(ValueError):
        aggregate_templates(x, [1, 1], [3, 3, 3], [1., 1., 1.])
    with pytest.raises(ValueError):
        aggregate_templates(x, [1, 1, 2], [3, 3, 3], [1., -1., 1.])
