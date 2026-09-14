"""Offline IJB-C template aggregation for cached affine-head calibration.

No inference operations are introduced. Raw original/flip features retain
their norms, receive detector weights, average within media, and sum within
templates. L2 normalization belongs to the subsequent plaintext pair score.
"""
import numpy as np
import torch


def template_split(templates, fit_count, validation_count, seed):
    """Select disjoint template IDs and every source row belonging to them."""
    templates = np.asarray(templates)
    if templates.ndim != 1 or fit_count < 1 or validation_count < 1:
        raise ValueError('one-dimensional template IDs and positive counts required')
    unique = np.unique(templates)
    if fit_count + validation_count > len(unique):
        raise ValueError('not enough distinct templates')
    chosen = np.random.default_rng(seed).permutation(unique)
    fit = np.sort(chosen[:fit_count])
    validation = np.sort(chosen[fit_count:fit_count+validation_count])
    return dict(fit_templates=fit, validation_templates=validation,
                fit_source_images=np.flatnonzero(np.isin(templates, fit)),
                validation_source_images=np.flatnonzero(np.isin(templates, validation)))


def aggregate_templates(orientations, templates, medias, detector_scores):
    """Return raw template sums, affine-bias weights, and sorted template IDs.

    Inputs are N x 2 x D raw embeddings and N-element metadata. A per-view
    affine map A*x+b commutes with aggregation as T*A.T + bias_weight*b.
    bias_weight includes both views and detector/media weights; it is not
    generally one or the number of images. Caller must supply complete
    templates (use template_split on the full metadata before extraction).
    """
    if orientations.ndim != 3 or orientations.shape[1] != 2:
        raise ValueError('expected N x 2 x D original/flip embeddings')
    device, dtype = orientations.device, orientations.dtype
    t = torch.as_tensor(templates, device=device, dtype=torch.long)
    m = torch.as_tensor(medias, device=device, dtype=torch.long)
    d = torch.as_tensor(detector_scores, device=device, dtype=dtype)
    n = len(orientations)
    if n == 0 or any(v.shape != (n,) for v in (t, m, d)):
        raise ValueError('nonempty matching metadata required')
    if not torch.isfinite(orientations).all() or not torch.isfinite(d).all() or (d < 0).any():
        raise ValueError('finite embeddings and nonnegative finite detector scores required')
    groups, inverse, counts = torch.unique(torch.stack((t, m), 1), dim=0,
                                           return_inverse=True, return_counts=True)
    weighted = orientations.sum(1) * d[:, None]
    media = weighted.new_zeros((len(groups), weighted.shape[1]))
    media.index_add_(0, inverse, weighted)
    media = media / counts[:, None]
    media_bias = d.new_zeros(len(groups))
    media_bias.index_add_(0, inverse, 2*d)
    media_bias = media_bias / counts
    ids, media_to_template = torch.unique(groups[:, 0], sorted=True, return_inverse=True)
    result = weighted.new_zeros((len(ids), weighted.shape[1]))
    result.index_add_(0, media_to_template, media)
    bias_weight = d.new_zeros(len(ids))
    bias_weight.index_add_(0, media_to_template, media_bias)
    return result, bias_weight, ids
