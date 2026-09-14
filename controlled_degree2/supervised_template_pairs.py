"""Explicitly supervised IJB-C calibration helpers (training only)."""
import numpy as np
import torch
from torch.nn import functional as F


def restrict_pairs(pairs, template_ids):
    """Keep only pairs with both endpoints in the supplied fitting partition.

    Return local row indices and labels; canonicalize unordered duplicates and
    reject contradictory labels. The caller records the label-file hash.
    """
    pairs = np.asarray(pairs)
    ids = np.asarray(template_ids)
    if pairs.ndim != 2 or pairs.shape[1] != 3:
        raise ValueError('pairs must have three columns')
    if ids.ndim != 1 or len(np.unique(ids)) != len(ids):
        raise ValueError('template IDs must be unique')
    if not np.issubdtype(pairs.dtype, np.integer):
        raise ValueError('pair IDs and labels must be integers')
    if not np.isin(pairs[:, 2], [0, 1]).all():
        raise ValueError('pair labels must be binary')
    keep = np.isin(pairs[:, 0], ids) & np.isin(pairs[:, 1], ids)
    selected = pairs[keep]
    lookup = {int(value): index for index, value in enumerate(ids)}
    unique = {}
    for first, second, label in selected:
        if first == second:
            raise ValueError('self pairs are not valid calibration supervision')
        key = tuple(sorted((lookup[int(first)], lookup[int(second)])))
        if key in unique and unique[key] != int(label):
            raise ValueError('conflicting labels for an unordered pair')
        unique[key] = int(label)
    return np.array([[*key, label] for key, label in sorted(unique.items())],
                    dtype=np.int64).reshape(-1, 3)


def supervised_margin_loss(embeddings, pairs, positive_margin=.4, negative_margin=.2):
    """Balanced squared hinge on cosine scores; no inference operations added.

    All provided positive/negative pairs participate. Any negative mining must
    be performed explicitly by the caller and recorded in experiment metadata.
    """
    if not -1 <= negative_margin < positive_margin <= 1:
        raise ValueError('require -1 <= negative margin < positive margin <= 1')
    if pairs.ndim != 2 or pairs.shape[1] != 3 or pairs.dtype != torch.long:
        raise ValueError('pairs must be an int64 tensor with three columns')
    if not len(pairs) or (pairs[:, :2] < 0).any() or (pairs[:, :2] >= len(embeddings)).any():
        raise ValueError('pair row indices are empty or out of bounds')
    labels = pairs[:, 2]
    if not ((labels == 0) | (labels == 1)).all():
        raise ValueError('pair labels must be binary')
    positive = labels == 1
    if not positive.any() or positive.all():
        raise ValueError('both positive and negative pairs are required')
    normalized = F.normalize(embeddings, dim=1)
    scores = (normalized[pairs[:, 0]]*normalized[pairs[:, 1]]).sum(dim=1)
    pos = (positive_margin-scores[positive]).clamp_min(0).square().mean()
    neg = (scores[~positive]-negative_margin).clamp_min(0).square().mean()
    return pos+neg
