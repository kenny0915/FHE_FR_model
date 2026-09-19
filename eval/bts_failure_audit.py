"""Fail closed at six BTS boundaries; zero both views of a failed source image."""
import json
import math
from pathlib import Path

import torch

from controlled_degree2.rescale_residual_graph import BOUNDARIES


class BTSFailureAudit:
    def __init__(self, prefix, check_range=True):
        self.check_range = bool(check_range)
        self.prefix = Path(prefix)
        self.prefix.parent.mkdir(parents=True, exist_ok=True)
        self.stream = open(str(self.prefix) + '.failures.jsonl', 'x')
        self.handles = []
        self.current = {}
        self.counts = dict(source_images=0, augmented_rows=0, failed_source_images=0,
                           failed_augmented_rows=0, range_failed_source_images=0,
                           nonfinite_source_images=0, embedding_nonfinite_source_images=0,
                           zeroed_augmented_rows=0)
        self.boundaries = {name: dict(out_of_range_augmented_rows=0,
                                    nonfinite_augmented_rows=0,
                                    failed_source_images=0,
                                    finite_row_min=None, finite_row_max=None)
                           for name in BOUNDARIES}

    def attach(self, model):
        for handle in self.handles:
            handle.remove()
        modules = [model.get_submodule(name) for name in BOUNDARIES]
        self.handles = [module.register_forward_hook(
            lambda m, i, o, name=name: self.observe(name, o))
            for name, module in zip(BOUNDARIES, modules)]

    def start_batch(self):
        self.current = {}

    def observe(self, name, output):
        flat = output.detach().flatten(1)
        # NaN/Inf in any element makes at least one extremum nonfinite.
        self.current[name] = torch.stack((flat.amin(1), flat.amax(1)), dim=1)

    def filter_embeddings(self, features, source_indices, source_names):
        if set(self.current) != set(BOUNDARIES):
            raise RuntimeError('Incomplete BTS boundary coverage')
        rows = len(features)
        if rows % 2 or len(source_indices) != rows // 2 or len(source_names) != rows // 2:
            raise ValueError('Expected interleaved original/flip rows for each source')
        limits = torch.stack([self.current[n] for n in BOUNDARIES], dim=1).cpu()
        if limits.shape != (rows, len(BOUNDARIES), 2):
            raise ValueError('BTS hook row count mismatch')
        finite = torch.isfinite(limits).all(2)
        outside = ((limits[:, :, 0] < -1) | (limits[:, :, 1] > 1)) if self.check_range else torch.zeros_like(finite)
        embedding_nonfinite = (~torch.isfinite(features).all(1)).cpu()
        range_bad = outside.any(1)
        nonfinite_bad = (~finite).any(1) | embedding_nonfinite
        bad = range_bad | nonfinite_bad
        source_bad = bad.reshape(-1, 2).any(1)
        source_indices = list(source_indices)
        self.counts['source_images'] += rows // 2
        self.counts['augmented_rows'] += rows
        self.counts['failed_source_images'] += int(source_bad.sum())
        self.counts['failed_augmented_rows'] += int(bad.sum())
        self.counts['range_failed_source_images'] += int(range_bad.reshape(-1, 2).any(1).sum())
        self.counts['nonfinite_source_images'] += int(nonfinite_bad.reshape(-1, 2).any(1).sum())
        self.counts['embedding_nonfinite_source_images'] += int(embedding_nonfinite.reshape(-1, 2).any(1).sum())
        self.counts['zeroed_augmented_rows'] += 2 * int(source_bad.sum())
        for j, name in enumerate(BOUNDARIES):
            entry = self.boundaries[name]
            entry['out_of_range_augmented_rows'] += int(outside[:, j].sum())
            entry['nonfinite_augmented_rows'] += int((~finite[:, j]).sum())
            entry['failed_source_images'] += int((outside[:, j] | ~finite[:, j]).reshape(-1, 2).any(1).sum())
            values = limits[finite[:, j], j]
            if len(values):
                minimum, maximum = values[:, 0].min().item(), values[:, 1].max().item()
                entry['finite_row_min'] = minimum if entry['finite_row_min'] is None else min(entry['finite_row_min'], minimum)
                entry['finite_row_max'] = maximum if entry['finite_row_max'] is None else max(entry['finite_row_max'], maximum)
        for source in source_bad.nonzero().flatten().tolist():
            record = dict(source_index=int(source_indices[source]), image_name=str(source_names[source]),
                          failed=True, embedding_zeroed=True, views=[])
            for orientation in range(2):
                row = 2 * source + orientation
                values = {}
                for j, name in enumerate(BOUNDARIES):
                    lo, hi = limits[row, j].tolist()
                    values[name] = dict(min=lo if math.isfinite(lo) else None,
                                        max=hi if math.isfinite(hi) else None,
                                        out_of_range=bool(outside[row, j]), nonfinite=not bool(finite[row, j]))
                record['views'].append(dict(orientation='flip' if orientation else 'original',
                    failed=bool(bad[row]), embedding_nonfinite=bool(embedding_nonfinite[row]), boundaries=values))
            self.stream.write(json.dumps(record, allow_nan=False) + '\n')
        self.stream.flush()
        result = features.clone()
        result[source_bad.repeat_interleave(2).to(features.device)] = 0
        if not torch.isfinite(result).all():
            raise FloatingPointError('Nonfinite embeddings survived failure filtering')
        return result

    def finish(self, expected_images):
        for handle in self.handles:
            handle.remove()
        self.handles = []
        self.stream.close()
        if self.counts['source_images'] != expected_images:
            raise RuntimeError('Incomplete IJB image coverage')
        summary = dict(self.counts, boundaries=self.boundaries, interval=[-1, 1] if self.check_range else None,
                       range_check_enabled=self.check_range,
                       failure_policy=('any boundary outside [-1,1] or ' if self.check_range else '')
                                      + 'nonfinite boundary/embedding; either view fails => zero both views before flip fusion/template aggregation',
                       failed_source_fraction=self.counts['failed_source_images'] / max(1, expected_images))
        Path(str(self.prefix) + '.summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
        return summary
