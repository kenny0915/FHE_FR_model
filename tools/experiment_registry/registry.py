"""Standard-library experiment inventory and source-backed result adapters."""
import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def family(name):
    """Naming hints only: never infer completion, degree or data usage."""
    if name in {'ms1mv3_r50', 'channelwise_prelu_verified_20260915'}:
        return 'baseline'
    if any(x in name for x in ('bts_', 'unscaled_ijbc', 'folded')):
        return 'deployment'
    if name.startswith('channelwise'):
        return 'channelwise_calibration'
    if any(x in name for x in ('poolformer', 'mbf', 'patch_cnn', '_nf')):
        return 'other_backbones'
    if any(x in name for x in ('controlled_', 'recipe_a', 'shared_d2', 'adaptive_conversion', 'polish_abcd', 'unclipped_round')):
        return 'controlled_degree2'
    if any(x in name for x in ('nl13', 'nl9', 'linear9', 'linear_sites', 'selective9')):
        return 'reduced_nonlinearity'
    if name.startswith(('ms1mv3_r50', 'casia_r50', 'quadT12', 'poly_run10')):
        return 'polynomial_conversion'
    return 'unclassified'


def inventory(root):
    base = root / 'work_dirs'
    if not base.is_dir():
        raise FileNotFoundError('work_dirs is absent; preserve the committed inventory snapshot')
    rows = []
    for directory in sorted(base.iterdir()):
        if not directory.is_dir() or directory.is_symlink():
            continue
        artifacts = dict(checkpoints=[], configs=[], metrics=[], logs=[])
        # Walk metadata only; no checkpoint loading, hashing or dataset traversal.
        for current, dirs, files in os.walk(directory, followlinks=False):
            dirs[:] = sorted(d for d in dirs if d not in {'__pycache__', 'tensorboard', '.git'}
                             and not (Path(current) / d).is_symlink())
            for name in sorted(files):
                path = Path(current) / name
                if path.is_symlink():
                    continue
                rel = path.relative_to(root).as_posix()
                if path.suffix in {'.pt', '.pth', '.ckpt'}:
                    artifacts['checkpoints'].append(rel)
                elif path.suffix == '.json' and any(x in name.lower() for x in ('config', 'manifest')):
                    artifacts['configs'].append(rel)
                elif path.suffix in {'.json', '.csv'}:
                    artifacts['metrics'].append(rel)
                elif path.suffix in {'.log', '.out', '.err'}:
                    artifacts['logs'].append(rel)
        rows.append(dict(run_id=directory.name, path=directory.relative_to(root).as_posix(),
                         family_hint=family(directory.name), classification_basis='directory_name_only',
                         review_status='needs_review', **artifacts))
    return rows


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized_results(root):
    source = 'docs/0915_result/metrics.json'
    data = json.loads((root / source).read_text())
    registry = json.loads((root / 'experiments/representatives.json').read_text())
    metadata = {r['metrics_key']: r for r in registry if r.get('metrics_key')}
    rows = []
    for key, model in sorted(data['models'].items()):
        meta = metadata[key]  # Fail rather than silently assign unknown calibration semantics.
        for point in model['points']:
            for rule, tar, actual in [('strict', 'tar_percent', 'actual_far'),
                                      ('nearest', 'nearest_tar_percent', 'nearest_actual_far')]:
                rows.append(dict(run_id=meta['run_id'], evaluation_id='0915_' + key,
                    checkpoint=model['checkpoint'], checkpoint_sha256=model['checkpoint_sha256'],
                    dataset='IJB-C', protocol='0915_full_original_flip', calibration=meta['calibration'],
                    finite_scope=meta['finite_scope'], nonfinite_embedding_rows=0,
                    far_rule=rule, requested_far=point['requested_far'], actual_far=point[actual],
                    tar_percent=point[tar], threshold=point['threshold'] if rule == 'strict' else 'unknown',
                    source=source, source_sha256=digest(root / source),
                    scores=model['scores'], scores_sha256=model['scores_sha256']))
    return rows


def historical_results(root):
    source = 'docs/0915_result/selection_evidence.json'
    rows = []
    for entry in json.loads((root / source).read_text()):
        reader = csv.DictReader(io.StringIO(entry['csv']))
        for record in reader:
            for far, value in record.items():
                if far == 'Methods':
                    continue
                float(far)
                rows.append(dict(artifact_group=entry['tar_csv'].split('/')[1],
                    method=record['Methods'], requested_far=far, tar_percent=float(value),
                    far_rule='historical_nearest', actual_far='unknown',
                    checkpoint_sha256='unknown', calibration='unknown', review_status='needs_review',
                    finite_evidence=json.dumps({'logs': entry['logs'], 'audits': entry['finite_audits']}, sort_keys=True),
                    source=source, source_sha256=digest(root / source), original_csv=entry['tar_csv']))
    return rows


def build(root, scan=False):
    if scan:
        rows = inventory(root)
        (root / 'experiments/inventory.json').write_text(json.dumps(rows, indent=2) + '\n')
        index = [dict(run_id=r['run_id'], path=r['path'], family_hint=r['family_hint'],
                      review_status=r['review_status'], **{k + '_count': len(r[k])
                      for k in ('checkpoints', 'configs', 'metrics', 'logs')}) for r in rows]
        write_csv(root / 'experiments/index.csv', index,
                  ['run_id', 'path', 'family_hint', 'review_status', 'checkpoints_count', 'configs_count', 'metrics_count', 'logs_count'])
    for name, rows in [('evaluations', normalized_results(root)), ('historical_evidence', historical_results(root))]:
        write_csv(root / ('reports/tables/' + name + '.csv'), rows, list(rows[0]))
        print('{}: {} rows'.format(name, len(rows)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scan', action='store_true', help='Refresh local work_dirs inventory; requires artifacts')
    args = parser.parse_args()
    build(ROOT, args.scan)
