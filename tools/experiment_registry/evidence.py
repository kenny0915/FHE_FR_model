"""Explicit local evidence capture; portable rebuilds read only saved snapshots."""
import csv
import hashlib
import io
import json
from pathlib import Path


def sha256(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def capture(root):
    inventory = json.loads((root / 'experiments/inventory.json').read_text())
    paths = {p for run in inventory for p in run['configs'] if Path(p).name == 'config.json'}
    paths.update(p for run in inventory for p in run['metrics'] if p.endswith('_tar_at_far.csv'))
    paths.update(['poly_run10/README.md', 'poly_run10/train/config.json'])
    records = []
    for path in sorted(paths):
        source = root / path
        if not source.is_file():
            raise FileNotFoundError('Evidence disappeared: ' + path)
        raw = source.read_bytes()
        text = raw.decode('utf-8')
        records.append(dict(path=path, sha256=hashlib.sha256(raw).hexdigest(),
                            format=source.suffix[1:], text=text))
    (root / 'experiments/local_evidence.json').write_text(json.dumps(records, indent=2) + '\n')
    return records


def local_rows(root):
    records = json.loads((root / 'experiments/local_evidence.json').read_text())
    rows = []
    for entry in records:
        if entry['format'] != 'csv':
            continue
        if hashlib.sha256(entry['text'].encode()).hexdigest() != entry['sha256']:
            raise ValueError('Snapshot hash mismatch: ' + entry['path'])
        reader = csv.DictReader(io.StringIO(entry['text']))
        if not reader.fieldnames or reader.fieldnames[0] != 'Methods':
            raise ValueError('Unrecognized TAR CSV header: ' + entry['path'])
        for record in reader:
            for far in reader.fieldnames[1:]:
                requested, tar = float(far), float(record[far])
                if not 0 <= tar <= 100 or not 0 < requested <= 1:
                    raise ValueError('Invalid metric: ' + entry['path'])
                rows.append(dict(artifact_group=entry['path'].split('/')[1],
                    original_csv=entry['path'], source_sha256=entry['sha256'],
                    method=record['Methods'], requested_far=requested, tar_percent=tar,
                    far_rule='unknown', actual_far='unknown', checkpoint_sha256='unknown',
                    calibration='unknown', finite_scope='unknown', review_status='needs_protocol_review'))
    return rows


def canonical_path(value):
    if not isinstance(value, str):
        return value
    # Historical paths under another checkout may still identify this repository's artifacts.
    if '/work_dirs/' in value:
        return 'work_dirs/' + value.split('/work_dirs/', 1)[1]
    return value


def lineage_rows(root):
    """Edges come from explicit config fields, not adjacency or directory naming."""
    records = json.loads((root / 'experiments/local_evidence.json').read_text())
    rows = []
    for entry in records:
        if entry['format'] != 'json':
            continue
        config = json.loads(entry['text'])
        for key in ('student_init', 'teacher', 'teacher_weights', 'resume', 'checkpoint', 'pretrained'):
            value = config.get(key)
            if not isinstance(value, str) or not value:
                continue
            rows.append(dict(child_config=entry['path'], relation=key,
                             parent_checkpoint=canonical_path(value), original_value=value,
                             parent_sha256='unknown', source=entry['path'], source_sha256=entry['sha256']))
    chain_path = 'docs/0915_result/calibrated_training_chain.json'
    chain = json.loads((root / chain_path).read_text())['chain']
    for item in chain:
        # Recorded source hashes are authoritative; do not guess the previous list item.
        parent = (item.get('calibration') or {}).get('source_sha256')
        if parent:
            rows.append(dict(child_config=canonical_path(item['checkpoint']), relation='calibration_source',
                             parent_checkpoint='unknown', original_value=parent,
                             parent_sha256=parent, source=chain_path, source_sha256=sha256(root / chain_path)))
    return rows
