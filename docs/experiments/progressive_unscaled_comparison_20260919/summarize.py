"""Verify full-evaluation coverage and compare with the fixed MS1MV3-scaled run.

Run from the repository root after the unscaled IJB-C job completes.
"""
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

ROOT = Path('work_dirs/progressive_best_unscaled_ijbc_20260919')
OLD_ROOT = Path('work_dirs/progressive_best_bts_ms1k_20260917')
DEST = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text())


def records(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def main():
    baseline_dir = ROOT / 'result/unscaled_nonfinite_only'
    scaled_dir = OLD_ROOT / 'ijbc_full/ms1k_bts_zero_failed'
    baseline = read(ROOT / 'result/failures.summary.json')
    scaled = read(OLD_ROOT / 'ijbc_full/bts.summary.json')
    assert baseline['source_images'] == scaled['source_images'] == 469375
    assert baseline['augmented_rows'] == scaled['augmented_rows'] == 938750
    assert not baseline['range_check_enabled'] and baseline['interval'] is None
    assert baseline['range_failed_source_images'] == 0
    before_records = records(ROOT / 'result/failures.failures.jsonl')
    after_records = records(OLD_ROOT / 'ijbc_full/bts.failures.jsonl')
    before_ids = {r['source_index'] for r in before_records}
    after_ids = {r['source_index'] for r in after_records}
    after_nonfinite_ids = {r['source_index'] for r in after_records if any(
        v['embedding_nonfinite'] or any(b['nonfinite'] for b in v['boundaries'].values())
        for v in r['views'])}
    names = Path('ijb/IJBC/meta/ijbc_name_5pts_score.txt').read_text().splitlines()
    assert len(before_records) == len(before_ids) == baseline['failed_source_images']
    assert baseline['zeroed_augmented_rows'] == 2 * len(before_ids)
    for r in before_records:
        assert r['image_name'] == names[r['source_index']].split()[0]
        assert r['failed'] and r['embedding_zeroed']
    before_scores = np.load(baseline_dir / 'ijbc.npy', mmap_mode='r')
    after_scores = np.load(scaled_dir / 'ijbc.npy', mmap_mode='r')
    assert before_scores.shape == after_scores.shape == (15658489,)
    assert np.isfinite(before_scores).all() and np.isfinite(after_scores).all()
    metadata = read(OLD_ROOT / 'scaled/calibration.json')
    source = Path(metadata['settings']['checkpoint'])
    digest = hashlib.sha256()
    with source.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    assert digest.hexdigest() == metadata['provenance']['source_sha256']
    before = read(baseline_dir / 'ijbc_tar_at_far_raw.json')[0]['points']
    after = read(scaled_dir / 'ijbc_tar_at_far_raw.json')[0]['points']
    points = []
    for far, b in before.items():
        a = after[far]
        bs, ass = b['at_or_below_requested_far'], a['at_or_below_requested_far']
        assert bs['actual_far'] <= float(far) and ass['actual_far'] <= float(far)
        points.append(dict(far=float(far), unscaled_strict_tar=bs['tar_percent'],
                           scaled_strict_tar=ass['tar_percent'],
                           scaled_minus_unscaled_strict_pp=ass['tar_percent'] - bs['tar_percent'],
                           unscaled_nearest_tar=b['tar_percent'], scaled_nearest_tar=a['tar_percent'],
                           scaled_minus_unscaled_nearest_pp=a['tar_percent'] - b['tar_percent']))
    result = dict(source_checkpoint=str(source), source_sha256=digest.hexdigest(),
                  baseline_job_id='403247', source_images=469375, views=938750, pairs=15658489,
                  all_scores_finite=True, unscaled_failed_images=len(before_ids),
                  scaled_failed_images=len(after_ids),
                  nonfinite_failure_ids_identical=before_ids == after_nonfinite_ids,
                  extra_scaled_failure_indices=sorted(after_ids - before_ids), points=points)
    (DEST / 'comparison.json').write_text(json.dumps(result, indent=2) + '\n')
    for source_file, name in [(baseline_dir / 'ijbc_tar_at_far_raw.json', 'unscaled_tar_at_far_raw.json'),
                              (ROOT / 'result/failures.summary.json', 'unscaled_failure_summary.json'),
                              (ROOT / 'result/failures.failures.jsonl', 'unscaled_failures.jsonl')]:
        shutil.copyfile(source_file, DEST / name)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
