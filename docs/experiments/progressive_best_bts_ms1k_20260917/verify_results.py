"""Verify this completed experiment; run from the repository root."""
import json,hashlib
from pathlib import Path
root=Path('work_dirs/progressive_best_bts_ms1k_20260917')
summary=json.loads((root/'ijbc_full/bts.summary.json').read_text())
records=[json.loads(s) for s in (root/'ijbc_full/bts.failures.jsonl').read_text().splitlines()]
images=Path('ijb/IJBC/meta/ijbc_name_5pts_score.txt').read_text().splitlines()
assert len(images)==summary['source_images']==469375
assert summary['augmented_rows']==2*len(images)
assert len(records)==summary['failed_source_images']==21
assert len(set(r['source_index'] for r in records))==len(records)
assert summary['zeroed_augmented_rows']==2*len(records)
assert sum(v['failed'] for r in records for v in r['views'])==summary['failed_augmented_rows']
for r in records:
 assert r['failed'] and r['embedding_zeroed']
 assert images[r['source_index']].split()[0]==r['image_name']
 assert any(v['failed'] for v in r['views'])
 for v in r['views']:
  assert v['failed']==(v['embedding_nonfinite'] or any(b['out_of_range'] or b['nonfinite'] for b in v['boundaries'].values()))
for name,s in summary['boundaries'].items():
 assert sum(v['boundaries'][name]['out_of_range'] for r in records for v in r['views'])==s['out_of_range_augmented_rows']
 assert sum(v['boundaries'][name]['nonfinite'] for r in records for v in r['views'])==s['nonfinite_augmented_rows']
 assert sum(any(v['boundaries'][name]['out_of_range'] or v['boundaries'][name]['nonfinite'] for v in r['views']) for r in records)==s['failed_source_images']
cal=json.loads((root/'scaled/calibration.json').read_text())
manifest=json.loads((root/'scaled/calibration_images.json').read_text())
assert len(manifest)==len(set(m['dataset_index'] for m in manifest))==1000
assert hashlib.sha256((root/'scaled/calibration_images.json').read_bytes()).hexdigest()==cal['provenance']['image_manifest_sha256']
for p,digest in [(Path(cal['settings']['checkpoint']),cal['provenance']['source_sha256']),(root/'scaled/student_scaled.pt',cal['checkpoint_sha256'])]:
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 assert h.hexdigest()==digest
print('Verified: complete coverage, unique failure image IDs/names, view and boundary counts, zeroed-row count, 1000-image manifest, source/scaled checkpoint hashes.')
