"""Generate run manifests and human-readable comparison views from preserved evidence."""
import json
from pathlib import Path


def publish(root, evaluations, write_csv, digest):
    representatives = json.loads((root / 'experiments/representatives.json').read_text())
    records = json.loads((root / 'experiments/local_evidence.json').read_text())
    snapshot_hashes = {r['path']: r['sha256'] for r in records}
    inventory = json.loads((root / 'experiments/inventory.json').read_text())
    summaries = []
    for row in inventory:
        prefix = row['path'] + '/'
        configs = [r for r in records if r['path'].startswith(prefix) and r['format'] == 'json']
        metrics = [r for r in records if r['path'].startswith(prefix) and r['format'] == 'csv']
        rep = next((r for r in representatives if r['run_id'] == row['run_id']), None)
        summaries.append(dict(run_id=row['run_id'], family=rep['family'] if rep else row['family_hint'],
            evidence_status='representative_documented' if rep else 'local_evidence_captured' if configs or metrics else 'inventory_only',
            protocol_review='documented_in_source' if rep and (rep.get('metrics_key') or rep.get('raw_metrics')) else 'needs_review',
            config_snapshots=len(configs), tar_csv_snapshots=len(metrics),
            checkpoint_sha256=rep.get('checkpoint_sha256', 'unknown') if rep else 'unknown'))
    write_csv(root / 'reports/tables/run_summary.csv', summaries, list(summaries[0]))
    for rep in representatives:
        config_records = [r for r in records if r['format'] == 'json' and
                          (r['path'].startswith('work_dirs/' + rep['run_id'] + '/') or
                           (rep['run_id'] == 'poly_run10' and r['path'].startswith('poly_run10/')))]
        sources = sorted(set([rep['source'], rep['lineage']] +
                             [rep[k] for k in ('raw_metrics', 'failure_summary') if k in rep]))
        evidence = [dict(path=p, sha256=snapshot_hashes.get(p) or (digest(root / p) if (root / p).is_file() else 'unknown')) for p in sources]
        manifest = dict(schema_version=1, run_id=rep['run_id'], experiment_id=rep['family'],
                        family=rep['family'], status='historical_documented',
                        checkpoint=dict(path=rep.get('checkpoint'), sha256=rep.get('checkpoint_sha256')),
                        parent_checkpoint=dict(path=rep.get('parent_checkpoint'), sha256=rep.get('parent_checkpoint_sha256')),
                        code=dict(commit=rep.get('training_commit', 'unknown'), dirty=None),
                        metadata_source='experiments/representatives.json',
                        config_snapshots=[dict(path=r['path'], sha256=r['sha256'], config=json.loads(r['text'])) for r in config_records],
                        calibration=rep.get('calibration', 'unknown'), polynomial=rep.get('polynomial'),
                        actual_fhe_execution=rep['actual_fhe_execution'], evidence=evidence,
                        lineage_table='reports/tables/lineage.csv',
                        evaluations=[r for r in evaluations if r['run_id'] == rep['run_id']],
                        notes=rep.get('notes'))
        path = root / 'experiments/runs' / rep['run_id'] / 'manifest.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest, indent=2) + '\n')
    lines = ['# 代表結果總覽', '',
             '此表由 registry 產生，僅選 strict FAR=1e-4 作導覽；不是跨條件排名。完整 FAR、hash 與來源在 [CSV](tables/evaluations.csv)。', '',
             '| Run | TAR (%) | Calibration | Evidence scope | Failed sources | Zeroed views |',
             '|---|---:|---|---|---:|---:|']
    for row in evaluations:
        if row['far_rule'] == 'strict' and row['requested_far'] == 1e-4:
            lines.append('| {} | {:.5f} | {} | {} | {} | {} |'.format(
                row['run_id'], row['tar_percent'], row['calibration'], row['evidence_scope'],
                row['failed_source_images'], row['zeroed_augmented_rows']))
    lines += ['', 'unknown 表示來源未提供該項計數；不代表 0。source image 與 augmented view 是不同分母。',
              '0915 head-only 只保留 embedding audit；channelwise 的 IJBC 校準集數字不是隔離測試。',
              'progressive 兩列包含失敗補零，且縮放版本額外施加 range gate；均為浮點評估，沒有實測 FHE。',
              'run10 與失敗案例若未建立精確 checkpoint→evaluation 對應，不移入此表。', '']
    (root / 'reports/summary.md').write_text('\n'.join(lines))
