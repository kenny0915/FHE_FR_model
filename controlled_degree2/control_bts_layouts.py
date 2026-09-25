"""Export and verify all five requested control BTS layouts on identical MS1MV3 rows."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from controlled_degree2.rescale_residual_graph import BTS_LAYOUTS, sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--dataset-root', required=True)
    parser.add_argument('--output-root', required=True)
    args = parser.parse_args()
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=False)
    source_hash = sha256_file(args.checkpoint)
    reports = {}
    manifest_hashes = set()
    for name, boundaries in BTS_LAYOUTS.items():
        output = root / name
        subprocess.run([sys.executable, '-m', 'controlled_degree2.calibrate_bts_ms1mv3',
            '--checkpoint', args.checkpoint, '--dataset-root', args.dataset_root,
            '--output-dir', str(output), '--images', '1000', '--seed', '42',
            '--target-absmax', '0.8', '--batch-size', '64', '--boundaries', *boundaries], check=True)
        report = json.loads((output / 'calibration.json').read_text())
        assert report['equivalence_passed'] and report['equivalence']['images'] == 1000
        assert report['provenance']['source_sha256'] == source_hash
        assert report['provenance']['boundaries'] == list(boundaries)
        manifest_hashes.add(report['provenance']['image_manifest_sha256'])
        reports[name] = report
    assert len(manifest_hashes) == 1
    assert sha256_file(args.checkpoint) == source_hash
    (root / 'summary.json').write_text(json.dumps(reports, indent=2, allow_nan=False) + '\n')
    print('All five exports reloaded and verified; identical calibration image manifests.', flush=True)


if __name__ == '__main__':
    main()
