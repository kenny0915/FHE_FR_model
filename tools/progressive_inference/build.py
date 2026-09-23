"""Build the five-file inference-only handoff; no training/evaluation repository."""
import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def main():
    checkpoint = ROOT / 'work_dirs/progressive_best_bts_ms1k_20260917/scaled/student_scaled.pt'
    expected = json.loads((ROOT / 'docs/experiments/progressive_best_bts_ms1k_20260917/calibration.json').read_text())['checkpoint_sha256']
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if digest != expected:
        raise ValueError('Checkpoint SHA-256 mismatch')
    output = ROOT / 'outputs/fhe_fr_progressive_ms1k_inference.zip'
    output.parent.mkdir(parents=True, exist_ok=True)
    files = {'model.pt': checkpoint, 'backbone.py': ROOT / 'backbones/iresnet.py',
             **{name: HERE / name for name in ('inference.py', 'README.md', 'requirements.txt')}}
    with zipfile.ZipFile(output, 'x', zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, source in files.items():
            archive.write(source, 'fhe_fr_progressive_inference/' + name)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix('.zip.sha256').write_text(digest + '  ' + output.name + '\n')
    print(json.dumps(dict(zip=str(output), bytes=output.stat().st_size, sha256=digest, files=list(files)), indent=2))


if __name__ == '__main__':
    main()
