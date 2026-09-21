"""Build a self-contained teammate handoff from committed source + verified weights."""
import argparse
import hashlib
import io
import json
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEMPLATES = Path(__file__).resolve().parent
BUNDLE_NAME = 'fhe_fr_progressive_ms1k_20260921'


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def git(*arguments):
    return subprocess.check_output(['git', '-C', str(ROOT), *arguments])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs' / (BUNDLE_NAME + '.zip'))
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    source_commit = git('rev-parse', 'HEAD').decode().strip()
    tracked = git('ls-tree', '-r', '--name-only', source_commit).decode().splitlines()
    doc_folders = ('docs/experiments/progressive_best_bts_ms1k_20260917/',
                   'docs/experiments/progressive_unscaled_comparison_20260919/')
    selected = [p for p in tracked if Path(p).suffix in ('.py', '.sh', '.slurm', '.yml', '.yaml', '.toml')
                or p.startswith(doc_folders)
                or p in ('docs/experiments/progressive_best_bts_scaling_20260917.md',
                         'docs/experiments/progressive_best_bts_scaling_20260917.json')
                or (p.startswith('configs/') and Path(p).suffix == '.json')
                or Path(p).name.upper().startswith(('LICENSE', 'COPYING'))]
    calibration = json.loads(git('show', source_commit + ':docs/experiments/progressive_best_bts_ms1k_20260917/calibration.json'))
    checkpoints = {
        'student_scaled.pt': (ROOT / 'work_dirs/progressive_best_bts_ms1k_20260917/scaled/student_scaled.pt', calibration['checkpoint_sha256']),
    }
    for source, expected in checkpoints.values():
        if sha256(source) != expected:
            raise ValueError('Checkpoint hash mismatch: ' + str(source))
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='progressive_release_') as temporary:
        bundle = Path(temporary) / BUNDLE_NAME
        bundle.mkdir()
        archive = git('archive', '--format=zip', source_commit, '--', *selected)
        with zipfile.ZipFile(io.BytesIO(archive)) as source_archive:
            source_archive.extractall(bundle)
        # Templates are explicit release inputs; never copy unrelated working-tree files.
        for name in ('loader.py', 'example.py', 'verify_bundle.py', 'README.md', 'requirements-inference.txt'):
            shutil.copyfile(TEMPLATES / name, bundle / name)
        (bundle / 'checkpoints').mkdir()
        for name, (source, _) in checkpoints.items():
            shutil.copyfile(source, bundle / 'checkpoints' / name)
        files = {str(p.relative_to(bundle)): sha256(p) for p in sorted(bundle.rglob('*')) if p.is_file()}
        manifest = dict(source_repository='https://github.com/kenny0915/FHE_FR_model', source_commit=source_commit,
                        primary_checkpoint='checkpoints/student_scaled.pt',
                        tested_environment=dict(python='3.10.20', torch='2.1.2+cu118', numpy='1.23.5', pillow='12.2.0'),
                        files=files)
        (bundle / 'MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n')
        with zipfile.ZipFile(output, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as destination:
            for path in sorted(bundle.rglob('*')):
                if path.is_file():
                    destination.write(path, str(path.relative_to(bundle.parent)))
    checksum = sha256(output)
    output.with_suffix(output.suffix + '.sha256').write_text(checksum + '  ' + output.name + '\n')
    print(json.dumps(dict(zip=str(output), bytes=output.stat().st_size, sha256=checksum,
                         source_commit=source_commit, manifest_files=len(files)), indent=2))


if __name__ == '__main__':
    main()
