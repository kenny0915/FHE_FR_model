"""Reversible same-filesystem relocation of root archives and scheduler logs.

No file deletion, copying, symlink traversal, or checkpoint loading. A journal is
written before every rename and supports partial-failure recovery.
"""
import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
JOURNAL = 'experiments/storage_migration.json'


def identity(path):
    stat = path.stat()
    return dict(bytes=stat.st_size, device=stat.st_dev, inode=stat.st_ino, mtime_ns=stat.st_mtime_ns)


def plan(root):
    rows = []
    for p in sorted(root.iterdir()):
        if p.is_symlink() or not p.is_file():
            continue
        if p.name.endswith(('.zip', '.tar', '.tar.gz', '.tgz')):
            group = 'archives'
        elif p.name.startswith('adaptive-') and p.suffix in ('.out', '.err'):
            group = 'scheduler_logs'
        else:
            continue
        rows.append(dict(source=p.name, destination='artifacts/' + group + '/' + p.name,
                         identity=identity(p), state='planned',
                         retention='retain_no_verified_external_backup', sha256='not_computed'))
    return rows


def save(root, rows):
    target = root / JOURNAL
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix('.tmp')
    temporary.write_text(json.dumps(rows, indent=2) + '\n')
    temporary.replace(target)


def safe_path(root, value):
    rel = Path(value)
    if rel.is_absolute() or '..' in rel.parts:
        raise ValueError('Journal path outside repository')
    path = root / rel
    if path.is_symlink() or any(p.is_symlink() for p in path.parents if p != root.parent):
        raise ValueError('Symlink in storage path')
    return path


def relocate(root, rows, restore=False):
    # Preflight all moves before modifying anything; support interrupted journal states.
    moves = []
    for row in rows:
        src = safe_path(root, row['destination'] if restore else row['source'])
        dst = safe_path(root, row['source'] if restore else row['destination'])
        if not src.exists() and dst.is_file() and identity(dst) == row['identity']:
            continue  # already moved, including crash between rename and journal update
        if not src.is_file() or identity(src) != row['identity']:
            raise ValueError('Source identity changed: ' + str(src))
        if dst.exists():
            raise FileExistsError(dst)
        parent = dst.parent
        while not parent.exists():
            parent = parent.parent
        if parent.stat().st_dev != row['identity']['device']:
            raise ValueError('Relocation must remain on the same filesystem')
        moves.append((src, dst))
    save(root, rows)
    for src, dst in moves:
        dst.parent.mkdir(parents=True, exist_ok=True)
        # Preflight was successful; recheck destination immediately before rename.
        if dst.exists():
            raise FileExistsError(dst)
        os.rename(src, dst)
        assert identity(dst) == next(r['identity'] for r in rows
                                     if r['source' if restore else 'destination'] == dst.relative_to(root).as_posix())
        for row in rows:
            target = safe_path(root, row['source'] if restore else row['destination'])
            if target.is_file() and identity(target) == row['identity']:
                row['state'] = 'restored' if restore else 'relocated'
        save(root, rows)
    # Reconcile a fully completed rename whose journal write was interrupted.
    for row in rows:
        row['state'] = 'restored' if restore else 'relocated'
    save(root, rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--apply', action='store_true')
    mode.add_argument('--restore', action='store_true')
    args = parser.parse_args()
    if args.restore:
        relocate(ROOT, json.loads((ROOT / JOURNAL).read_text()), restore=True)
    elif args.apply:
        rows = json.loads((ROOT / JOURNAL).read_text()) if (ROOT / JOURNAL).exists() else plan(ROOT)
        relocate(ROOT, rows)
    else:
        print(json.dumps(plan(ROOT), indent=2))
