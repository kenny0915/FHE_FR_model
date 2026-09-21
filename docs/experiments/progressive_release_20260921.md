# Scaled progressive teammate ZIP (2026-09-21)

Archive: `outputs/fhe_fr_progressive_ms1k_20260921.zip` (163,513,004 bytes,
approximately 155.94 MiB). External checksum file has the `.zip.sha256` suffix.

SHA-256: `01a4ff27bdb50c32fd434b1d2835518bafb063f41066d489d4c1c4a01c9eef52`.

The **only checkpoint** is `checkpoints/student_scaled.pt`, copied unchanged
from `work_dirs/progressive_best_bts_ms1k_20260917/scaled/student_scaled.pt`.
Its SHA-256 matches the original calibration report:
`f6b8fddafc8d8cfca5127d1732781b6eeaa7d9e7727524d5123e002bd1165dc1`.
The user explicitly requested that the unscaled checkpoint not be included.

The ZIP has one enclosing directory, `fhe_fr_progressive_ms1k_20260921`, with
450 hashed payload files and a `MANIFEST.json`. It includes committed Python
and shell source, the original environment description, selected calibration
and accuracy reports, and a Chinese quick-start README. It does not include
face datasets, optimizer states, or other model weights.

Runtime source snapshot: commit `46f2b1a397e2955303c2f97272fa3b4765eeae23`.
Explicit release templates come from `tools/progressive_release/` and are
hashed in the manifest. Unrelated working-tree edits are not included.

## Verification

The ZIP was CRC-checked and extracted under an independent temporary directory.
With `PYTHONPATH` removed, the bundled loader strictly loaded the checkpoint
using source from that extracted directory. All 450 manifest hashes passed.
The loaded model has 25 DirectQuadratic modules and no PReLU modules.

`verify_bundle.py` and `example.py` passed on CPU using a zero tensor. The
image-reading/output path also passed using a synthetic 112x112 solid RGB image
and `--zero-failed --output smoke_embedding.pt`. Output shape was [1,512],
embeddings and all six boundary maps were finite, and the synthetic inputs did
not fail the range check. These are package smoke tests, not new accuracy tests.
The archived full IJB-C results remain the accuracy evidence.

[Build and isolated validation records](progressive_release_20260921.json).

## Rebuilding and teammate use

From the original repository with the scaled checkpoint available:

```bash
python tools/progressive_release/build_bundle.py --output outputs/NEW_BUNDLE.zip
```

The builder reads committed runtime source from the current HEAD, verifies the
checkpoint against the committed calibration hash, and copies only the
explicit release templates from the working tree. It refuses to overwrite an
existing ZIP. Rebuilding at a later HEAD can intentionally change source content;
the manifest records the exact commit and hashes for each build.

The recipient should unzip the archive, enter its directory, and run:

```bash
python -m pip install -r requirements-inference.txt
python verify_bundle.py
python example.py
```

See the bundled README for aligned RGB input format, actual-image inference,
BTS failure filtering, frozen-BN assumptions, and optional calibration/evaluation
commands. The archive and weights remain outside Git; the builder, templates,
and release record are committed.
