"""A relocation must be reversible and must never overwrite another artifact."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location('storage', Path(__file__).resolve().parents[1] / 'tools/experiment_registry/storage.py')
storage = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(storage)


class StorageOrganizationTests(unittest.TestCase):
    def test_move_restore_and_interrupted_journal_recovery(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'model.zip').write_bytes(b'archive evidence')
            rows = storage.plan(root)
            storage.relocate(root, rows)
            self.assertFalse((root / 'model.zip').exists())
            rows[0]['state'] = 'planned'  # simulate rename completed before journal update
            storage.relocate(root, rows)
            self.assertEqual(rows[0]['state'], 'relocated')
            storage.relocate(root, rows, restore=True)
            self.assertEqual((root / 'model.zip').read_bytes(), b'archive evidence')

    def test_preflight_collision_changes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'a.zip').write_bytes(b'a')
            (root / 'b.zip').write_bytes(b'b')
            dest = root / 'artifacts/archives'
            dest.mkdir(parents=True)
            (dest / 'b.zip').write_bytes(b'other backup')
            with self.assertRaises(FileExistsError):
                storage.relocate(root, storage.plan(root))
            self.assertEqual((root / 'a.zip').read_bytes(), b'a')
            self.assertEqual((dest / 'b.zip').read_bytes(), b'other backup')

    def test_changed_source_and_symlink_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / 'a.zip'
            src.write_bytes(b'a')
            rows = storage.plan(root)
            src.write_bytes(b'changed')
            with self.assertRaises(ValueError):
                storage.relocate(root, rows)
            (root / 'linked.zip').symlink_to(src)
            self.assertEqual(len(storage.plan(root)), 1)
            rows[0]['source'] = '../escape.zip'
            with self.assertRaises(ValueError):
                storage.relocate(root, rows)


if __name__ == '__main__':
    unittest.main()
