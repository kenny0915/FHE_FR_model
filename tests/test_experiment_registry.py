"""Regression checks for scientific provenance and read-only inventory."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('registry', ROOT / 'tools/experiment_registry/registry.py')
registry = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(registry)


class RegistryTests(unittest.TestCase):
    def test_roc_rules_and_calibration_are_not_conflated(self):
        rows = registry.normalized_results(ROOT)
        self.assertEqual(len(rows), 36)
        baseline = [r for r in rows if r['run_id'] == 'ms1mv3_r50' and r['requested_far'] == 1e-4]
        strict = next(r for r in baseline if r['far_rule'] == 'strict')
        nearest = next(r for r in baseline if r['far_rule'] == 'nearest')
        self.assertLessEqual(strict['actual_far'], 1e-4)
        self.assertGreater(nearest['actual_far'], 1e-4)
        self.assertNotEqual(strict['tar_percent'], nearest['tar_percent'])
        self.assertEqual(nearest['threshold'], 'unknown')
        head = next(r for r in rows if 'head_only' in r['run_id'])
        self.assertIn('ijbb_overlap', head['calibration'])
        self.assertEqual(head['finite_scope'], 'augmented_embeddings_only')
        self.assertTrue(all(len(r['checkpoint_sha256']) == 64 for r in rows))

    def test_historical_failures_remain_unranked_and_unknown(self):
        rows = registry.historical_results(ROOT)
        failed = next(r for r in rows if r['artifact_group'] == 'quadT12c_ijbc')
        evidence = json.loads(failed['finite_evidence'])
        self.assertEqual(evidence['logs'][0]['nonfinite_augmented_embedding_rows'], [193])
        self.assertEqual(failed['checkpoint_sha256'], 'unknown')
        self.assertEqual(failed['review_status'], 'needs_review')
        self.assertEqual(failed['actual_far'], 'unknown')

    def test_inventory_does_not_follow_symlinks_or_load_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / 'work_dirs' / 'unknown_run'
            run.mkdir(parents=True)
            (run / 'bad.pt').write_bytes(b'not a checkpoint')
            (run / 'config.json').write_text('{}')
            external = root / 'external'
            external.mkdir()
            (external / 'secret.pt').touch()
            (run / 'linked').symlink_to(external, target_is_directory=True)
            rows = registry.inventory(root)
            self.assertEqual(rows[0]['checkpoints'], ['work_dirs/unknown_run/bad.pt'])
            self.assertEqual(rows[0]['family_hint'], 'unclassified')
            self.assertEqual(rows[0]['review_status'], 'needs_review')
            self.assertEqual((run / 'bad.pt').read_bytes(), b'not a checkpoint')

    def test_missing_artifacts_do_not_erase_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                registry.inventory(Path(tmp))

    def test_committed_tables_are_reproducible_without_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ['docs/0915_result/metrics.json', 'docs/0915_result/selection_evidence.json',
                         'experiments/representatives.json']:
                target = root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes((ROOT / name).read_bytes())
            registry.build(root)
            for name in ['evaluations', 'historical_evidence']:
                path = 'reports/tables/' + name + '.csv'
                self.assertEqual((root / path).read_bytes(), (ROOT / path).read_bytes())


if __name__ == '__main__':
    unittest.main()
