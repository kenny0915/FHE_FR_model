"""Config imports preserve legacy recipes without leaking state between loads."""
import importlib
import json
from pathlib import Path
import unittest
from utils.utils_config import get_config

ROOT = Path(__file__).resolve().parents[1]


class ConfigOrganizationTests(unittest.TestCase):
    def test_all_legacy_and_family_paths_match(self):
        mapping = json.loads((ROOT / 'experiments/config_migration.json').read_text())
        self.assertEqual(len(mapping), 131)
        for old, new in mapping.items():
            with self.subTest(config=old):
                self.assertEqual(get_config(old), get_config(new))
                self.assertEqual(get_config(old[:-3]), get_config(new[:-3]))

    def test_independent_nested_values_and_no_base_pollution(self):
        module = importlib.import_module('configs.base')
        before = dict(module.config)
        first = get_config('configs/baseline/ms1mv3_r50')
        first.val_targets.append('must_not_leak')
        first.new_field = True
        second = get_config('configs/ms1mv3_r50')
        self.assertNotIn('must_not_leak', second.val_targets)
        self.assertNotIn('new_field', second)
        get_config('configs/ms1mv3_r50_pillar_espn')
        self.assertEqual(get_config('configs/ms1mv3_r50'), second)
        self.assertEqual(dict(module.config), before)
        self.assertEqual(second.output, 'work_dirs/ms1mv3_r50')

    def test_reject_outside_config_paths(self):
        for value in ['outside/model', 'configs/../outside', '/configs/test', 'configs/x.json']:
            with self.subTest(path=value), self.assertRaises(ValueError):
                get_config(value)

    def test_conflict_resolution_uses_documented_grouped_fresh_start(self):
        cfg = get_config('configs/other_backbones/ms1mv3_poolformer_s24_fully_gated_affine_fp32')
        self.assertFalse(cfg.resume)
        self.assertTrue(cfg.output.endswith('_grouped_fp32'))
        self.assertEqual(cfg.affine_blocks_per_group, 1)


if __name__ == '__main__':
    unittest.main()
