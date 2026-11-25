"""Read-only regression against the retained, independently pinned invented releases."""
import json
from pathlib import Path
import unittest

from analysis.alpha_matched import report_development as report, run_fold as core
from analysis.alpha_matched.artifacts import Store, sha
from analysis.alpha_matched.partitions import canonical_json
from qualification import qualify_vectors as fixture

ROOT = Path(__file__).resolve().parents[1]/'outputs/integrated-storage-final'
ORIGINAL = '04c2c57570c1a66f7e0f8b4828fa66e568a7464f41bad2dc279f5a4365c745fe'
COMPACT = '5307ba951c76e82308bf31b8d1254996e34f2a5976c75887e1a5e999b97f7e09'


class PinnedReports(unittest.TestCase):
    def replay(self, category, marker):
        fixture.check()
        root = ROOT/category
        entry = core.active(root)['final_refit_release']
        header = (root/'reviews'/entry['file']).read_bytes()
        self.assertEqual(sha(header), entry['sha256'])
        rec = json.loads(header)
        cfg, splits, meta = report.released_inputs(root, rec)
        self.assertNotIn('ricci_storage', cfg)
        proc = core.Procedure(meta, splits, None, {}, [], None)
        store = Store(root/'jobs/development-final-refits'/entry['sha256']/core.FINAL)
        receipt = (store.root/'receipt.json').read_bytes()
        self.assertEqual(sha(receipt), marker)
        body = json.loads(store.read(json.loads(receipt)['record']))
        objects = set(body['objects'])
        self.assertEqual(body['inputs'], rec['inputs'])
        observed = 0
        for arm, saved in body['arms'].items():
            rules = json.loads(store.read(saved['decision_rules']))
            for method in ('Ricci', 'K0', 'joint'):
                for ref in rules['heldout_fits'][method]:
                    fit = json.loads(store.read(ref))
                    train, valid = proc.frames(fit['split'])
                    def table(source, signed, key, config):
                        return report.fit_table(source, signed, key, arm, method, fit['split'], train, valid, config)
                    result = table(store, objects, ref, cfg)
                    self.assertEqual(result.to_numpy().tobytes(), store.load_array(fit['probabilities']).tobytes())
                    observed += 1
                    if observed != 1:
                        continue
                    for altered in (dict(cfg, ricci_storage='dense'), {k:v for k,v in cfg.items() if k != 'candidates'}):
                        with self.assertRaisesRegex(ValueError, 'configuration differs'):
                            table(store, objects, ref, altered)
                    # Read-only overlay. A wrapper must retain the exact producing fit-unit binding.
                    rewritten = dict(fit, review_probe=True)
                    if category == 'clean':
                        rewritten['blocks'] = [{k:v for k,v in b.items() if k != 'contract'} for b in fit['blocks']]
                    data = canonical_json(rewritten)
                    fake = sha(data) + '.json'
                    class Overlay(Store):
                        def read(self, name):
                            return data if name == fake else super().read(name)
                    expected = 'split contract' if category == 'clean' else 'completed source-bound fit'
                    with self.assertRaisesRegex(ValueError, expected):
                        table(Overlay(store.root), objects | {fake}, fake, cfg)
        self.assertGreater(observed, 0)
        # Execution/report release gates continue to compare reviewed and installed source.
        with self.assertRaisesRegex(ValueError, 'Implementation changed since review'):
            core.release(root, core.FINAL)
        return observed

    def test_original_dense_and_implicit_compact(self):
        self.assertEqual(self.replay('reference', ORIGINAL), self.replay('clean', COMPACT))


if __name__ == '__main__':
    unittest.main()
