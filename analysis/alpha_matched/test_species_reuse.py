"""Reuse identical fitted species controls while preserving each arm's provenance."""
import json
import tempfile
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

from . import run_fold as core
from .test_artifacts import fixture


class SpeciesReuse(TestCase):
    def test_exact_shared_fit_and_cold_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture(root, final=True, masked=True)
            permit, rid = core.release(root, core.FINAL)
            config, splits, meta, graph, taxa = core.read_inputs(root, permit)
            store = core.Store(root / 'units')
            head = ('linear', 1.0)
            def worker(arm, cache, frame=meta):
                factory = core.Factory(graph, arm, root / 'geometry', config['tolerances'])
                proc = core.Procedure(frame, splits, factory, {}, [], None)
                return core.Fold(proc, factory, store, config, taxa, lambda: None, rid, cache)
            def fit(f, node):
                block = f.build('species', {}, node)
                _, record = f.fit('species', node, [block], head)
                return block, json.loads(store.read(record))
            for node in ('outer', core.FINAL):
                raw, source = fit(worker('RAW', {}), node)
                cache = {}
                # Resume from the completed first-arm unit before the second arm.
                with patch.object(core, 'fit_head', side_effect=AssertionError('Repeated species fit')):
                    fit(worker('RAW', cache), node)
                    other = worker('RELATIVE', cache)
                    relative, reused = fit(other, node)
                self.assertNotEqual(raw['model'], relative['model'])
                self.assertEqual(reused['arm'], 'RELATIVE')
                self.assertEqual(reused['blocks'], [relative])
                self.assertEqual(reused['fitted'], source['fitted'])
                if node == core.FINAL:
                    self.assertNotEqual(reused['generators'], source['generators'])
                    gens = json.loads(store.read(reused['generators']))
                    self.assertEqual(gens['blocks'], [relative['model']])
                else:
                    for name in ('probabilities', 'participant_means', 'participants', 'samples'):
                        self.assertEqual(source[name], reused[name])
                changed = meta.copy()
                changed.loc[:, 'y'] = 1 - changed.y
                changed_worker = worker('RELATIVE', cache, changed)
                self.assertNotEqual(other.species_fit_key(node, relative, head),
                                    changed_worker.species_fit_key(node, relative, head))
