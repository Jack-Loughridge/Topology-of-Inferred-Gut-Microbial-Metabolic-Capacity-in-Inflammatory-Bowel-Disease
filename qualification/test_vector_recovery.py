"""Interrupted synthetic build, durable object replay and crossed scaler rejection."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from analysis.alpha_matched.artifacts import Store
from analysis.alpha_matched.features import Factory
from analysis.alpha_matched.nesting import Procedure
from analysis.alpha_matched.run_fold import Fold
from qualification import qualify_vectors as fixture
from qualification.vector_recipe import RecipeStore, block_expectation, copy_objects, ordinary_closure, record


class RecipeRecovery(unittest.TestCase):
    def test_interrupted_build_and_cold_resume(self):
        fixture.check()
        with tempfile.TemporaryDirectory(dir=fixture.BASE) as folder, patch.object(fixture, 'WIDTH', 23), patch(
                'analysis.alpha_matched.features.compute', side_effect=AssertionError('Unexpected solve')):
            root = Path(folder)
            graph, meta = fixture.panel(8)
            meta['group'] = meta.person
            train, valid = meta.iloc[:4], meta.iloc[4:]
            fac = Factory(graph, 'RELATIVE', root / 'geometry', fixture.TOL)
            fixture.seed(fac, train, meta.index, fixture.SETTINGS[0], 0, [])
            proc = Procedure(meta, {'node': dict(train=train.person.unique(), validate=valid.person.unique())},
                             fac, {}, [], None)
            dense = Store(root / 'dense')
            full = Fold(proc, fac, dense, {'heads': [], 'ricci_storage': 'dense'}, (), fixture.check)._build('Ricci', fixture.SETTINGS[0], 'node')
            store = RecipeStore(root / 'recipes')
            copy_objects(dense, store, ordinary_closure(dense, full['model']))
            train_vec = dense.load_vector(full['train'])
            valid_vec = dense.load_vector(full['validate'])
            expected = block_expectation(dense, full, train, valid, 'RELATIVE', fac)
            train_key = store.capture(train_vec, fac)

            def interrupted():
                self.assertEqual(store.capture(train_vec, fac), train_key)
                raise InterruptedError

            def build():
                return dict(model=full['model'], train=store.capture(train_vec, fac), validate=store.capture(valid_vec, fac))

            with self.assertRaises(InterruptedError):
                store.run_block('invented', 'build', expected, interrupted)
            self.assertFalse(list(store.root.glob('unit-*.json')))
            store = RecipeStore(store.root)
            block = store.run_block('invented', 'build', expected, build)
            self.assertEqual(block['train'], train_key)
            store = RecipeStore(store.root)
            again = store.run_block('invented', 'build', expected, lambda: self.fail('Repeated completed build'))
            self.assertEqual(again, block)
            fixture.equal(store.load_vector(again['train']), train_vec)
            fixture.equal(store.load_vector(again['validate']), valid_vec)
            spec = record(store, again['validate'])
            other = record(store, spec['model'])
            state = other['state']
            state['mean'] = store.array(store.load_array(state['mean']) + 1.)
            with self.assertRaisesRegex(ValueError, 'Frozen generator identity mismatch'):
                store.restore(dict(spec, model=store.record(other)))
            with self.assertRaisesRegex(ValueError, 'Frozen generator identity mismatch'):
                store.restore(dict(spec, model=store.record(dict(other, partition='different-training-roster'))))
            # Existing unit traversal cannot vouch for an external raw NPY by itself.
            (store.root / spec['rows'][0]['payload']).unlink()
            with self.assertRaises(FileNotFoundError):
                store.run_block('invented', 'build', expected, lambda: self.fail('Repeated build'))
