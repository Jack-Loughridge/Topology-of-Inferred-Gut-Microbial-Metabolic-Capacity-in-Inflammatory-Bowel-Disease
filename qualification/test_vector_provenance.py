"""Invented provenance challenges for v2; no numerical solver or classifier."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from analysis.alpha_matched.artifacts import Store, sha
from analysis.alpha_matched.features import Factory
from analysis.alpha_matched.nesting import Procedure
from analysis.alpha_matched.partitions import canonical_json
from analysis.alpha_matched.run_fold import Fold
from qualification import qualify_vectors as fixture
from qualification.vector_recipe import RecipeStore, block_expectation, copy_objects, ordinary_closure, record


class VectorProvenance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture.check()
        tmp = tempfile.TemporaryDirectory(dir=fixture.BASE)
        cls.addClassCleanup(tmp.cleanup)
        cls.root = Path(tmp.name)
        with patch.object(fixture, 'WIDTH', 23), patch('analysis.alpha_matched.features.compute',
                                                      side_effect=AssertionError('Unexpected solve')):
            graph, meta = fixture.panel(8)
            meta['group'] = meta.person
            train, valid = meta.iloc[:4], meta.iloc[4:]
            fac = Factory(graph, 'RELATIVE', cls.root / 'geometry', fixture.TOL)
            proc = Procedure(meta, {'node': dict(train=train.person.unique(), validate=valid.person.unique())},
                             fac, {}, [], None)
            dense = Store(cls.root / 'dense')
            cls.store = RecipeStore(cls.root / 'compact')
            worker = Fold(proc, fac, dense, {'heads': []}, (), fixture.check)
            cls.blocks, cls.contracts = [], []
            for i, setting in enumerate(fixture.SETTINGS):
                fixture.seed(fac, train, meta.index, setting, i, [])
                original = worker._build('Ricci', setting, 'node')
                copy_objects(dense, cls.store, ordinary_closure(dense, original['model']))
                block = dict(model=original['model'])
                for side in ('train', 'validate'):
                    block[side] = cls.store.capture(dense.load_vector(original[side]), fac)
                cls.blocks.append(block)
                cls.contracts.append(block_expectation(dense, original, train, valid, fac.arm, fac))
            cls.ref = record(cls.store, record(cls.store, cls.blocks[0]['model'])['reference'])

    def test_finalized_rho_b_with_both_rho_a_splits(self):
        a, b = self.blocks
        bad = dict(a, model=b['model'])
        expected = self.contracts[1]
        dest = RecipeStore(self.root / 'crossed')
        with self.assertRaisesRegex(ValueError, 'Split generator'):
            self.store.export(dest, [bad], [expected])
        self.assertFalse((dest.root / 'complete.json').exists())
        with self.assertRaisesRegex(ValueError, 'Split generator'):
            self.store.dense_block(dest, bad, expected)
        with self.assertRaisesRegex(ValueError, 'Split generator'):
            self.store.run_block('test', 'crossed', expected, lambda: bad)
        self.assertFalse(list(self.store.root.glob('unit-*.json')))
        # Even a consistent-looking destination marker must be verified against its blocks.
        names = self.store.closure([bad])
        copy_objects(self.store, dest, names)
        contract = dest.record([expected])
        dest.publish('complete.json', canonical_json(dest.seal_record([bad], contract, names | {contract})))
        with self.assertRaisesRegex(ValueError, 'Split generator'):
            RecipeStore(dest.root).restart([expected])

    def test_ordered_sides_training_scope_and_layout(self):
        block = self.blocks[0]
        expected = self.contracts[0]
        with self.assertRaisesRegex(ValueError, 'ordered supervised split'):
            self.store.verify_block(dict(block, validate=block['train']), expected)
        for key, value in [('arm', 'RAW'), ('partition', 'wrong')]:
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'block scope'):
                self.store.verify_block(block, dict(expected, **{key: value}))
        for side in ('train', 'validate'):
            for key in ('ids', 'layout'):
                swapped = copy.deepcopy(expected)
                swapped[side][key] = list(reversed(swapped[side][key])) if key == 'ids' else 'wrong'
                with self.subTest(side=side, key=key), self.assertRaisesRegex(ValueError, 'ordered supervised split'):
                    self.store.verify_block(block, swapped)

    def test_authentic_foreign_empty_receipt(self):
        a, b = [record(self.store, block['train']) for block in self.blocks]
        first, foreign = a['rows'][0], b['rows'][0]
        self.assertEqual(first['receipt']['record']['active_edges'], 0)
        self.assertEqual(first['payload'], foreign['payload'])
        self.assertEqual(first['sample'], foreign['sample'])
        self.assertNotEqual(first['shard'], foreign['shard'])
        a['rows'][0] = foreign
        with self.assertRaisesRegex(ValueError, 'Invalid sample evidence'):
            self.store.restore(a)
        bad = dict(self.blocks[0], train=self.store.record(a))
        with self.assertRaisesRegex(ValueError, 'Invalid sample evidence'):
            self.store.export(RecipeStore(self.root / 'empty'), [bad], [self.contracts[0]])

    def test_every_scientific_key_field_even_with_rehashed_receipt(self):
        source = record(self.store, self.blocks[0]['train'])
        from analysis.alpha_matched.cache import FIELDS
        self.assertEqual(set(source['rows'][0]['receipt']['record']['key']), FIELDS)
        for field in sorted(FIELDS):
            spec = copy.deepcopy(source)
            row = spec['rows'][0]
            receipt = row['receipt']
            key = receipt['record']['key']
            key[field] = 'foreign-value'
            row['shard'] = sha(canonical_json(key))
            receipt['digest'] = sha(canonical_json(receipt['record']))
            row['body']['solver_receipt'] = receipt['digest']
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'Invalid sample evidence'):
                self.store.restore(spec)

    def test_independent_source_attestation_and_input_witness(self):
        original = record(self.store, self.blocks[0]['train'])
        for what in ('proof', 'inputs'):
            spec = copy.deepcopy(original)
            row = spec['rows'][0]
            info = row['receipt']['record']
            if what == 'proof':
                info['certificate']['curvature_error'] = 1e-10
                row['body']['numerical_errors'] = info['certificate']
            else:
                from analysis.alpha_matched.partitions import digest
                inputs = record(self.store, spec['inputs'])
                w = inputs[0]
                w['support'] = sha(b'another support')
                info['key']['input_digest'] = digest(w['vertices'].encode(), w['indices'].encode(), w['support'].encode())
                row['shard'] = sha(canonical_json(info['key']))
                spec['inputs'] = self.store.record(inputs)
            row['receipt']['digest'] = sha(canonical_json(info))
            row['body']['solver_receipt'] = row['receipt']['digest']
            bad = dict(self.blocks[0], train=self.store.record(spec))
            with self.subTest(what=what), self.assertRaisesRegex(ValueError, 'Wrong original'):
                self.store.verify_block(bad, self.contracts[0])

    def test_missing_relative_dependencies_at_seal_and_after_cold_restart(self):
        store, block, want = self.store, self.blocks[0], self.contracts[0]
        for field in ('log_reference', 'unseen'):
            name = self.ref[field]
            dest = RecipeStore(self.root / field)
            # A stale in-memory object registration cannot count as copied evidence.
            dest.objects.add(name)
            with self.subTest(field=field), self.assertRaises(FileNotFoundError):
                store.export(dest, [block], [want])
            self.assertFalse((dest.root / 'complete.json').exists())
            store.export(RecipeStore(dest.root), [block], [want])
            self.assertEqual(RecipeStore(dest.root).restart(sha(canonical_json([want]))), [block])
            with self.assertRaisesRegex(ValueError, 'Wrong portable expectation'):
                RecipeStore(dest.root).restart(sha(b'foreign split contract'))
            (dest.root / name).unlink()
            cold = RecipeStore(dest.root)
            with self.assertRaises(FileNotFoundError):
                cold.load_vector(block['train'])
            with self.assertRaises(FileNotFoundError):
                cold.restart([want])

    def test_missing_dependency_in_nested_non_numeric_model_state(self):
        # Verify even an unused, transitive dependency reached via a JSON list of references.
        source = RecipeStore(self.root / 'nested-source')
        leaf = source.array(self.store.load_array(self.ref['log_reference']))
        inner = source.record([leaf])
        model = source.record(dict(name='invented-evidence', state=dict(auxiliary=inner)))
        self.assertEqual(source.closure(model), {model, inner, leaf})
        (source.root / leaf).unlink()
        with self.assertRaises(FileNotFoundError):
            RecipeStore(source.root).closure(model)
