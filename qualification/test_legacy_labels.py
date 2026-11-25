"""Opaque labels and actual Store references through build, restart and portable copy."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.sparse import csr_matrix

from analysis.alpha_matched import comparator_run, run_fold as core
from analysis.alpha_matched.artifacts import Store, sha
from analysis.alpha_matched.partitions import FrozenVectors, canonical_json
from analysis.alpha_matched.vector_storage import RecipeStore, record
from qualification import qualify_vectors as fixture


class OpaqueLabels(unittest.TestCase):
    def setUp(self):
        fixture.check()
        tmp = tempfile.TemporaryDirectory(dir=fixture.BASE)
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)

    def test_reference_context_is_preserved(self):
        store = RecipeStore(self.root)
        payload = store.array(np.ones(2))
        absent = 'a' * 64 + '.npy'
        labels = store.record([payload, absent])
        obj = dict(ids=labels, features=[absent], rows=[dict(sample=absent)], contract=dict(ids=[payload]))
        self.assertEqual(store.closure(obj), {labels})
        actual_refs = store.record([payload])
        self.assertEqual(store.closure(dict(features=actual_refs, sources=actual_refs)), {actual_refs, payload})
        for schema in (dict(ids=absent), dict(features=absent), dict(sources=labels)):
            with self.subTest(schema=schema), self.assertRaises(FileNotFoundError):
                store.closure(schema)

    @patch('analysis.alpha_matched.features.compute', side_effect=AssertionError('unexpected solve'))
    def test_build_cold_unit_and_comparator_copy(self, _solve):
        with patch.object(fixture, 'WIDTH', 64):
            graph, meta = fixture.panel(8)
        ids = list(meta.index)
        ids[0], ids[5] = 'a' * 64 + '.npy', 'b' * 64 + '.jsonz'
        feature = 'c' * 64 + '.npyz'
        graph = replace(graph, sample_ids=tuple(ids), species_abundance=csr_matrix(np.ones((len(ids), 1))),
                        species_ids=(feature,), species_nondetection_zero=True)
        meta.index = ids
        meta['group'] = meta.person
        train, valid = meta.iloc[:4], meta.iloc[4:]
        arm, key, setting = 'RELATIVE', 'node', fixture.SETTINGS[0]
        fac = core.Factory(graph, arm, self.root/'geometry', fixture.TOL)
        fac.version = sha(canonical_json(core.source_hashes()))
        fixture.seed(fac, train, meta.index, setting, 0, ())
        proc = core.Procedure(meta, {key: dict(train=list(train.person.unique()),
                                               validate=list(valid.person.unique()))}, fac, {}, [], None)
        def build(store, name='Ricci', **config):
            fold = core.Fold(proc, fac, store, dict(heads=[], **config), (feature,), fixture.check,
                             release_id='invented')
            return fold.build(name, setting if name == 'Ricci' else {}, key)

        dense, store = Store(self.root/'dense'), Store(self.root/'compact')
        baseline, block = build(dense, ricci_storage='dense'), build(store)
        species = build(store, name='species')
        cold = Store(store.root)
        with patch.object(core.Fold, '_build', side_effect=AssertionError('unit recomputed')):
            self.assertEqual(build(cold), block)
            self.assertEqual(build(cold, name='species'), species)
        self.assertEqual(cold.load_vector(species['train']).features, (feature,))
        deps = RecipeStore(store.root).closure(block)
        self.assertLessEqual(deps, cold.objects)
        for side in ('train', 'validate'):
            a, b = dense.load_vector(baseline[side]), cold.load_vector(block[side])
            self.assertEqual(a.identity(), b.identity())
            self.assertEqual(a.values.tobytes(), b.values.tobytes())

        # FrozenPrimary accepts a dense H0 block alongside the compact Ricci block.
        names = store.record([feature])
        part = record(store, block['model'])['partition']
        h0 = dict(model=store.record(dict(name='H0', features=names, partition=part, arm=arm)))
        for side, frame in [('train', train), ('validate', valid)]:
            h0[side] = store.vector(FrozenVectors(np.ones((len(frame), 1)), tuple(frame.index),
                                                   (feature,), arm, part, h0['model'], 'H0'))
        primary = comparator_run.FrozenPrimary.__new__(comparator_run.FrozenPrimary)
        primary.store, primary.objects = store, set(store.objects)
        primary.selected = {(arm, key): [h0, block]}
        dest = Store(self.root/'copy')
        self.assertEqual(primary.blocks(dest, arm, 'joint', key, train, valid), [h0, block])
        closure = RecipeStore(dest.root).closure([h0, block])
        self.assertEqual(dest.objects, closure)
        self.assertTrue({record(store, h0['train'])['ids'], names,
                         record(store, block['train'])['rows'][0]['payload']} <= closure)
        self.assertTrue({ids[0], ids[5], feature}.isdisjoint(closure))
        for item in (block, h0):
            for side in ('train', 'validate'):
                self.assertEqual(Store(dest.root).load_vector(item[side]).identity(),
                                 store.load_vector(item[side]).identity())
