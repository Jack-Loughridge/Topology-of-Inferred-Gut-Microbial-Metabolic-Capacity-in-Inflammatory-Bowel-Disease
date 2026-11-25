"""Exercise the dense and default compact Fold paths on the same certified source arrays."""
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from analysis.alpha_matched import run_fold as core, vector_storage as vs
from analysis.alpha_matched.artifacts import Store, sha
from analysis.alpha_matched.partitions import canonical_json
from qualification import qualify_vectors as fixture


class LegacyGeometry(unittest.TestCase):
    def setUp(self):
        fixture.check()
        tmp = tempfile.TemporaryDirectory(dir=fixture.BASE)
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        trap = patch('analysis.alpha_matched.features.compute', side_effect=AssertionError('unexpected solve'))
        trap.start()
        self.addCleanup(trap.stop)

    def compare(self, dtype, arm='RELATIVE'):
        fixture.check()
        root = self.root / (arm + dtype.str)
        width = 2304
        pattern = np.array([-0., 0., -.125, .10001, np.nextafter(np.longdouble(1), np.longdouble(2))],
                           dtype=np.longdouble)
        values = [np.resize(pattern, n).astype(dtype) for n in (0, 19, width, 9, 15, width, 0, 33)]
        with patch.object(fixture, 'WIDTH', width):
            graph, meta = fixture.panel(len(values), values)
        meta['group'] = meta.person
        train, valid = meta.iloc[:4], meta.iloc[4:]
        fac = core.Factory(graph, arm, root/'geometry', fixture.TOL)
        fac.version = sha(canonical_json(core.source_hashes()))
        setting = fixture.SETTINGS[0]
        fixture.seed(fac, train, meta.index, setting, 0, values)
        splits = {'node': dict(train=list(train.person.unique()), validate=list(valid.person.unique()))}
        proc = core.Procedure(meta, splits, fac, {}, [], None)
        dense, compact = Store(root/'dense'), Store(root/'compact')
        dense_fold = core.Fold(proc, fac, dense, {'heads': [], 'ricci_storage': 'dense'}, (), fixture.check)
        compact_fold = core.Fold(proc, fac, compact, {'heads': []}, (), fixture.check)
        reference = dense_fold.build('Ricci', setting, 'node')
        ricci = compact_fold.build('Ricci', setting, 'node')
        cold = Store(compact.root)
        for a, b in [(reference, ricci), (dense_fold.curvature_block(reference), compact_fold.curvature_block(ricci))]:
            self.assertEqual(a['model'], b['model'])
            for side in ('train', 'validate'):
                original, restored = dense.load_vector(a[side]), cold.load_vector(b[side])
                self.assertEqual(original.values.dtype, np.dtype(np.float64))
                self.assertEqual(restored.values.dtype, original.values.dtype)
                self.assertEqual(restored.values.shape, original.values.shape)
                self.assertEqual(restored.values.tobytes(), original.values.tobytes())
                self.assertEqual(restored.identity(), original.identity())

        extensions = set()
        for side in ('train', 'validate'):
            for row in vs.record(cold, ricci[side])['rows']:
                info = row['receipt']['record']
                source = fac.cache_dir/row['shard']/row['payload']
                pinned = cold.root/row['payload']
                self.assertEqual(pinned.read_bytes(), source.read_bytes())
                self.assertEqual(sha(source.read_bytes()), info.get('zlib_sha256', info['sha256']))
                self.assertEqual(sha(cold.read(row['payload'])), info['sha256'])
                decoded = cold.load_array(row['payload'])
                seeded = values[graph.lookup[row['sample']]]
                self.assertEqual(decoded.dtype.str, seeded.dtype.str)
                self.assertEqual(decoded.tobytes(), seeded.tobytes())
                extensions.add(pinned.suffix)
        self.assertEqual(extensions, {'.npy', '.npyz'})
        return cold, ricci

    def test_native_float64_control(self):
        self.compare(np.dtype('=f8'))

    def test_float32(self):
        self.compare(np.dtype('=f4'))

    def test_big_endian_float64(self):
        self.compare(np.dtype('>f8'))

    def test_float_widths_and_endianness_raw(self):
        for floating in (np.float16, np.float32, np.float64, np.longdouble):
            for order in ('<', '>'):
                dtype = np.dtype(floating).newbyteorder(order)
                with self.subTest(dtype=dtype.str):
                    self.compare(dtype, 'RAW')

    def test_invalid_payloads_after_valid_digest_checks(self):
        store, block = self.compare(np.dtype('=f8'))
        bad = [np.ones(19, dtype=np.int32), np.ones(19, dtype=np.complex64),
               np.full(19, np.nan), np.full(19, np.inf), np.ones((19, 1))]
        cases = [(value, 'Invalid finite floating-point geometry') for value in bad]
        if np.finfo(np.longdouble).max > np.finfo(np.float64).max:
            cases.append((np.full(19, np.finfo(np.longdouble).max), 'Expected finite canonical'))
        for values, message in cases:
            with self.subTest(dtype=values.dtype.str, shape=values.shape, value=values.flat[0]):
                spec = vs.record(store, block['train'])
                row = spec['rows'][1]
                stream = io.BytesIO()
                np.save(stream, values, allow_pickle=False)
                data = stream.getvalue()
                row['payload'] = store.put(data, '.npy')
                receipt = row['receipt']
                receipt['record']['sha256'] = sha(data)
                receipt['digest'] = sha(canonical_json(receipt['record']))
                row['body']['solver_receipt'] = receipt['digest']
                with np.errstate(over='ignore'), self.assertRaisesRegex(ValueError, message):
                    store.load_vector(store.record(spec))
