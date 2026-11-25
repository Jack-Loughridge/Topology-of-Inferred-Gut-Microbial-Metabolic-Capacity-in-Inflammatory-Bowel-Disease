"""Physical representation changes must preserve evidence and exact float bits."""
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from .cache import CurvatureCache, FIELDS, encode_payload, payload_name, record_sample
from .partitions import canonical_json
from .run_fold import geometry_files
from .unitcache import database


class GeometryStorage(unittest.TestCase):
    def test_lossless_resume_inventory_and_partition_links(self):
        values = np.resize(np.array([-0., 0., np.nextafter(0., 1.), -7.25], dtype='>f8'), 8192)
        pairs = np.column_stack((np.arange(len(values)), np.arange(1, len(values) + 1))).astype('<i8')
        errors = dict.fromkeys(('first_hit_error', 'transport_error', 'marginal_error', 'curvature_error'), 0.)
        key = dict.fromkeys(FIELDS, 'fixture')
        key.update(arm='RAW', reference_digest='identity', tolerance=errors,
                   active_pairs_digest=hashlib.sha256(pairs.tobytes()).hexdigest())
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / 'geometry'
            raw = CurvatureCache(folder / '.raw-numerics', dict(key, training_people_digest='fixed-RAW-numerical-operator'))
            with patch('analysis.alpha_matched.cache.os.replace', side_effect=InterruptedError):
                with self.assertRaises(InterruptedError):
                    raw.save(values, pairs, errors)
            self.assertIsNone(raw.read(pairs))
            raw.save(values, pairs, errors)
            rec = json.loads((raw.folder / 'receipt.json').read_bytes())['record']
            path = raw.folder / payload_name(rec)
            self.assertEqual(path.suffix, '.npyz')
            npy = io.BytesIO()
            np.save(npy, values, allow_pickle=False)
            self.assertEqual(rec['sha256'], hashlib.sha256(npy.getvalue()).hexdigest())
            expected = set()
            for part in ('train-1', 'train-2'):
                cache = CurvatureCache(folder, dict(key, training_people_digest=part))
                cache.bind_raw_calculation(raw, pairs)
                restored = cache.read(pairs)
                self.assertEqual((restored.dtype, restored.tobytes()), (values.dtype, values.tobytes()))
                self.assertTrue((cache.folder / path.name).samefile(path))
                record_sample(cache, 'sample', pairs, folder)
            with database(folder) as db:
                for sample, shard, body in db.execute('SELECT sample,geometry,body FROM receipts'):
                    expected.add((sample, shard, hashlib.sha256(body).hexdigest()))
            inventory = geometry_files(root, expected)
            self.assertEqual(len(expected), 2)
            self.assertEqual(sum(n.endswith('.npyz') for n in inventory), 3)
            tampered = values.copy()
            tampered[0] = 0.
            with self.assertRaisesRegex(ValueError, 'Conflicting results'):
                raw.save(tampered, pairs, errors)
            original = path.read_bytes()
            path.write_bytes(original[:-2])
            with self.assertRaisesRegex(ValueError, 'digest'):
                raw.read(pairs)
            with self.assertRaisesRegex(ValueError, 'Changed'):
                geometry_files(root, expected)
            path.write_bytes(original)
            marker = raw.folder / 'receipt.json'
            rec['sha256'] = '0' * 64
            marker.write_bytes(canonical_json(dict(record=rec, digest=hashlib.sha256(canonical_json(rec)).hexdigest())))
            with self.assertRaisesRegex(ValueError, 'digest'):
                raw.read(pairs)

    def test_raw_fallback_empty_and_incompressible(self):
        noise = np.random.default_rng(973).integers(0, 256, size=4000 * 8, dtype=np.uint8)
        values = noise.view('<f8')
        values[~np.isfinite(values)] = 0.
        for arr in (np.empty(0, dtype='>f4'), values):
            record, body = encode_payload(arr, 4096)
            self.assertNotIn('zlib_sha256', record)
            restored = np.load(io.BytesIO(body), allow_pickle=False)
            self.assertEqual((restored.shape, restored.dtype, restored.tobytes()), (arr.shape, arr.dtype, arr.tobytes()))
