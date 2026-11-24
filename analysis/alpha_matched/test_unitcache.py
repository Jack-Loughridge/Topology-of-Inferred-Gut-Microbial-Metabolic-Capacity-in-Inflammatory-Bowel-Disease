"""Exercise a real interruption between completed units using invented visits."""
import json
from pathlib import Path
import shutil
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from . import run_fold as runner
from .artifacts import Store
from .partitions import canonical_json
from .test_artifacts import fixture


class Restart(unittest.TestCase):
    def test_matched_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'interrupted'
            clean = Path(tmp) / 'clean'
            status = fixture(root)
            shutil.copytree(root, clean)
            release_id = status['development_fold_release']['sha256']
            job = Path('jobs/development-folds') / release_id / 'outer'
            completed = set()
            fit = runner.Fold._fit

            def identity(worker, name, key, blocks, head):
                return canonical_json([worker.factory.arm, name, key, blocks, head])

            def interrupt(worker, *args):
                if worker.factory.arm == 'RELATIVE' and args[0] == 'Ricci':
                    raise InterruptedError('synthetic interruption')
                result = fit(worker, *args)
                completed.add(identity(worker, *args))
                return result

            builds = set()
            build = runner.Fold._build

            def track_build(worker, name, setting, key):
                result = build(worker, name, setting, key)
                builds.add(canonical_json([worker.factory.arm, name, setting, key]))
                return result

            with patch.object(runner.Fold, '_fit', interrupt), patch.object(
                    runner.Fold, '_build', track_build), self.assertRaises(InterruptedError):
                runner.execute(root, 'outer')
            self.assertGreater(len(completed), 10)
            self.assertFalse((root / job / 'receipt.json').exists())

            def resume(worker, *args):
                self.assertNotIn(identity(worker, *args), completed)
                return fit(worker, *args)

            def resume_build(worker, name, setting, key):
                self.assertNotIn(canonical_json([worker.factory.arm, name, setting, key]), builds)
                return build(worker, name, setting, key)

            with patch.object(runner.Fold, '_fit', resume), patch.object(runner.Fold, '_build', resume_build):
                resumed = runner.execute(root, 'outer')
            self.assertEqual(resumed, runner.execute(clean, 'outer'))

            def predictions(base):
                store = Store(base / job)
                marker = json.loads((store.root / 'receipt.json').read_bytes())
                arms = json.loads(store.read(marker['record']))['arms']
                for arm, result in sorted(arms.items()):
                    for name, ref in sorted(result['outputs'].items()):
                        rec = json.loads(store.read(ref))
                        yield arm, name, rec['samples'], store.load_array(rec['probabilities'])

            for a, b in zip(predictions(root), predictions(clean), strict=True):
                self.assertEqual(a[:3], b[:3])
                np.testing.assert_array_equal(a[3], b[3])
            geometry = root / job / 'geometry'
            primitive = next((geometry / '.raw-numerics').glob('*/receipt.json'))
            saved = primitive.read_bytes()
            primitive.unlink()
            try:
                with self.assertRaisesRegex(ValueError, 'Missing RAW numerical receipt'):
                    runner.execute(root, 'outer')
            finally:
                primitive.write_bytes(saved)
            path = geometry / 'sample-certificates.sqlite'
            with sqlite3.connect(path) as db:
                db.execute('DELETE FROM receipts WHERE rowid=(SELECT MIN(rowid) FROM receipts)')
            with self.assertRaisesRegex(ValueError, 'attestation'):
                runner.execute(root, 'outer')
