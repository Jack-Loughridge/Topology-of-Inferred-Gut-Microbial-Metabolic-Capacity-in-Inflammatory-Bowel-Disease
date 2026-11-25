"""Challenge candidate anchor, scaler and physical encoding on fabricated certified geometry."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from analysis.alpha_matched import run_fold as core, vector_storage as vs
from analysis.alpha_matched.artifacts import Store
from analysis.alpha_matched.features import Factory
from analysis.alpha_matched.nesting import Procedure
from analysis.alpha_matched.partitions import array_id, canonical_json, digest
from qualification import qualify_vectors as fixture


class IntegratedEvidence(unittest.TestCase):
    def test_adaptive_producer_and_source_challenges(self):
        fixture.check()
        with tempfile.TemporaryDirectory(dir=fixture.BASE) as tmp, patch.object(fixture, 'WIDTH', 4096), patch(
                'analysis.alpha_matched.features.compute', side_effect=AssertionError('unexpected solve')):
            root = Path(tmp)
            values = [np.empty(0)] + [np.resize(np.array([-0., 0., .5, -.125]), 3072) for _ in range(7)]
            graph, meta = fixture.panel(8, values)
            meta['group'] = meta.person
            train, valid = meta.iloc[:4], meta.iloc[4:]
            fac = Factory(graph, 'RELATIVE', root/'geometry', fixture.TOL)
            fac.version = core.sha(canonical_json(core.source_hashes()))
            store = Store(root/'store')
            proc = Procedure(meta, {'node':dict(train=list(train.person.unique()), validate=list(valid.person.unique()))},
                             fac, {}, [], None)
            worker = core.Fold(proc, fac, store, {'heads':[], 'ricci_storage':'active-v2'}, (), fixture.check)
            blocks = []
            for i, setting in enumerate(fixture.SETTINGS):
                fixture.seed(fac, train, meta.index, setting, i, values)
                blocks.append(worker.build('Ricci',setting,'node'))
            a, b = blocks
            specs = [vs.record(store, block['train']) for block in blocks]
            row = specs[0]['rows'][1]
            self.assertIn('zlib_sha256', row['receipt']['record'])
            self.assertTrue((store.root/row['payload']).samefile(fac.cache_dir/row['shard']/row['payload']))
            with self.assertRaisesRegex(ValueError, 'block model'):
                vs.verify_evidence(store, dict(a,model=b['model']),train,valid,fac.arm)
            altered = vs.record(store,a['train'])
            altered['rows'][0] = specs[1]['rows'][0]
            self.assertEqual(altered['rows'][0]['payload'],specs[0]['rows'][0]['payload'])
            with self.assertRaisesRegex(ValueError,'sample evidence'):
                store.load_vector(store.record(altered))
            with self.assertRaisesRegex(ValueError,'inventory'):
                vs.verify_evidence(store,a,train,valid,fac.arm,objects=store.objects-{row['payload']})

            # Create an internally consistent alternative frozen scaler/generator with authentic row proofs.
            altered = vs.record(store,a['train'])
            model = vs.record(store,a['model'])
            state = model['state']
            mean = store.load_array(state['mean']) + 1.
            state['mean'] = store.array(mean)
            ref = vs.record(store,model['reference'])
            state['generator'] = digest(canonical_json([fac.arm,model['partition'],'Ricci',sorted(state['setting'].items()),
                [],array_id(store.load_array(ref['log_reference'])),array_id(mean),array_id(store.load_array(state['scale'])),
                fac.version,model['catalogue'],model['mapping'],vs.record(store,model['features'])]))
            altered['model'] = store.record(model)
            _, inv, count = np.unique(train.person,return_inverse=True,return_counts=True)
            with self.assertRaisesRegex(ValueError,'certified training rows'):
                vs.RecipeStore(store.root).restore(altered, 1 / count[inv])
            (store.root/ref['unseen']).unlink()
            with self.assertRaises(FileNotFoundError):
                store.load_vector(a['train'])
