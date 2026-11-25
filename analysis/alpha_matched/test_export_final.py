"""Final-refit export test using only invented data and temporary releases."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from . import comparator_run as driver
from .export_final import export
from .test_artifacts import fixture
from .test_comparator_run import register


class FinalExport(unittest.TestCase):
    def test_mixed_export(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture(root,final=True,masked=True)
            fold = driver.core.FINAL
            parent = driver.core.execute(root,fold)
            store = register(root,fold,primary=parent)
            driver.execute(root,fold)
            marker = driver.core.bundle.sha(store.root/'receipt.json')
            with patch.object(driver.heads,'fit',side_effect=AssertionError('Export refitted estimator')):
                path = export(root,parent,marker)
                checksum = driver.core.bundle.sha(path)
                self.assertEqual(driver.core.bundle.sha(export(root,parent,marker)),checksum)
            record = json.loads(path.read_bytes())
            self.assertEqual(len(record['procedures']),42)
            sources={record['files'][n]:n for n in record['decision_sources']}
            for procedure in record['procedures']:
                source=sources[procedure['decision_source_sha256']]
                self.assertEqual(driver.core.bundle.sha(path.parent/source),procedure['decision_source_sha256'])
            for method in ('logistic','random_forest','xgboost','tabm'):
                shared=[p for p in record['procedures'] if p['method']=='cmp.shared.species.'+method]
                self.assertEqual(len(shared),2)
                self.assertEqual(len({x['fit'] for x in shared}),1)
            with self.assertRaisesRegex(ValueError,'marker'):
                export(root,'0'*64,marker)
            (root/'STOP').touch()
            with self.assertRaisesRegex(RuntimeError,'STOP'):
                export(root,parent,marker)

    def test_three_class_argmax(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture(root,nclasses=3,final=True,masked=True)
            parent = driver.core.execute(root,driver.core.FINAL)
            store = register(root,driver.core.FINAL)
            driver.execute(root,driver.core.FINAL)
            marker = driver.core.bundle.sha(store.root/'receipt.json')
            data = json.loads(export(root,parent,marker).read_bytes())
            self.assertEqual(len(data['procedures']),36)
            self.assertTrue(all(p['rule']=={'rule':'argmax','threshold':None} for p in data['procedures']))
            # Replace the final direct-support head with a real child-trained head,
            # then rehash its deployment, fit and enclosing receipt as a bad caller could.
            _, parent_id = driver.core.release(root,driver.core.FINAL)
            src = driver.Store(root/'jobs/development-final-refits'/parent_id/driver.core.FINAL)
            pin = src.root/'receipt.json'
            record = driver.read(src,json.loads(pin.read_bytes())['record'])
            omitted = json.loads(next(src.root.glob('unit-*.json')).read_bytes())['record']
            partial = dict(record,objects=[name for name in record['objects'] if name!=omitted])
            self.assertLess(len(partial['objects']),len(record['objects']))
            replacement = pin.with_suffix('.replacement')
            replacement.write_bytes(driver.canonical_json(dict(record=src.record(partial))))
            replacement.replace(pin)
            with self.assertRaisesRegex(ValueError,'inventory'):
                export(root,driver.core.bundle.sha(pin),marker)
            arm = record['arms']['RAW']
            before = arm['outputs']['direct-raw']
            fit = driver.read(src,before)
            deploy = driver.read(src,fit['deployment'])
            child = next(item for key in arm['fits'] for item in [driver.read(src,key)]
                         if item['name']=='direct-raw' and item['split']!=driver.core.FINAL
                         and item['fitted']!=fit['fitted'])
            deploy['head'] = fit['fitted'] = child['fitted']
            fit['deployment'] = src.record(deploy)
            after = src.record(fit)
            arm['outputs']['direct-raw'] = after
            arm['fits'] = [after if key==before else key for key in arm['fits']]
            record['objects'] = sorted(set(record['objects']) | {after,fit['deployment']})
            replacement = pin.with_suffix('.replacement')
            replacement.write_bytes(driver.canonical_json(dict(record=src.record(record))))
            replacement.replace(pin)
            with self.assertRaisesRegex(ValueError,'producer unit'):
                export(root,driver.core.bundle.sha(pin),marker)
