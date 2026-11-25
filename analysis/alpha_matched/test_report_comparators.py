"""Invented comparator report receipts, including shared species and held-out groups."""
import json
import shutil
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from . import comparator_run as driver, report_comparators as report
from .test_artifacts import fixture
from .test_comparator_run import register, save


class ReportComparators(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pool = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.pool.cleanup)
        cls.cached = {}

    def make(self, classes=2, neural=True, optional=False, qualified=None):
        key = classes, neural, optional, tuple(qualified or [])
        if key not in self.cached:
            base = Path(self.pool.name) / str(len(self.cached))
            self.cached[key] = self.produce(base, classes, neural, optional, qualified)
        source, rid, markers = self.cached[key]
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        shutil.copytree(source, root, dirs_exist_ok=True)
        return root, rid, markers.copy()

    def produce(self, root, classes, neural, optional, qualified):
        fixture(root, classes, masked=True)
        inputs = root / 'manifests/development-fold-inputs'
        meta = pd.read_csv(inputs / 'meta.tsv', sep='\t', dtype=str)
        cutoff = 12 if classes == 2 else 10
        meta['cohort'] = ['North' if int(p[1:]) < cutoff else 'West' for p in meta.person]
        meta['family'] = meta.cohort
        meta.to_csv(inputs / 'meta.tsv', sep='\t', index=False)
        people = meta.drop_duplicates('person').set_index('person')
        splits = {}
        def partition(node, train, valid, depth):
            splits[node] = dict(train=list(train.index), validate=list(valid.index))
            if depth:
                n = min(3 if classes == 3 else 2, int(train.y.value_counts().min()))
                self.assertGreaterEqual(n, 2)
                assignment = train.groupby('y').cumcount() % n
                for i in range(n):
                    partition(node + '/' + str(i), train[assignment != i], train[assignment == i], depth - 1)
        folds = []
        for name in ('North', 'West'):
            held = people.cohort == name
            folds.append('hold-' + name)
            partition(folds[-1], people[~held], people[held], 2 if classes == 2 or optional else 1)
        save(inputs / 'splits.json', splits)
        review = root / 'reviews/test.json'
        parent = json.loads(review.read_bytes())
        parent['folds'] = folds
        for k in ('meta', 'splits'):
            spec = parent['inputs'][k]
            spec['sha256'] = driver.sha((inputs / spec['path']).read_bytes())
        status = json.loads((root / 'status.json').read_bytes())
        status['development_fold_release'] = save(review, parent)
        save(root / 'status.json', status)
        primary = {f: driver.core.execute(root, f) for f in folds} if optional else {}
        register(root, folds[0], neural=neural, primary=primary.get(folds[0]))
        rec_path = root / 'reviews/test-comparator.json'
        rec = json.loads(rec_path.read_bytes())
        rec['folds'] = folds
        if optional:
            rec['primary_receipts'] = primary
        if qualified is not None:
            budget_path = root / 'manifests' / rec['budget']['file']
            budget = json.loads(budget_path.read_bytes())
            budget.update(neural_inputs=qualified, unmet_reason='Invented width limit')
            rec['budget'] = save(budget_path, budget)
            approval_path = root / 'reviews' / rec['neural_review']['file']
            approval = json.loads(approval_path.read_bytes())
            approval.update(budget_sha256=rec['budget']['sha256'],
                            evidence={k: approval['evidence'][k] for k in qualified + budget['frozen_vectors']})
            rec['neural_review'] = save(approval_path, approval)
        ref = save(rec_path, rec)
        rid = ref['sha256']
        status = json.loads((root / 'status.json').read_bytes())
        save(root / 'status.json', dict(status, comparator_release=ref))
        markers = {}
        for fold in folds:
            driver.execute(root, fold)
            path = root / 'jobs/comparator-folds' / rid / fold / 'receipt.json'
            markers[fold] = driver.sha(path.read_bytes())
        return root, rid, markers

    def test_matched_metrics_shared_alias_and_disclosure(self):
        root, rid, pins = self.make()
        out = report.generate(root, rid, pins)
        public = json.loads((out / 'aggregate/report.json').read_bytes())
        private = json.loads((out / 'private/report.json').read_bytes())
        self.assertEqual(public['mandatory_registered'], 12)
        self.assertEqual(len(public['methods']), 12)
        self.assertEqual(len(public['cohorts']['North']['models']), 12)
        for m, entry in public['methods'].items():
            if entry['representation'] == 'species':
                self.assertEqual(entry['applies_to'], list(driver.ARMS))
            values = [next(r['metrics']['roc_auc'] for r in cohort['models'] if r['method'] == m)
                      for cohort in private['cohorts'].values()]
            self.assertAlmostEqual(private['equal_cohort'][m]['roc_auc']['mean'], np.mean(values))
            self.assertIn('withheld', public['equal_cohort'][m]['brier']['status'])
            self.assertIn('threshold', private['folds']['hold-West']['methods'][m])
        self.assertIn('withheld', public['cohorts']['West']['status'])
        for token in ('visit-', 'person-', 'house-', 'sha256', 'threshold', 'hold-West', 'heldout_fits'):
            self.assertNotIn(token, json.dumps(public))
        self.assertEqual(out, report.generate(root, rid, pins))

    def test_partial_multiclass(self):
        root, rid, pins = self.make(3, neural=False)
        out = report.generate(root, rid, pins)
        result = json.loads((out / 'aggregate/report.json').read_bytes())
        self.assertEqual((result['status'], len(result['unmet']), len(result['methods'])),
                         ('partial-neural-unmet', 3, 9))
        self.assertIn('withheld', result['cohorts']['North']['status'])
        private = json.loads((out / 'private/report.json').read_bytes())
        self.assertEqual(len(private['cohorts']['North']['models']), 9)
        prior = out / 'private/report.json'
        private['reporting_sources']['report_comparators.py'] = '0' * 64
        prior.chmod(0o600)
        save(prior, private)
        snapshot = prior.read_bytes()
        verified = report.generate(root, rid, pins)
        self.assertNotEqual(out, verified)
        self.assertEqual(prior.read_bytes(), snapshot)
        self.assertEqual((out / 'aggregate/report.json').read_bytes(), (verified / 'aggregate/report.json').read_bytes())

    def test_width_limited_neural_family(self):
        root, rid, pins = self.make(3, optional=True, qualified=['species'])
        out = report.generate(root, rid, pins)
        result = json.loads((out / 'aggregate/report.json').read_bytes())
        self.assertEqual(result['mandatory_registered'], 10)
        self.assertEqual(result['unmet'], [dict(family='tabm', representation=r)
                                           for r in ('direct-raw', 'direct-relative')])
        self.assertEqual(len(result['equal_cohort']), 16)

    def test_corrupt_evidence(self):
        root, rid, pins = self.make()
        fold = next(iter(pins))
        store = driver.Store(root / 'jobs/comparator-folds' / rid / fold)
        path = store.root / 'receipt.json'
        path.chmod(0o600)
        original = driver.read(store, json.loads(path.read_bytes())['record'])
        shared = next(k for k, v in original['registry'].items() if v['arm'] == 'SHARED')
        method = 'cmp.raw.direct-raw.logistic'
        for change in ('inputs', 'registry', 'roster', 'threshold', 'classes', 'alias', 'coverage'):
            with self.subTest(change=change):
                body = json.loads(json.dumps(original))
                obj = body['results'][method]
                if change == 'inputs':
                    body['inputs']['meta']['sha256'] = 'f'*64
                elif change == 'registry':
                    del body['results'][method]
                elif change == 'alias':
                    body['registry'][shared]['applies_to'] = ['RAW']
                elif change in ('classes', 'roster'):
                    fit = driver.read(store, obj['fit'])
                    fit['classes' if change == 'classes' else 'samples'].reverse()
                    obj['fit'] = store.record(fit)
                else:
                    rule = driver.read(store, obj['decision'])
                    if change == 'coverage':
                        rule['heldout_fits'] = rule['heldout_fits'][:1]
                    else:
                        rule['threshold'] = -1
                    obj['decision'] = store.record(rule)
                body['objects'] = sorted(set(body['objects']) | store.objects)
                pins[fold] = save(path, dict(record=store.record(body)))['sha256']
                with self.assertRaises(ValueError):
                    report.generate(root, rid, pins)
                self.assertFalse((root / 'jobs/comparator-reports').exists())

    def test_trial_lineage_and_checkpoint_omissions(self):
        root, rid, pins = self.make()
        fold = next(iter(pins))
        store = driver.Store(root / 'jobs/comparator-folds' / rid / fold)
        marker = store.root / 'receipt.json'
        marker.chmod(0o600)
        original = driver.read(store, json.loads(marker.read_bytes())['record'])
        parent, rec, budget, _ = driver.release(root, fold)
        config, splits, meta = report.released_inputs(root, parent)
        proc = driver.Procedure(meta, splits, None, {}, [], None)
        method = 'cmp.raw.direct-raw.logistic'
        for change in ('empty-trials', 'duplicate-trial', 'absent-nested-trial', 'score', 'short-children',
                       'substituted-child', 'wrong-selection', 'wrong-oof', 'missing-head', 'missing-generator',
                       'missing-training-vector', 'stub', 'wrong-blocks', 'missing-unit'):
            with self.subTest(change=change):
                receipt = json.loads(json.dumps(original))
                result = receipt['results'][method]
                trials = result['trials']
                fit = driver.read(store, result['fit'])
                indexes = [i for i, ref in enumerate(trials) if driver.read(store, ref)['parent'] == fold]
                i, j = indexes
                t = driver.read(store, trials[i])
                other = driver.read(store, trials[j])
                remove = None
                if change == 'empty-trials':
                    result['trials'] = []
                elif change == 'duplicate-trial':
                    trials[i] = trials[j]
                elif change == 'absent-nested-trial':
                    trials.pop(next(k for k in range(len(trials)) if k not in indexes))
                elif change in ('score', 'short-children', 'substituted-child'):
                    if change == 'score':
                        t['participant_cohort_auc'] += .01
                    elif change == 'short-children':
                        t['child_fits'].pop()
                    else:
                        t['child_fits'][0] = other['child_fits'][0]
                    trials[i] = store.record(t)
                elif change == 'wrong-selection':
                    fit['candidate'] = next(c for c in budget['candidates']['logistic'] if c != fit['candidate'])
                    result['fit'] = store.record(fit)
                elif change == 'wrong-oof':
                    rule = driver.read(store, result['decision'])
                    current = rule['heldout_fits'][0]
                    rule['heldout_fits'][0] = next(r['child_fits'][0] for r in (t, other) if r['child_fits'][0] != current)
                    result['decision'] = store.record(rule)
                elif change == 'stub':
                    model = driver.read(store, fit['fitted'])
                    fit['fitted'] = store.record({k: model[k] for k in ('config', 'classes')})
                    result['fit'] = store.record(fit)
                elif change == 'wrong-blocks':
                    fit['blocks'] = driver.read(store, t['child_fits'][0])['blocks']
                    result['fit'] = store.record(fit)
                elif change == 'missing-unit':
                    remove = next(r for r in receipt['objects'] if r.endswith('.json') and
                                  isinstance(obj := driver.read(store, r), dict) and obj.get('result') == result['fit'])
                elif change == 'missing-head':
                    model = driver.read(store, fit['fitted'])
                    remove = driver.read(store, model['state'])['coef']
                else:
                    block = fit['blocks'][0]
                    remove = (driver.read(store, block['model'])['state']['mean'] if change == 'missing-generator'
                              else driver.read(store, block['train'])['values'])
                receipt['objects'] = sorted((set(receipt['objects']) | store.objects) - {remove})
                pin = save(marker, dict(record=store.record(receipt)))['sha256']
                with self.assertRaises(ValueError):
                    report.read_fold(store.root, pin, rid, rec, parent, budget, driver.registry(budget), fold, proc, config)

    def test_driver_produced_fold(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root, masked=True)
            store = register(root, 'outer')
            driver.execute(root, 'outer')
            parent, rec, budget, rid = driver.release(root, 'outer')
            config, splits, meta = report.released_inputs(root, parent)
            proc = driver.Procedure(meta, splits, None, {}, [], None)
            marker = driver.sha((store.root / 'receipt.json').read_bytes())
            tables, cuts, link = report.read_fold(store.root, marker, rid, rec, parent, budget,
                                                  driver.registry(budget), 'outer', proc, config)
            self.assertEqual(len(tables), 12)
            self.assertEqual(set(tables), set(cuts))
            self.assertEqual(set(tables), set(link['methods']))
            for method, frame in tables.items():
                self.assertEqual(list(frame.index), list(proc.frames('outer')[1].index))
                self.assertTrue(0 <= cuts[method] <= 1)

    def test_signed_inputs_stop_and_pins(self):
        root, rid, pins = self.make(3, neural=False)
        with self.assertRaises(ValueError):
            report.generate(root, rid, dict.fromkeys(pins, '0'*64))
        with self.assertRaises(ValueError):
            report.generate(root, rid, {})
        (root / 'STOP').touch()
        with self.assertRaisesRegex(RuntimeError, 'STOP'):
            report.generate(root, rid, pins)
        (root / 'STOP').unlink()
        path = root / 'manifests/development-fold-inputs/meta.tsv'
        path.write_bytes(path.read_bytes() + b'\n')
        with self.assertRaisesRegex(ValueError, 'Released input changed'):
            report.generate(root, rid, pins)
