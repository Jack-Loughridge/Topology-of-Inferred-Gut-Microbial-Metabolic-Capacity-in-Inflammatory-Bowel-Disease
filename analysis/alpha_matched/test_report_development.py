"""Exercise reporting CLI against temporary invented release, split and Store artifacts."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd

from . import report_development as report, run_fold, thresholding
from .artifacts import Store, sha
from .nesting import Procedure
from .partitions import FrozenVectors, canonical_json


def write(path, data):
    path.write_bytes(canonical_json(data))


class Fixture:
    def __init__(self, root, nclasses):
        self.root = root
        classes = ['control', 'case', 'other'][:nclasses]
        self.config = dict(endpoint='invented', classes=classes)
        inputs = root / 'manifests/development-fold-inputs'
        inputs.mkdir(parents=True)
        (root / 'reviews').mkdir()
        meta = []
        for c, n in [('North', 24), ('West', 24), ('East', 4)]:
            for i in range(n):
                for j in range(1 + int(i % 4 == 0)):
                    meta.append(dict(sample=f'visit-{c}-{i}-{j}', person=f'person-{c}-{i}',
                                     group=f'house-{c}-{i//2}', cohort=c, family=c, role='development',
                                     y=0 if c == 'East' else i % nclasses))
        frame = pd.DataFrame(meta).set_index('sample')
        self.folds = ['hold-' + c for c in frame.cohort.unique()]
        self.splits = {}
        for f in self.folds:
            valid = frame.cohort.eq(f.removeprefix('hold-'))
            train = frame[~valid]
            self.splits[f] = self.split(train, frame[valid])
            half = train.group.str.rsplit('-', n=1).str[-1].astype(int) % 2 == 0
            self.splits[f + '/a'] = self.split(train[half], train[~half])
            self.splits[f + '/b'] = self.split(train[~half], train[half])
        frame.to_csv(inputs / 'meta.tsv', sep='\t')
        write(inputs / 'config.json', self.config)
        write(inputs / 'splits.json', self.splits)
        self.release = dict(stage='G4', decision='PASS', scope='development-only-fold', reviewer='test-reviewer',
                            blocking_findings=[], endpoint='invented', folds=self.folds,
                            timestamp='2026-01-01T00:00:00+00:00', source_sha256=run_fold.source_hashes(),
                            inputs={key: dict(path=name, sha256=sha((inputs / name).read_bytes()))
                                    for key, name in [('config', 'config.json'), ('splits', 'splits.json'),
                                                      ('meta', 'meta.tsv')]})
        # Only the graph receipt identity is carried through this reporting join.
        self.release['inputs']['bundle'] = dict(path='panel/receipt.json', sha256='f'*64)
        self.sign()
        self.proc = Procedure(frame, self.splits, None, {}, [], None)
        self.names = report.methods_by_arm(self.config)
        self.stores = {}
        for fold in self.folds:
            store = Store(root / 'jobs/development-folds' / self.sid / fold)
            self.stores[fold] = store
            arms = {}
            for arm, names in self.names.items():
                def emit(key):
                    train, valid = self.proc.frames(key)
                    outputs = {}
                    for name in names:
                        # Distinct threshold and outer score distributions. Rank West in reverse.
                        if nclasses == 2:
                            pos = np.where(valid.y == 1, .8, .6)
                            if '/' not in key:
                                pos = np.where(valid.y == 1, .85, .7)
                                if key == 'hold-West':
                                    pos = 1 - pos
                            p = np.column_stack((1-pos, pos))
                        else:
                            p = np.full((len(valid), 3), .1)
                            p[np.arange(len(valid)), valid.y] = .8
                        means = pd.DataFrame(p, index=valid.index).groupby(valid.person).mean()
                        outputs[name] = store.record(dict(arm=arm, name=name, split=key,
                            train_groups=sorted(set(train.group)), classes=classes, endpoint='invented',
                            probabilities=store.array(p), samples=list(valid.index),
                            participant_means=store.array(means.to_numpy()), participants=list(means.index)))
                    return dict(outputs=outputs, fits=list(outputs.values()), trials=[])
                arms[arm] = emit(fold)
                runner = SimpleNamespace(store=store, proc=self.proc, config=self.config, run=emit)
                arms[arm]['decision_rules'] = thresholding.select_thresholds(runner, fold)
            receipt = dict(fold=fold, release=self.sid, inputs=self.release['inputs'], arms=arms,
                           objects=sorted(store.objects), geometry={})
            write(store.root / 'receipt.json', dict(record=store.record(receipt)))

    @staticmethod
    def split(train, valid):
        return dict(train=list(train.person.unique()), validate=list(valid.person.unique()))

    def sign(self):
        path = self.root / 'reviews/invented.json'
        write(path, self.release)
        self.sid = sha(path.read_bytes())
        write(self.root / 'status.json', dict(status='ACTIVE', stage='G4',
              development_fold_release=dict(file=path.name, sha256=self.sid)))
        write(self.root / 'roles.json', {'independent reviewer': 'test-reviewer'})

    def mutate(self, fn):
        store = self.stores[self.folds[0]]
        path = store.root / 'receipt.json'
        rec = json.loads(store.read(json.loads(path.read_bytes())['record']))
        fn(rec, store)
        rec['objects'] = sorted(store.objects)
        write(path, dict(record=store.record(rec)))


class ReportingTests(unittest.TestCase):
    def fixture(self, nclasses=2):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        return Fixture(Path(tmp.name), nclasses)

    def test_cli_binary_all_procedures_projection_and_summaries(self):
        fx = self.fixture()
        out = io.StringIO()
        with redirect_stdout(out):
            report.main(['--release-sha256', fx.sid], root=fx.root)
        directory = Path(out.getvalue().strip())
        private = json.loads((directory / 'private/report.json').read_bytes())
        public = json.loads((directory / 'aggregate/report.json').read_bytes())
        self.assertEqual(private['eligible_auc_cohorts'], ['North', 'West'])
        self.assertEqual(len(private['equal_cohort_auc']), 20)
        self.assertTrue(all(r['roc_auc'] == .5 for r in private['equal_cohort_auc']))
        self.assertEqual(public['equal_cohort_auc'], private['equal_cohort_auc'])
        north = private['cohorts']['North']
        self.assertEqual(north['coverage']['eligible_visits'], 30)
        self.assertEqual(north['coverage']['complete_people'], 24)
        self.assertEqual(north['coverage']['groups'], 12)
        self.assertEqual(len(north['contrasts']), 7)
        self.assertEqual(north['contrasts'][0]['metrics']['roc_auc']['interval'], [0, 0])
        self.assertTrue(all(r['metrics']['balanced_accuracy'] == 1 for r in north['models']))
        self.assertEqual({r['method'] for r in north['models'] if r['arm'] == 'RELATIVE'},
                         {'H0','Ricci','joint','direct-relative','K0','fixed-alpha','species',
                          'activity','historical','graph-count'})
        east = private['cohorts']['East']
        self.assertEqual(east['coverage']['missing_people'], 0)
        self.assertIsNone(east['models'][0]['metrics']['roc_auc'])
        self.assertIn('recall/control', east['contrasts'][0]['metrics'])
        self.assertIn('withheld', public['cohorts']['East']['status'])
        pub_metrics = public['cohorts']['North']['models'][0]['metrics']
        self.assertAlmostEqual(pub_metrics['calibration']['case']['mean_predicted'], .775)
        self.assertEqual(pub_metrics['calibration']['case']['observed'], .5)
        self.assertEqual(len(private['folds']['hold-North']['roster_sha256']), 64)
        public_bytes = json.dumps(public)
        for banned in ('house-', 'visit-', 'person-', 'sha256', 'heldout_fits', 'hold-North', 'decision_rules'):
            self.assertNotIn(banned, public_bytes)

    def test_multiclass_argmax(self):
        fx = self.fixture(3)
        out = report.generate(fx.root, fx.sid)
        document = json.loads((out / 'private/report.json').read_bytes())
        self.assertEqual(document['class_order'], ['control', 'case', 'other'])
        self.assertTrue(all(r['roc_auc'] == 1 for r in document['equal_cohort_auc']))
        r = document['cohorts']['North']['models'][0]['metrics']
        self.assertAlmostEqual(r['brier'], .06)
        self.assertEqual(r['macro_f1'], 1)

    def test_incomplete_fold_fails_without_report(self):
        fx = self.fixture()
        (fx.stores['hold-North'].root / 'receipt.json').unlink()
        with self.assertRaisesRegex(report.IncompleteCoverage, 'Missing completed fold'):
            report.generate(fx.root, fx.sid)
        self.assertFalse((fx.root / 'jobs/development-reports').exists())

    def test_duplicate_and_missing_registered_folds(self):
        fx = self.fixture()
        for folds, error in [([fx.folds[0]]*2, 'duplicate'), (fx.folds[1:], 'outer split registry')]:
            fx.release['folds'] = folds
            fx.sign()
            with self.assertRaisesRegex(ValueError, error):
                report.generate(fx.root, fx.sid)

    def test_duplicate_people_across_nominally_distinct_folds(self):
        fx = self.fixture()
        path = fx.root / 'manifests/development-fold-inputs/splits.json'
        fx.splits[fx.folds[1]] = fx.splits[fx.folds[0]]
        write(path, fx.splits)
        fx.release['inputs']['splits']['sha256'] = sha(path.read_bytes())
        fx.sign()
        with self.assertRaisesRegex(ValueError, 'Duplicate held-out people'):
            report.generate(fx.root, fx.sid)

    def test_reject_swapped_prediction_fit_class_axis_and_roster(self):
        fx = self.fixture()
        def change(rec, store):
            arm = rec['arms']['RAW']
            info = json.loads(store.read(arm['outputs']['species']))
            info.update(self.overrides)
            key = store.record(info)
            arm['fits'].append(key)
            arm['outputs']['species'] = key
        marker = fx.stores[fx.folds[0]].root / 'receipt.json'
        original = marker.read_bytes()
        for overrides in [dict(classes=['case', 'control']), dict(samples=['wrong']),
                          dict(participants=['wrong']), dict(split='hold-West'), dict(arm='RELATIVE')]:
            with self.subTest(overrides=overrides):
                marker.write_bytes(original)
                self.overrides = overrides
                fx.mutate(change)
                with self.assertRaisesRegex(ValueError, 'Fit identity'):
                    report.generate(fx.root, fx.sid)

    def test_missing_baseline_and_mismatched_release(self):
        fx = self.fixture()
        fx.mutate(lambda rec, s: rec['arms']['RAW']['outputs'].pop('historical'))
        with self.assertRaisesRegex(report.IncompleteCoverage, 'methods'):
            report.generate(fx.root, fx.sid)
        fx.mutate(lambda rec, s: rec.update(release='0'*64))
        with self.assertRaisesRegex(ValueError, 'receipt release'):
            report.generate(fx.root, fx.sid)

    def test_wrong_threshold_and_duplicate_inner_fit(self):
        fx = self.fixture()
        def modify(rec, store):
            arm = rec['arms']['RAW']
            rules = json.loads(store.read(arm['decision_rules']))
            rules['thresholds']['species'] = self.stored
            if self.duplicate:
                fits = rules['heldout_fits']['species']
                fits[1] = fits[0]
            arm['decision_rules'] = store.record(rules)
        for stored, duplicate, error in [(.5, False, 'Stored threshold'), (.8, True, 'OOF split')]:
            self.stored, self.duplicate = stored, duplicate
            fx.mutate(modify)
            with self.assertRaisesRegex(ValueError, error):
                report.generate(fx.root, fx.sid)

    def test_producer_saved_species_fit_reads_through_statistics_join(self):
        fx = self.fixture(3)
        key = fx.folds[0]
        train, valid = fx.proc.frames(key)
        store = fx.stores[key]
        factory = SimpleNamespace(arm='RAW', graphs=SimpleNamespace(axis=['invented-feature']))
        config = dict(fx.config, heads=[['linear', .1]])
        producer = run_fold.Fold(fx.proc, factory, store, config, ['invented-taxon'], lambda: None)
        block = {}
        for side, frame in [('train', train), ('validate', valid)]:
            values = np.arange(len(frame), dtype=float).reshape(-1, 1)
            block[side] = store.vector(FrozenVectors(values, tuple(frame.index), ('feature',),
                                                     'RAW', 'invented-fit', 'generator', 'species'))
        observed, fit = producer.fit('species', key, [block], ('linear', .1))
        table = report.fit_table(store, store.objects, fit, 'RAW', 'species', key, train, valid, config)
        np.testing.assert_array_equal(table.to_numpy(), observed)
        self.assertEqual(list(table.index), list(valid.index))

    def test_stop_and_changed_release_before_read(self):
        fx = self.fixture()
        with self.assertRaisesRegex(ValueError, 'not current'):
            report.generate(fx.root, '0'*64)
        (fx.root / 'STOP').touch()
        with self.assertRaisesRegex(RuntimeError, 'STOP'):
            report.generate(fx.root, fx.sid)


if __name__ == '__main__':
    unittest.main()
