"""Invented matched comparator folds, release rejection, restart and mixed G6 sealing."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from . import comparator_run as driver
from .test_artifacts import fixture, application_fixture
from .test_comparator_heads import PARAMS


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(driver.canonical_json(obj))
    return dict(file=path.name, sha256=driver.core.bundle.sha(path))


def register(root, fold, neural=True, primary=None):
    parent, pid = driver.core.release(root, fold)
    optional = ['H0', 'Ricci', 'joint'] if primary else []
    params = {k: dict(v) for k, v in PARAMS.items()}
    params['tabm'].update(epochs=1, k=2, d_block=4)
    candidates = {k: [dict(family=k, seed=15, params=p)] for k, p in params.items() if neural or k != 'tabm'}
    candidates['logistic'].append(dict(family='logistic', seed=15, params=dict(C=.1, max_iter=120)))
    inputs = list(driver.DIRECT) if neural else []
    budget = dict(candidates=candidates, packages=driver.packages(), neural_family='tabm' if neural else None,
                  neural_inputs=inputs, frozen_vectors=optional, budget_rationale='Tiny invented coverage grid',
                  unmet_reason='' if neural else 'Synthetic pending operational review',
                  max_features={n: 32 for n in inputs + optional})
    budget_ref = save(root / 'manifests/fixture-budget.json', budget)
    rec = dict(scope='development-comparators', stage=parent['stage'], reviewer=parent['reviewer'],
               timestamp=parent['timestamp'], decision='PASS', blocking_findings=[], folds=[fold],
               source_sha256=driver.source_hashes(), parent_sha256=pid, budget=budget_ref)
    if primary:
        rec['primary_receipts'] = {fold: primary}
    if neural:
        probe = save(root / 'jobs/neural-feasibility/test-probe.json', dict(family='TabM', version='0.0.3',
            score_inspected=False, replay='exact', torch_threads=1, checkpoint_sha256='d'*64, width=32,
            fit_rows=20, application_rows=2, wall_seconds=1, peak_rss_kib=100))
        access = save(root / 'jobs/neural-feasibility/test-access.json',
                      dict(model='TabPFN-3.5', licensed_access=False, reason='invented access outcome'))
        rec['neural_review'] = save(root / 'reviews/test-feasibility.json', dict(decision='PASS',
            reviewer=parent['reviewer'], scope='local-operational-feasibility', blocking_findings=[],
            family='tabm', selection_basis='operational', preferred_family_probe=access, fallback_reason='fixture',
            budget_sha256=budget_ref['sha256'], timestamp=parent['timestamp'],
            evidence=dict.fromkeys(inputs + optional, probe)))
    ref = save(root / 'reviews/test-comparator.json', rec)
    path = root / 'status.json'
    status = json.loads(path.read_bytes())
    status['comparator_final_refit_release' if fold == driver.core.FINAL else 'comparator_release'] = ref
    save(path, status)
    return driver.Store(root / 'jobs/comparator-folds' / ref['sha256'] / fold)


def add_comparators(root, submission, output, store, receipt):
    path = submission / 'submission.json'
    sub = json.loads(path.read_bytes())
    core_item = sub['planned_predictions'][0]
    for method, result in receipt['results'].items():
        model = result['deployment']
        spec = driver.read(store, model)
        for name in driver.deployment_files(store, model):
            dest = submission / 'objects' / name
            dest.write_bytes((store.root / name).read_bytes())
            relative = 'objects/' + name
            sub['files'][relative] = driver.core.bundle.sha(dest)
            role = 'models' if name == model else 'generators'
            if relative not in sub['roles'][role]:
                sub['roles'][role].append(relative)
        entry = receipt['registry'][method]
        for arm in entry['applies_to']:
            sub['planned_predictions'].append(dict(core_item, id=arm + '-' + method, arm=arm, method=method,
                          requires_curvature=entry['representation'] in ('Ricci','joint'),
                          model_sha256=model.split('.')[0], generator_sha256=spec['generators'].split('.')[0]))
    sub['files']['comparisons.json'] = save(submission / 'comparisons.json',
                                           {'predictions': sub['planned_predictions']})['sha256']
    sid = save(path, sub)['sha256']
    review_path = root / 'reviews/clinical.json'
    review = json.loads(review_path.read_bytes())
    review.update(reviewed_files=sub['files'], submission_sha256=sid)
    ref = save(review_path, review)
    save(root / 'status.json', dict(status='ACTIVE', stage='G7', final_prediction_release=ref))
    roster = output / 'graph_roster.json'
    save(roster, dict(json.loads(roster.read_bytes()), submission_sha256=sid))


class Comparators(unittest.TestCase):
    def test_development_and_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root, masked=True)
            store = register(root, 'outer')
            original = driver.heads.fit
            calls = []
            def interrupt(*args):
                calls.append(args[-1])
                if len(calls) == 6:
                    raise InterruptedError('synthetic interruption')
                return original(*args)
            with patch.object(driver.heads, 'fit', side_effect=interrupt), self.assertRaises(InterruptedError):
                driver.execute(root, 'outer')
            self.assertFalse((store.root / 'receipt.json').exists())
            with patch.object(driver.heads, 'fit', wraps=original) as fit:
                key = driver.execute(root, 'outer')
                # Six selection partitions per candidate and one winning full-parent
                # fit per family and representation. Five fits survived interruption.
                self.assertEqual(fit.call_count + 5, 3 * (6 * 5 + 4))
            with patch.object(driver.heads, 'fit', side_effect=AssertionError('unexpected refit')):
                self.assertEqual(driver.execute(root, 'outer'), key)
            receipt = driver.read(store, key)
            _, reported = driver.report_fold(store.root, receipt['release'], receipt['parent'], 'outer',
                                             driver.core.bundle.sha(store.root / 'receipt.json'))
            self.assertEqual(reported, receipt)
            self.assertEqual((receipt['status'], len(receipt['results'])), ('complete', 12))
            rows = {arm: {r['method']: r for r in driver.report_rows(store, receipt, arm)} for arm in driver.ARMS}
            shared = set(rows['RAW']) & set(rows['RELATIVE'])
            self.assertEqual(len(shared), 4)
            for method in shared:
                self.assertEqual(rows['RAW'][method]['fit'], rows['RELATIVE'][method]['fit'])
                self.assertEqual(rows['RAW'][method]['fitted_arm'], 'SHARED')
            for fit in receipt['results'].values():
                rule = driver.read(store, fit['decision'])
                for ref in rule['heldout_fits']:
                    child = driver.read(store, ref)
                    self.assertTrue(set(child['train_groups']).isdisjoint('g'+s[1:].split('.')[0] for s in child['samples']))
            (root / 'jobs/neural-feasibility/test-probe.json').write_bytes(b'{}')
            with self.assertRaisesRegex(ValueError, 'Changed comparator registration'):
                driver.execute(root, 'outer')

    def test_missing_approval_is_partial_or_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root, 3)
            store = register(root, 'outer', neural=False)
            receipt = driver.read(store, driver.execute(root, 'outer'))
            self.assertEqual((receipt['status'], len(receipt['results'])), ('partial-neural-unmet', 9))
            self.assertTrue(all(driver.read(store, v['decision'])['rule']=='argmax' for v in receipt['results'].values()))
            register(root, 'outer')
            (root / 'reviews/test-feasibility.json').unlink()
            with patch.object(driver.core, 'read_inputs', side_effect=AssertionError), self.assertRaises(FileNotFoundError):
                driver.execute(root, 'outer')
            (root / 'STOP').touch()
            with self.assertRaisesRegex(RuntimeError, 'STOP'):
                driver.execute(root, 'outer')

    def test_frozen_primary_and_sealed_application(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            status = fixture(root, final=True, masked=True)
            fold = driver.core.FINAL
            marker = driver.core.execute(root, fold)
            parent = driver.Store(root / 'jobs/development-final-refits' / status['final_refit_release']['sha256'] / fold)
            primary = driver.read(parent, json.loads((parent.root / 'receipt.json').read_bytes())['record'])
            store = register(root, fold, primary=marker)
            with patch('analysis.alpha_matched.features.Generator.__init__', side_effect=AssertionError('upstream refit')):
                result = driver.read(store, driver.execute(root, fold))
            self.assertEqual(len(result['results']), 18)
            for key, entry in result['registry'].items():
                rep = entry['representation']
                if rep not in driver.DIRECT:
                    fit = driver.read(store, result['results'][key]['fit'])
                    core_fit = driver.read(parent, primary['arms'][entry['arm']]['outputs'][rep])
                    self.assertEqual(fit['blocks'], core_fit['blocks'])
                self.assertNotIn('probabilities', driver.read(store, result['results'][key]['fit']))
            submission, panel, out = application_fixture(root, [(parent, primary)])
            add_comparators(root, submission, out, store, result)
            with patch.object(driver.heads, 'fit', side_effect=AssertionError('external refit')):
                outputs = driver.apply_final(root, submission, panel, 'invented-new', 'binary', out)
            self.assertEqual(len(outputs), 42)
            for family in (*driver.BASE, 'tabm'):
                def probs(arm):
                    artifact = outputs[arm+'-cmp.shared.species.'+family]
                    return [json.loads(row)['probabilities'] for row in (out / artifact).read_bytes().splitlines()]
                np.testing.assert_array_equal(probs('RAW'), probs('RELATIVE'))
            seal = driver.core.clinical.verify_and_seal(submission, root / 'reviews/clinical.json',
                                                       'test-reviewer','test-evaluator',out)
            self.assertEqual(len(json.loads(seal.read_bytes())['predictions']), 42)


if __name__ == '__main__':
    unittest.main()
