"""Synthetic integration fixture for the release-gated development runner."""
import io
import json
import shutil
from contextlib import ExitStack
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, save_npz

from . import run_fold as runner
from .artifacts import Store
from .bundle import sha
from .nesting import fit_head
from .partitions import canonical_json


def fixture(root, nclasses=2, *, final=False, masked=False):
    inputs = root / 'manifests/development-fold-inputs'
    panel = inputs / 'panel'
    panel.mkdir(parents=True)
    reviews = root / 'reviews'
    reviews.mkdir()
    rows = []
    raw = []
    for i in range(20):
        for j in range(1 + (i == 0)):
            rows.append([f's{i}.{j}', f'p{i}', f'g{i}', str(i >= 16), i % nclasses])
            raw.append([.05 + .01 * (i % 5), .2 if i % 2 else .1, .1 if i >= 16 else 0])
    frame = pd.DataFrame(rows, columns='sample person group cohort y'.split())
    frame['family'] = frame.cohort
    frame['role'] = 'development'
    frame.to_csv(inputs / 'meta.tsv', sep='\t', index=False)
    x = csr_matrix(raw)
    species = ['a', 'b', 'c']
    abundance = x
    if final:
        species.append('unassigned')
        abundance = csr_matrix(np.column_stack((raw, np.linspace(.07, .45, len(rows)))))
    for name, matrix in [('raw', x), ('abundance', abundance), ('vertices', csr_matrix(np.ones((len(rows), 3))))]:
        save_npz(panel / (name + '.npz'), matrix)
    np.save(panel / 'pairs.npy', [[0, 1], [0, 2], [1, 2]])
    axes = dict(samples=list(frame['sample']), source_species=species)
    if masked:
        axes['source_rosters'] = {'full': species, 'limited': [t for t in species if t != 'c']}
        axes['sample_releases'] = ['full' if p == 'p0' or int(p[1:]) >= 16 else 'limited' for p in frame.person]
    (panel / 'axes.json').write_bytes(canonical_json(axes))
    receipt = dict(outputs={p.name: sha(p) for p in panel.iterdir()},
                   catalogue_digest='invented', mapping_digest='invented')
    (panel / 'receipt.json').write_bytes(canonical_json(receipt))
    splits = {}
    def split(key, train, valid, depth):
        splits[key] = dict(train=train, validate=valid)
        if depth:
            half = len(train) // 2
            split(key + '/a', train[:half], train[half:], depth - 1)
            split(key + '/b', train[half:], train[:half], depth - 1)
    p = list(frame.person.unique())
    split('outer', p[:16], p[16:], 2)
    if final:
        split(runner.FINAL, p, [], 3 if nclasses == 2 else 2)
    (inputs / 'splits.json').write_bytes(canonical_json(splits))
    config = dict(endpoint='invented', classes=['control', 'case', 'other'][:nclasses], heads=[['linear', 1.0]], candidates=dict(
        H0=[dict(q=25, learned=learned, epochs=1, seed=17) for learned in (True, False)],
        Ricci=[dict(family='linear', parameter=None, rho=.75)]),
        historical=[dict(quantile_step=25, epochs=1, seed=17)],
        tolerances=dict(first_hit_error=1e-7, marginal_error=1e-8, transport_error=1e-6,
                        curvature_error=1e-7))
    (inputs / 'config.json').write_bytes(canonical_json(config))
    files = dict(meta='meta.tsv', config='config.json', splits='splits.json', bundle='panel/receipt.json')
    release = dict(endpoint=config['endpoint'], stage='G4', decision='PASS', scope='development-only-fold', reviewer='test-reviewer',
                   blocking_findings=[], folds=['outer'], timestamp='2026-01-01T00:00:00+00:00',
                   source_sha256=runner.source_hashes(),
                   inputs={k: dict(path=v, sha256=sha(inputs / v)) for k, v in files.items()})
    if final:
        config['fit_thresholds'] = True
        config['endpoint'] = 'binary' if nclasses == 2 else 'subtypes'
        (inputs / 'config.json').write_bytes(canonical_json(config))
        release['inputs']['config']['sha256'] = sha(inputs / 'config.json')
        release.update(stage='G5', gate='G5', scope='development-final-refit', folds=[runner.FINAL],
                       endpoint=config['endpoint'])
    path = reviews / 'test.json'
    path.write_bytes(canonical_json(release))
    (root / 'roles.json').write_bytes(canonical_json({'independent reviewer': 'test-reviewer', 'reserved evaluator': 'test-evaluator'}))
    status = dict(stage='G4', status='ACTIVE', development_fold_release=dict(file='test.json', sha256=sha(path)))
    if final:
        status['stage'] = 'G5'
        status['final_refit_release'] = status.pop('development_fold_release')
    (root / 'status.json').write_bytes(canonical_json(status))
    return status


def application_fixture(root, fits):
    """Independent, unlabeled panel with a new taxon, missing b, and positive unassigned mass."""
    panel = root / 'new-bundle'
    panel.mkdir()
    samples = ['independent-1', 'independent-2']
    taxa = ['only-new', 'c', 'a', 'unassigned']
    pairs = np.array([[0, 1], [0, 2], [1, 2]], dtype='<i8')
    np.save(panel / 'pairs.npy', pairs)
    for name, values in [('raw', [[.13, 0, .27], [0, 0, 0]]), ('vertices', [[1, 1, 1], [1, 0, 0]]),
                         ('abundance', [[.17, .27, .13, .38], [.23, 0, 0, .62]])]:
        save_npz(panel / (name + '.npz'), csr_matrix(values))
    def write(path, obj):
        path.write_bytes(canonical_json(obj))
    write(panel / 'axes.json', dict(samples=samples, source_species=taxa))
    write(panel / 'receipt.json', dict(outputs={p.name: sha(p) for p in panel.iterdir()},
          species_nondetection_zero=True, catalogue_digest='invented', mapping_digest='invented'))

    submission = root / 'package'
    submission.mkdir()
    portable = submission / 'objects'
    portable.mkdir()
    roles = {role: [] for role in ('models', 'generators')}
    plan = []
    rosters = {}
    rules = submission / 'graph-rules.json'
    write(rules, {'catalogue': 'invented', 'mapping': 'invented'})
    for store, receipt in fits:
        for arm, result in receipt['arms'].items():
            for method, ref in result['outputs'].items():
                fit = json.loads(store.read(ref))
                endpoint = fit['endpoint']
                rosters[endpoint] = samples
                model = fit['deployment']
                for name in store.deployment_files(model):
                    shutil.copyfile(store.root / name, portable / name)
                    role = 'models' if name == model else 'generators'
                    relative = 'objects/' + name
                    if relative not in roles[role]:
                        roles[role].append(relative)
                plan.append(dict(id='-'.join((endpoint, arm, method)), endpoint=endpoint,
                                 cohort='invented-new', arm=arm, method=method, class_order=fit['classes'],
                                 row_count=len(samples), eligible_row_hash=runner.sha(canonical_json(sorted(samples))),
                                 requires_curvature=method in ('joint', 'Ricci', 'K0'),
                                 model_sha256=model.split('.')[0], generator_sha256=fit['generators'].split('.')[0],
                                 graph_rules_sha256=sha(rules)))
    write(submission / 'eligibility.json', {'rosters': {'invented-new': rosters}})
    write(submission / 'comparisons.json', {'predictions': plan})
    roles.update(test_eligibility=['eligibility.json'], comparisons=['comparisons.json'])
    for role in 'code environment data splits config thresholds prediction_schema statistical_analysis'.split():
        roles[role] = [rules.name]
    files = {n: sha(submission / n) for names in roles.values() for n in names}
    sub = dict(planned_predictions=plan, eligible_primary_binary_cohorts=['invented-new'], files=files, roles=roles)
    write(submission / 'submission.json', sub)
    sid = sha(submission / 'submission.json')
    review = root / 'reviews' / 'clinical.json'
    write(review, dict(reviewer='test-reviewer', gate='G6', decision='PASS', blocking_findings=[], resolved_findings=[],
                       timestamp='2026-01-01T00:00:00+00:00', submission_sha256=sid, reviewed_files=files))
    write(root / 'status.json', dict(status='ACTIVE', stage='G7', final_prediction_release=dict(file=review.name,
                                                                                                sha256=sha(review))))
    output = root / 'predictions'
    output.mkdir()
    registry = {}
    for sample, edges in zip(samples, [pairs[[0, 2]], pairs[:0]]):
        buf = io.BytesIO()
        np.save(buf, edges, allow_pickle=False)
        registry[sample] = dict(sample=sample, profile_sha256=runner.sha(sample.encode()),
                                graph_rules_sha256=sha(rules), active_pairs_sha256=runner.sha(buf.getvalue()))
    write(output / 'graph_roster.json', dict(submission_sha256=sid, evaluator='test-evaluator',
          bundle_receipts={'invented-new': sha(panel / 'receipt.json')}, cohorts={'invented-new': registry}))
    return submission, panel, output


class TestFold(unittest.TestCase):
    def test_masked_species_transfer(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root, final=True, masked=True)
            rec, _ = runner.release(root, runner.FINAL)
            config, splits, meta, train_graph, native = runner.read_inputs(root, rec)
            store = Store(root / 'model')
            factory = runner.Factory(train_graph, 'RAW', root / 'cache', config['tolerances'])
            proc = runner.Procedure(meta, splits, factory, {}, [], None)
            fold = runner.Fold(proc, factory, store, config, native, lambda: None)
            block = fold.build('species', {}, runner.FINAL)
            model = block['model']
            state = json.loads(store.read(model))['state']
            impute = store.load_array(state['species_impute'])
            # c was observed in four positive people and one zero person with two visits.
            self.assertAlmostEqual(impute[2], .4 / 5)
            self.assertEqual(json.loads(store.read(state['species_taxa'])), ['a', 'b', 'c', 'unassigned'])
            trained = store.load_vector(block['train'])
            self.assertEqual(trained.features, tuple(k + ':"' + t + '"' for k in ('abundance', 'unreported')
                                                    for t in ('a', 'b', 'c', 'unassigned')))
            replay = store.transform(model, factory, trained.ids, native)
            self.assertEqual(replay.identity(), trained.identity())

            from .features import Graphs
            species = ('z', 'c', 'unassigned', 'a')
            observed = np.array([[True, True, True, True], [True, False, True, True]])
            abundance = csr_matrix([[.3, 0, .4, .1], [.35, 0, .5, 0]])
            g = Graphs(csr_matrix((2, 3)), train_graph.pairs, (np.arange(3), np.arange(3)),
                       ('new-1', 'new-2'), abundance, train_graph.catalogue_digest,
                       train_graph.mapping_digest, False, species, observed)
            app = runner.Factory(g, 'RAW', root / 'cache', config['tolerances'])
            before = {p.name: sha(p) for p in store.root.iterdir()}
            values = store.transform(model, app, g.sample_ids, species).values
            unscaled = values * store.load_array(state['scale']) + store.load_array(state['mean'])
            np.testing.assert_allclose(unscaled, [[.1, impute[1], 0, .4, 0, 1, 0, 0],
                                                  [0, impute[1], .08, .5, 0, 1, 1, 0]], atol=1e-15)
            g.species_observed = None
            with self.assertRaisesRegex(ValueError, 'qualified measurement'):
                store.transform(model, app, g.sample_ids, species)
            g.species_nondetection_zero = True
            values = store.transform(model, app, g.sample_ids, species).values
            unscaled = values * store.load_array(state['scale']) + store.load_array(state['mean'])
            np.testing.assert_allclose(unscaled, [[.1, 0, 0, .4, 0, 0, 0, 0],
                                                  [0, 0, 0, .5, 0, 0, 0, 0]], atol=1e-15)
            self.assertEqual(before, {p.name: sha(p) for p in store.root.iterdir()})

    def test_final_gates_and_roster(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root)
            with self.assertRaisesRegex(RuntimeError, 'completed G5'):
                runner.execute(root, runner.FINAL)
            with self.assertRaisesRegex(RuntimeError, 'G6 release'):
                runner.apply_final(root, root, root / 'does-not-exist', 'test', 'binary', root)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root, final=True)
            rec, _ = runner.release(root, runner.FINAL)
            path = root / 'manifests/development-fold-inputs/splits.json'
            splits = json.loads(path.read_bytes())
            splits[runner.FINAL]['validate'] = ['p0']
            path.write_bytes(canonical_json(splits))
            rec['inputs']['splits']['sha256'] = sha(path)
            with self.assertRaisesRegex(ValueError, 'empty application'):
                runner.read_inputs(root, rec)

    def test_binary_g5_requires_thresholds(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root, final=True)
            rec, _ = runner.release(root, runner.FINAL)
            path = root / 'manifests/development-fold-inputs/config.json'
            config = json.loads(path.read_bytes())
            for value in (None, False, 'true'):
                config.pop('fit_thresholds', None)
                if value is not None:
                    config['fit_thresholds'] = value
                path.write_bytes(canonical_json(config))
                rec['inputs']['config']['sha256'] = sha(path)
                with self.assertRaisesRegex(ValueError, 'fit_thresholds'):
                    runner.read_inputs(root, rec)

    def test_final_refits_apply_and_seal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fits = []
            for count in (2, 3):
                dev = root / str(count)
                status = fixture(dev, count, final=True, masked=(count == 3))
                marker = runner.execute(dev, runner.FINAL)
                self.assertEqual(marker, runner.execute(dev, runner.FINAL))
                directory = dev / 'jobs/development-final-refits' / status['final_refit_release']['sha256'] / runner.FINAL
                store = Store(directory)
                receipt = json.loads(store.read(json.loads((directory / 'receipt.json').read_bytes())['record']))
                for arm in receipt['arms'].values():
                    rules = json.loads(store.read(arm['decision_rules']))
                    if count == 2:
                        self.assertEqual(rules['parent'], runner.FINAL)
                        self.assertEqual(set(rules['thresholds']), set(arm['outputs']))
                        for refs in rules['heldout_fits'].values():
                            self.assertTrue(set(refs) <= set(arm['fits']))
                            self.assertEqual({json.loads(store.read(ref))['split'] for ref in refs},
                                             {runner.FINAL + '/a', runner.FINAL + '/b'})
                    else:
                        self.assertEqual(rules, dict(rule='argmax', class_order=['control', 'case', 'other']))
                    self.assertEqual(len(arm['outputs']), 10)
                    for ref in arm['outputs'].values():
                        fit = json.loads(store.read(ref))
                        self.assertFalse({'samples', 'probabilities', 'participant_means'} & fit.keys())
                        for block in fit['blocks']:
                            trained = store.load_vector(block['train'])
                            self.assertEqual(len(trained.ids), 21)
                            self.assertEqual(store.load_vector(block['validate']).ids, ())
                    splits = json.loads((dev / 'manifests/development-fold-inputs/splits.json').read_bytes())
                    for ref in arm['fits']:
                        fit = json.loads(store.read(ref))
                        for block in fit['blocks']:
                            people = {'p' + i[1:].split('.')[0] for i in store.load_vector(block['train']).ids}
                            self.assertEqual(people, set(splits[fit['split']]['train']))
                    self.assertTrue({runner.FINAL, runner.FINAL + '/a', runner.FINAL + '/b'} <=
                                    {json.loads(store.read(ref))['split'] for ref in arm['trials']})
                fits.append((store, receipt))

            submission, panel, out = application_fixture(dev, fits)
            def snapshot():
                return {p.relative_to(submission): sha(p) for p in submission.rglob('*') if p.is_file()}
            before = snapshot()
            graph = runner.bundle.load(panel)
            abundance = graph.species_abundance.toarray().copy()
            with ExitStack() as guards:
                for target in ('run_fold.fit_head', 'run_fold.fit_historical', 'features.Factory.__call__',
                               'features.Generator.__init__', 'comparators.Baseline.__init__',
                               'partitions.RelativeReference.fit'):
                    guards.enter_context(patch('analysis.alpha_matched.' + target, side_effect=AssertionError('refit')))
                for store, receipt in fits:
                    fit = json.loads(store.read(receipt['arms']['RAW']['outputs']['species']))
                    endpoint = fit['endpoint']
                    with patch.object(Store, 'probabilities', autospec=True, side_effect=Store.probabilities) as calls:
                        results = runner.apply_final(dev, submission, panel, 'invented-new', endpoint, out)
                    species_calls = [c for c in calls.call_args_list if c.args[1] == 'species']
                    self.assertEqual(len(species_calls), len(graph.sample_ids))
                    self.assertEqual(len(results), 20)
                    rows = [json.loads(line) for line in (out / results[endpoint + '-RAW-species']).read_text().splitlines()]
                    state = json.loads(store.read(fit['blocks'][0]['model']))['state']
                    expected = np.array([[.13, 0, .27, .38], [0, 0, 0, .62]])
                    if state['species_impute']:
                        expected = np.column_stack((expected, np.zeros_like(expected)))
                    x = (expected - store.load_array(state['mean'])) / store.load_array(state['scale'])
                    np.testing.assert_allclose([r['probabilities'] for r in rows], store.predict(fit['fitted'], x))
                    self.assertEqual([r['sample'] for r in rows], list(graph.sample_ids))
                    self.assertEqual(results, runner.apply_final(dev, submission, panel, 'invented-new', endpoint, out))
                    factory = runner.Factory(graph, 'RAW', out / 'geometry', {})
                    model = fit['blocks'][0]['model']
                    graph.species_nondetection_zero = False
                    with self.assertRaisesRegex(ValueError, 'nondetection semantics'):
                        store.transform(model, factory, graph.sample_ids, graph.species_ids)
                    graph.species_nondetection_zero = True
                    store.transform(model, factory, graph.sample_ids, graph.species_ids)
                    np.testing.assert_array_equal(graph.species_abundance.toarray(), abundance)
            seal = runner.clinical.verify_and_seal(submission, dev / 'reviews/clinical.json',
                                                   'test-reviewer', 'test-evaluator', out)
            self.assertEqual(len(json.loads(seal.read_bytes())['predictions']), 40)
            self.assertEqual(snapshot(), before)
            with (dev / 'reviews/clinical.json').open('ab') as stream:
                stream.write(b' ')
            with self.assertRaisesRegex(ValueError, 'Changed G6 review'):
                runner.apply_final(dev, submission, panel, 'invented-new', 'binary', out)

    def test_synthetic_fold(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            status = fixture(root)
            marker = runner.execute(root, 'outer')
            self.assertEqual(runner.execute(root, 'outer'), marker)
            directory = root / 'jobs/development-folds' / status['development_fold_release']['sha256'] / 'outer'
            store = Store(directory)
            def record(ref):
                return json.loads(store.read(ref))
            receipt = record(json.loads((directory / 'receipt.json').read_bytes())['record'])
            release, _ = runner.release(root, 'outer')
            config, _, _, graph, species = runner.read_inputs(root, release)
            for arm, result in receipt['arms'].items():
                outputs = result['outputs']
                factory = runner.Factory(graph, arm, directory / 'geometry', config['tolerances'])
                factory.version = runner.sha(canonical_json(release['source_sha256']))
                for ref in outputs.values():
                    for block in record(ref)['blocks']:
                        held = store.load_vector(block['validate'])
                        replay = store.transform(block['model'], factory, held.ids, species)
                        self.assertEqual(replay.identity(), held.identity())
                self.assertEqual(set(outputs), {'H0', 'Ricci', 'joint', 'species', 'graph-count', 'activity',
                                                'direct-' + arm.lower(), 'historical', 'K0', 'fixed-alpha'})
                for ref in result['fits']:
                    fit = record(ref)
                    for block in fit['blocks']:
                        trained = store.load_vector(block['train'])
                        held = store.load_vector(block['validate'])
                        self.assertFalse(set(trained.ids) & set(held.ids))
                        if fit['split'] != 'outer':
                            self.assertTrue(all(int(i[1:].split('.')[0]) < 16 for i in trained.ids + held.ids))
                species_block = record(outputs['species'])['blocks'][0]
                species_fit = record(species_block['model'])
                # The third fixture taxon is positive only in the outer validation rows.
                self.assertEqual(store.load_array(species_fit['state']['species_axis']).tolist(), [0, 1])
                frozen = store.load_vector(species_block['validate'])
                self.assertEqual(frozen.features, ('a', 'b'))
                self.assertEqual(store.transform(species_block['model'], factory, frozen.ids, species).identity(),
                                 frozen.identity())
                with self.assertRaisesRegex(ValueError, 'abundance coordinates'):
                    store.transform(species_block['model'], factory, frozen.ids, ['c', 'b', 'a'])
                self.assertEqual(record(outputs['joint'])['blocks'],
                                 [record(outputs[m])['blocks'][0] for m in ('H0', 'Ricci')])
                if arm == 'RELATIVE':
                    refs = [record(record(outputs[m])['blocks'][0]['model'])['reference']
                            for m in ('direct-relative', 'Ricci', 'H0', 'historical')]
                    self.assertEqual(len(set(refs)), 1)
                    self.assertTrue(store.load_array(record(refs[0])['unseen'])[-1])
                self.assertGreater(len(result['trials']), 10)
            target = directory / record(outputs['joint'])['probabilities']
            target.chmod(0o600)
            target.write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError, 'digest mismatch'):
                runner.execute(root, 'outer')

    def test_gate_before_input_read(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(runner, 'read_inputs') as reader:
            root = Path(tmp)
            status = fixture(root)
            for modified in [dict(status, stage='G3'), dict(status, status='PAUSED'),
                             dict(status, development_fold_release=dict(file='test.json', sha256='stale'))]:
                (root / 'status.json').write_bytes(canonical_json(modified))
                with self.assertRaises((ValueError, RuntimeError)):
                    runner.execute(root, 'outer')
            (root / 'STOP').touch()
            with self.assertRaisesRegex(RuntimeError, 'STOP'):
                runner.execute(root, 'outer')
            reader.assert_not_called()

    def test_reject_mixed_role_and_cross_family_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root)
            rec, _ = runner.release(root, 'outer')
            path = root / 'manifests/development-fold-inputs/meta.tsv'
            meta = pd.read_csv(path, sep='\t')
            meta.loc[0, 'role'] = 'invented-other-role'
            meta.to_csv(path, sep='\t', index=False)
            rec['inputs']['meta']['sha256'] = sha(path)
            with self.assertRaisesRegex(ValueError, 'non-development'):
                runner.read_inputs(root, rec)
            meta['role'] = 'development'
            meta['family'] = 'shared'
            meta.to_csv(path, sep='\t', index=False)
            rec['inputs']['meta']['sha256'] = sha(path)
            with patch.object(runner, 'release', return_value=(rec, 'synthetic')):
                with self.assertRaisesRegex(ValueError, 'cohort partition'):
                    runner.execute(root, 'outer')

    def test_three_class_fold(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            status = fixture(root, 3)
            runner.execute(root, 'outer')
            directory = root / 'jobs/development-folds' / status['development_fold_release']['sha256'] / 'outer'
            store = Store(directory)
            receipt = json.loads(store.read(json.loads((directory / 'receipt.json').read_bytes())['record']))
            for result in receipt['arms'].values():
                for fitted in result['outputs'].values():
                    record = json.loads(store.read(fitted))
                    self.assertEqual(record['classes'], ['control', 'case', 'other'])
                    self.assertEqual(store.load_array(record['probabilities']).shape, (4, 3))

    def test_compression_preserves_numeric_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(tmp)
            for arr in (np.zeros((32, 1024), dtype='>f8'),
                        np.array([[0., -0., 1.e-38]], dtype='>f4').T,
                        np.arange(60, dtype='<i8')[::3], np.empty((0, 7), dtype='bool')):
                key = store.array(arr)
                restored = store.load_array(key)
                self.assertEqual((arr.dtype, arr.shape, arr.tobytes()),
                                 (restored.dtype, restored.shape, restored.tobytes()))
                if arr.nbytes > 65536:
                    self.assertLess((store.root / key).stat().st_size, arr.nbytes // 50)

    def test_portable_forest(self):
        with tempfile.TemporaryDirectory() as tmp:
            s = Store(tmp)
            x = np.arange(36, dtype=float).reshape(-1, 3)
            frame = pd.DataFrame(dict(y=np.arange(len(x)) % 2, person=np.arange(len(x))))
            model = fit_head(('forest', 2), x, frame)
            np.testing.assert_allclose(s.predict(s.head(model), x), model.predict_proba(x))


if __name__ == '__main__':
    unittest.main()
