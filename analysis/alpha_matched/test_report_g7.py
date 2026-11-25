"""Invented-only G7 tests with real local final-refit model and rule descriptors."""
import json
import os
from pathlib import Path
from contextlib import redirect_stdout
import io
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from . import comparator_run as cmp, report_g7 as g7
from .artifacts import Store, sha
from .partitions import canonical_json
from .test_artifacts import fixture, application_fixture
from .test_comparator_run import register, save

THREAD = "12345678-abcd-4321-bcde-0123456789ab"
ROLE = "/root/evaluator"


def get(path):
    return json.loads(path.read_bytes())


def make_package(base):
    fits, extras, training = [], [], {}
    axes = {'binary':['control','case'], 'subtypes':['control','case','other'], 'cd-vs-non-ibd':['non-IBD','CD']}
    for ep, classes in axes.items():
        dev = base / ('training-' + ep)
        status = fixture(dev, len(classes), final=True)
        inputs = dev / 'manifests/development-fold-inputs'
        if ep == 'cd-vs-non-ibd':
            config = get(inputs / 'config.json')
            config.update(endpoint=ep,classes=classes)
            cfg_ref = save(inputs / 'config.json',config)
            ref = get(dev / 'reviews/test.json')
            ref['endpoint'] = ep
            ref['inputs']['config']['sha256'] = cfg_ref['sha256']
            status['final_refit_release'] = save(dev / 'reviews/test.json',ref)
            save(dev / 'status.json',status)
        meta = pd.read_csv(inputs / 'meta.tsv',sep='\t',dtype=str)
        training[ep] = [[sha(canonical_json(x)) for x in row] for row in meta[['sample','person','group']].values.tolist()]
        fold = cmp.core.FINAL
        marker = cmp.core.execute(dev, fold)
        s = Store(dev / 'jobs/development-final-refits' / status['final_refit_release']['sha256'] / fold)
        fits.append((s, cmp.read(s, get(s.root / 'receipt.json')['record'])))
        comp = register(dev, fold, primary=marker if ep == 'cd-vs-non-ibd' else None)
        extras.append((comp, cmp.read(comp, cmp.execute(dev, fold))))
    root = base / 'run'
    work = root / 'invented-custody'
    work.mkdir(parents=True)
    (work / 'reviews').mkdir()
    subdir, _, pdir = application_fixture(work, fits)
    subdir.rename(work / 'submission')
    subdir = work / 'submission'
    (root / 'reviews').mkdir()
    save(root / 'roles.json', {'reserved evaluator': ROLE, 'independent reviewer': 'test-reviewer'})
    sub = get(subdir / 'submission.json')
    sub['roles']['thresholds'] = []
    rules, exports = {}, {}
    portable = Store(subdir / 'objects')

    def copy_object(s, name, role):
        portable.publish(name, (s.root / name).read_bytes())
        relative = 'objects/' + name
        if relative not in sub['roles'][role]:
            sub['roles'][role].append(relative)
        return relative

    for (s, receipt), (cs, cr) in zip(fits, extras):
        ep = cmp.read(s,next(iter(receipt['arms']['RAW']['outputs'].values())))['endpoint']
        export = dict(endpoint=ep,classes=axes[ep],parent_release=receipt['release'],comparator_release=cr['release'],
                      primary_marker=g7.release.sha(s.root / 'receipt.json'),comparator_marker=g7.release.sha(cs.root / 'receipt.json'),
                      exporter_sha256=g7.release.sha(Path(__file__)), files={}, models=[], decision_sources=[], procedures=[])
        exports[ep] = export
        for arm, result in receipt['arms'].items():
            source = copy_object(s, result['decision_rules'], 'thresholds')
            rule = cmp.read(s, result['decision_rules'])
            for method, ref in result['outputs'].items():
                fit = cmp.read(s, ref)
                ep = fit['endpoint']
                model = cmp.read(s, fit['deployment'])
                assert 'decision_source_sha256' not in model
                item = next(i for i in sub['planned_predictions'] if (i['endpoint'],i['arm'],i['method']) == (ep,arm,method))
                assert item['model_sha256'] == fit['deployment'].split('.')[0]
                for obj in s.deployment_files(fit['deployment']):
                    export['files']['objects/'+obj] = obj.split('.')[0]
                export['models'].append('objects/'+fit['deployment'])
                cut = rule.get('thresholds',{}).get(method)
                rules[ep, arm, method] = source, dict(rule=rule['rule'],threshold=cut), ref
        for method, result in cr['results'].items():
            model = cmp.read(cs, result['deployment'])
            entry = cr['registry'][method]
            export['models'].append('objects/'+result['deployment'])
            for obj in cmp.deployment_files(cs, result['deployment']):
                export['files']['objects/'+obj] = obj.split('.')[0]
                copy_object(cs, obj, 'models' if obj == result['deployment'] else 'generators')
            source = copy_object(cs, result['decision'], 'thresholds')
            for arm in entry['applies_to']:
                item = dict(item, arm=arm, method=method, endpoint=model['endpoint'], class_order=model['classes'],
                            requires_curvature=entry['representation'] in ('Ricci','joint'),
                            model_sha256=result['deployment'].split('.')[0], generator_sha256=model['generators'].split('.')[0])
                sub['planned_predictions'].append(item)
                rules[model['endpoint'],arm,method] = source, model['decision'], result['fit']

    roles = {'North':dict(binary='primary',subtypes='secondary'),
             'South':dict(binary='secondary',subtypes='unsupported-secondary'),
             'Harbour':dict(binary='structurally-inapplicable',subtypes='structurally-inapplicable')}
    for c in roles:
        roles[c]['cd-vs-non-ibd'] = 'primary' if c == 'Harbour' else 'structurally-inapplicable'
    cfg = dict(version=1, implementation=g7.sources(), bootstrap=dict(draws=2000, seed=20261124),
               contrasts=g7.statistics.PRIMARY_PAIRS, public_min_groups=5, aggregate_disclosure=g7.AGGREGATE_DISCLOSURE,
               procedure_exports={ep:ep+'-export.json' for ep in axes}, training_identities='training-identities.json',
               panel_roles=roles, endpoints=axes, unmet_methods=[], unavailable_panels=[])
    rows, rosters = [], {}
    for cohort, n in [('North',18), ('South',2), ('Harbour',16)]:
        rosters[cohort] = {}
        for ep, classes in axes.items():
            if roles[cohort][ep] not in ('primary','secondary'):
                cfg['unavailable_panels'].append(dict(cohort=cohort,endpoint=ep,reason='invented eligibility'))
                continue
            visits = []
            for i in range(n):
                for j in range(1 + int(i == 0 and cohort != 'Harbour')):
                    sample = f'visit-{cohort}-{i}-{j}'
                    visits.append(sample)
                    rows.append(dict(sample=sample,person=f'person-{cohort}-{i}',group=f'group-{cohort}-{i//2}',
                                     cohort=cohort,endpoint=ep,label=classes[i % len(classes) if cohort != 'South' else 1]))
            rosters[cohort][ep] = visits
    plan = []
    for cohort, eps in rosters.items():
        for i in sub['planned_predictions']:
            if i['endpoint'] not in eps:
                continue
            pid = '-'.join([cohort,i['endpoint'],i['arm'],i['method']])
            visits = eps[i['endpoint']]
            item = dict(i, id=pid, cohort=cohort, row_count=len(visits), eligible_row_hash=sha(canonical_json(sorted(visits))))
            plan.append(item)
    for item in sub['planned_predictions']:
        ep, arm, method = (item[k] for k in ('endpoint','arm','method'))
        source, rule, fit = rules[ep,arm,method]
        export = exports[ep]
        digest = Path(source).name.split('.')[0]
        export['files'][source] = digest
        if source not in export['decision_sources']:
            export['decision_sources'].append(source)
        fields = ('endpoint','arm','method','class_order','requires_curvature','model_sha256','generator_sha256')
        export['procedures'].append(dict({k:item[k] for k in fields},fit=fit,rule=rule,decision_source_sha256=digest))
    for ep, export in exports.items():
        name = cfg['procedure_exports'][ep]
        save(subdir / name,export)
        sub['roles']['config'].append(name)
    save(subdir / cfg['training_identities'],training)
    sub['roles']['splits'].append(cfg['training_identities'])
    sub.update(planned_predictions=plan, eligible_primary_binary_cohorts=['North'],g7_reporting='report-policy.json')
    cfg['identities_sha256'] = sha(canonical_json(sorted([r[k] for k in ('cohort','endpoint','sample','person','group')] for r in rows)))
    (work / 'outcomes').mkdir()
    save(work / 'outcomes/identities.json', sorted([r[k] for k in g7.AXES] for r in rows))
    data = b''.join(canonical_json(r) + b'\n' for r in rows)
    (work / 'outcomes/outcomes.jsonl').write_bytes(data)
    cfg['outcomes_sha256'] = sha(data)
    sub['roles']['statistical_analysis'] = ['report-policy.json']
    for name in cfg['implementation']:
        relative = 'code/' + name
        (subdir / 'code').mkdir(exist_ok=True)
        shutil.copyfile(Path(__file__).with_name(name), subdir / relative)
        sub['roles']['code'].append(relative)
    for name, obj in [('report-policy.json',cfg),('comparisons.json',dict(predictions=plan)),
                      ('eligibility.json',dict(rosters=rosters)),('submission.json',sub)]:
        save(subdir / name, obj)
    return root


def activate(root):
    """Invent a fresh signed G6 and whole-run seal after a fixture mutation."""
    work = root / 'invented-custody'
    s, p = work / 'submission', work / 'predictions'
    sub = get(s / 'submission.json')
    sub['files'] = {n:g7.release.sha(s / n) for names in sub['roles'].values() for n in names}
    sid = save(s / 'submission.json', sub)['sha256']
    rev = dict(gate='G6',decision='PASS',blocking_findings=[],resolved_findings=[],reviewer='test-reviewer',
               evaluator_session=dict(role=ROLE,thread_id=THREAD),timestamp='2026-01-01T00:00:00+00:00',submission_sha256=sid,reviewed_files=sub['files'])
    review = save(root / 'reviews/invented.json', rev)
    rosters = get(s / 'eligibility.json')['rosters']
    np.save(p / 'edges.npy', np.empty((0,2), dtype=np.int64), allow_pickle=False)
    np.savez(p / 'geom.npz', pairs=np.empty((0,2),dtype=np.int64),curvature=np.empty(0))
    edge_hash, geom_hash = (g7.release.sha(p / n) for n in ('edges.npy','geom.npz'))
    graphs = {}
    for item in sub['planned_predictions']:
        cohort, ep, pid = item['cohort'],item['endpoint'],item['id']
        lines = []
        for n, sample in enumerate(rosters[cohort][ep]):
            evidence = dict(sample=sample,profile_sha256=sha(sample.encode()),graph_rules_sha256=item['graph_rules_sha256'],
                            active_pairs_sha256=edge_hash)
            graphs.setdefault(cohort,{})[sample] = evidence
            positive = .2 + .05 * (n % 9)
            probs = [1. - positive, positive] if len(item['class_order']) == 2 else [.1,.2,.7]
            fields = {k:item[k] for k in ('class_order','generator_sha256','model_sha256')}
            lines.append(canonical_json(dict(**fields,sample=sample,prediction_id=pid,submission_sha256=sid,
                probabilities=probs,geometry=dict(**evidence,prediction_id=pid,generator_sha256=item['generator_sha256'],
                values='geom.npz',values_sha256=geom_hash,active_pairs='edges.npy'))))
        artifact = pid + '.jsonl'
        data = b'\n'.join(lines) + b'\n'
        (p / artifact).write_bytes(data)
        save(p / (pid + '.json'),dict(prediction_id=pid,submission_sha256=sid,eligible_row_hash=item['eligible_row_hash'],
             row_count=len(lines),artifact=artifact,sha256=sha(data),expected_edges=0,computed_finite_edges=0,failed_edges=0,
             schema_validation_passed=True))
    save(p / 'graph_roster.json',dict(submission_sha256=sid,evaluator=ROLE,cohorts=graphs))
    sealfile = p / 'SEALED.json'
    sealfile.unlink(missing_ok=True)
    seal = g7.release.verify_and_seal(s, root / 'reviews/invented.json','test-reviewer',ROLE,p)
    save(root / 'status.json',dict(status='ACTIVE',stage='G7',final_prediction_release=review,
         g7_report_release=dict(submission_sha256=sid, review_sha256=review['sha256'],seal_sha256=g7.release.sha(seal),
                                workspace=str(work),evaluator=ROLE)))


class PostSeal(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(tmp.cleanup)
        cls.base = make_package(Path(tmp.name))
        activate(cls.base)

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name) / 'run'
        shutil.copytree(self.base, self.root)
        self.work = self.root / 'invented-custody'
        self.s, self.p = self.work / 'submission', self.work / 'predictions'
        self.status = self.root / 'status.json'
        status = get(self.status)
        status['g7_report_release']['workspace'] = str(self.work)
        save(self.status,status)
        self.enterContextCompat(patch.dict(os.environ,CODEX_THREAD_ID=THREAD))

    def enterContextCompat(self, ctx):
        ctx.__enter__()
        self.addCleanup(ctx.__exit__,None,None,None)

    def test_global_report(self):
        path = g7.generate(self.root)
        private, public = (get(path / scope / 'report.json') for scope in ('private','aggregate'))
        self.assertEqual(private['status'],'complete-with-declared-limitations')
        for ep in ('binary','subtypes'):
            panel = private['cohorts']['North'][ep]
            self.assertEqual((len(panel['models']),len(panel['contrasts'])),(36,7))
            self.assertEqual(panel['coverage']['eligible_people'],18)
            self.assertEqual(panel['coverage']['eligible_visits'],19)
            self.assertEqual([(tuple(r['left']),tuple(r['right'])) for r in panel['contrasts']],
                             list(g7.statistics.PRIMARY_PAIRS))
            if ep == 'binary':
                model = panel['models'][0]
                self.assertAlmostEqual(model['probabilities'][model['participants'].index('person-North-0')][1], .225)
            self.assertIn('withheld',public['cohorts']['North'][ep]['status'])
        harbour = public['cohorts']['Harbour']['cd-vs-non-ibd']
        self.assertIn('calibration', harbour['models'][0]['metrics'])
        self.assertEqual(harbour['role'], 'primary')
        self.assertEqual(len(harbour['models']),42)
        self.assertIn('cmp.relative.joint.tabm', {row['method'] for row in harbour['models']})
        self.assertTrue(any(r['endpoint']=='binary' and r['cohort']=='Harbour' and
                            r['classification']=='structurally-inapplicable' for r in public['unavailable_panels']))
        south = private['cohorts']['South']['binary']
        self.assertIsNone(south['models'][0]['metrics']['roc_auc'])
        self.assertIn('recall/case',south['contrasts'][0]['metrics'])
        self.assertIn('withheld',public['cohorts']['South']['binary']['status'])
        for term in ('visit-', 'group-', 'person-', 'threshold','sha256','decision_source'):
            self.assertNotIn(term,json.dumps(public))
        args = ['report_g7', '--root', str(self.root)]
        printed = io.StringIO()
        with patch('sys.argv', args), redirect_stdout(printed):
            g7.main()
        self.assertEqual(str(path), printed.getvalue().strip())

    def assert_before_outcomes(self):
        original = Path.open
        def guard(path, *args, **kwargs):
            if path == self.work / 'outcomes/outcomes.jsonl':
                raise AssertionError('Premature outcome file access')
            return original(path, *args, **kwargs)
        with patch.object(Path,'open',guard), patch.object(g7,'_parse_outcomes',side_effect=AssertionError('premature outcome read')) as read:
            with self.assertRaises((ValueError,PermissionError,FileNotFoundError,RuntimeError,KeyError)):
                g7.generate(self.root)
            read.assert_not_called()
        self.assertFalse((self.work / 'reports').exists())
        self.assertFalse((self.work / 'exposure').exists())

    def test_preseal_and_actor_gates(self):
        with patch.dict(os.environ,CODEX_THREAD_ID='c66cd228-a3a0-4749-ab6c-e1d5ae644af4'):
            self.assert_before_outcomes()
        st = get(self.status)
        save(self.status,dict(st,status='PAUSED'))
        self.assert_before_outcomes()
        save(self.status,dict(st,stage='G6'))
        self.assert_before_outcomes()
        save(self.status,st)
        stop = self.root / 'STOP'
        stop.touch()
        self.assert_before_outcomes()
        stop.unlink()
        seal = self.p / 'SEALED.json'
        raw = seal.read_bytes()
        seal.unlink()
        self.assert_before_outcomes()
        self.assertFalse(seal.exists())
        seal.write_bytes(raw + b' ')
        self.assert_before_outcomes()

    def test_obsolete_disclosure_policy_before_outcomes(self):
        path = self.s / 'report-policy.json'
        cfg = get(path)
        cfg['aggregate_disclosure'] = 'independent-panel-floors'
        save(path, cfg)
        activate(self.root)
        self.assert_before_outcomes()

    def test_predictions_and_review_before_outcomes(self):
        snap = get(self.p / 'SEALED.json')['predictions'][-1]
        last = self.p / snap['artifact']
        original = last.read_bytes()
        last.unlink()
        self.assert_before_outcomes()
        last.write_bytes(original + b' ')
        self.assert_before_outcomes()
        last.write_bytes(original)
        with (self.root / 'reviews/invented.json').open('ab') as f:
            f.write(b' ')
        self.assert_before_outcomes()

    def test_replaced_valid_receipt_cannot_reseal(self):
        snap = get(self.p / 'SEALED.json')['predictions'][-1]
        path = self.p / snap['artifact']
        rows = [json.loads(line) for line in path.read_bytes().splitlines()]
        rows[0]['probabilities'].reverse()
        path.write_bytes(b''.join(canonical_json(r)+b'\n' for r in rows))
        receipt = self.p / (snap['prediction_id']+'.json')
        save(receipt,dict(get(receipt),sha256=g7.release.sha(path)))
        self.assert_before_outcomes()

    def test_sealed_shared_alias_disagreement(self):
        sub = get(self.s / 'submission.json')
        item = next(i for i in sub['planned_predictions'] if i['cohort']=='North' and i['arm']=='RELATIVE'
                    and i['method']=='cmp.shared.species.logistic' and i['endpoint']=='binary')
        rec_path = self.p / (item['id']+'.json')
        rec = get(rec_path)
        path = self.p / rec['artifact']
        rows = [json.loads(line) for line in path.read_bytes().splitlines()]
        rows[0]['probabilities'].reverse()
        path.write_bytes(b''.join(canonical_json(r)+b'\n' for r in rows))
        save(rec_path,dict(rec,sha256=g7.release.sha(path)))
        (self.p / 'SEALED.json').unlink()
        seal = g7.release.verify_and_seal(self.s,self.root/'reviews/invented.json','test-reviewer',ROLE,self.p)
        status = get(self.status)
        status['g7_report_release']['seal_sha256'] = g7.release.sha(seal)
        save(self.status,status)
        self.assert_before_outcomes()

    def test_outcome_class_failure_before_metrics(self):
        path = self.work / 'outcomes/outcomes.jsonl'
        rows = [json.loads(line) for line in path.read_bytes().splitlines()]
        rows[-1]['label'] = 'unregistered class'
        path.write_bytes(b''.join(canonical_json(r)+b'\n' for r in rows))
        policy = self.s / 'report-policy.json'
        save(policy,dict(get(policy),outcomes_sha256=g7.release.sha(path)))
        activate(self.root)
        with patch.object(g7,'_score',side_effect=AssertionError('premature scoring')):
            with self.assertRaisesRegex(ValueError,'class order'):
                g7.generate(self.root)
        self.assertFalse((self.work / 'reports').exists())

    def test_wrong_frozen_rules(self):
        path = self.s / 'binary-export.json'
        export = get(path)
        row = next(r for r in export['procedures'] if r['method'] == 'H0')
        row['rule']['threshold'] = .123456789
        save(path,export)
        activate(self.root)
        self.assert_before_outcomes()

    def test_omitted_amended_method_before_outcomes(self):
        path = self.s / 'submission.json'
        sub = get(path)
        sub['planned_predictions'] = [i for i in sub['planned_predictions']
                                     if i['method'] != 'cmp.shared.species.xgboost']
        save(path,sub)
        save(self.s / 'comparisons.json',dict(predictions=sub['planned_predictions']))
        activate(self.root)
        self.assert_before_outcomes()

    def test_wrong_group_with_matching_commitments(self):
        path = self.work / 'outcomes/outcomes.jsonl'
        rows = [json.loads(l) for l in path.read_bytes().splitlines()]
        rows[0]['group'] = 'group-South-0'
        data = b''.join(canonical_json(r)+b'\n' for r in rows)
        path.write_bytes(data)
        cfg = get(self.s / 'report-policy.json')
        cfg['identities_sha256'] = sha(canonical_json(sorted([r[k] for k in
            ('cohort','endpoint','sample','person','group')] for r in rows)))
        cfg['outcomes_sha256'] = sha(data)
        save(self.s / 'report-policy.json',cfg)
        activate(self.root)
        save(self.work / 'outcomes/identities.json', sorted([r[k] for k in g7.AXES] for r in rows))
        self.assert_before_outcomes()

    def test_unmet_mandatory_cell(self):
        path = self.s / 'report-policy.json'
        cfg = get(path)
        cfg['panel_roles']['South']['subtypes'] = 'missing-mandatory'
        save(path,cfg)
        activate(self.root)
        report = get(g7.generate(self.root) / 'aggregate/report.json')
        self.assertEqual(report['status'],'partial')
        self.assertIn(dict(cohort='South',endpoint='subtypes',classification='missing-mandatory'), report['unavailable_panels'])

    def test_roster_before_metrics(self):
        path = self.work / 'outcomes/outcomes.jsonl'
        rows = [json.loads(l) for l in path.read_bytes().splitlines()]
        rows[0]['person'] = 'invented-different-person'
        data = b''.join(canonical_json(r)+b'\n' for r in rows)
        path.write_bytes(data)
        cfg = get(self.s / 'report-policy.json')
        cfg['outcomes_sha256'] = sha(data)
        save(self.s / 'report-policy.json',cfg)
        activate(self.root)
        with patch.object(g7,'_score',side_effect=AssertionError('premature scoring')) as score:
            with self.assertRaisesRegex(ValueError,'commitment'):
                g7.generate(self.root)
            score.assert_not_called()
        self.assertFalse((self.work / 'reports').exists())

    def test_recheck_global_seal_at_publication(self):
        original = g7._score
        def change(*args):
            reports = original(*args)
            snap = get(self.p / 'SEALED.json')['predictions'][-1]
            (self.p / snap['artifact']).unlink()
            return reports
        with patch.object(g7,'_score',side_effect=change), self.assertRaises(FileNotFoundError):
            g7.generate(self.root)
        dest = self.work / 'reports' / get(self.status)['g7_report_release']['seal_sha256']
        self.assertFalse(dest.exists())
        self.assertEqual(len(list((self.work / 'exposure').glob('*.json'))),1)

    def test_recheck_at_publication(self):
        original = g7._score
        def change(*args):
            reports = original(*args)
            (self.root / 'STOP').touch()
            return reports
        with patch.object(g7,'_score',side_effect=change), self.assertRaisesRegex(RuntimeError,'STOP'):
            g7.generate(self.root)
        dest = self.work / 'reports' / get(self.status)['g7_report_release']['seal_sha256']
        self.assertFalse(dest.exists())


class AggregateDisclosure(unittest.TestCase):
    @staticmethod
    def panel(cohort, labels):
        ids = [f'{cohort}-{i}' for i in range(len(labels))]
        return pd.DataFrame(dict(cohort=cohort, person=ids, group=ids, y=labels), index=ids)

    @staticmethod
    def roles(rosters):
        roles = {}
        for cohort, endpoint in rosters:
            roles.setdefault(cohort, {})[endpoint] = 'primary' if endpoint == 'a' else 'secondary'
        return roles

    def reports(self, rosters):
        axes = ['negative', 'positive']
        panels = {}
        for key, frame in rosters.items():
            positive = 0.2 + 0.5 * frame.y + 0.01 * (np.arange(len(frame)) % 7)
            probs = pd.DataFrame(dict(negative=1-positive, positive=positive), index=frame.index)
            tables = {(a,m):probs for a in g7.statistics.ARMS for m in g7.statistics.HEADS}
            panels[key] = (list(frame.index), tables, dict.fromkeys(tables, .5), {k:{} for k in tables})
        cfg = dict(panel_roles=self.roles(rosters), endpoints={e:axes for c,e in rosters},
                   unavailable_panels=[], unmet_methods=[], identities_sha256='invented', outcomes_sha256='invented',
                   bootstrap=dict(draws=8,seed=123))
        args = (cfg, panels, rosters, dict(submission_sha256='s',review_sha256='r',evaluator=ROLE),
                dict(seal_sha256='x'))
        with patch.dict(os.environ, CODEX_THREAD_ID=THREAD):
            private, public = g7._score(*args)
            with patch.object(g7, '_public_panels', return_value=set(rosters)):
                baseline, _ = g7._score(*args)
        self.assertEqual(canonical_json(private), canonical_json(baseline))
        return private, public

    def test_nested_one_group_residual_and_complementary_suppression(self):
        north = self.panel('North', [0]*6 + [1]*6)
        south = self.panel('South', [0]*6 + [1]*11)
        rosters = {('North','a'):north, ('North','b'):north.iloc[:-1],
                   ('South','a'):south, ('South','b'):south.iloc[:-5]}
        private, public = self.reports(rosters)
        hidden = public['cohorts']['North']['b']
        self.assertEqual(set(hidden), {'status','class_order','role'})
        self.assertIn('withheld', hidden['status'])
        self.assertEqual(private['cohorts']['North']['b']['coverage']['class_groups'],
                         {'negative':6,'positive':5})
        self.assertEqual(len(private['cohorts']['North']['b']['contrasts']), 7)
        for cohort, ep in set(rosters) - {('North','b')}:
            self.assertEqual(len(public['cohorts'][cohort][ep]['contrasts']), 7)
        a,b = public['equal_cohort_auc']
        self.assertEqual(a['eligible_cohorts'], ['North','South'])
        self.assertEqual(b, dict(endpoint='b',role='secondary',status='withheld'))
        self.assertEqual(private['equal_cohort_auc'][1]['eligible_cohorts'], ['North','South'])

    def triples(self, last):
        rows = [bits for bits in np.ndindex(2,2,2) for _ in range(last if bits==(1,1,1) else 5)]
        return {('North',ep): self.panel('North', [r[i] for r in rows]) for i,ep in enumerate('abc')}

    def test_three_class_masks_pairwise_safe_joint_residual_one(self):
        rosters = self.triples(1)
        for excluded in rosters:
            pair = {k:v for k,v in rosters.items() if k != excluded}
            self.assertEqual(g7._public_panels(self.roles(pair),pair),set(pair))
        reversed_rosters = dict(reversed(rosters.items()))
        self.assertEqual(g7._public_panels(self.roles(rosters), reversed_rosters), {('North','a'),('North','b')})
        private, public = self.reports(rosters)
        self.assertEqual(set(public['cohorts']['North']['c']), {'status','class_order','role'})
        self.assertEqual(public['equal_cohort_auc'][-1],dict(endpoint='c',role='secondary',status='withheld'))
        self.assertEqual(len(private['cohorts']['North']['c']['models']), 6)
        self.assertEqual(len(private['cohorts']['North']['c']['contrasts']), 7)

    def test_joint_floor_exactly_five_groups(self):
        rosters = self.triples(5)
        self.assertEqual(g7._public_panels(self.roles(rosters),rosters), set(rosters))

    def test_repeated_visits_and_related_people_cannot_supply_group_support(self):
        original = self.panel('North', [0]*30 + [1]*30)
        for repeated in (True, False):
            with self.subTest(repeated=repeated):
                frame = original.copy()
                frame['group'] = [f'family-{i//5}' for i in range(len(frame))]
                if repeated:
                    frame['person'] = frame['group']
                rosters = {('North','a'):frame, ('North','b'):frame.iloc[:-5]}
                self.assertEqual(g7._public_panels(self.roles(rosters),rosters), {('North','a')})

    def test_within_panel_visit_person_group_differences_and_eligibility(self):
        south = self.panel('South', [0]*6 + [1]*6)
        for repeated in (True, False):
            public_by_support = []
            for full_support in (True, False):
                with self.subTest(repeated=repeated, full_support=full_support):
                    north = self.panel('North', [0]*6 + [1]*7 if full_support else [0]*13)
                    north.loc['North-12','group'] = 'North-11'
                    if repeated:
                        north.loc['North-12','person'] = 'North-11'
                    private, public = self.reports({('North','a'):north, ('South','a'):south})
                    coverage = private['cohorts']['North']['a']['coverage']
                    self.assertEqual((coverage['eligible_visits'], coverage['eligible_people'], coverage['groups']),
                                     (13, 12 if repeated else 13, 12))
                    expected = ['North','South'] if full_support else ['South']
                    self.assertEqual(private['equal_cohort_auc'][0]['eligible_cohorts'], expected)
                    self.assertEqual(set(public['cohorts']['North']['a']), {'role','class_order','status'})
                    self.assertEqual(public['equal_cohort_auc'], [dict(endpoint='a',role='primary',status='withheld')])
                    public_by_support.append(canonical_json(public))
            self.assertEqual(*public_by_support)
