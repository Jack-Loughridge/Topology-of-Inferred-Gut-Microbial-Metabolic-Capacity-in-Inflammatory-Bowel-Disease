"""Report one released endpoint from completed development folds without fitting models."""
import argparse
import io
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from . import run_fold, statistics
from .artifacts import Store, sha
from .nesting import Procedure
from .partitions import canonical_json


# Required per arm by the analysis table and Fold.run. A signed config may
# extend this list per arm using reporting_methods (the complete method list).
COMMON = (*statistics.HEADS, 'K0', 'species', 'activity', 'graph-count', 'fixed-alpha', 'historical')
PUBLIC_MIN_GROUPS = 5


class IncompleteCoverage(ValueError):
    """Required prediction, fold or decision evidence is absent."""


def released_inputs(root, rec):
    base = root / 'manifests/development-fold-inputs'

    def read(key):
        spec = rec['inputs'][key]
        data = run_fold.inside(base, spec['path']).read_bytes()
        if sha(data) != spec['sha256']:
            raise ValueError('Released input changed: ' + key)
        return data

    config = json.loads(read('config'))
    splits = json.loads(read('splits'))
    meta = pd.read_csv(io.BytesIO(read('meta')), sep='\t', dtype=str,
                       keep_default_na=False).set_index('sample')
    classes = config['classes']
    if (config['endpoint'] != rec['endpoint'] or not isinstance(classes, list) or len(classes) not in (2, 3)
            or not meta.role.eq('development').all() or not len(meta)
            or '' in meta.index or (meta[['person', 'group', 'family', 'cohort']] == '').any().any()
            or not meta.y.isin([str(i) for i in range(len(classes))]).all()):
        raise ValueError('Invalid development endpoint or metadata')
    meta['y'] = meta.y.astype(int)
    statistics._check_roster(meta, classes)
    if (meta.groupby('cohort').family.nunique() != 1).any():
        raise ValueError('Cohort spans recruitment families')
    for spec in splits.values():
        for side in ('train', 'validate'):
            if not isinstance(spec[side], list) or len(spec[side]) != len(set(spec[side])):
                raise ValueError('Duplicate or invalid split roster')
    return config, splits, meta


def methods_by_arm(config):
    required = {a: [*COMMON, 'direct-' + a.lower()] for a in statistics.ARMS}
    registered = config.get('reporting_methods', required)
    if set(registered) != set(required):
        raise ValueError('Invalid registered arm list')
    for arm, names in registered.items():
        if (not isinstance(names, list) or len(names) != len(set(names))
                or not set(required[arm]).issubset(names)
                or not all(isinstance(n, str) and n for n in names)):
            raise ValueError('Invalid registered procedures')
    return registered


def fit_table(store, objects, fit_id, arm, method, split, train, valid, config):
    if fit_id not in objects:
        raise IncompleteCoverage('Fit absent from fold object manifest')
    fit = json.loads(store.read(fit_id))
    expected = dict(arm=arm, name=method, split=split, classes=config['classes'],
                    endpoint=config['endpoint'], train_groups=sorted(set(train.group)),
                    samples=valid.index.tolist(), participants=sorted(set(valid.person)))
    if any(fit.get(k) != v for k, v in expected.items()):
        raise ValueError('Fit identity, class order, training groups or held-out roster mismatch')
    if any(fit[k] not in objects for k in ('probabilities', 'participant_means')):
        raise IncompleteCoverage('Prediction arrays absent from fold object manifest')
    p = store.load_array(fit['probabilities'])
    run_fold.validate_probabilities(p, len(valid), config['classes'])
    probs = pd.DataFrame(p, index=valid.index, columns=config['classes'])
    means = probs.groupby(valid.person, sort=True).mean()
    if not np.array_equal(store.load_array(fit['participant_means']), means.to_numpy()):
        raise ValueError('Saved participant probabilities differ from the eligible visit means')
    return probs


def decision_rules(store, objects, rule_id, arm, methods, fold, train, children, config):
    if rule_id not in objects:
        raise IncompleteCoverage('Missing fitted decision rules')
    rules = json.loads(store.read(rule_id))
    classes = config['classes']
    if rules.get('class_order') != classes:
        raise ValueError('Decision class order mismatch')
    if len(classes) > 2:
        if rules != dict(rule='argmax', class_order=classes):
            raise ValueError('Invalid multiclass decision rule')
        return dict.fromkeys(methods)
    if (rules.get('rule') != 'max-participant-balanced-accuracy' or rules.get('parent') != fold
            or set(rules.get('thresholds', {})) != set(methods)
            or set(rules.get('heldout_fits', {})) != set(methods)):
        raise ValueError('Wrong binary rule, parent or method roster')
    for method in methods:
        parts = []
        seen = set()
        for fit_id in rules['heldout_fits'][method]:
            if fit_id not in objects:
                raise IncompleteCoverage('Missing inner held-out fit')
            child = json.loads(store.read(fit_id))['split']
            if child not in children or child in seen:
                raise ValueError('Duplicate or foreign OOF split')
            a, b = children[child]
            probs = fit_table(store, objects, fit_id, arm, method, child, a, b, config)
            parts.append((set(a.group), probs))
            seen.add(child)
        if seen != set(children):
            raise IncompleteCoverage('Incomplete inner OOF fold coverage')
        stored = rules['thresholds'][method]
        if type(stored) not in (float, int) or stored != statistics.select_inner_oof_threshold(train, classes, parts):
            raise ValueError('Stored threshold disagrees with its inner OOF evidence')
    return rules['thresholds']


def read_fold(directory, fold, rec, release_id, proc, config, methods):
    if not (directory / 'receipt.json').is_file():
        raise IncompleteCoverage('Missing completed fold receipt for ' + fold)
    store = Store(directory)
    marker = (directory / 'receipt.json').read_bytes()
    record_id = json.loads(marker)['record']
    receipt = json.loads(store.read(record_id))
    if ((receipt['release'], receipt['fold'], receipt['inputs']) != (release_id, fold, rec['inputs'])
            or set(receipt['arms']) != set(statistics.ARMS)):
        raise ValueError('Fold receipt release, inputs, arms or fold ID mismatch')
    objects = set(receipt['objects'])
    train, valid = proc.frames(fold)
    children = ({k: (a, b) for k, a, b in proc.children(fold, train)}
                if len(config['classes']) == 2 else {})
    probs, thresholds = {}, {}
    linkage = dict(marker_sha256=sha(marker), record=record_id,
                   roster_sha256=sha(canonical_json(valid.reset_index()[
                       ['sample', 'person', 'group', 'cohort', 'family', 'y']].values.tolist())), arms={})
    for arm, names in methods.items():
        saved = receipt['arms'][arm]
        outputs = saved['outputs']
        if set(outputs) != set(names):
            raise IncompleteCoverage('Fold output methods differ from the approved roster')
        if not set(outputs.values()).issubset(saved['fits']):
            raise IncompleteCoverage('Output fit missing from arm ledger')
        rule_id = saved.get('decision_rules')
        rules = decision_rules(store, objects, rule_id, arm, names, fold, train, children, config)
        linkage['arms'][arm] = dict(outputs=outputs, decision_rules=rule_id)
        for method in names:
            key = (arm, method)
            probs[key] = fit_table(store, objects, outputs[method], arm, method, fold, train, valid, config)
            thresholds[key] = rules[method]
    return valid, probs, thresholds, linkage


def limitations(coverage, contrasts):
    notes = []
    if not all(coverage['class_people'].values()):
        notes.append('Incomplete class support. Whole-task discrimination and macro decision metrics are undefined.')
    if len(coverage['visits_per_person']) > 1:
        notes.append('Participants contribute different numbers of visits to the available-visits assessment.')
    if any(v['interval'] is None for entry in contrasts.values() for v in entry['metrics'].values()):
        notes.append('Some descriptive intervals are undefined under the independent-group and valid-draw rules.')
    return notes


def public_cohort(cohort):
    """Project aggregate fields explicitly. Singleton reliability bins can expose scores."""
    cov = cohort['coverage']
    group_support = cov['class_groups']
    if cov['groups'] < PUBLIC_MIN_GROUPS or any(0 < n < PUBLIC_MIN_GROUPS for n in group_support.values()):
        return dict(status='withheld for small independent-group counts')
    result = {k: cohort[k] for k in ('status', 'auc_status', 'limitations', 'contrasts')}
    result['coverage'] = {k: cov[k] for k in ('eligible_people', 'eligible_visits', 'groups', 'class_people')}
    result['models'] = []
    for row in cohort['models']:
        s = row['metrics']
        metrics = {k: s[k] for k in ('roc_auc', 'pr_auc', 'balanced_accuracy', 'macro_f1', 'brier')}
        # Keep per-bin predicted means private. Even a bin with many people
        # can represent only one family. Publish rates at cohort level only.
        metrics['calibration'] = {
            c: dict(mean_predicted=sum(b['people'] * b['predicted'] for b in v['calibration'] if b['people'])
                                   / cov['eligible_people'],
                    observed=v['people'] / cov['eligible_people'])
            for c, v in s['by_class'].items()}
        metrics['recall'] = {c: v['recall'] for c, v in s['by_class'].items()}
        result['models'].append(dict(arm=row['arm'], method=row['method'], metrics=metrics))
    return result


def generate(root, release_id):
    root = Path(root).resolve(strict=True)
    status = run_fold.active(root)
    entry = status['development_fold_release']
    if entry['sha256'] != release_id or not re.fullmatch('[a-f0-9]{64}', release_id):
        raise ValueError('Requested development release is not current')
    header = run_fold.inside(root / 'reviews', entry['file']).read_bytes()
    if sha(header) != release_id:
        raise ValueError('Changed release header')
    folds = json.loads(header)['folds']
    if (not isinstance(folds, list) or not folds or len(set(folds)) != len(folds)
            or any(f == run_fold.FINAL or not re.fullmatch(r'[A-Za-z0-9_-][A-Za-z0-9_.-]*', f) for f in folds)):
        raise ValueError('Invalid or duplicate released outer fold')
    rec, verified = run_fold.release(root, folds[0])
    if verified != release_id:
        raise ValueError('Release changed during setup')
    config, splits, roster = released_inputs(root, rec)
    if set(folds) != {k for k in splits if '/' not in k and k != run_fold.FINAL}:
        raise IncompleteCoverage('Released folds differ from the full outer split registry')
    proc = Procedure(roster, splits, None, {}, [], None)
    methods = methods_by_arm(config)
    people, seen = set(roster.person), set()
    for fold in folds:
        train, valid = proc.frames(fold)
        a, b = set(train.person), set(valid.person)
        if (not a or not b or a | b != people or seen & b or set(train.family) & set(valid.family)
                or set(train.y) != set(range(len(config['classes'])))):
            raise ValueError('Duplicate held-out people, broken family boundary or invalid training partition')
        seen |= b
    if seen != people:
        raise IncompleteCoverage('Incomplete held-out participant coverage')

    classes = config['classes']
    base = root / 'jobs/development-folds' / release_id
    private = dict(endpoint=config['endpoint'], class_order=classes, release_sha256=release_id,
                   inputs=rec['inputs'], folds={}, cohorts={},
                   reporting_sources={p.name: sha(p.read_bytes()) for p in (
                       Path(__file__), Path(statistics.__file__))})
    reports = {}
    for fold in folds:
        run_fold.active(root)
        valid, probs, thresholds, linkage = read_fold(
            run_fold.inside(base, fold), fold, rec, release_id, proc, config, methods)
        private['folds'][fold] = linkage
        for cohort, panel in valid.groupby('cohort'):
            if cohort in reports:
                raise ValueError('Cohort spans outer folds')
            run_fold.active(root)
            predictions = {key: frame.loc[panel.index] for key, frame in probs.items()}
            report = statistics.cohort_report(panel, predictions, classes, thresholds)
            contrasts = statistics.primary_contrasts(panel, predictions, classes, thresholds)
            coverage = next(iter(report.values()))['coverage']
            if any(r['coverage'] != coverage or r['status'] != 'complete' for r in report.values()):
                raise IncompleteCoverage('Incomplete matched model coverage')
            reports[cohort] = report
            notes = limitations(coverage, contrasts)
            if 'control_type' not in panel and 'non-IBD' in classes:
                notes.append('Comparator subtype ascertainment is unavailable in this reporting export.')
            private['cohorts'][cohort] = dict(status='complete', coverage=coverage,
                auc_status='defined' if all(coverage['class_people'].values()) else 'undefined under class support',
                limitations=notes,
                models=[dict(arm=a, method=m, metrics=r['metrics']) for (a, m), r in report.items()],
                contrasts=[dict(left=left, right=right, **v) for (left, right), v in contrasts.items()])
    expected = sorted(roster.cohort.unique())
    private['equal_cohort_auc'] = [
        dict(arm=a, method=m, roc_auc=statistics.equal_cohort_auc(reports, (a, m), expected_cohorts=expected))
        for a, names in methods.items() for m in names]
    private['eligible_auc_cohorts'] = [c for c in expected if all(private['cohorts'][c]['coverage']['class_people'].values())]
    public = dict(endpoint=config['endpoint'], class_order=classes,
                  interpretation='Adaptive development. Intervals are marginal descriptive group-bootstrap intervals.',
                  cohorts={c: public_cohort(private['cohorts'][c]) for c in expected})
    # A summary including a suppressed cohort could reveal its result by subtraction.
    if any(public['cohorts'][c]['status'] != 'complete' for c in private['eligible_auc_cohorts']):
        public['equal_cohort_auc_status'] = 'withheld for small independent-group counts'
    else:
        public['equal_cohort_auc'] = private['equal_cohort_auc']
        public['eligible_auc_cohorts'] = private['eligible_auc_cohorts']

    if run_fold.release(root, folds[0])[1] != release_id:
        raise ValueError('Release changed during reporting')
    for fold, linkage in private['folds'].items():
        if sha((base / fold / 'receipt.json').read_bytes()) != linkage['marker_sha256']:
            raise ValueError('Fold receipt changed during reporting')
    output = root / 'jobs/development-reports' / release_id
    for scope, result in [('private', private), ('aggregate', public)]:
        run_fold.active(root)
        store = Store(run_fold.inside(root / 'jobs', output / scope))
        store.publish('report.json', canonical_json(result))
    return output


def main(argv=None, *, root=run_fold.ROOT):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release-sha256', required=True, help='Current signed development-only-fold release digest')
    args = parser.parse_args(argv)
    try:
        print(generate(root, args.release_sha256))
    except (IncompleteCoverage, FileNotFoundError) as exc:
        parser.exit(2, f'Incomplete development evidence: {exc}\n')
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        parser.exit(2, f'Invalid development evidence or run state: {exc}\n')


if __name__ == '__main__':
    main()
