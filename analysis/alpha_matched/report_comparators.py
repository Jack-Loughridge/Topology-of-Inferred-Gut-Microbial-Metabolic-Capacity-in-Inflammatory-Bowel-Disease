"""Matched, read-only development scoring for the separate comparator release."""
import argparse
import json
from pathlib import Path
import re

import numpy as np

from . import comparator_run as driver, report_development as development, run_fold, statistics
from .artifacts import Store, sha
from .nesting import Procedure, score
from .partitions import canonical_json, digest, fusion_input

METRICS = ('roc_auc', 'pr_auc', 'balanced_accuracy', 'macro_f1', 'brier')


def registered_folds(root, release_id, markers):
    entry = run_fold.active(root)['comparator_release']
    if entry['sha256'] != release_id or not re.fullmatch('[0-9a-f]{64}', release_id):
        raise ValueError('Requested comparator release is not current')
    folds = driver.document(root / 'reviews', entry)['folds']
    if (not isinstance(folds, list) or not folds or len(set(folds)) != len(folds)
            or any(f == run_fold.FINAL or not re.fullmatch(r'[a-zA-Z0-9_-][a-zA-Z0-9_.-]*', f) for f in folds)):
        raise ValueError('Invalid development fold registry')
    if (not isinstance(markers, dict) or set(markers) != set(folds) or
            any(not re.fullmatch('[0-9a-f]{64}', value) for value in markers.values())):
        raise ValueError('Expected one pinned receipt digest per released fold')
    return folds


def released_inputs(root, parent):
    # Verify the signed graph receipt as well as the three tabular/configuration inputs.
    # Reporting only needs the graph receipt's digest.
    base = root / 'manifests/development-fold-inputs'
    inputs = parent['inputs']
    for key in set(inputs) - {'config', 'splits', 'meta'} | {'bundle'}:
        spec = inputs[key]
        if run_fold.bundle.sha(run_fold.inside(base, spec['path'])) != spec['sha256']:
            raise ValueError('Released input changed: ' + key)
    return development.released_inputs(root, parent)


class FitEvidence:
    """Read producer unit receipts and independently score each registered selection."""
    def __init__(self, store, objects, proc, config, budget, rid, fold, primary, bundle):
        self.store, self.objects, self.proc = store, objects, proc
        self.config, self.budget, self.primary = config, budget, primary
        runtime = json.loads((store.root / 'runtime.json').read_bytes())
        if runtime[1] != budget['packages']:
            raise ValueError('Unexpected fitted package registration')
        self.context = sha(canonical_json([rid, fold, runtime]))
        self.bundle = bundle
        self.checked = {}

    def record(self, ref):
        if ref not in self.objects:
            raise ValueError('Dependency missing from completed fold inventory')
        return driver.read(self.store, ref)

    def unit(self, identity, expected):
        key = sha(canonical_json([self.context, identity]))
        marker = json.loads((self.store.root / ('unit-' + key + '.json')).read_bytes())
        if self.record(marker['record']) != dict(key=key, result=expected):
            raise ValueError('Fit or representation differs from its completed unit')

    def vectors(self, blocks, entry, node, train, valid):
        arm, name = entry['arm'], entry['representation']
        self.unit(['representation', arm, name, node], blocks)
        deps = driver.references(self.store, blocks)
        if not deps <= self.objects:
            raise ValueError('Missing transitive representation dependency')
        names = ['H0', 'Ricci'] if name == 'joint' else [name]
        if not isinstance(blocks, list) or len(names) != len(blocks):
            raise ValueError('Wrong representation block layout')
        if name not in driver.DIRECT:
            h0, ricci = self.primary.selected[arm, node]
            if blocks != {'H0': [h0], 'Ricci': [ricci], 'joint': [h0, ricci]}[name] or not deps <= self.primary.objects:
                raise ValueError('Frozen blocks differ from signed primary selection')
        partition = digest(canonical_json(sorted(zip(train.index, train.person))))
        vectors = []
        for block, kind in zip(blocks, names):
            model = self.record(block['model'])
            if (model['arm'], model['name'], model['partition'], model['training_people_sha256']) != (
                    arm, kind, partition, sha(canonical_json(sorted(set(train.person))))):
                raise ValueError('Wrong generator partition or identity')
            if ((model['catalogue'], model['mapping']) !=
                    (self.bundle['catalogue_digest'], self.bundle['mapping_digest'])
                    or not re.fullmatch('[0-9a-f]{64}', model['axis'])):
                raise ValueError('Generator source catalogue or mapping mismatch')
            features = self.record(model['features'])
            state = model['state']
            width = len(features)
            mean, scale = [self.store.load_array(state[k]) for k in ('mean', 'scale')]
            if mean.shape != (width,) or scale.shape != mean.shape or not (scale > 0).all():
                raise ValueError('Invalid fitted generator scaling')
            ref = model['reference']
            if (ref is not None) != (arm == 'RELATIVE'):
                raise ValueError('Missing or unexpected fold-local reference')
            if ref is not None:
                r = self.record(ref)
                refshape = self.store.load_array(r['log_reference']).shape
                if (r['axis'] != model['axis'] or not r['partition'] or len(refshape) != 1 or
                        self.store.load_array(r['unseen']).shape != refshape):
                    raise ValueError('Incomplete relative reference')
            if kind in ('H0', 'Ricci'):
                if (state['block'] != kind or state['setting'] not in self.config['candidates'][kind] or
                        state['tolerances'] != self.config['tolerances'] or
                        not re.fullmatch('[0-9a-f]{64}', state['generator'])):
                    raise ValueError('Incomplete primary generator configuration')
                if kind == 'H0':
                    image = state['image']
                    if (self.store.load_array(image['alpha']).shape != (width,)
                            or self.store.load_array(image['bounds']).shape != (width + 1,)
                            or len(image['source_cuts']) != width - 1 or not image['bandwidth'] > 0):
                        raise ValueError('Incomplete persistence image state')
            if kind == 'species':
                taxa = self.record(state['species_taxa'])
                axis = self.store.load_array(state['species_axis'])
                impute = state['species_impute']
                if axis.shape != (len(taxa),) or (impute is not None and self.store.load_array(impute).shape != axis.shape):
                    raise ValueError('Incomplete fitted species axis')
                expected = (taxa if impute is None else
                            [part + ':' + canonical_json(t).decode() for part in ('abundance','unreported') for t in taxa])
                if expected != features:
                    raise ValueError('Species coordinates disagree')
            for side, frame in [('train', train), ('validate', valid)]:
                v = self.store.load_vector(block[side])
                if ((v.block, v.arm, v.partition, v.generator, v.ids, list(v.features)) !=
                        (kind, arm, partition, block['model'], tuple(frame.index), features)):
                    raise ValueError('Representation vector lineage mismatch')
                if side == 'validate':
                    vectors.append(v)
        return vectors[0].values if len(vectors) == 1 else fusion_input(*vectors)

    def predictions(self, ref, method, entry, node, candidate):
        identity = (ref, method, node, canonical_json(candidate))
        if identity in self.checked:
            return self.checked[identity]
        fit = self.record(ref)
        if fit['candidate'] != candidate:
            raise ValueError('Fit candidate differs from registered selection')
        train, valid = self.proc.frames(node)
        table = development.fit_table(self.store, self.objects, ref, entry['arm'], method, node, train, valid, self.config)
        model = self.record(fit['fitted'])
        if model['config'] != candidate or model['classes'] != list(range(len(self.config['classes']))):
            raise ValueError('Fitted head candidate or class mismatch')
        if not set(driver.heads.dependencies(self.store, fit['fitted'])) <= self.objects:
            raise ValueError('Missing fitted head dependency')
        if entry['family'] in ('logistic', 'random_forest'):
            state = self.record(model['state'])
            family = 'linear' if entry['family'] == 'logistic' else 'forest'
            if (state['family'] != family or state['classes'] != model['classes'] or
                    state['n_features'] != model['n_features'] or
                    family == 'forest' and len(state['trees']) != candidate['params']['n_estimators']):
                raise ValueError('Incomplete fitted classifier state')
        blocks = fit['blocks']
        self.unit(['fit', method, node, blocks, candidate], ref)
        x = self.vectors(blocks, entry, node, train, valid)
        replay = driver.heads.predict(self.store, fit['fitted'], x)
        if not np.allclose(replay, table.to_numpy(), atol=1e-14, rtol=1e-12):
            raise ValueError('Stored probabilities differ from fitted head replay')
        self.checked[identity] = table
        return table

    def selected(self, row, entry, fold, refs):
        method, rule = row['method'], row['rule']
        classes = self.config['classes']
        binary = len(classes) == 2
        if (row['decision'] not in self.objects or rule['method'] != method or rule['classes'] != classes
                or rule['parent'] != fold):
            raise ValueError('Decision provenance mismatch')
        inner = self.proc.children(fold, self.proc.frames(fold)[0])
        nodes = [fold] + ([k for k, _, _ in inner] if binary else [])
        choices = self.budget['candidates'][entry['family']]
        trials = {}
        for ref in refs:
            t = self.record(ref)
            key = t['parent'], canonical_json(t['candidate'])
            if (t['method'] != method or t['candidate'] not in choices or t['parent'] not in nodes or key in trials):
                raise ValueError('Duplicate or unregistered comparator trial')
            trials[key] = t
        if len(trials) != len(nodes) * len(choices):
            raise ValueError('Incomplete registered comparator trials')
        winners, children_by_candidate = {}, {}
        for node in nodes:
            splits = self.proc.children(node, self.proc.frames(node)[0])
            scores = []
            for choice in choices:
                trial = trials[node, canonical_json(choice)]
                children = trial['child_fits']
                if len(children) != len(splits):
                    raise ValueError('Incomplete trial child fit coverage')
                parts = []
                for child_ref, (child, _, valid) in zip(children, splits):
                    table = self.predictions(child_ref, method, entry, child, choice)
                    parts.append((valid, table.to_numpy()))
                    children_by_candidate[node, child, canonical_json(choice)] = child_ref
                value = score(parts)
                if value != trial['participant_cohort_auc']:
                    raise ValueError('Unreproducible trial score')
                scores.append(value)
            winners[node] = choices[int(np.argmax(scores))]
        table = self.predictions(row['fit'], method, entry, fold, winners[fold])
        cut = rule['threshold']
        if binary:
            parts, expected = [], []
            for child, a, b in inner:
                choice = winners[child]
                ref = children_by_candidate[fold, child, canonical_json(choice)]
                expected.append(ref)
                parts.append((set(a.group), self.predictions(ref, method, entry, child, choice)))
            if (rule['rule'] != 'max-participant-balanced-accuracy' or rule['heldout_fits'] != expected or
                    type(cut) not in (float, int) or
                    cut != statistics.select_inner_oof_threshold(self.proc.frames(fold)[0], classes, parts)):
                raise ValueError('Threshold differs from selected group-excluded fits')
        elif rule['rule'] != 'argmax' or rule['heldout_fits'] != [] or cut is not None:
            raise ValueError('Unexpected multiclass rule')
        return table, cut


def read_fold(directory, marker, rid, rec, parent, budget, entries, fold, proc, config):
    store, receipt = driver.report_fold(directory, rid, rec['parent_sha256'], fold, marker)
    objects = set(receipt['objects'])
    complete = set(budget['neural_inputs']) == set(driver.DIRECT)
    expected = dict(entries=entries, implementation=rec['source_sha256'], parent=rec['parent_sha256'],
                    release=rid, packages=budget['packages'], neural_review=rec.get('neural_review'),
                    mandatory_complete=complete, budget=rec['budget'])
    if (receipt['registration'] not in objects or driver.read(store, receipt['registration']) != expected
            or receipt['inputs'] != parent['inputs'] or receipt['registry'] != entries
            or receipt['status'] != ('complete' if complete else 'partial-neural-unmet')):
        raise ValueError('Fold inputs, registration or feasibility differ from the signed release')
    root = store.root.parents[3]
    spec = parent['inputs']['bundle']
    bundle = driver.document(root / 'manifests/development-fold-inputs', dict(file=spec['path'], sha256=spec['sha256']))
    primary = (driver.FrozenPrimary(root, fold, parent, rec)
               if budget['frozen_vectors'] else None)
    evidence = FitEvidence(store, objects, proc, config, budget, rid, fold, primary, bundle)
    tables, cuts, links = {}, {}, {}
    for arm in driver.ARMS:
        seen = set()
        for row in driver.report_rows(store, receipt, arm):
            method = row['method']
            entry = entries[method]
            if (method in seen or arm not in entry['applies_to'] or row['arm'] != arm
                    or row['fit'] != receipt['results'][method]['fit']
                    or row['decision'] != receipt['results'][method]['decision']
                    or row['fitted_arm'] != entry['arm'] or row['family'] != entry['family']
                    or row['representation'] != entry['representation']):
                raise ValueError('Invalid comparator alias')
            seen.add(method)
            if method in links:
                if (entry['arm'] != 'SHARED' or row['fit'] != links[method]['fit']
                        or row['decision'] != links[method]['decision']):
                    raise ValueError('Shared species aliases disagree')
                continue
            try:
                tables[method], cuts[method] = evidence.selected(row, entry, fold, receipt['results'][method]['trials'])
            except (KeyError, IndexError, TypeError, FileNotFoundError) as exc:
                raise ValueError('Incomplete fitted comparator evidence for ' + method) from exc
            evidence.checked.clear()
            links[method] = dict(fit=row['fit'], decision=row['decision'], threshold=cuts[method])
        if seen != {k for k, v in entries.items() if arm in v['applies_to']}:
            raise ValueError('Incomplete alias roster')
    return tables, cuts, dict(marker_sha256=marker, registration=receipt['registration'], methods=links)


def generate(root, release_id, markers):
    root = Path(root).resolve(strict=True)
    folds = registered_folds(root, release_id, markers)
    parent, rec, budget, verified = driver.release(root, folds[0])
    if verified != release_id:
        raise ValueError('Comparator release changed')
    config, splits, roster = released_inputs(root, parent)
    if (set(folds) != set(parent['folds']) or len(parent['folds']) != len(folds)
            or set(folds) != {k for k in splits if '/' not in k and k != run_fold.FINAL}):
        raise ValueError('Incomplete outer fold registry')
    entries = driver.registry(budget)
    neural = budget['neural_family']
    missing = [dict(representation=name, family=neural or 'neural') for name in driver.DIRECT
               if name not in budget['neural_inputs']]
    required = {(rep, fam) for rep in driver.DIRECT for fam in driver.BASE}
    required.update((rep, neural) for rep in budget['neural_inputs'])
    actual = [(v['representation'], v['family']) for v in entries.values() if v['representation'] in driver.DIRECT]
    if len(actual) != len(required) or set(actual) != required or len(actual) + len(missing) != 12:
        raise ValueError('Invalid mandatory family and representation registry')
    manifest = dict(status='partial-neural-unmet' if missing else 'complete',
                    mandatory_expected=12, mandatory_registered=len(actual), unmet=missing, methods=entries)
    classes = config['classes']
    private = dict(**manifest, endpoint=config['endpoint'], class_order=classes, release_sha256=release_id,
                   parent_sha256=rec['parent_sha256'], inputs=parent['inputs'], budget=rec['budget'],
                   unmet_reason=budget.get('unmet_reason'), folds={}, cohorts={},
                   reporting_sources={Path(m).name: sha(Path(m).read_bytes()) for m in
                                      (__file__, development.__file__, statistics.__file__)})
    public = dict(**manifest, endpoint=config['endpoint'], class_order=classes, cohorts={},
                  interpretation='Held-out development cohorts. Each eligible cohort has equal weight in the summaries.')
    proc = Procedure(roster, splits, None, {}, [], None)
    seen = set()
    reports = {}
    base = run_fold.inside(root / 'jobs', Path('comparator-folds') / release_id)
    for fold in folds:
        run_fold.active(root)
        train, valid = proc.frames(fold)
        if (valid.empty or set(train.y) != set(range(len(classes))) or set(valid.person) & seen
                or set(train.person) | set(valid.person) != set(roster.person)
                or set(train.family) & set(valid.family)):
            raise ValueError('Invalid or overlapping held-out development partition')
        seen.update(valid.person)
        tables, cuts, link = read_fold(run_fold.inside(base, fold), markers[fold], release_id,
                                       rec, parent, budget, entries, fold, proc, config)
        link['roster_sha256'] = sha(canonical_json(valid.reset_index()[
            ['sample', 'person', 'group', 'cohort', 'family', 'y']].values.tolist()))
        private['folds'][fold] = link
        for cohort, panel in valid.groupby('cohort'):
            if cohort in reports:
                raise ValueError('Cohort spans outer folds')
            rows = statistics.cohort_report(panel, {m: t.loc[panel.index] for m, t in tables.items()}, classes, cuts)
            coverage = next(iter(rows.values()))['coverage']
            if any(r['status'] != 'complete' or r['coverage'] != coverage for r in rows.values()):
                raise ValueError('Incomplete matched comparator coverage')
            reports[cohort] = rows
            detail = dict(status='complete', coverage=coverage, contrasts=[],
                          auc_status='defined' if all(coverage['class_people'].values()) else 'undefined under class support',
                          limitations=development.limitations(coverage, {}),
                          models=[dict(arm=entries[m]['arm'], method=m, metrics=r['metrics']) for m, r in rows.items()])
            private['cohorts'][cohort] = detail
            public['cohorts'][cohort] = development.public_cohort(detail)
    if seen != set(roster.person):
        raise ValueError('Incomplete held-out development coverage')
    cohorts = sorted(roster.cohort.unique())
    private['equal_cohort'] = {}
    public['equal_cohort'] = {}
    for method in entries:
        summary, safe = {}, {}
        for metric in METRICS:
            eligible = [c for c in cohorts if reports[c][method]['metrics'][metric] is not None]
            value = None
            if metric == 'roc_auc':
                value = statistics.equal_cohort_auc(reports, method, expected_cohorts=cohorts)
            elif eligible:
                value = float(np.mean([reports[c][method]['metrics'][metric] for c in eligible]))
            summary[metric] = dict(mean=value, eligible_cohorts=eligible)
            safe[metric] = (summary[metric] if all(public['cohorts'][c]['status'] == 'complete' for c in eligible)
                            else dict(status='withheld for small independent-group counts'))
        private['equal_cohort'][method], public['equal_cohort'][method] = summary, safe
    if driver.release(root, folds[0])[3] != release_id:
        raise ValueError('Release changed during reporting')
    released_inputs(root, parent)
    for fold in folds:
        if sha((base / fold / 'receipt.json').read_bytes()) != markers[fold]:
            raise ValueError('Fold receipt changed during reporting')
    out = run_fold.inside(root / 'jobs', Path('comparator-reports') / release_id)
    prior = out / 'private/report.json'
    if prior.exists():
        sources = json.loads(prior.read_bytes())['reporting_sources']
        if sources != private['reporting_sources']:
            # Keep reports issued by earlier source versions available at their original paths.
            out = run_fold.inside(out, 'verified-' + sha(canonical_json(private['reporting_sources'])))
    for scope, report in [('private', private), ('aggregate', public)]:
        run_fold.active(root)
        Store(run_fold.inside(out, scope)).publish('report.json', canonical_json(report))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release-sha256', required=True)
    parser.add_argument('--fold-markers', type=Path, required=True, help='JSON mapping of outer fold IDs to receipt.json SHA-256')
    args = parser.parse_args()
    print(generate(run_fold.ROOT, args.release_sha256, json.loads(args.fold_markers.read_bytes())))


if __name__ == '__main__':
    main()
