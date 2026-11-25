"""Evaluator-only, global post-seal reporting. All endpoint panels publish together."""
import argparse
from collections import Counter, defaultdict
import fcntl
import json
import os
from pathlib import Path
import re
import tempfile
from datetime import datetime, timezone
import uuid
import zlib

import numpy as np
import pandas as pd

from . import comparator_run, release, report_development as dev, run_fold, statistics
from .artifacts import Store, sha
from .partitions import canonical_json


AGGREGATE_DISCLOSURE = 'related-group-membership-profiles-v1'


def sources():
    return {Path(p).name: release.sha(Path(p)) for p in
            (__file__, statistics.__file__, dev.__file__, release.__file__, comparator_run.__file__, run_fold.__file__)}


def checked_bytes(path, digest):
    data = path.read_bytes()
    if sha(data) != digest:
        raise ValueError('Frozen artifact changed')
    return data


def authority(root):
    status = run_fold.active(root)
    if status['stage'] != 'G7':
        raise PermissionError('G7 required')
    roles = json.loads((root / 'roles.json').read_bytes())
    reviewer = roles['independent reviewer'].split()[0]
    evaluator = roles['reserved evaluator'].split()[0]
    grant = status['g7_report_release']
    review = status['final_prediction_release']
    if review['sha256'] != grant['review_sha256']:
        raise PermissionError('Wrong G6 review for G7 activation')
    review_path = run_fold.inside(root / 'reviews', review['file'])
    reviewed = json.loads(checked_bytes(review_path, grant['review_sha256']))
    actor = reviewed['evaluator_session']
    thread = os.environ.get('CODEX_THREAD_ID')
    if (not evaluator or reviewer == evaluator or grant['evaluator'] != evaluator or actor['role'] != evaluator
            or not thread or thread != actor['thread_id'] or str(uuid.UUID(thread)) != thread):
        raise PermissionError('G6-designated evaluator session required')
    workspace = Path(grant['workspace'])
    if workspace.resolve(strict=True) != workspace or not workspace.is_dir():
        raise ValueError('Invalid evaluator workspace')
    return grant, workspace, reviewer, evaluator, review_path


def global_seal(root):
    auth = authority(root)
    grant, work, reviewer, evaluator, review = auth
    submission = run_fold.inside(work, 'submission')
    predictions = run_fold.inside(work, 'predictions')
    seal_path = run_fold.inside(predictions, 'SEALED.json')
    # Require the pre-existing, independently activated seal even though the
    # common verifier can also create a seal during initial evaluator application.
    raw = checked_bytes(seal_path, grant['seal_sha256'])
    manifest = run_fold.inside(submission, 'submission.json')
    sub = json.loads(checked_bytes(manifest, grant['submission_sha256']))
    if any(not re.fullmatch(r'[A-Za-z0-9_-][A-Za-z0-9_.-]*', p['id']) for p in sub['planned_predictions']):
        raise ValueError('Invalid obligation ID')
    verified = release.verify_and_seal(submission, review, reviewer, evaluator, predictions)
    if (verified != seal_path or seal_path.read_bytes() != raw
            or release.sha(manifest) != grant['submission_sha256'] or release.sha(review) != grant['review_sha256']
            or authority(root) != auth):
        raise ValueError('Seal or authorization changed during verification')
    return auth, sub, json.loads(raw)


def role_file(submission, sub, role, name):
    if name not in sub['roles'][role]:
        raise ValueError('Wrong G6 artifact role ' + role)
    path = release.resolve(submission, name)
    data = checked_bytes(path, sub['files'][name])
    return json.loads(zlib.decompress(data) if path.suffix == '.jsonz' else data)


def frozen_rule(submission, sub, item, model, exported, source_paths):
    digest = exported['decision_source_sha256']
    name = source_paths[digest]
    if any(exported[k] != item[k] for k in ('method','endpoint','class_order','model_sha256',
                                              'generator_sha256','requires_curvature')):
        raise ValueError('Final-refit procedure export disagrees with G6 plan')
    source = role_file(submission, sub, 'thresholds', name)
    method, classes = item['method'], item['class_order']
    cmp = method.startswith('cmp.')
    if cmp:
        if source['method'] != method or source['classes'] != classes or source['parent'] != run_fold.FINAL:
            raise ValueError('Comparator decision source identity mismatch')
        cut = source['threshold']
    else:
        if source['class_order'] != classes:
            raise ValueError('Core deployment decision source mismatch')
        if len(classes) == 2:
            if source['parent'] != run_fold.FINAL or not source['heldout_fits'][method]:
                raise ValueError('Missing training decision provenance')
            cut = source['thresholds'][method]
        else:
            cut = None
    binary = len(classes) == 2
    rule = dict(rule='max-participant-balanced-accuracy' if binary else 'argmax', threshold=cut)
    if (source['rule'] != rule['rule'] or canonical_json(exported['rule']) != canonical_json(rule) or cmp and model['decision'] != rule
            or binary and (type(cut) not in (float, int) or not 0 <= cut <= 1)
            or not binary and cut is not None or cmp and binary and not source['heldout_fits']):
        raise ValueError('Invalid frozen decision')
    return cut


def mandatory():
    methods = {(arm, name) for arm in statistics.ARMS for name in (*dev.COMMON, 'direct-' + arm.lower())}
    for rep, owner in comparator_run.DIRECT.items():
        for family in (*comparator_run.BASE, 'tabm'):
            for arm in statistics.ARMS if owner == 'SHARED' else [owner]:
                methods.add((arm, f'cmp.{owner.lower()}.{rep}.{family}'))
    return methods


def prepare(work, sub, seal):
    submission = work / 'submission'
    cfg = role_file(submission, sub, 'statistical_analysis', sub['g7_reporting'])
    if (cfg['version'] != 1 or cfg['implementation'] != sources() or not set(sources().values()) <=
            {sub['files'][p] for p in sub['roles']['code']} or
            cfg['contrasts'] != json.loads(canonical_json(statistics.PRIMARY_PAIRS)) or
            cfg['bootstrap'] != dict(draws=2000, seed=20261124) or
            cfg['public_min_groups'] != dev.PUBLIC_MIN_GROUPS or cfg['aggregate_disclosure'] != AGGREGATE_DISCLOSURE):
        raise ValueError('Unrecognized frozen statistical analysis')
    roles, endpoints = cfg['panel_roles'], cfg['endpoints']
    if (not roles or any(set(r) != set(endpoints) for r in roles.values())
            or 'binary' not in endpoints or len(endpoints['binary']) != 2
            or any(not isinstance(cs, list) or len(cs) not in (2, 3) or len(set(cs)) != len(cs)
                   or any(not isinstance(c, str) or not c.strip() for c in cs) for cs in endpoints.values())):
        raise ValueError('Invalid frozen cohort/endpoint registry')
    exported = {}
    source_paths = {sub['files'][n]:n for n in sub['roles']['thresholds']
                    if Path(n).name in (sub['files'][n]+'.json', sub['files'][n]+'.jsonz')}
    for endpoint, name in cfg['procedure_exports'].items():
        export = role_file(submission, sub, 'config', name)
        if (export['endpoint'] != endpoint or export['classes'] != endpoints[endpoint]
                or any(not re.fullmatch('[0-9a-f]{64}', export[k]) for k in
                       ('parent_release','comparator_release','primary_marker','comparator_marker','exporter_sha256'))):
            raise ValueError('Invalid endpoint final-refit export')
        models = {export['files'][n] for n in export['models']}
        sources_used = {export['files'][n] for n in export['decision_sources']}
        for row in export['procedures']:
            k = endpoint, row['arm'], row['method']
            if (row['endpoint'] != endpoint or row['class_order'] != endpoints[endpoint] or k in exported
                    or row['arm'] not in statistics.ARMS or row['model_sha256'] not in models
                    or row['decision_source_sha256'] not in sources_used
                    or row['decision_source_sha256'] not in source_paths):
                raise ValueError('Duplicate or unbound final-refit export row')
            exported[k] = row
    snapshots = {p['prediction_id']: p for p in seal['predictions']}
    groups = {}
    for item in sub['planned_predictions']:
        key = item['cohort'], item['endpoint']
        if key[0] not in roles or roles[key[0]][key[1]] not in ('primary','secondary') or item['class_order'] != endpoints[key[1]] or item['arm'] not in statistics.ARMS:
            raise ValueError('Unexpected procedure roster or class order')
        groups.setdefault(key, []).append(item)
    omitted = {(r['cohort'], r['endpoint']): r for r in cfg['unavailable_panels']}
    if (len(omitted) != len(cfg['unavailable_panels']) or set(groups) & omitted.keys()
            or set(groups) | omitted.keys() != {(c, e) for c in roles for e in endpoints}
            or any(not r['reason'] or roles[c][e] not in ('structurally-inapplicable','unsupported-secondary','missing-mandatory')
                   for (c,e), r in omitted.items())
            or {c for c, e in groups if e == 'binary' and roles[c][e] == 'primary'} != set(sub['eligible_primary_binary_cohorts'])):
        raise ValueError('Incomplete approved cohort/endpoint accounting')
    unmet = {(r['endpoint'], r['arm'], r['method']) for r in cfg['unmet_methods']}
    if (len(unmet) != len(cfg['unmet_methods']) or any(not r['reason'] for r in cfg['unmet_methods'])
            or any(e not in endpoints or (a, m) not in mandatory() for e, a, m in unmet)):
        raise ValueError('Invalid unmet method registry')
    planned = {(i['endpoint'], i['arm'], i['method']) for i in sub['planned_predictions']}
    if set(exported) != planned:
        raise ValueError('Procedure export differs from the global required model set')
    eligibility = role_file(submission, sub, 'test_eligibility', sub['roles']['test_eligibility'][0])['rosters']
    if {(c,e) for c, rows in eligibility.items() for e in rows} != set(groups):
        raise ValueError('Eligibility contains an unaccounted panel')
    training = role_file(submission, sub, 'splits', cfg['training_identities'])
    training_keys = [set(), set(), set()]
    visits_by_endpoint = {}
    for ep, rows in training.items():
        if (not rows or any(not isinstance(row,list) or len(row) != 3 or
                            any(not re.fullmatch('[0-9a-f]{64}', x) for x in row) for row in rows)
                or len({r[0] for r in rows}) != len(rows)):
            raise ValueError('Invalid hashed training identity registry')
        visits_by_endpoint[ep] = sorted(r[0] for r in rows)
        for keys, values in zip(training_keys,zip(*rows)):
            keys.update(values)
    if set(training) != {ep for _,ep in groups}:
        raise ValueError('Training identity registry differs from scored endpoints')
    identities, panel_methods, panels = {}, {}, {}
    checked_training = set()
    for (cohort, endpoint), items in groups.items():
        methods = {(i['arm'], i['method']) for i in items}
        missing = {(a,m) for e,a,m in unmet if e == endpoint}
        if (methods != panel_methods.setdefault(endpoint, methods) or mandatory() - methods != missing or
                not {(a,h) for a in statistics.ARMS for h in statistics.HEADS} <= methods):
            raise ValueError('Incomplete matched procedure inventory')
        deployments = (comparator_run.load_deployments(submission, sub, items)
                       if any(m.startswith('cmp.') for _, m in methods)
                       else run_fold.released_models(submission, sub, items))
        ids = eligibility[cohort][endpoint]
        tables, cuts, links, shared = {}, {}, {}, {}
        for item in items:
            pid = item['id']
            key = item['arm'], item['method']
            snap = snapshots[pid]
            path = release.resolve(work / 'predictions', snap['artifact'])
            rows = [json.loads(line) for line in checked_bytes(path, snap['sha256']).splitlines()]
            if any(r['submission_sha256'] != seal['submission_sha256'] or r['prediction_id'] != pid
                   or any(r[k] != item[k] for k in ('class_order','model_sha256','generator_sha256')) for r in rows):
                raise ValueError('Sealed row procedure binding changed')
            samples = [r['sample'] for r in rows]
            if len(samples) != len(ids) or set(samples) != set(ids):
                raise ValueError('Prediction rows differ from eligibility')
            tables[key] = pd.DataFrame([r['probabilities'] for r in rows], index=samples,
                                       columns=item['class_order']).loc[ids]
            run_fold.validate_probabilities(tables[key].to_numpy(), len(ids), item['class_order'])
            store, model, gen = deployments[key]
            token = endpoint, gen['training_visits']
            if token not in checked_training:
                if comparator_run.read(store, gen['training_visits']) != visits_by_endpoint[endpoint]:
                    raise ValueError('Frozen generator and training identity registry differ')
                checked_training.add(token)
            if key[1].startswith('cmp.'):
                reg = comparator_run.read(store, model['registry'])['entries'][key[1]]
                rep, fam, owner = reg['representation'], reg['family'], reg['arm']
                if (key[1] != f'cmp.{owner.lower()}.{rep}.{fam}' or
                        owner != comparator_run.DIRECT.get(rep, key[0]) or
                        reg['applies_to'] != (list(statistics.ARMS) if owner == 'SHARED' else [owner])):
                    raise ValueError('Comparator role or representation identity mismatch')
            row = exported[(endpoint, *key)]
            cuts[key] = frozen_rule(submission, sub, item, model, row, source_paths)
            identity = (item['model_sha256'], item['generator_sha256'], row['decision_source_sha256'], cuts[key])
            if identities.setdefault((endpoint, key), identity) != identity:
                raise ValueError('Procedure or decision changes across cohorts')
            if key[1].startswith('cmp.shared.species.'):
                signature = (identity, tables[key].to_numpy().tobytes())
                if shared.setdefault(key[1], signature) != signature:
                    raise ValueError('Shared species predictions or decisions disagree')
            links[key] = dict(id=pid, model=item['model_sha256'], generator=item['generator_sha256'],
                              decision_source=row['decision_source_sha256'], threshold=cuts[key])
        panels[cohort,endpoint] = ids, tables, cuts, links
    return cfg, panels, training_keys


AXES = ('cohort', 'endpoint', 'sample', 'person', 'group')


def validate_identity_roster(path, cfg, panels, training_keys):
    rows = json.loads(checked_bytes(path, cfg['identities_sha256']))
    if (not isinstance(rows,list) or not rows or any(not isinstance(r,list) or len(r) != len(AXES)
            or any(not isinstance(x,str) or not x.strip() for x in r) for r in rows)
            or rows != sorted(rows)):
        raise ValueError('Invalid frozen visit/participant/group registry')
    frame = pd.DataFrame(rows, columns=AXES)
    if (frame.duplicated(['cohort', 'endpoint', 'sample']).any()
            or (frame.groupby('sample')[['cohort','person','group']].nunique() != 1).any().any()
            or (frame.groupby('person')[['cohort','group']].nunique() != 1).any().any()
            or (frame.groupby('group').cohort.nunique() != 1).any()):
        raise ValueError('Duplicate visit or inconsistent participant/group roster')
    for column, known in zip(('sample','person','group'), training_keys):
        if any(sha(canonical_json(i)) in known for i in frame[column].unique()):
            raise ValueError('Reserved roster overlaps a fitted visit, participant or related group')
    by_panel = {k:set(v['sample']) for k,v in frame.groupby(['cohort','endpoint'])}
    if by_panel != {k:set(value[0]) for k,value in panels.items()}:
        raise ValueError('Identity roster differs from approved panel eligibility')


def _parse_outcomes(data, cfg, panels):
    rows = [json.loads(line) for line in data.splitlines()]
    if any(set(row) != {*AXES, 'label'} or any(type(row[k]) is not str or not row[k].strip() for k in row)
           for row in rows):
        raise ValueError('Invalid outcome roster schema')
    if sha(canonical_json(sorted([r[k] for k in AXES] for r in rows))) != cfg['identities_sha256']:
        raise ValueError('Visit/participant/related-group commitment mismatch')
    frame = pd.DataFrame(rows)
    results = {}
    for k, panel in frame.groupby(['cohort','endpoint']):
        if k not in panels or set(panel['sample']) != set(panels[k][0]):
            raise ValueError('Outcome roster disagrees with global eligibility')
        cs = cfg['endpoints'][k[1]]
        if not panel.label.isin(cs).all():
            raise ValueError('Outcome outside frozen class order')
        roster = panel.set_index('sample').loc[panels[k][0]].copy()
        roster['y'] = roster.label.map({c: i for i,c in enumerate(cs)})
        statistics._check_roster(roster, cs)
        results[k] = roster
    if results.keys() != panels.keys():
        raise ValueError('Missing outcome panel')
    return results


def _supported_overlap(cohort, endpoints, rosters, minimum):
    """Group by the full membership pattern, without using identifiers as features.

    A visit signature carries every endpoint/class inclusion. Keeping the visits
    within each person, and people within each group, preserves multiplicities
    in group, participant and visit counts for any combination of masks.
    """
    members = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for endpoint in sorted(endpoints):
        for visit, person, group, label in rosters[cohort, endpoint][['person', 'group', 'y']].itertuples():
            members[group][person][visit].append((endpoint, label))
    profiles = Counter()
    for people in members.values():
        person_patterns = []
        for visits in people.values():
            person_patterns.append(tuple(sorted(tuple(masks) for masks in visits.values())))
        profiles[tuple(sorted(person_patterns))] += 1
    return all(count >= minimum for count in profiles.values())


def _public_panels(roles, rosters):
    """Greedily admit panels in frozen role/endpoint order within each cohort."""
    allowed = set()
    components = defaultdict(list)
    order = sorted(rosters, key=lambda k: (k[0], roles[k[0]][k[1]] != 'primary', k[1]))
    for cohort, endpoint in order:
        frame = rosters[cohort, endpoint]
        groups = set(frame.group)
        if len(groups) < dev.PUBLIC_MIN_GROUPS:
            continue
        previous = components[cohort]
        merged = {endpoint}
        independent = []
        for other_endpoints, other_groups in previous:
            if groups.isdisjoint(other_groups):
                independent.append((other_endpoints, other_groups))
            else:
                merged.update(other_endpoints)
                groups.update(other_groups)
        if not _supported_overlap(cohort, merged, rosters, dev.PUBLIC_MIN_GROUPS):
            continue
        components[cohort] = independent + [(merged, groups)]
        allowed.add((cohort, endpoint))
    return allowed


def _score(cfg, panels, rosters, seal, grant):
    roles = cfg['panel_roles']
    omitted = [dict(r, classification=roles[r['cohort']][r['endpoint']]) for r in cfg['unavailable_panels']]
    unmet = cfg['unmet_methods'] or any(r['classification'] == 'missing-mandatory' for r in omitted)
    status = 'complete'
    if unmet:
        status = 'partial'
    elif any(r['classification'] == 'unsupported-secondary' for r in omitted):
        status = 'complete-with-declared-limitations'
    private = dict(status=status, seal_sha256=grant['seal_sha256'], submission_sha256=seal['submission_sha256'],
                   review_sha256=seal['review_sha256'], evaluator=seal['evaluator'], evaluator_thread=os.environ['CODEX_THREAD_ID'],
                   outcomes_sha256=cfg['outcomes_sha256'], identities_sha256=cfg['identities_sha256'],
                   unavailable_panels=omitted, unmet_methods=cfg['unmet_methods'], cohorts={})
    public = dict(status=status, cohorts={}, unavailable_panels=[
        {k:r[k] for k in ('cohort','endpoint','classification')} for r in omitted],
        unmet_methods=[{k:r[k] for k in ('endpoint','arm','method')} for r in cfg['unmet_methods']],
        interpretation='Available-visit participant means. Paired intervals are descriptive and marginal.')
    allowed = _public_panels(roles, rosters)
    summary = {}
    for (cohort, endpoint), (ids, tables, cuts, links) in panels.items():
        roster, cs = rosters[cohort, endpoint], cfg['endpoints'][endpoint]
        models = statistics.cohort_report(roster, tables, cs, cuts)
        coverage = next(iter(models.values()))['coverage']
        if any(v['status'] != 'complete' or v['coverage'] != coverage for v in models.values()):
            raise ValueError('Incomplete matched statistical coverage')
        contrasts = statistics.primary_contrasts(roster, tables, cs, cuts, **cfg['bootstrap'])
        detail = dict(status='complete', class_order=cs, role=roles[cohort][endpoint], coverage=coverage,
                      auc_status='defined' if all(coverage['class_people'].values()) else 'undefined under class support',
                      limitations=dev.limitations(coverage, contrasts), models=[],
                      contrasts=[dict(left=a,right=b,**v) for (a,b),v in contrasts.items()])
        for (arm, method), result in models.items():
            means = statistics.participant_probabilities(roster, tables[arm,method], cs)
            detail['models'].append(dict(arm=arm, method=method, metrics=result['metrics'], **links[arm,method],
                                         participants=means.index.tolist(), probabilities=means.to_numpy().tolist()))
        detail['visits'] = roster[['person','group','y']].rename_axis('sample').reset_index().to_dict('records')
        private['cohorts'].setdefault(cohort,{})[endpoint] = detail
        projected = (dev.public_cohort(detail) if (cohort, endpoint) in allowed
                     else dict(status='withheld for independent-group disclosure control'))
        projected.update(class_order=cs, role=roles[cohort][endpoint])
        public['cohorts'].setdefault(cohort,{})[endpoint] = projected
        summary.setdefault((endpoint, roles[cohort][endpoint]), {})[cohort] = models
    private['equal_cohort_auc'], public['equal_cohort_auc'] = [], []
    for (endpoint, role), reports in sorted(summary.items()):
        cohorts = sorted(reports)
        eligible = [c for c in cohorts if all(private['cohorts'][c][endpoint]['coverage']['class_people'].values())]
        rows = [dict(arm=a, method=m, roc_auc=statistics.equal_cohort_auc(reports, (a,m), expected_cohorts=cohorts))
                for a,m in sorted(reports[cohorts[0]])]
        item = dict(endpoint=endpoint, role=role, eligible_cohorts=eligible, models=rows)
        private['equal_cohort_auc'].append(item)
        suppressed = any((c, endpoint) not in allowed for c in cohorts)
        public['equal_cohort_auc'].append(dict(endpoint=endpoint,role=role,status='withheld') if suppressed else item)
    return private, public


def generate(root):
    root = Path(root).resolve(strict=True)
    auth, sub, seal = gate = global_seal(root)
    grant, work = auth[:2]
    cfg, panels, training_keys = prepare(work, sub, seal)
    identity_path = run_fold.inside(work, 'outcomes/identities.json')
    validate_identity_roster(identity_path, cfg, panels, training_keys)
    if global_seal(root) != gate:
        raise ValueError('Frozen evidence changed before outcome access')
    path = run_fold.inside(work, 'outcomes/outcomes.jsonl')
    event = dict(event='authorized-outcome-read', utc=datetime.now(timezone.utc).isoformat(),
                 submission_sha256=seal['submission_sha256'], seal_sha256=grant['seal_sha256'],
                 evaluator=seal['evaluator'], thread_id=os.environ['CODEX_THREAD_ID'])
    journal = run_fold.inside(work, 'exposure')
    journal.mkdir(mode=0o700, exist_ok=True)
    release.immutable_write(journal, uuid.uuid4().hex + '.json', event)
    if authority(root) != auth or release.sha(work / 'predictions/SEALED.json') != grant['seal_sha256']:
        raise ValueError('Activation changed at outcome access')
    outcome_bytes = checked_bytes(path, cfg['outcomes_sha256'])
    rosters = _parse_outcomes(outcome_bytes, cfg, panels)
    if global_seal(root) != gate:
        raise ValueError('Frozen evidence changed before scoring')
    private, aggregate = _score(cfg, panels, rosters, seal, grant)
    output = run_fold.inside(work, 'reports')
    output.mkdir(mode=0o700, exist_ok=True)
    dest = run_fold.inside(output, grant['seal_sha256'])
    with run_fold.inside(output, '.publication.lock').open('a') as lock, tempfile.TemporaryDirectory(dir=output) as tmp:
        fcntl.flock(lock, fcntl.LOCK_EX)
        stage = Path(tmp) / 'bundle'
        for scope, result in [('private', private), ('aggregate', aggregate)]:
            Store(stage / scope).publish('report.json', canonical_json(result))
        existing = dest.exists()
        if existing:
            for scope in ('private','aggregate'):
                published = run_fold.inside(dest, Path(scope) / 'report.json')
                if (stage / scope / 'report.json').read_bytes() != published.read_bytes():
                    raise ValueError('Conflicting published G7 report')
        if (global_seal(root) != gate or release.sha(identity_path) != cfg['identities_sha256']
                or release.sha(path) != cfg['outcomes_sha256'] or authority(root) != auth):
            raise ValueError('Global evidence changed before publication')
        if not existing:
            os.rename(stage, dest)
    return dest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    try:
        result = generate(args.root)
    except (KeyError, IndexError, TypeError, OSError, ValueError, RuntimeError):
        parser.exit(2, 'G7 authorization or evidence check failed.\n')
    print(result)


if __name__ == '__main__':
    main()
