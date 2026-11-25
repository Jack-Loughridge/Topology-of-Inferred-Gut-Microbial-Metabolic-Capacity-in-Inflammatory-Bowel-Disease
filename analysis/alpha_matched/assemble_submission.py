"""Prepare a private G6 candidate from pinned development exports and blinded staging inputs."""
import argparse
import csv
import io
import json
from pathlib import Path
import re

from . import comparator_run as driver
from . import export_final, release, report_g7
from .artifacts import Store, sha
from .partitions import canonical_json

ASSEMBLER_SHA256 = release.sha(Path(__file__))

ASSET_ROLES = {'code','environment','data','splits','config','prediction_schema',
               'test_eligibility','statistical_analysis'}


def staged_path(root, name):
    if not isinstance(name,str) or not name or Path(name).is_absolute() or '..' in Path(name).parts:
        raise ValueError('Unsafe staging name')
    return release.resolve(root,name)


def pinned(root, ref):
    path = staged_path(root,ref['file'])
    data = path.read_bytes()
    if sha(data) != ref['sha256']:
        raise ValueError('Staged input changed: '+ref['file'])
    return data


def verify_training_origin(root, staging, source, exported, triples):
    review = json.loads(pinned(root/'reviews',source['review']))
    reviewer = json.loads((root/'roles.json').read_bytes())['independent reviewer'].split()[0]
    if (source['review']['sha256'] != exported['parent_release'] or review['gate'] != 'G5'
            or review['decision'] != 'PASS' or review['blocking_findings'] or review['reviewer'] != reviewer
            or review['scope'] != 'development-final-refit' or review['endpoint'] != exported['endpoint']
            or driver.core.FINAL not in review['folds']):
        raise ValueError('Unbound final-refit source')
    def released(key):
        return pinned(staging,dict(file=source[key],sha256=review['inputs'][key]['sha256']))
    split = json.loads(released('splits'))[driver.core.FINAL]
    config = json.loads(released('config'))
    if split['validate'] or config['endpoint'] != exported['endpoint'] or config['classes'] != exported['classes']:
        raise ValueError('Inconsistent released final-refit partition')
    rows = list(csv.DictReader(io.StringIO(released('meta').decode()),delimiter='\t'))
    if set(split['train']) != {r['person'] for r in rows} or any(r['role']!='development' for r in rows):
        raise ValueError('Invalid released development roster')
    expected = sorted([sha(canonical_json(r[k])) for k in ('sample','person','group')] for r in rows)
    if sorted(triples) != expected:
        raise ValueError('Training identity differs from signed development metadata')


def exclusive_path(root, path):
    for part in (path,*path.parents):
        if part == root:
            return
        if part.is_symlink():
            raise ValueError('Symlink in frozen output')
    raise ValueError('Output outside project')


def permitted(root):
    if driver.core.active(root)['stage'] not in ('G5','G6'):
        raise ValueError('Final-refit stage required to prepare a G6 candidate')
    if release.sha(Path(__file__)) != ASSEMBLER_SHA256:
        raise ValueError('Assembler changed since import')


def assemble(root, staging, spec_id):
    root, staging = Path(root).resolve(strict=True),Path(staging).resolve(strict=True)
    permitted(root)
    spec = json.loads(pinned(staging,dict(file='assembly.json',sha256=spec_id)))
    if spec['version'] != 1 or set(spec['assets']) != ASSET_ROLES:
        raise ValueError('Incomplete G6 assembly asset roles')
    # Archives and candidates may include private eligibility pseudonyms. The staging
    # manifest contains hashes of the outcome commitments, never the outcomes.
    destination = root/'jobs/submission-candidates'/sha(canonical_json([spec_id,ASSEMBLER_SHA256]))/'submission'
    exclusive_path(root,destination)
    if destination.is_relative_to(staging) or staging.is_relative_to(destination):
        raise ValueError('Staged inputs overlap frozen output')
    files, roles = {}, {}
    def put(name,data,role):
        if not re.fullmatch(r'[A-Za-z0-9_.\-/]+',name) or any(s in ('','.','..') for s in name.split('/')):
            raise ValueError('Unsafe frozen filename')
        exclusive_path(root,destination/name)
        checksum = sha(data)
        if files.setdefault(name,checksum) != checksum:
            raise ValueError('Conflicting frozen artifact')
        names=roles.setdefault(role,set())
        if name not in names:
            Store(destination/Path(name).parent).publish(Path(name).name,data)
            names.add(name)

    for role, assets in spec['assets'].items():
        if not assets:
            raise ValueError('Empty mandatory frozen role')
        for asset in assets:
            if not asset['file'].startswith('assets/'):
                raise ValueError('Expected explicit staged asset')
            put(asset['file'],pinned(staging,asset),role)
    if len(roles['test_eligibility']) != 1 or spec['report_policy'] not in roles['statistical_analysis']:
        raise ValueError('Missing blinded eligibility or reporting policy')
    policy=json.loads((destination/spec['report_policy']).read_bytes())
    code_hashes={files[name] for name in roles['code']}
    required_hashes=set(driver.source_hashes().values()) | set(report_g7.sources().values())
    if (policy['version']!=1 or policy['implementation']!=report_g7.sources()
            or not required_hashes <= code_hashes
            or policy.get('aggregate_disclosure')!=report_g7.AGGREGATE_DISCLOSURE
            or policy['bootstrap']!=dict(draws=2000,seed=20261124) or policy['public_min_groups']!=report_g7.dev.PUBLIC_MIN_GROUPS
            or policy['contrasts']!=json.loads(canonical_json(report_g7.statistics.PRIMARY_PAIRS))):
        raise ValueError('Missing or inconsistent frozen scientific runtime or reporting policy')
    eligible=json.loads((destination/next(iter(roles['test_eligibility']))).read_bytes())['rosters']
    if spec['graph_rules'] not in roles['config'] or policy['training_identities'] not in roles['splits']:
        raise ValueError('Missing graph or training identity binding')
    for field in ['identities_sha256','outcomes_sha256']:
        if not re.fullmatch('[a-f0-9]{64}',policy[field]):
            raise ValueError('Invalid steward commitment')
    trained=json.loads((destination/policy['training_identities']).read_bytes())
    panels={ (c,e) for c,endpoints in policy['panel_roles'].items() for e,role in endpoints.items()
             if role in ('primary','secondary') }
    if { (c,e) for c,ep in eligible.items() for e in ep } != panels:
        raise ValueError('Scored eligibility disagrees with panel registry')
    scored_endpoints={e for _,e in panels}
    omitted={(r['cohort'],r['endpoint']) for r in policy['unavailable_panels']}
    matrix={(c,e) for c in policy['panel_roles'] for e in policy['endpoints']}
    if (any(set(cells)!=set(policy['endpoints']) for cells in policy['panel_roles'].values())
            or len(omitted)!=len(policy['unavailable_panels']) or panels & omitted or panels|omitted!=matrix
            or any(policy['panel_roles'][c][e] not in ('missing-mandatory','structurally-inapplicable','unsupported-secondary')
                   for c,e in omitted) or any(not row['reason'] for row in policy['unavailable_panels'])):
        raise ValueError('Incomplete panel applicability')
    export_names = list(policy['procedure_exports'].values())
    if (len(set(export_names)) != len(export_names)
            or any(not re.fullmatch(r'exports/[A-Za-z0-9_.-]+\.json', name) for name in export_names)
            or set(spec['development_sources']) != scored_endpoints):
        raise ValueError('Invalid export destination or development source set')
    if set(spec['exports']) != scored_endpoints or set(policy['procedure_exports']) != scored_endpoints or set(trained)!=scored_endpoints:
        raise ValueError('Unmatched final-refit export, development roster or panel')
    primary=sorted(c for c,e in panels if e=='binary' and policy['panel_roles'][c][e]=='primary')
    if not primary:
        raise ValueError('No eligible registered primary binary cohort')
    missing = {(r['endpoint'],r['arm'],r['method']) for r in policy['unmet_methods']}
    if (len(missing)!=len(policy['unmet_methods']) or any(not r['reason'] for r in policy['unmet_methods'])
            or any(e not in policy['endpoints'] or (a,m) not in report_g7.mandatory() for e,a,m in missing)):
        raise ValueError('Invalid unmet method registry')
    plan=[]
    for endpoint,ref in sorted(spec['exports'].items()):
        driver.core.active(root)
        raw=pinned(staging,ref)
        exported=json.loads(raw)
        if exported['endpoint']!=endpoint or exported['classes']!=policy['endpoints'][endpoint]:
            raise ValueError('Final-refit export endpoint mismatch')
        exporter_hash = release.sha(Path(export_final.__file__))
        validators = {Path(m.__file__).name:release.sha(Path(m.__file__))
                      for m in (export_final.amended,export_final.development)}
        if (exported['exporter_sha256'] != exporter_hash or exported['validator_sha256'] != validators
                or not {exporter_hash,*validators.values()} <= code_hashes):
            raise ValueError('Unbound fitted procedure exporter or evidence validator')
        keys = [exported[k] for k in ('parent_release','comparator_release','primary_marker','comparator_marker')]
        if any(not isinstance(k,str) or not re.fullmatch('[0-9a-f]{64}',k) for k in keys):
            raise ValueError('Invalid producer identity')
        producer = sha(canonical_json([*keys,exporter_hash,validators]))
        original = staged_path(root,'jobs/final-procedure-exports/'+producer+'/export.json')
        if original.read_bytes() != raw:
            raise ValueError('Staged export differs from original final-refit producer')
        for label,folder,marker in [('primary','development-final-refits',keys[2]),('comparator','comparator-folds',keys[3])]:
            rid = keys[0] if label == 'primary' else keys[1]
            if release.sha(staged_path(root,'jobs/'+folder+'/'+rid+'/'+driver.core.FINAL+'/receipt.json')) != marker:
                raise ValueError('Final-refit producer marker changed')
        verify_training_origin(root,staging,spec['development_sources'][endpoint],exported,trained[endpoint])
        put(policy['procedure_exports'][endpoint],raw,'config')
        methods={(r['arm'],r['method']) for r in exported['procedures']}
        primary_methods={(a,m) for a in report_g7.statistics.ARMS for m in report_g7.statistics.HEADS}
        if (report_g7.mandatory()-methods!={(a,m) for e,a,m in missing if e==endpoint}
                or not primary_methods<=methods):
            raise ValueError('Contradictory or incomplete method registry')
        src=staged_path(staging,ref['file']).parent
        for name,h in exported['files'].items():
            if not re.fullmatch(r'objects/[a-f0-9]{64}\.[a-z0-9]+',name) or Path(name).name.split('.')[0]!=h:
                raise ValueError('Unbound export object')
            role=('models' if name in exported['models'] else 'thresholds' if name in exported['decision_sources'] else 'generators')
            put(name,pinned(src,dict(file=name,sha256=h)),role)
        for cohort in sorted(c for c,e in panels if e==endpoint):
            ids=eligible[cohort][endpoint]
            if not ids or len(set(ids))!=len(ids) or not all(isinstance(i,str) and i for i in ids):
                raise ValueError('Invalid private eligibility roster')
            cohort_items=[]
            for proc in exported['procedures']:
                if proc['endpoint']!=endpoint or proc['class_order']!=exported['classes']:
                    raise ValueError('Exported method differs from endpoint')
                key=[cohort,endpoint,proc['arm'],proc['method']]
                item={k:proc[k] for k in ('arm','endpoint','method','class_order','model_sha256','generator_sha256','requires_curvature')}
                item.update(id='pred-'+sha(canonical_json(key)),cohort=cohort,row_count=len(ids),
                            eligible_row_hash=sha(canonical_json(sorted(ids))),graph_rules_sha256=files[spec['graph_rules']])
                cohort_items.append(item)
            if len(cohort_items)!=len({i['id'] for i in cohort_items}):
                raise ValueError('Duplicate procedure in export')
            loaded=driver.load_deployments(destination,dict(files=files,roles=roles),cohort_items)
            triples=trained[endpoint]
            if any(not isinstance(row,list) or len(row)!=3 or not all(re.fullmatch('[a-f0-9]{64}',v) for v in row) for row in triples):
                raise ValueError('Invalid training identity triple')
            train_ids=sorted(r[0] for r in triples)
            if len(set(train_ids))!=len(train_ids) or set(train_ids).intersection(sha(canonical_json(v)) for v in ids):
                raise ValueError('Crossed or duplicated training visit')
            for store,_,generator in loaded.values():
                if driver.read(store,generator['training_visits'])!=train_ids:
                    raise ValueError('Training population differs from deployed model')
            plan.extend(cohort_items)
    ids_ref=spec['reserved_identities']
    if ids_ref['sha256']!=policy['identities_sha256'] or not ids_ref['file'].startswith('identity-only/'):
        raise ValueError('Wrong blinded identity reference')
    # This file contains only the five key fields, without labels. Do not copy it
    # into the candidate: the evaluator retains the original sidecar in custody.
    keys=[{row[k] for triples in trained.values() for row in triples} for k in range(3)]
    report_g7.validate_identity_roster(staged_path(staging,ids_ref['file']),policy,
        {(c,e):(eligible[c][e],) for c,e in panels},keys)
    plan.sort(key=lambda i:i['id'])
    put('comparisons.json',canonical_json(dict(predictions=plan)),'comparisons')
    submission=dict(planned_predictions=plan,eligible_primary_binary_cohorts=primary,
                    roles={role:sorted(names) for role,names in roles.items()},files=files,
                    g7_reporting=spec['report_policy'],assembly_spec_sha256=spec_id,
                    assembler_sha256=ASSEMBLER_SHA256)
    driver.core.active(root)
    for name,checksum in files.items():
        if release.sha(staged_path(destination,name))!=checksum:
            raise ValueError('Frozen candidate changed before publication')
    exclusive_path(root,destination/'submission.json')
    permitted(root)
    Store(destination).publish('submission.json',canonical_json(submission))
    return destination/'submission.json'


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('root',type=Path)
    p.add_argument('staging',type=Path)
    p.add_argument('spec_sha256')
    args=p.parse_args()
    print(assemble(args.root,args.staging,args.spec_sha256))
