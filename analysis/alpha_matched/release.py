"""Content-bound G6 handoff and all-prediction sealing for the evaluator.

Roles use distinct contexts on one shared account. These integrity checks verify
the named review and bytes. They do not authenticate a separate OS identity.
"""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from .partitions import canonical_json


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):
            h.update(b)
    return h.hexdigest()


def resolve(folder,name):
    candidate=(folder/name).resolve(strict=True)
    if not candidate.is_relative_to(folder.resolve()) or not candidate.is_file():
        raise ValueError('Artifact outside declared root')
    return candidate


def verify_release(folder,review_file,expected_reviewer,expected_evaluator):
    folder=Path(folder)
    submission_file=folder/'submission.json'
    sub_id=sha(submission_file)
    submission=json.loads(submission_file.read_bytes())
    review=json.loads(Path(review_file).read_bytes())
    if (not expected_reviewer or not expected_evaluator or expected_reviewer==expected_evaluator
            or review['reviewer']!=expected_reviewer or review['gate']!='G6'
            or review['decision']!='PASS' or review['blocking_findings']!=[]
            or review['submission_sha256']!=sub_id):
        raise PermissionError('Exact independent G6 decision required')
    if (datetime.fromisoformat(review['timestamp']).utcoffset() is None
            or not isinstance(review['resolved_findings'],list)):
        raise ValueError('G6 needs attributed time and a finding resolution register')
    for finding in review['resolved_findings']:
        if not finding['id'] or sha(resolve(folder,finding['artifact']))!=finding['sha256']:
            raise ValueError('G6 finding resolution evidence changed')
    files=submission['files']
    groups=('code','environment','data','splits','config','models','generators','thresholds',
            'prediction_schema','test_eligibility','statistical_analysis','comparisons')
    if any(not submission['roles'].get(k) for k in groups):
        raise ValueError('Incomplete submission roles')
    listed=set().union(*(set(submission['roles'][k]) for k in groups))
    if listed!=set(files) or review['reviewed_files']!=files:
        raise ValueError('Incomplete digest-bound review scope')
    for path,expected in files.items():
        if sha(resolve(folder,path))!=expected:
            raise ValueError('G6-covered artifact changed: '+path)
    plan=submission['planned_predictions']
    if not plan or len({p['id'] for p in plan})!=len(plan):
        raise ValueError('Missing or duplicate prediction obligations')
    if any(set(m for m in ('cohort','endpoint','arm','method','eligible_row_hash','row_count') if m not in p)
           or p['row_count']<1 for p in plan):
        raise ValueError('Incomplete prediction plan')
    tables=submission['roles']['comparisons']
    eligibility=submission['roles']['test_eligibility']
    if len(tables)!=1 or len(eligibility)!=1:
        raise ValueError('Use one frozen obligations and one eligibility table')
    obligations=json.loads(resolve(folder,tables[0]).read_bytes())['predictions']
    rosters=json.loads(resolve(folder,eligibility[0]).read_bytes())['rosters']
    if sorted(plan,key=lambda p:p['id'])!=sorted(obligations,key=lambda p:p['id']):
        raise ValueError('Prediction plan differs from frozen analysis obligations')
    for p in plan:
        ids=rosters[p['cohort']][p['endpoint']]
        if (not ids or len(ids)!=len(set(ids)) or any(not isinstance(x,str) for x in ids)
                or p['row_count']!=len(ids) or p['eligible_row_hash']!=hashlib.sha256(canonical_json(sorted(ids))).hexdigest()
                or type(p['requires_curvature']) is not bool or len(set(p['class_order']))!=len(p['class_order'])
                or len(p['class_order'])<2):
            raise ValueError('Unmatched frozen eligibility for '+p['id'])
        for field,role in [('model_sha256','models'),('generator_sha256','generators'),('graph_rules_sha256','config')]:
            if p[field] not in {files[f] for f in submission['roles'][role]}:
                raise ValueError('Unknown frozen procedure for '+p['id'])
        if p['method'] in ('Ricci','joint','K0') and not p['requires_curvature']:
            raise ValueError('Curvature evidence required for '+p['method'])
    primary=submission['eligible_primary_binary_cohorts']
    if not primary or len(set(primary))!=len(primary):
        raise ValueError('Missing eligible primary cohort registry')
    required={(c,'binary',arm,model) for c in primary for arm in ('RAW','RELATIVE')
              for model in ('H0','Ricci','joint')}
    registered={(p['cohort'],p['endpoint'],p['arm'],p['method']) for p in plan}
    if not required<=registered or len(registered)!=len(plan):
        raise ValueError('Missing or duplicate matched primary prediction obligations')
    return sub_id,submission


def immutable_write(directory,name,body):
    directory.mkdir(parents=True,exist_ok=True)
    path=directory/name
    payload=canonical_json(body)
    with tempfile.NamedTemporaryFile(dir=directory,delete=False) as f:
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())
    tmp=Path(f.name)
    try:
        try:
            os.link(tmp,path)
        except FileExistsError:
            if path.read_bytes()!=payload:
                raise ValueError('Conflicting attempt to modify sealed record')
    finally:
        tmp.unlink()
    return path


def verify_prediction(pdir,artifact,item,sid,graphs):
    ids=set()
    count=0
    with artifact.open() as f:
        for line in f:
            row=json.loads(line)
            sample=row['sample']
            if not isinstance(sample,str) or sample in ids or row['submission_sha256']!=sid:
                raise ValueError('Duplicate or incorrectly bound prediction')
            for field in ('class_order','model_sha256','generator_sha256'):
                if row[field]!=item[field]:
                    raise ValueError('Prediction procedure or class identity mismatch')
            if row['prediction_id']!=item['id']:
                raise ValueError('Prediction belongs to another obligation')
            ids.add(sample)
            raw=row['probabilities']
            if any(type(x) not in (float,int) for x in raw):
                raise ValueError('Non-numeric probability')
            p=np.asarray(raw,dtype=float)
            if (p.shape!=(len(item['class_order']),) or not np.isfinite(p).all()
                    or np.any(p<0) or np.any(p>1) or abs(p.sum()-1)>1e-10):
                raise ValueError('Invalid class probabilities')
            if item['requires_curvature']:
                geom=row['geometry']
                expected=graphs[item['cohort']][sample]
                if (expected['sample']!=sample or expected['graph_rules_sha256']!=item['graph_rules_sha256']
                        or geom['sample']!=sample or geom['graph_rules_sha256']!=expected['graph_rules_sha256']
                        or geom['profile_sha256']!=expected['profile_sha256']
                        or geom['active_pairs_sha256']!=expected['active_pairs_sha256']
                        or geom['prediction_id']!=item['id'] or geom['generator_sha256']!=item['generator_sha256']):
                    raise ValueError('Wrong sample graph or procedure evidence')
                gpath=resolve(pdir,geom['values'])
                epath=resolve(pdir,geom['active_pairs'])
                if sha(gpath)!=geom['values_sha256'] or sha(epath)!=geom['active_pairs_sha256']:
                    raise ValueError('Curvature evidence digest mismatch')
                edges=np.load(epath,allow_pickle=False)
                with np.load(gpath,allow_pickle=False) as data:
                    curv=data['curvature']
                    if (edges.ndim!=2 or edges.shape[1]!=2 or edges.dtype.kind not in 'iu'
                            or np.any(edges<0) or np.any(edges[:,0]==edges[:,1])
                            or len(np.unique(edges,axis=0))!=len(edges)
                            or not np.array_equal(data['pairs'],edges) or curv.shape!=(len(edges),)
                            or curv.dtype.kind!='f' or not np.isfinite(curv).all()):
                        raise ValueError('Incomplete active-edge evidence')
                count+=len(edges)
    if len(ids)!=item['row_count'] or hashlib.sha256(canonical_json(sorted(ids))).hexdigest()!=item['eligible_row_hash']:
        raise ValueError('Predictions disagree with frozen eligibility')
    return count


def verify_and_seal(submission_root,review_file,reviewer,evaluator,prediction_root):
    """Evaluator calls only after validating rows, probabilities and all-edge receipts.

    Repeat this check both on resume and before outcome disclosure.
    """
    sid,sub=verify_release(submission_root,review_file,reviewer,evaluator)
    pdir=Path(prediction_root)
    graph_file=resolve(pdir,'graph_roster.json')
    graph_registry=json.loads(graph_file.read_bytes())
    if graph_registry['submission_sha256']!=sid or graph_registry['evaluator']!=evaluator:
        raise ValueError('Wrong evaluator graph roster')
    snapshots=[]
    for item in sub['planned_predictions']:
        path=resolve(pdir,item['id']+'.json')
        rec=json.loads(path.read_bytes())
        art=resolve(pdir,rec['artifact'])
        actual=verify_prediction(pdir,art,item,sid,graph_registry['cohorts'])
        if (any(type(rec[k]) is not int or rec[k]<0 for k in ('row_count','computed_finite_edges','expected_edges','failed_edges'))
                or rec['computed_finite_edges']!=actual or rec['submission_sha256']!=sid or rec['prediction_id']!=item['id']
                or rec['eligible_row_hash']!=item['eligible_row_hash'] or rec['row_count']!=item['row_count']
                or rec['sha256']!=sha(art) or rec['computed_finite_edges']!=rec['expected_edges']
                or rec['failed_edges']!=0 or rec['schema_validation_passed'] is not True):
            raise ValueError('Unsealed or mismatched prediction: '+item['id'])
        snapshots.append(dict(prediction_id=item['id'],artifact=rec['artifact'],sha256=rec['sha256'],receipt=sha(path)))
    return immutable_write(pdir,'SEALED.json',dict(submission_sha256=sid,review_sha256=sha(Path(review_file)),
                                                   predictions=snapshots,evaluator=evaluator,graph_roster_sha256=sha(graph_file)))
