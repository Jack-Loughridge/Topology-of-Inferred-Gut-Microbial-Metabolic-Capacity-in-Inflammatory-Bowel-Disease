"""Strict source/split provenance; no fresh outer split is substituted."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib,json,platform,sys,shutil
from pathlib import Path
import numpy as np
import pandas as pd
import scipy,sklearn
from scipy import sparse
from sklearn.model_selection import StratifiedGroupKFold
from vendor.original_core import data as core_data
from vendor.original_core.tasks import TASKS,mask_for_task,labels_for_task,normalise_label
from .settings import Settings,TASK_ORDER,MATRIX_SHA256
from .io import sha256_file,fingerprint,read_json,stat_record,atomic_json,atomic_csv
from . import SCHEMA
ROOT=Path(__file__).resolve().parents[1]
TASK_BY_FOLDER={t.folder:t for t in TASKS}

def runtime_versions():
    return {'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,
            'scipy':scipy.__version__,'scikit_learn':sklearn.__version__}

def package_fingerprint():
    return fingerprint({str(p.relative_to(ROOT)):sha256_file(p) for p in sorted(ROOT.rglob('*.py'))})

def update_array_digest(h, array):
    a = np.ascontiguousarray(array)
    h.update(str(a.dtype).encode()); h.update(np.asarray(a.shape,dtype=np.int64).tobytes())
    h.update(a.view(np.uint8))


def core_input_fingerprints(inputs):
    # Same content identity algorithm as the original repeated-CV orchestrator.
    meta = pd.DataFrame({'sample_id':np.asarray(inputs.sample_ids,dtype=str),
        'participant_id':np.asarray(inputs.participant_ids,dtype=str),
        'label':np.asarray(inputs.labels3,dtype=str),
        'h0_path':np.asarray(inputs.h0_paths,dtype=str)}).sort_values('sample_id',kind='mergesort')
    h0 = hashlib.sha256()
    for sid, values in sorted(zip(np.asarray(inputs.sample_ids,dtype=str),inputs.h0_deaths),key=lambda x:x[0]):
        b=sid.encode(); h0.update(len(b).to_bytes(8,'little')); h0.update(b)
        update_array_digest(h0,np.asarray(values,dtype=np.float64))
    X=inputs.ricci.tocsr(copy=False)
    if not X.has_sorted_indices: X=X.copy(); X.sort_indices()
    hr=hashlib.sha256()
    for a in [np.asarray(X.shape,dtype=np.int64),X.indptr,X.indices,X.data]:update_array_digest(hr,a)
    return {'sample_metadata_sha256':hashlib.sha256(meta.to_csv(index=False).encode()).hexdigest(),
            'h0_diagrams_sha256':h0.hexdigest(),'ricci_matrix_sha256':hr.hexdigest()}


@dataclass(frozen=True)
class Fold:
    task: str
    repeat: int
    number: int
    seed: int
    train: np.ndarray  # GLOBAL rows, in original matrix order
    test: np.ndarray
    inner: tuple  # (global_train, global_validation), same original seed arithmetic

    @property
    def key(self):return f'{self.task}/repeat_{self.repeat:02d}/fold_{self.number}'

    def identity(self):
        return fingerprint({'task':self.task,'repeat':self.repeat,'fold':self.number,'seed':self.seed,
            'train':self.train.tolist(),'test':self.test.tolist(),
            'inner':[(a.tolist(),b.tolist()) for a,b in self.inner]})


def require_split_classes(y, indices, classes, name):
    if set(y[indices]) != set(classes):
        raise ValueError(f'{name} does not contain all classes; no alternative split is substituted.')


def plan_from_manifest(inputs, manifest: pd.DataFrame, settings: Settings,
                       reference: Path | None = None):
    required={'task_folder','repeat','fold','split_seed','role','sample_id','participant_id','label'}
    if required-set(manifest):raise ValueError(f'Split manifest lacks {required-set(manifest)}')
    df=manifest.copy()
    if df[list(required)].isna().any().any():raise ValueError('Missing split-manifest identity or label.')
    for c in ['task_folder','role','sample_id','participant_id']:
        df[c]=df[c].astype(str).str.strip()
    for c in ['repeat','fold','split_seed']:
        raw=pd.to_numeric(df[c],errors='raise')
        if not np.isfinite(raw).all() or not np.equal(raw,np.floor(raw)).all():
            raise ValueError('Nonintegral split identifier.')
        df[c]=raw.astype(np.int64)
    df['label']=df['label'].map(normalise_label)
    if set(df.task_folder)!=set(TASK_ORDER):raise ValueError('Expected exactly the five joint tasks.')
    if set(df.role)!={'train','test'}:raise ValueError('Unknown train/test role.')
    if df.duplicated(['task_folder','repeat','fold','sample_id']).any():
        raise ValueError('Duplicate or overlapping sample rows in split manifest.')
    plan=[]; checks=[]
    for taskname in TASK_ORDER:
        task=TASK_BY_FOLDER[taskname]
        global_rows=np.flatnonzero(mask_for_task(inputs.labels3,task))
        sid=np.asarray(inputs.sample_ids)[global_rows].astype(str)
        pid=np.asarray(inputs.participant_ids)[global_rows].astype(str)
        y=labels_for_task(inputs.labels3[global_rows],task)
        master=pd.DataFrame({'sample_id':sid,'participant_id':pid,'label':y})
        if master.groupby('participant_id').label.nunique().max()!=1:
            raise ValueError('Conflicting participant diagnoses.')
        task_df=df[df.task_folder==taskname]
        if set(task_df['repeat'])!=set(range(1,settings.repeats+1)):
            raise ValueError('Unexpected repetition inventory.')
        for r in range(1,settings.repeats+1):
            rdf=task_df[task_df['repeat']==r]
            if set(rdf.fold)!=set(range(1,settings.outer_folds+1)) or rdf.split_seed.nunique()!=1:
                raise ValueError('Unexpected outer folds or seeds.')
            seed=int(rdf.split_seed.iloc[0]); seen_test=[]
            generated=list(StratifiedGroupKFold(n_splits=settings.outer_folds,shuffle=True,
                random_state=seed).split(np.zeros(len(y)),y,pid))
            for f in range(1,settings.outer_folds+1):
                fdf=rdf[rdf.fold==f].set_index('sample_id')
                if set(fdf.index)!=set(sid):raise ValueError('Outer-fold sample inventory differs.')
                aligned=fdf.loc[sid]
                if not np.array_equal(aligned.participant_id.to_numpy(),pid):raise ValueError('Participant mapping differs.')
                if not np.array_equal(aligned.label.to_numpy(),y):raise ValueError('Task label mapping differs.')
                a=np.flatnonzero(aligned.role.to_numpy()=='train')
                b=np.flatnonzero(aligned.role.to_numpy()=='test')
                if set(pid[a]) & set(pid[b]):raise ValueError('Outer participant leakage.')
                if not np.array_equal(a,generated[f-1][0]) or not np.array_equal(b,generated[f-1][1]):
                    raise ValueError('Current grouped splitter differs from original locked folds. '
                                     'Use the original joint virtual environment; do not regenerate splits.')
                require_split_classes(y,a,task.classes,'Outer training')
                require_split_classes(y,b,task.classes,'Outer test')
                inner=[]
                for ia,ib in StratifiedGroupKFold(n_splits=settings.inner_folds,shuffle=True,
                    random_state=seed+100*f).split(np.zeros(len(a)),y[a],pid[a]):
                    ia=a[ia]; ib=a[ib]
                    if set(pid[ia]) & set(pid[ib]):raise ValueError('Inner participant leakage.')
                    require_split_classes(y,ia,task.classes,'Inner training')
                    require_split_classes(y,ib,task.classes,'Inner validation')
                    inner.append((global_rows[ia],global_rows[ib]))
                if reference is not None:
                    path=reference/'runs'/taskname/f'repeat_{r:02d}'/taskname/f'fold_{f}'/'participant_group_split.csv'
                    old=pd.read_csv(path,low_memory=False)
                    old['sample_id']=old.sample_id.astype(str).str.strip()
                    old['participant_id']=old.participant_id.astype(str).str.strip()
                    old['label']=old.label.map(normalise_label)
                    if old.sample_id.duplicated().any():raise ValueError(f'Duplicate original split rows: {path}')
                    if set(old.sample_id)!=set(sid):raise ValueError(f'Original fitted split sample set differs: {path}')
                    old=old.set_index('sample_id').loc[sid]
                    if not np.array_equal(old['split'].to_numpy(),aligned.role.to_numpy()) or not np.array_equal(old.participant_id.to_numpy(),pid) or not np.array_equal(old.label.to_numpy(),y):
                        raise ValueError(f'Original fitted split differs from locked manifest: {path}')
                fold=Fold(taskname,r,f,seed,global_rows[a],global_rows[b],tuple(inner))
                plan.append(fold); seen_test.extend(b.tolist())
                checks.append({'task':taskname,'repeat':r,'fold':f,'seed':seed,
                    'train_samples':len(a),'test_samples':len(b),'split_sha256':fold.identity(),
                    'outer_exact':True,'inner_all_classes':True})
            if sorted(seen_test)!=list(range(len(y))):raise ValueError('Not exactly one held-out prediction per repetition.')
    return plan,pd.DataFrame(checks)


def preflight(reference,output,settings,log=print):
    reference=Path(reference).expanduser().resolve();output=Path(output).expanduser().resolve()
    cfg_path=reference/'orchestrator_config.json';pre_path=reference/'preflight/preflight_summary.json'
    mf_path=reference/'splits/all_task_split_manifest.csv'
    required=[cfg_path,pre_path,mf_path,reference/'preflight/input_fingerprints.json']
    for p in required:
        if not p.is_file():raise FileNotFoundError('Required original run provenance missing: '+str(p))
    cfg=read_json(cfg_path);oldver=read_json(pre_path)['software']['core_virtual_environment']
    now=runtime_versions()
    if oldver!=now:
        raise RuntimeError(f'Original joint virtual environment required; no fitting. Recorded={oldver}, current={now}')
    feature=Path(cfg['ricci_feature_dir']).expanduser().resolve()
    pd_dir=Path(cfg['h0_pd_dir']).expanduser().resolve()
    if output==reference or output in (feature,pd_dir) or reference in output.parents:
        raise ValueError('Output must be a separate new directory, outside original inputs/results.')
    core=Path(cfg['joint_repo']).expanduser().resolve()
    locks=cfg.get('source_lock',{}).get('core_source_fingerprints',{})
    if not locks:raise ValueError('Missing original source lock.')
    for rel,h in locks.items():
        if sha256_file(core/rel)!=h:raise ValueError(f'Original core source changed: {rel}')
    for name in ('data.py','tasks.py'):
        if sha256_file(core/'src/h0_ricci_joint_sparse'/name)!=sha256_file(ROOT/'vendor/original_core'/name):
            raise ValueError('Input parser differs from recorded core: '+name)
    matrix=feature/'feature_matrix_B_K0.npz';log('Checking original matrix hash, diagrams and cohort identities.')
    h=sha256_file(matrix)
    if h!=MATRIX_SHA256:raise ValueError('Saved matrix does not match the audited source SHA-256.')
    raw=sparse.load_npz(matrix)
    if not np.isfinite(raw.data).all():raise ValueError('Nonfinite input matrix; refusing automatic imputation.')
    del raw
    X,meta,cols=core_data.load_ricci_matrix(feature)
    sid=meta['_sample_id'].to_numpy();pid=meta['_participant_id'].to_numpy();labels=meta['_label3'].to_numpy()
    deaths,paths=core_data.load_h0_for_samples(pd_dir,sid)
    inputs=core_data.LoadedInputs(X,meta,sid,pid,labels,deaths,paths,pd.DataFrame(),cols,pd_dir)
    if X.shape!=(1317,33864) or len(set(pid))!=106:raise ValueError('Unexpected source cohort shape.')
    measured=core_input_fingerprints(inputs)
    if measured!=read_json(reference/'preflight/input_fingerprints.json'):
        raise ValueError('Original loaded H0, Ricci or metadata fingerprint differs.')
    manifest=pd.read_csv(mf_path,low_memory=False)
    log('Checking 500 locked outer folds and reproducing 1,500 grouped inner partitions.')
    plan,checks=plan_from_manifest(inputs,manifest,settings,reference)
    watched=list(dict.fromkeys(required+[matrix,feature/'matched_metadata.csv']+[Path(p) for p in paths]+
        [reference/'runs'/f.task/f'repeat_{f.repeat:02d}'/f.task/f'fold_{f.number}'/'participant_group_split.csv' for f in plan]))
    stats={str(p):stat_record(p) for p in watched}
    hashes={str(p):sha256_file(p) for p in watched}
    check_stats(stats)
    mem=0
    if Path('/proc/meminfo').exists():
        mem=next((int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines()
                  if x.startswith('MemAvailable:')),0)
    if mem and mem<3*1024**3:raise MemoryError('Less than 3 GiB available; no fitting started.')
    output.parent.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(output.parent).free<10*1024**3:raise OSError('Less than 10 GiB free disk; no fitting started.')
    from .basis import filter_deaths
    filtered=[filter_deaths(d) for d in inputs.h0_deaths]
    filtered_hash=hashlib.sha256()
    for sid,d in zip(inputs.sample_ids,filtered):
        b=str(sid).encode();filtered_hash.update(len(b).to_bytes(8,'little'));filtered_hash.update(b)
        update_array_digest(filtered_hash,d)
    h0_filter_summary={'original_positive_finite_deaths':sum(map(len,inputs.h0_deaths)),
                       'eligible_deaths':sum(map(len,filtered)),
                       'empty_after_boundary_filter':sum(len(d)==0 for d in filtered),
                       'filtered_deaths_sha256':filtered_hash.hexdigest(),
                       'rule':'finite 0 < d < 1-1e-12; repetitions retained'}
    identity={'schema':SCHEMA,'settings':settings.payload(),'code':package_fingerprint(),'versions':now,
              'input_fingerprints':measured,'matrix_sha256':h,'manifest_sha256':sha256_file(mf_path),
              'output':str(output),'reference':str(reference),'input_hashes':hashes}
    payload={**identity,'id':fingerprint(identity),'outer_folds':len(plan),'inner_folds':sum(len(f.inner) for f in plan),
             'original_cohort_counts':{'samples':1317,'participants':106,'ricci_columns':33864},
             'inner_provenance':'Reproduced from original row order, seed+100*outer_fold and unchanged sklearn. '
                                'Old inner memberships were not saved. New memberships are saved explicitly.',
             'memory_available_bytes':mem,'h0_filter_summary':h0_filter_summary}
    return inputs,plan,checks,payload,stats

def check_stats(stats):
    for p,s in stats.items():
        if stat_record(Path(p))!=s:raise RuntimeError('Consumed input changed: '+p)
