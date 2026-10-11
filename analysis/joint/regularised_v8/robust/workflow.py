"""IBD only, original first five grouped repetitions. Labels never enter prediction."""
from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor,as_completed,wait,FIRST_COMPLETED
from dataclasses import replace
import hashlib,json,multiprocessing,os,shutil,tarfile,time,traceback
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score,roc_auc_score,log_loss,confusion_matrix
from audit.io import atomic_json,atomic_csv,atomic_npz,read_json,sha256_file,fingerprint,finish,completed
from audit.basis import fit_intervals,tensor_many,Intervals
from .model import Problem,FittedModel,State
from .config import Config,SCHEMA,TASK,CLASSES

_GLOBAL={}

def log(s):print(time.strftime('%Y-%m-%d %H:%M:%S')+' '+str(s),flush=True)

def chosen_folds(data,original_plan,repeats=5,folds=5):
    plan=[replace(f,inner=()) for f in original_plan if f.task==TASK and 1<=f.repeat<=repeats]
    expected={(r,k) for r in range(1,repeats+1) for k in range(1,folds+1)}
    if len(plan)!=len(expected) or {(f.repeat,f.number) for f in plan}!=expected:
        raise ValueError('Missing or duplicate locked IBD partitions.')
    eligible=set(range(len(data.sample_ids)))
    for r in range(1,repeats+1):
        seen=[]
        for f in (x for x in plan if x.repeat==r):
            tr,te=list(f.train),list(f.test)
            if set(tr)&set(te) or set(tr)|set(te)!=eligible or len(tr)!=len(set(tr)) or len(te)!=len(set(te)):
                raise ValueError('Invalid original partition.')
            if set(data.participant_ids[f.train])&set(data.participant_ids[f.test]):
                raise ValueError('Participant overlap.')
            if not np.array_equal(f.train,np.sort(f.train)) or not np.array_equal(f.test,np.sort(f.test)):
                raise ValueError('Original row order changed.')
            for rows in (f.train,f.test):
                if set((np.asarray(data.labels3)[rows]!='nonIBD').astype(int))!={0,1}:
                    raise ValueError('Partition lacks a class.')
            seen.extend(te)
        if sorted(seen)!=sorted(eligible):raise ValueError('Not exactly one test prediction per repetition.')
    return sorted(plan,key=lambda x:(x.repeat,x.number))

def folder_for(out,fold):return Path(out)/'jobs'/f'repeat_{fold.repeat:02d}'/f'fold_{fold.number}'

def validate_probabilities(y,p):
    y=np.asarray(y);p=np.asarray(p,float)
    if p.shape!=(len(y),2) or not np.isfinite(p).all() or np.any((p<0)|(p>1)) or not np.allclose(p.sum(1),1,rtol=0,atol=1e-10):
        raise ValueError('Invalid probability table.')
    if set(y)!={0,1}:raise ValueError('Both evaluation classes required.')

def metrics(y,p):
    validate_probabilities(y,p);pred=np.argmax(p,axis=1)
    return {'accuracy':float(accuracy_score(y,pred)),
      'balanced_accuracy':float(balanced_accuracy_score(y,pred)),
      'macro_f1':float(f1_score(y,pred,average='macro',zero_division=0)),
      'roc_auc':float(roc_auc_score(y,p[:,1])),
      'log_loss':float(log_loss(y,p,labels=[0,1]))}

def participant_table(df):
    if df.groupby('participant_id').y.nunique().max()!=1:raise ValueError('Conflicting participant labels.')
    return df.groupby('participant_id',sort=True,as_index=False).agg(y=('y','first'),p0=('p0','mean'),p1=('p1','mean'))

def pair_worker(f):
    data,out,cfg,run_id=_GLOBAL['data'],_GLOBAL['out'],_GLOBAL['config'],_GLOBAL['id']
    job=folder_for(out,f);job.mkdir(parents=True,exist_ok=True)
    pid=fingerprint({'run':run_id,'fold':f.identity()})
    if completed(job,pid):return read_json(job/'report.json')
    prefix=f'R{f.repeat:02d} F{f.number}'
    def info(s):
        log(prefix+' '+s)
        atomic_json(job/'STATUS.json',{'repeat':f.repeat,'fold':f.number,'pid':os.getpid(),'message':str(s),'time':time.time()})
    tr=np.asarray(f.train);te=np.asarray(f.test);t=time.monotonic()
    raw=data.ricci[tr].toarray() if sparse.issparse(data.ricci) else np.asarray(data.ricci)[tr].copy()
    y=(np.asarray(data.labels3)[tr]!='nonIBD').astype(np.int64)
    test_R=data.ricci[te].toarray() if sparse.issparse(data.ricci) else np.asarray(data.ricci)[te].copy()
    train_d=[data.h0_deaths[i] for i in tr];test_d=[data.h0_deaths[i] for i in te]
    basis=job/'basis';basis.mkdir(exist_ok=True)
    bid=fingerprint({'fold':pid,'basis':cfg.requested_intervals,'rule':'source_before_original_axis_integration'})
    if completed(basis,bid):
        ivd=read_json(basis/'intervals.json');iv=Intervals(np.array(ivd['bounds']),ivd['sigma'],ivd['requested'])
        with np.load(basis/'train_tensor.npz',allow_pickle=False) as z:A=z['A'].copy()
    else:
        info('building training-only H0 intervals and tensor')
        iv=fit_intervals(train_d,cfg.requested_intervals,cfg.interval_min_gap)
        A=tensor_many(train_d,iv,cfg.quadrature_points,info)
        atomic_npz(basis/'train_tensor.npz',A=A);atomic_json(basis/'intervals.json',iv.payload())
        finish(basis,bid,['train_tensor.npz','intervals.json'])
    report={'repeat':f.repeat,'fold':f.number,'task':TASK,'train_samples':len(tr),'test_samples':len(te),
        'train_participants':len(set(data.participant_ids[tr])),'test_participants':len(set(data.participant_ids[te])),
        'requested_M':cfg.requested_intervals,'realised_M':iv.size,'models':{}}
    for arm in ('joint','ricci'):
        d=job/arm;d.mkdir(exist_ok=True);aid=fingerprint({'pair':pid,'arm':arm})
        if completed(d,aid):report['models'][arm]=read_json(d/'numerical.json');continue
        info('fitting '+arm+' model')
        aa=A if arm=='joint' else np.empty((len(tr),0,0))
        problem=Problem(aa,raw,y,cfg)
        cp=d/'CHECKPOINT.npz';resume=None
        if cp.is_file():
            with np.load(cp,allow_pickle=False) as z:
                meta=json.loads(str(z['metadata'].item()))
                if meta['identity']!=aid:raise ValueError('Same-fit checkpoint identity mismatch.')
                resume=(State(float(z['intercept']),z['h0'].copy(),z['alpha'].copy(),z['ricci'].copy()),meta['progress'])
        def save_checkpoint(state,progress):
            atomic_npz(cp,intercept=np.array(state.intercept),h0=state.h0,alpha=state.alpha,ricci=state.ricci,
                       metadata=np.array(json.dumps({'identity':aid,'progress':progress})))
            atomic_json(d/'training_history_progress.json',progress['history'])
        model=problem.fit(lambda s:info(arm+': '+s),save_checkpoint,resume)
        model.save(d/'model.npz',aid)
        saved=FittedModel.load(d/'model.npz',aid)
        # No held-out labels are passed to tensor generation or prediction.
        info('predicting '+arm+' on held-out observations')
        at=tensor_many(test_d,iv,cfg.quadrature_points) if arm=='joint' else np.empty((len(te),0,0))
        probs,scores,pcheck=saved.predict(at,test_R)
        nb=min(16,len(te));p2,z2,_=model.predict(at[:nb],test_R[:nb])
        if not np.array_equal(p2,probs[:nb]) or not np.array_equal(z2,scores[:nb]):
            raise RuntimeError('Saved-model round trip differs.')
        pcheck.update(saved_reload_equal=True,samples=len(te),classes=list(CLASSES),uses_saved_training_transforms=True)
        truth=(np.asarray(data.labels3)[te]!='nonIBD').astype(int)
        frame=pd.DataFrame({'sample_id':np.asarray(data.sample_ids)[te],
            'participant_id':np.asarray(data.participant_ids)[te],'y':truth,'p0':probs[:,0],'p1':probs[:,1]})
        validate_probabilities(truth,probs)
        atomic_csv(d/'predictions_PRIVATE.csv',frame)
        diag=dict(model.diagnostics);hist=diag.pop('history')
        atomic_json(d/'numerical.json',diag);atomic_json(d/'training_history.json',hist)
        atomic_json(d/'prediction_checks.json',pcheck)
        finish(d,aid,['model.npz','numerical.json','training_history.json','predictions_PRIVATE.csv','prediction_checks.json'])
        report['models'][arm]=diag
        info(arm+' fitted: '+diag['status'])
    report['elapsed_seconds']=time.monotonic()-t
    report['both_converged']=all(x['converged'] for x in report['models'].values())
    atomic_json(job/'report.json',report)
    atomic_json(job/'intervals.json',iv.payload())
    finish(job,pid,['report.json','intervals.json','joint/COMPLETE.json','ricci/COMPLETE.json'])
    info('PAIR COMPLETE'+('' if report['both_converged'] else ' WITH OPTIMISATION WARNING'))
    return report

def validate_pair(out,f,run_id):
    d=folder_for(out,f);pid=fingerprint({'run':run_id,'fold':f.identity()})
    if not completed(d,pid):return False
    for arm in ('joint','ricci'):
        if not completed(d/arm,fingerprint({'pair':pid,'arm':arm})):return False
    return True

def aggregate(out,plan,data,run_id):
    out=Path(out);available=[f for f in plan if validate_pair(out,f,run_id)]
    warnings=[];diag=[]
    for f in available:
        rep=read_json(folder_for(out,f)/'report.json')
        for arm,d in rep['models'].items():
            diag.append(dict(repeat=f.repeat,fold=f.number,model=arm,**d))
            if not d['converged']:warnings.append({'repeat':f.repeat,'fold':f.number,'model':arm,
                'head_kkt':d['head_kkt'],'alpha_gap':d['alpha_conditional_gap'],'stop_reason':d['stop_reason']})
    atomic_csv(out/'NUMERICAL_SUMMARY.csv',pd.DataFrame(diag))
    base={'schema':SCHEMA,'completed_pairs':len(available),'required_pairs':len(plan),
          'analysis_complete':len(available)==len(plan),'optimisation_warnings':warnings,
          'all_fits_converged':len(available)==len(plan) and not warnings,
          'original_unpenalised_model':False,'new_h0_ridge':_GLOBAL.get('config',Config()).h0_ridge,
          'no_fold_omitted':True,'no_results_selected_for_performance':True}
    dest=out/'aggregate';dest.mkdir(exist_ok=True)
    if len(available)!=len(plan):
        atomic_json(dest/'completeness.json',base);return base
    rows=[];conf=[];all_predictions={}
    repeats=sorted(set(f.repeat for f in plan))
    expected=set(map(str,data.sample_ids))
    for arm in ('joint','ricci'):
        all_predictions[arm]=[]
        for r in repeats:
            pieces=[pd.read_csv(folder_for(out,f)/arm/'predictions_PRIVATE.csv',dtype={'sample_id':str,'participant_id':str})
                    for f in plan if f.repeat==r]
            frame=pd.concat(pieces,ignore_index=True)
            if frame.sample_id.duplicated().any() or set(frame.sample_id)!=expected:
                raise ValueError('Repeated prediction inventory differs from the complete source cohort.')
            master=pd.DataFrame({'sample_id':np.asarray(data.sample_ids).astype(str),
              'participant_id':np.asarray(data.participant_ids).astype(str),
              'y':(np.asarray(data.labels3)!='nonIBD').astype(int)}).set_index('sample_id')
            aligned=frame.set_index('sample_id').loc[master.index]
            if not np.array_equal(aligned.y.to_numpy(),master.y.to_numpy()) or not np.array_equal(aligned.participant_id.to_numpy(),master.participant_id.to_numpy()):
                raise ValueError('Saved prediction label/participant mapping differs.')
            all_predictions[arm].append(frame.assign(repeat=r))
            for level,df in [('sample',frame),('participant',participant_table(frame))]:
                mm=metrics(df.y.to_numpy(),df[['p0','p1']].to_numpy())
                for metric,v in mm.items():rows.append({'model':arm,'repeat':r,'level':level,'metric':metric,'value':v,'n':len(df)})
                cm=confusion_matrix(df.y,np.argmax(df[['p0','p1']].to_numpy(),axis=1),labels=[0,1])
                conf.append({'model':arm,'repeat':r,'level':level,'TN':int(cm[0,0]),'FP':int(cm[0,1]),'FN':int(cm[1,0]),'TP':int(cm[1,1])})
    df=pd.DataFrame(rows);atomic_csv(dest/'repetition_metrics.csv',df)
    sums=df.groupby(['model','level','metric'],sort=True).value.agg(['mean','std','count']).reset_index()
    atomic_csv(dest/'performance_summary.csv',sums)
    pivot=df.pivot(index=['repeat','level','metric'],columns='model',values='value')
    delta=(pivot['joint']-pivot['ricci']).rename('joint_minus_ricci').reset_index()
    atomic_csv(dest/'paired_differences.csv',delta)
    atomic_csv(dest/'paired_summary.csv',delta.groupby(['level','metric']).joint_minus_ricci.agg(['mean','std','count']).reset_index())
    atomic_csv(dest/'repetition_confusions.csv',pd.DataFrame(conf))
    consensus=[]
    for arm,frames in all_predictions.items():
        raw=pd.concat(frames,ignore_index=True)
        s=raw.groupby(['sample_id','participant_id','y'],as_index=False)[['p0','p1']].mean()
        for level,ff in [('sample',s),('participant',participant_table(s))]:
            mm=metrics(ff.y.to_numpy(),ff[['p0','p1']].to_numpy())
            cm=confusion_matrix(ff.y,np.argmax(ff[['p0','p1']].to_numpy(),axis=1),labels=[0,1])
            consensus.append({'model':arm,'level':level,'n':len(ff),**mm,'TN':int(cm[0,0]),'FP':int(cm[0,1]),'FN':int(cm[1,0]),'TP':int(cm[1,1])})
    atomic_csv(dest/'consensus_SEPARATE.csv',pd.DataFrame(consensus))
    base['status']='COMPLETE_ALL_CONVERGED' if not warnings else 'COMPLETE_WITH_OPTIMISATION_WARNINGS'
    atomic_json(dest/'completeness.json',base);return base

def export_review(out):
    """Strict allowlist. No model, training data, identifiers or individual predictions."""
    out=Path(out);names=['FINAL_REPORT.json','STATUS.json','SCIENTIFIC_SPECIFICATION.json','PREFLIGHT_SUMMARY.json',
                      'NUMERICAL_SUMMARY.csv','END_INTEGRITY.json','FAILURE_SUMMARY.json']
    files=[out/n for n in names if (out/n).is_file()]
    files.extend(p for p in sorted((out/'aggregate').glob('*')) if p.is_file())
    for p in sorted((out/'jobs').glob('repeat_*/fold_*/*/numerical.json')):files.append(p)
    for p in sorted((out/'jobs').glob('repeat_*/fold_*/*/training_history.json')):files.append(p)
    for p in sorted((out/'jobs').glob('repeat_*/fold_*/*/training_history_progress.json')):files.append(p)
    for p in sorted((out/'jobs').glob('repeat_*/fold_*/*/prediction_checks.json')):files.append(p)
    import io
    manifest={str(p.relative_to(out)):sha256_file(p) for p in files}
    dest=out/'joint_ibd_regularised_v8_review.tar.gz';tmp=dest.with_suffix('.tmp')
    with tarfile.open(tmp,'w:gz') as tar:
        for p in files:tar.add(p,arcname=str(p.relative_to(out)),recursive=False)
        b=(json.dumps(manifest,indent=2)+'\n').encode();ti=tarfile.TarInfo('MANIFEST.json');ti.size=len(b);tar.addfile(ti,io.BytesIO(b))
    os.replace(tmp,dest)
    (out/(dest.name+'.sha256')).write_text(sha256_file(dest)+'  '+dest.name+'\n')
    return dest

def run_schedule(data,plan,out,config,run_id,workers=2):
    out=Path(out)
    _GLOBAL.update(data=data,out=out,config=config,id=run_id)
    pending=[f for f in plan if not validate_pair(out,f,run_id)]
    def record():
        a=aggregate(out,plan,data,run_id);atomic_json(out/'STATUS.json',a);export_review(out)
    if not pending:return aggregate(out,plan,data,run_id)
    # First complete pair verifies new outer-training construction and prediction.
    first=plan[0]
    if not validate_pair(out,first,run_id):
        pair_worker(first);record();pending=[f for f in pending if f.key!=first.key]
        log('FIRST OUTER PAIR SAVED. Continuing the remaining fixed evaluation automatically.')
    if workers==1:
        for f in pending:pair_worker(f);record()
    elif pending:
        pool=ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('fork'))
        futures={pool.submit(pair_worker,f):f for f in pending}
        try:
            remaining=set(futures)
            while remaining:
                ready,remaining=wait(remaining,timeout=15,return_when=FIRST_COMPLETED)
                if not ready:log(f'HEARTBEAT: {len(remaining)} fold pairs remaining or running')
                for ft in ready:
                    ft.result();record()
        except BaseException:
            for fu in futures:fu.cancel()
            for proc in pool._processes.values():
                if proc.is_alive():proc.terminate()
            pool.shutdown(wait=True,cancel_futures=True)
            raise
        else:pool.shutdown(wait=True)
    return aggregate(out,plan,data,run_id)
