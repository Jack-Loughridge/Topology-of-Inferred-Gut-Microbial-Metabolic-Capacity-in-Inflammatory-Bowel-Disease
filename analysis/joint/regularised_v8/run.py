#!/usr/bin/env python3
"""Run the agreed small regularised IBD experiment. No new approval stage."""
import os,sys
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
sys.dont_write_bytecode=True
from pathlib import Path
import argparse,contextlib,fcntl,json,traceback,time
from audit.inputs import preflight,check_stats
from audit.settings import Settings as HistoricalAuditSettings
from audit.io import read_json,atomic_json,sha256_file,fingerprint,canonical_json,exclusive_lock
from robust.config import Config,specification,SCHEMA
from robust.workflow import chosen_folds,run_schedule,aggregate,export_review,log
from verify_package import verify

ROOT=Path(__file__).resolve().parent
OLD=('H0_Ricci_StandaloneAlpha_C002_v4','H0_Ricci_V4_CoupledGates_1','H0_Ricci_FirstGate_Stability_1',
     'H0_Ricci_StagedCV_v5','H0_Ricci_TensorCV_v6','H0_Ricci_IBD_FixedM32_v7')

def safe_output(path,protected):
    p=Path(path).expanduser()
    if p.is_symlink():raise ValueError('Output must not be a symlink.')
    p=p.resolve()
    for other in protected:
        other=Path(other).expanduser().resolve()
        if p==other or p in other.parents or other in p.parents:
            raise ValueError('Output overlaps a protected input, source or historical result directory.')
    if p.exists() and any(p.iterdir()):
        marker=p/'IDENTITY.json'
        if not marker.is_file():marker=p/'V8_DIRECTORY_RECEIPT.json'
        if not marker.is_file() or read_json(marker).get('schema')!=SCHEMA:
            raise ValueError('Refusing to use an unknown populated output directory.')
    return p

@contextlib.contextmanager
def old_lock(path):
    if path.is_symlink():raise ValueError('Historical lock is a symlink.')
    if not path.is_file():yield;return
    with path.open('r') as f:
        fcntl.flock(f,fcntl.LOCK_SH|fcntl.LOCK_NB)
        try:yield
        finally:fcntl.flock(f,fcntl.LOCK_UN)

def main(argv=None):
    base=Path.home()/'Real_Data'
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--reference',type=Path,default=base/'H0_Ricci_JointSparse_RepeatedCV_AllTasks')
    ap.add_argument('--output',type=Path,default=base/'H0_Ricci_IBD_Regularised_v8')
    ap.add_argument('--workers',type=int,default=2)
    args=ap.parse_args(argv)
    if args.workers not in (1,2,3):ap.error('Use 1, 2 or 3 workers, unchanged across resumes.')
    version=verify();cfg=Config();spec=specification(cfg);spec['workers']=args.workers
    ref=args.reference.expanduser().resolve();base=ref.parent
    old=read_json(ref/'orchestrator_config.json')
    protected=[ROOT,ref,*[base/n for n in OLD]]
    protected += [Path(old[k]).expanduser() for k in ('joint_repo','ricci_feature_dir','h0_pd_dir','h0_results_dir','ricci_original_dir') if isinstance(old.get(k),str)]
    out=safe_output(args.output,protected)
    if (out/'IDENTITY.json').is_file() and read_json(out/'IDENTITY.json')['specification']['workers']!=args.workers:
        raise ValueError('Keep the worker count unchanged for this output directory.')
    with contextlib.ExitStack() as stack:
        for name in OLD:
            for mark in ('RUN.lock','GATES.lock'):stack.enter_context(old_lock(base/name/mark))
        out.mkdir(parents=True,exist_ok=True)
        stack.enter_context(exclusive_lock(out/'RUN.lock'))
        directory_receipt={'schema':SCHEMA,'reference':str(ref),'package':version,'specification':spec}
        dr=out/'V8_DIRECTORY_RECEIPT.json'
        if dr.exists() and canonical_json(read_json(dr))!=canonical_json(directory_receipt):
            raise ValueError('Directory belongs to a different v8 configuration or source package.')
        atomic_json(dr,directory_receipt)
        data=plan=run_id=payload=None
        try:
            log('Checking original source data and locked partitions. No historical fitted models are loaded.')
            data,original_plan,checks,payload,stats=preflight(ref,out,HistoricalAuditSettings(),lambda s:log('INPUT AUDIT: '+s))
            plan=chosen_folds(data,original_plan)
            idinfo={'schema':SCHEMA,'specification':spec,'package':version,'source_identity':payload['id'],
                   'folds':[f.identity() for f in plan],'reference':str(ref),'output':str(out)}
            run_id=fingerprint(idinfo);idinfo['id']=run_id
            existing=out/'IDENTITY.json'
            if existing.exists() and canonical_json(read_json(existing))!=canonical_json(idinfo):
                raise RuntimeError('Source, configuration or run identity changed. No incompatible resume.')
            atomic_json(existing,idinfo);atomic_json(out/'SCIENTIFIC_SPECIFICATION.json',spec)
            atomic_json(out/'SOURCE_PROVENANCE_PRIVATE.json',payload)
            atomic_json(out/'PREFLIGHT_SUMMARY.json',{'schema':SCHEMA,'matrix_sha256':payload['matrix_sha256'],
                'source_counts':payload['original_cohort_counts'],'versions':payload['versions'],
                'selected_outer_folds':25,'repetitions':[1,2,3,4,5],'requested_M':32,
                'workers':args.workers,'h0_ridge':cfg.h0_ridge,'no_inner_fitting':True})
            log('V8 PREFLIGHT PASSED. Fixed M32, five IBD repetitions, explicit H0 ridge=0.001.')
            result=run_schedule(data,plan,out,cfg,run_id,args.workers)
            check_stats(stats)
            for p,h in payload['input_hashes'].items():
                if sha256_file(Path(p))!=h:raise RuntimeError('Consumed source file changed.')
            if verify()!=version:raise RuntimeError('Package changed during fitting.')
            atomic_json(out/'END_INTEGRITY.json',{'passed':True,'source_files_unchanged':True,'package_unchanged':True})
            atomic_json(out/'FINAL_REPORT.json',result);atomic_json(out/'STATUS.json',result);export_review(out)
            log(result['status']+'. Review archive and checksum saved.')
            return 0
        except KeyboardInterrupt:
            if data is not None and plan is not None and run_id:
                report=aggregate(out,plan,data,run_id);report['status']='INTERRUPTED'
                atomic_json(out/'FINAL_REPORT.json',report);atomic_json(out/'STATUS.json',report);export_review(out)
            log('INTERRUPTED. Completed pairs and same-fit checkpoints retained.');return 130
        except BaseException as exc:
            report={'status':'INPUT_OR_EXECUTION_FAILURE','type':type(exc).__name__,'message':str(exc),
                    'analysis_complete':False}
            atomic_json(out/'FAILURE_SUMMARY.json',report)
            # Full traceback is private and excluded from the review archive.
            atomic_json(out/'ERROR_PRIVATE.json',{'traceback':traceback.format_exc()})
            atomic_json(out/'STATUS.json',report)
            if data is not None and plan is not None and run_id:
                summ=aggregate(out,plan,data,run_id);atomic_json(out/'FINAL_REPORT.json',{**summ,**report})
            export_review(out);log('STOPPED: '+str(exc));traceback.print_exc();return 2
if __name__=='__main__':
    try:sys.exit(main())
    except Exception as exc:
        print('STARTUP STOPPED: '+str(exc),flush=True);traceback.print_exc();sys.exit(2)
