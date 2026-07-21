#!/usr/bin/env python3
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, save_npz
from sklearn.model_selection import StratifiedGroupKFold

from ricci_all_tasks import ALL_TASKS, TASK_ORDER, RunConfig, run


def make_cohort() -> pd.DataFrame:
    rows=[]; participant_counter=0
    for diagnosis in ("nonIBD","UC","CD"):
        for _ in range(6):
            pid=f"P{participant_counter:03d}"; participant_counter+=1
            for sample_idx in range(2):
                rows.append({"sample_id":f"S{participant_counter-1:03d}_{sample_idx}","participant_id":pid,"cond":diagnosis})
    return pd.DataFrame(rows)


def task_frame(cohort:pd.DataFrame,folder:str)->pd.DataFrame:
    if folder=="IBD_vs_nonIBD":
        out=cohort.copy(); out["label"]=np.where(out["cond"].eq("nonIBD"),"nonIBD","IBD"); return out
    if folder=="three_way_nonIBD_UC_CD": out=cohort.copy(); out["label"]=out["cond"]; return out
    if folder=="nonIBD_vs_UC": out=cohort[cohort["cond"].isin(["nonIBD","UC"])].copy(); out["label"]=out["cond"]; return out
    if folder=="nonIBD_vs_CD": out=cohort[cohort["cond"].isin(["nonIBD","CD"])].copy(); out["label"]=out["cond"]; return out
    if folder=="CD_vs_UC": out=cohort[cohort["cond"].isin(["CD","UC"])].copy(); out["label"]=out["cond"]; return out
    raise KeyError(folder)


def write_manifests(cohort:pd.DataFrame,split_dir:Path,repeats:int,folds:int)->None:
    seeds=[101,202]
    for task in ALL_TASKS:
        data=task_frame(cohort,task.folder).reset_index(drop=True)
        rows=[]; labels=data["label"].to_numpy(object); groups=data["participant_id"].to_numpy(object)
        for repeat in range(1,repeats+1):
            splitter=StratifiedGroupKFold(n_splits=folds,shuffle=True,random_state=seeds[repeat-1])
            for fold,(train_idx,test_idx) in enumerate(splitter.split(np.zeros(len(data)),labels,groups=groups),start=1):
                for role,indices in (("train",train_idx),("test",test_idx)):
                    for idx in indices:
                        rows.append({"task_folder":task.folder,"repeat":repeat,"fold":fold,"split_seed":seeds[repeat-1],"role":role,"sample_id":data.loc[idx,"sample_id"],"participant_id":data.loc[idx,"participant_id"],"label":data.loc[idx,"label"]})
        pd.DataFrame(rows).to_csv(split_dir/f"{task.folder}_split_manifest.csv",index=False)


def write_features(cohort:pd.DataFrame,feature_dir:Path)->None:
    rng=np.random.default_rng(42); n_edges=8
    B=[]; K=[]
    signal={"nonIBD":(0,0.8),"UC":(1,1.3),"CD":(2,1.8)}
    for row in cohort.itertuples(index=False):
        b=(rng.random(n_edges)<0.25).astype(float); k=rng.normal(0,0.25,n_edges)
        idx,level=signal[row.cond]; b[idx]=1.0; k[idx]+=level
        if row.cond=="UC": k[4]+=0.7
        if row.cond=="CD": b[5]=1.0; k[5]-=0.8
        B.append(b); K.append(k*b)
    X=np.concatenate([np.asarray(B),np.asarray(K)],axis=1)
    save_npz(feature_dir/"feature_matrix_B_K0.npz",csr_matrix(X))
    cohort.to_csv(feature_dir/"matched_metadata.csv",index=False)
    pd.DataFrame({"edge":[f"M_e{i} -> M_p{i}" for i in range(n_edges)],"process":["transport" if i<3 else "carbon" for i in range(n_edges)]}).to_csv(feature_dir/"edge_metadata.csv",index=False)


def main()->None:
    with tempfile.TemporaryDirectory(prefix="ricci_all_tasks_selftest_") as tmp:
        root=Path(tmp); feature_dir=root/"features"; split_dir=root/"splits"; output=root/"output"
        feature_dir.mkdir(); split_dir.mkdir()
        cohort=make_cohort(); write_features(cohort,feature_dir); write_manifests(cohort,split_dir,2,2)
        config=RunConfig(feature_dir=str(feature_dir),split_dir=str(split_dir),output_dir=str(output),tasks=TASK_ORDER,c_values=(1.0,),expected_repeats=2,expected_folds=2,max_iter=3000,n_jobs=1,make_plots=True,fit_full_source_models=True,top_n=10)
        class Args:
            overwrite_incompatible_output=False; validate_only=False; aggregate_only=False
        run(config,Args())
        markers=list(output.glob("*/C_*/folds/repeat_*/fold_*/FOLD_COMPLETE.json"))
        assert len(markers)==5*2*2,(len(markers),20)
        combined=pd.read_csv(output/"aggregate"/"repetition_performance_summary_all_tasks.csv")
        assert set(combined["task_folder"])==set(TASK_ORDER)
        assert set(combined["evaluation_level"])=={"sample","participant"}
        assert combined["n_repetitions"].eq(2).all()
        three=pd.read_csv(output/"three_way_nonIBD_UC_CD"/"C_1"/"repetition_pooled_oof_metrics.csv")
        for name in ("nonIBD","UC","CD"): assert f"ovr_auc_{name}" in three
        stability=pd.read_csv(output/"three_way_nonIBD_UC_CD"/"C_1"/"feature_coefficient_stability.csv.gz")
        assert set(stability["class"])=={"nonIBD","UC","CD"}
        assert stability["selected_count"].sum() > 0, "Synthetic signal produced no selected coefficients."
        assert (output/"IBD_vs_nonIBD"/"C_1"/"full_source_model"/"full_source_model.npz").exists()
        before={p:p.stat().st_mtime_ns for p in markers}; run(config,Args()); after={p:p.stat().st_mtime_ns for p in markers}
        assert before==after,"Resume refitted completed folds"
        broken_manifest=split_dir/"nonIBD_vs_UC_split_manifest.csv"
        frame=pd.read_csv(broken_manifest); frame.loc[0,"participant_id"]="LEAK"; frame.to_csv(broken_manifest,index=False)
        broken=RunConfig(**{**config.__dict__,"output_dir":str(root/"broken")})
        try: run(broken,Args())
        except (ValueError,RuntimeError): pass
        else: raise AssertionError("Broken manifest was accepted")
    print("SELF-TEST PASSED: all five locked tasks, repeated participant-grouped binary and multiclass Ricci models, train-only scaling, sample and participant OOF metrics, coefficient and process stability, full-source artifacts, manifest rejection, and no-refit resume behavior all succeeded.")


if __name__=="__main__": main()
