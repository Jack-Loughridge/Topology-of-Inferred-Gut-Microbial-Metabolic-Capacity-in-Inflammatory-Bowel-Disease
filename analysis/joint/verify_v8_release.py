#!/usr/bin/env python3
"""Verify the published v8 source and aggregates without observations or fitting.

AUC and log loss are reaggregated from archived scalars. This is not independent
prediction-level verification. The standard library is sufficient.
"""
from pathlib import Path
import csv, hashlib, json, math, statistics, subprocess, sys

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'results/figure_source_data/joint/regularised_v8'

def load(name):
    with (DATA/name).open(newline='') as stream:
        return list(csv.DictReader(stream))

def close(a,b):
    if not math.isclose(float(a), float(b), rel_tol=0, abs_tol=2e-12):
        raise AssertionError((a,b))

def manifest(path, expected):
    seen=set()
    for line in path.read_text().splitlines():
        if not line.strip(): continue
        want, rel=line.split(maxsplit=1); rel=rel.removeprefix('*')
        if rel in seen or Path(rel).is_absolute() or '..' in Path(rel).parts:
            raise AssertionError('Unsafe or duplicate manifest entry')
        seen.add(rel); p=ROOT/rel
        assert p.is_file() and not p.is_symlink(),rel
        assert hashlib.sha256(p.read_bytes()).hexdigest()==want,rel
    assert len(seen)==expected,(len(seen),expected)

def main():
    manifest(ROOT/'provenance/joint_v8/source_manifest.sha256',29)
    manifest(ROOT/'provenance/joint_v8/result_manifest.sha256',9)
    subprocess.run([sys.executable,str(ROOT/'analysis/joint/regularised_v8/verify_package.py')],check=True)
    spec=json.loads((DATA/'SCIENTIFIC_SPECIFICATION.json').read_text())
    c=spec['configuration']
    assert (c['C'],c['h0_ridge'],c['requested_intervals'],c['repeats'],c['outer_folds'])==(.02,.001,32,5,5)
    assert spec['inner_fits']==0 and spec['resolution_selection'] is False
    final=json.loads((DATA/'FINAL_REPORT.json').read_text())
    assert final['status']=='COMPLETE_ALL_CONVERGED'
    rows=load('NUMERICAL_SUMMARY.csv')
    keys={(int(x['repeat']),int(x['fold']),x['model']) for x in rows}
    expected={(r,f,a) for r in range(1,6) for f in range(1,6) for a in ('joint','ricci')}
    assert len(rows)==50 and keys==expected
    for row in rows:
        assert row['converged']=='True' and row['status']=='CONVERGED'
        assert row['approximation_warning']=='' and row['global_optimum_certified']=='False'
        assert float(row['head_kkt'])<=1e-5 and float(row['alpha_conditional_gap'])<=1e-5
        close(row['objective'],sum(float(row[k]) for k in ('cross_entropy','ricci_penalty','ridge_penalty')))
        close(row['ridge_penalty'],.001/2*float(row['h0_l2_norm'])**2)
    rep=load('repetition_metrics.csv')
    indexed={(x['model'],x['level'],x['metric'],int(x['repeat'])):float(x['value']) for x in rep}
    assert len(rep)==100 and len(indexed)==100
    for row in load('performance_summary.csv'):
        vals=[indexed[row['model'],row['level'],row['metric'],r] for r in range(1,6)]
        close(row['mean'],statistics.mean(vals));close(row['std'],statistics.stdev(vals))
        assert int(row['count'])==5
    diffs={}
    for row in load('paired_differences.csv'):
        l,m,r=row['level'],row['metric'],int(row['repeat'])
        d=indexed['joint',l,m,r]-indexed['ricci',l,m,r]
        close(row['joint_minus_ricci'],d);diffs[l,m,r]=d
    assert len(diffs)==50
    for row in load('paired_summary.csv'):
        vals=[diffs[row['level'],row['metric'],r] for r in range(1,6)]
        close(row['mean'],statistics.mean(vals));close(row['std'],statistics.stdev(vals))
    for row in load('repetition_confusions.csv'):
        tn,fp,fn,tp=(int(row[x]) for x in ('TN','FP','FN','TP'))
        n=tn+fp+fn+tp
        assert (tn+fp,fn+tp)==((26,80) if row['level']=='participant' else (359,958))
        values={'accuracy':(tn+tp)/n, 'balanced_accuracy':.5*(tn/(tn+fp)+tp/(tp+fn)),
                'macro_f1':.5*(2*tn/(2*tn+fp+fn)+2*tp/(2*tp+fp+fn))}
        for m,v in values.items():close(v,indexed[row['model'],row['level'],m,int(row['repeat'])])
    assert all(diffs['participant','roc_auc',r]>0 for r in range(1,6))
    assert all(diffs['sample','log_loss',r]>0 for r in range(1,6))
    print('PASS: exact v8 source, 50 converged records, all aggregate means/SDs, paired differences and confusion metrics')
    print('Boundary: raw prediction ranks, production gradients, exact executed VM source manifest and split membership are not in this export.')

if __name__=='__main__':main()
