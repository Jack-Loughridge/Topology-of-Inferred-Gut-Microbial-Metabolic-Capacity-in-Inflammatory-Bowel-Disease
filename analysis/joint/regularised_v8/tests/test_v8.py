import unittest,tempfile,json,hashlib,tarfile
from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import expit
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from audit.basis import fit_intervals,tensor_one,tensor_many,direct_oracle,filter_deaths,Scale
from audit.inputs import Fold
from audit.io import atomic_json,read_json,completed
from robust.config import Config,specification,SCHEMA,TASK
from robust.model import Problem,State,FittedModel,simplex_floor,l1_residual,NumericalInputError
from robust import workflow as wf


def toy(n=64,m=4,p=12,seed=2):
    rng=np.random.default_rng(seed)
    x=rng.normal(size=n);y=(x+.7*rng.normal(size=n)>0).astype(int)
    A=rng.lognormal(0,.3,(n,m,m))
    if m:A[:,0,0]*=np.exp(.5*x)
    R=rng.normal(size=(n,p))
    if p:R[:,0]=x
    return A,R,y


def workflow_fixture(repeats=5,folds=5):
    rng=np.random.default_rng(444);subjects=20;samples=2
    pid=np.repeat(np.array([f'PRIVATE_PERSON_{i:02d}' for i in range(subjects)]),samples)
    base=(np.arange(subjects)%2);y=np.repeat(base,samples)
    sid=np.array([f'PRIVATE_SAMPLE_{i:03d}' for i in range(len(pid))])
    labels=np.where(y==0,'nonIBD','CD')
    R=np.maximum(0,rng.normal(1,.4,(len(pid),8)));R[:,0]+=.2*y
    d=[rng.uniform(.02,.97,15) for _ in range(len(pid))]
    data=SimpleNamespace(sample_ids=sid,participant_ids=pid,labels3=labels,ricci=sparse.csr_matrix(R),h0_deaths=d)
    plan=[]
    for r in range(1,repeats+1):
        for k,(a,b) in enumerate(StratifiedKFold(n_splits=folds,shuffle=True,random_state=13+r).split(np.zeros(subjects),base),1):
            tr=np.flatnonzero(np.isin(np.arange(len(pid))//samples,a));te=np.flatnonzero(np.isin(np.arange(len(pid))//samples,b))
            plan.append(Fold(TASK,r,k,13+r,tr,te,()))
    return data,plan

class BasisTests(unittest.TestCase):
    def test_filter_boundaries(self):
        np.testing.assert_array_equal(filter_deaths([0,1,.2,.2,np.nan,np.inf,-1]),[.2,.2])
    def test_training_quantiles_only(self):
        iv=fit_intervals([np.linspace(.1,.2,20)],4)
        self.assertEqual(iv.bounds[0],0);self.assertEqual(iv.bounds[-1],1)
        self.assertLess(iv.bounds[-2],.3)
    def test_source_weights(self):
        d=np.r_[np.linspace(.02,.94,45),.4,.4];iv=fit_intervals([d],6);a=np.arange(iv.size)+.3
        np.testing.assert_allclose(tensor_one(d,iv)@a,direct_oracle(d,iv,a),rtol=1e-13,atol=1e-13)
    def test_not_output_weighting(self):
        d=np.linspace(.01,.91,30);iv=fit_intervals([d],4);a=np.array([1.,.1,2.,.9]);A=tensor_one(d,iv)
        self.assertGreater(np.max(np.abs(A@a-(A@np.ones(4))*a)),1e-3)
    def test_duplicates_preserved(self):
        d=np.linspace(.1,.9,30);iv=fit_intervals([d],4)
        np.testing.assert_allclose(tensor_one(np.tile(d,2),iv),2*tensor_one(d,iv))
    def test_empty_is_zero(self):
        iv=fit_intervals([np.linspace(.1,.9,15)],4);self.assertFalse(tensor_one([],iv).any())
    def test_ties_merged(self):
        iv=fit_intervals([np.repeat([.1,.3,.8],50)],32);self.assertLess(iv.size,32)
    def test_scale_frozen_uniform_alpha(self):
        A,R,y=toy();p=Problem(A,R,y);np.testing.assert_allclose(p.hscale.mean,A.sum(2).mean(0))
        np.testing.assert_allclose(p.features(np.ones(4)),p.hscale.transform(A.sum(2)))
    def test_scale_zero_variance(self):
        s=Scale.fit(np.ones((8,3)));np.testing.assert_array_equal(s.sd,np.ones(3))

class ObjectiveTests(unittest.TestCase):
    def test_penalty_units(self):
        A,R,y=toy();p=Problem(A,R,y);self.assertAlmostEqual(p.W,len(y));self.assertAlmostEqual(p.penalty_R,1/(.02*len(y)))
    def test_h0_ridge_explicit(self):
        A,R,y=toy();p=Problem(A,R,y);s=p.initial();s.h0[:]=.2
        cert=p.certificate(s);self.assertAlmostEqual(cert['ridge_penalty'],.5*.001*np.dot(s.h0,s.h0))
    def test_no_alpha_penalty(self):
        A,R,y=toy();p=Problem(A,R,y);s=p.initial();j=p.objective(s);s.alpha=np.array([.1,.3,2.,1.6]);self.assertAlmostEqual(j,p.objective(s))
    def test_alpha_gradient_independent_difference(self):
        A,R,y=toy();p=Problem(A,R,y);s=p.initial();s.h0=np.array([.1,.02,-.03,.1]);s.ricci[0]=.01
        g=p.certificate(s)['ga'];eps=1e-6
        for j in range(4):
            a=s.copy();b=s.copy();a.alpha[j]+=eps;b.alpha[j]-=eps
            self.assertAlmostEqual(g[j],(p.objective(a)-p.objective(b))/(2*eps),places=7)
    def test_ricci_gradient_independent_difference(self):
        A,R,y=toy();p=Problem(A,R,y);s=p.initial();s.ricci[:]=.1;g=p.certificate(s)['gr'];eps=1e-6
        for j in range(4):
            a=s.copy();b=s.copy();a.ricci[j]+=eps;b.ricci[j]-=eps
            self.assertAlmostEqual(g[j]+p.penalty_R,(p.objective(a)-p.objective(b))/(2*eps),places=7)
    def test_l1_zero_kkt(self):
        np.testing.assert_allclose(l1_residual(np.array([.2,-.2,1.]),np.array([0.,0.,0.]),.3),[0,0,.7])
    def test_projection(self):
        a=simplex_floor(np.array([-20.,1.,5.]),1e-8);self.assertAlmostEqual(a.sum(),3);self.assertGreaterEqual(a.min(),1e-8)
    def test_nonfinite_rejected(self):
        A,R,y=toy();A[0,0,0]=np.nan
        with self.assertRaises(NumericalInputError):Problem(A,R,y)
    def test_single_class_rejected(self):
        A,R,y=toy()
        with self.assertRaises(NumericalInputError):Problem(A,R,np.zeros_like(y))
    def test_invalid_config_rejected(self):
        with self.assertRaises(ValueError):replace(Config(),h0_ridge=0.).validate()
    def test_original_axis_gradient(self):
        A,R,y=toy();p=Problem(A,R,y);s=p.initial();s.alpha=np.array([.5,1.5,.2,1.8]);s.h0[:]=.1
        z=p.hscale.transform(A@s.alpha)@s.h0+p.rscale.transform(R)@s.ricci+s.intercept
        np.testing.assert_allclose(p.score(s),z,rtol=1e-13,atol=1e-13)

class FitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.A,cls.R,cls.y=toy(n=80,p=6)
        cls.model=Problem(cls.A,cls.R,cls.y).fit(lambda s:None)
    def test_joint_converges(self):self.assertTrue(self.model.diagnostics['converged'])
    def test_constraints(self):
        self.assertAlmostEqual(self.model.state.alpha.sum(),4)
        self.assertGreaterEqual(self.model.state.alpha.min(),1e-8-1e-12)
    def test_full_scan(self):self.assertEqual(self.model.diagnostics['full_ricci_columns_checked'],6)
    def test_objective_nonincreasing(self):
        vals=[x['objective'] for x in self.model.diagnostics['history']]
        self.assertTrue(np.all(np.diff(vals)<=1e-9))
    def test_coefficient_bound(self):
        self.assertLessEqual(np.linalg.norm(self.model.state.h0),self.model.diagnostics['coefficient_norm_upper_bound'])
    def test_reference_saga(self):
        A,R,y=toy(n=200,m=0,p=5);cfg=replace(Config(),C=.5)
        model=Problem(A,R,y,cfg).fit(lambda s:None)
        ref=LogisticRegression(C=.5,penalty='l1',solver='saga',class_weight='balanced',max_iter=20000,tol=1e-10,random_state=31)
        ref.fit(model.rscale.transform(R),y)
        pp=model.predict(A,R)[0]
        np.testing.assert_allclose(pp,ref.predict_proba(model.rscale.transform(R)),atol=1e-4,rtol=0)
    def test_separable_h0_has_finite_ridge_solution(self):
        n=60;y=np.arange(n)%2;A=np.zeros((n,2,2));A[:,0,0]=1+y;A[:,1,1]=2-y
        m=Problem(A,np.zeros((n,0)),y).fit(lambda s:None)
        self.assertTrue(m.diagnostics['converged']);self.assertLess(np.linalg.norm(m.state.h0),38)
    def test_exact_aliases(self):
        A,R,y=toy();A[:,1]=A[:,0];R[:,1]=R[:,0];m=Problem(A,R,y).fit(lambda s:None)
        self.assertTrue(m.diagnostics['converged']);self.assertLess(m.diagnostics['max_abs_h0_coefficient'],38)
    def test_near_aliases(self):
        A,R,y=toy();A[:,1]=A[:,0]+np.random.default_rng(15).uniform(0,1e-14,A[:,1].shape)
        m=Problem(A,R,y).fit(lambda s:None);self.assertTrue(m.diagnostics['converged'])
    def test_finite_budget_label_not_convergence(self):
        A,R,y=toy(n=100,p=20);cfg=replace(Config(),C=.5,max_working_rounds=1,working_batch=1,joint_max_iterations=3,rescue_cycles=0)
        m=Problem(A,R,y,cfg).fit(lambda s:None)
        self.assertEqual(m.diagnostics['status'],'FINITE_BUDGET_APPROXIMATION');self.assertFalse(m.diagnostics['converged'])
        self.assertIsNotNone(m.diagnostics['approximation_warning']);self.assertLessEqual(m.diagnostics['objective'],np.log(2)+1e-10)
    def test_resume_same_fit(self):
        cp=[];A,R,y=toy();p=Problem(A,R,y)
        model=p.fit(lambda s:None,lambda s,d:cp.append((s.copy(),d.copy())))
        self.assertTrue(cp)
        resumed=Problem(A,R,y).fit(lambda s:None,resume=cp[-1])
        self.assertLessEqual(resumed.diagnostics['objective'],model.diagnostics['objective']+1e-9)
    def test_roundtrip(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'model.npz';self.model.save(path,'test')
            m=FittedModel.load(path,'test')
            np.testing.assert_array_equal(self.model.predict(self.A,self.R)[0],m.predict(self.A,self.R)[0])
    def test_wrong_model_identity(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'m.npz';self.model.save(path,'yes')
            with self.assertRaises(ValueError):FittedModel.load(path,'no')
    def test_unseen_prediction(self):
        A,R,y=toy(n=11,p=6,seed=99);p,z,ch=self.model.predict(A,R)
        self.assertEqual(p.shape,(11,2));self.assertTrue(np.isfinite(p).all())
        self.assertLess(ch['max_float64_extended_score_difference'],1e-9)
    def test_nonfinite_prediction_rejected(self):
        R=self.R.copy();R[0,0]=np.inf
        with self.assertRaises(NumericalInputError):self.model.predict(self.A,R)
    def test_zero_h0_reference(self):
        A,R,y=toy(m=0,p=4);m=Problem(A,R,y).fit(lambda s:None);self.assertEqual(len(m.state.h0),0)
        self.assertEqual(m.diagnostics['ridge_penalty'],0.)

class PlanAndMetricTests(unittest.TestCase):
    def test_plan_all_25(self):
        data,p=workflow_fixture();self.assertEqual(len(wf.chosen_folds(data,p)),25)
    def test_missing_fold(self):
        data,p=workflow_fixture()
        with self.assertRaises(ValueError):wf.chosen_folds(data,p[:-1])
    def test_overlap(self):
        data,p=workflow_fixture();p[0]=replace(p[0],train=np.r_[p[0].train,p[0].test[0]])
        with self.assertRaises(ValueError):wf.chosen_folds(data,p)
    def test_participant_overlap(self):
        data,p=workflow_fixture();data.participant_ids[p[0].train[0]]=data.participant_ids[p[0].test[0]]
        with self.assertRaises(ValueError):wf.chosen_folds(data,p)
    def test_exact_tie_negative(self):
        m=wf.metrics(np.array([0,1]),np.full((2,2),.5));self.assertEqual(m['accuracy'],.5)
    def test_no_nans_scored(self):
        with self.assertRaises(ValueError):wf.metrics(np.array([0,1]),np.array([[.5,.5],[np.nan,.5]]))
    def test_participant_average(self):
        df=pd.DataFrame({'participant_id':['a','a','b'],'y':[0,0,1],'p0':[.9,.7,.2],'p1':[.1,.3,.8]})
        p=wf.participant_table(df);self.assertAlmostEqual(p.loc[0,'p1'],.2)
    def test_conflicting_labels(self):
        df=pd.DataFrame({'participant_id':['a','a'],'y':[0,1],'p0':[.9,.7],'p1':[.1,.3]})
        with self.assertRaises(ValueError):wf.participant_table(df)
    def test_declared_scope(self):
        s=specification(Config());self.assertFalse(s['original_ridge_free_model']);self.assertEqual(s['expected_pairs'],25)
        self.assertEqual(s['alpha_penalty'],0);self.assertFalse(s['resolution_selection'])

class FullWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();cls.out=Path(cls.temp.name)/'output';cls.out.mkdir()
        cls.data,cls.plan=workflow_fixture()
        cls.cfg=replace(Config(),requested_intervals=4,max_working_rounds=10)
        atomic_json(cls.out/'SCIENTIFIC_SPECIFICATION.json',specification(cls.cfg))
        with patch('robust.workflow.log',lambda s:None):
            cls.result=wf.run_schedule(cls.data,cls.plan,cls.out,cls.cfg,'synthetic',workers=2)
        atomic_json(cls.out/'FINAL_REPORT.json',cls.result);wf.export_review(cls.out)
    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()
    def test_all_pairs(self):self.assertEqual(self.result['completed_pairs'],25);self.assertTrue(self.result['analysis_complete'])
    def test_no_inner_models(self):self.assertEqual(len(list(self.out.glob('jobs/repeat_*/fold_*/joint/model.npz'))),25)
    def test_all_models_reported(self):self.assertEqual(len(pd.read_csv(self.out/'NUMERICAL_SUMMARY.csv')),50)
    def test_repeated_summaries(self):
        rows=pd.read_csv(self.out/'aggregate/repetition_metrics.csv');s=pd.read_csv(self.out/'aggregate/performance_summary.csv')
        self.assertEqual(len(rows),100);self.assertTrue((s['count']==5).all())
        self.assertAlmostEqual(s.iloc[0]['mean'],rows[(rows.model==s.iloc[0]['model'])&(rows.level==s.iloc[0]['level'])&(rows.metric==s.iloc[0]['metric'])].value.mean())
    def test_paired_differences(self):
        d=pd.read_csv(self.out/'aggregate/paired_differences.csv');self.assertEqual(len(d),50)
    def test_resume_does_not_refit(self):
        with patch('robust.workflow.pair_worker',side_effect=AssertionError('Attempted refit')):
            a=wf.run_schedule(self.data,self.plan,self.out,self.cfg,'synthetic',workers=1)
        self.assertTrue(a['analysis_complete'])
    def test_export_private_exclusion(self):
        with tarfile.open(self.out/'joint_ibd_regularised_v8_review.tar.gz') as t:
            for member in t.getmembers():
                self.assertNotIn('PRIVATE',member.name);self.assertFalse(member.name.endswith('.npz'))
                content=t.extractfile(member).read()
                self.assertNotIn(b'PRIVATE_PERSON',content);self.assertNotIn(b'PRIVATE_SAMPLE',content)
    def test_manifest_hashes(self):
        with tarfile.open(self.out/'joint_ibd_regularised_v8_review.tar.gz') as t:
            m=json.load(t.extractfile('MANIFEST.json'))
            for n,h in m.items():self.assertEqual(hashlib.sha256(t.extractfile(n).read()).hexdigest(),h)
    def test_receipt_corruption_blocks(self):
        p=self.out/'jobs/repeat_01/fold_1/joint/numerical.json';old=p.read_bytes()
        try:
            p.write_bytes(old+b' ')
            with self.assertRaises(RuntimeError):wf.validate_pair(self.out,self.plan[0],'synthetic')
        finally:p.write_bytes(old)
    def test_all_prediction_roundtrips(self):
        for p in self.out.glob('jobs/repeat_*/fold_*/*/prediction_checks.json'):
            self.assertTrue(read_json(p)['saved_reload_equal'])
    def test_partial_not_complete(self):
        with tempfile.TemporaryDirectory() as td:
            x=wf.aggregate(Path(td),self.plan,self.data,'synthetic')
            self.assertFalse(x['analysis_complete']);self.assertFalse((Path(td)/'aggregate/performance_summary.csv').exists())

class OutputSafetyTests(unittest.TestCase):
    def test_old_output_refused(self):
        from run import safe_output
        with tempfile.TemporaryDirectory() as td:
            old=Path(td)/'old';old.mkdir()
            with self.assertRaises(ValueError):safe_output(old,[old])
    def test_unknown_populated_refused(self):
        from run import safe_output
        with tempfile.TemporaryDirectory() as td:
            (Path(td)/'anything.txt').write_text('keep')
            with self.assertRaises(ValueError):safe_output(Path(td),[])
    def test_new_output_accepted(self):
        from run import safe_output
        with tempfile.TemporaryDirectory() as td:
            self.assertEqual(safe_output(Path(td)/'new',[]),Path(td)/'new')
    def test_symlink_output_refused(self):
        from run import safe_output
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'alias';p.symlink_to(Path(td),target_is_directory=True)
            with self.assertRaises(ValueError):safe_output(p,[])

if __name__=='__main__':unittest.main()
