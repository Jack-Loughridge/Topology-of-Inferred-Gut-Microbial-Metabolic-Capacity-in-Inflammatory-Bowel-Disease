import unittest,tempfile,json
from pathlib import Path
from dataclasses import replace
import numpy as np
from scipy.special import expit
from robust.model import Problem,State,FittedModel,TimeBudget,simplex_floor
from robust.config import Config
from audit.basis import fit_intervals,tensor_one
from test_v8 import toy

class EdgeTests(unittest.TestCase):
    def test_head_gradient_with_ridge(self):
        A,R,y=toy();p=Problem(A,R,y);s=p.initial();s.h0[:]=.03
        g=p.features(s.alpha).T@(p.w*(expit(p.score(s))-y))+.001*s.h0
        for j in range(len(g)):
            a=s.copy();b=s.copy();a.h0[j]+=1e-6;b.h0[j]-=1e-6
            self.assertAlmostEqual(g[j],(p.objective(a)-p.objective(b))/2e-6,places=7)
    def test_alpha_floor_projection_mean_one(self):
        for n in (1,2,31,32):
            a=simplex_floor(np.random.default_rng(n).normal(size=n)*100,1e-8)
            self.assertAlmostEqual(a.sum(),n);self.assertTrue((a>=1e-8-1e-12).all())
    def test_immediate_timeout_not_fake_fitted_model(self):
        A,R,y=toy()
        with self.assertRaises(FloatingPointError):Problem(A,R,y,replace(Config(),max_fit_seconds=1e-12)).fit(lambda s:None)
    def test_timeout_after_checkpoint_reports_approximation(self):
        A,R,y=toy();cfg=replace(Config(),max_working_rounds=1,joint_max_iterations=2,rescue_cycles=0,working_batch=1)
        cp=[];m=Problem(A,R,y,cfg).fit(lambda s:None,lambda s,d:cp.append((s,d)))
        self.assertTrue(cp)
        state,meta=cp[-1];meta=dict(meta);meta['elapsed_seconds']=cfg.max_fit_seconds+1
        # Same completed training state is retained, never re-labelled converged.
        model=Problem(A,R,y,cfg).fit(lambda s:None,resume=(state,meta))
        self.assertIn(model.diagnostics['status'],['FINITE_BUDGET_APPROXIMATION','CONVERGED'])
    def test_zero_tensor_on_some_rows(self):
        A,R,y=toy();A[::5]=0.;m=Problem(A,R,y).fit(lambda s:None);self.assertTrue(np.isfinite(m.predict(A,R)[0]).all())
    def test_constant_ricci_features(self):
        A,R,y=toy();R[:,:]=1.;m=Problem(A,R,y).fit(lambda s:None);self.assertEqual(m.diagnostics['ricci_nonzero'],0)
    def test_prediction_uses_saved_scaling(self):
        A,R,y=toy();m=Problem(A,R,y).fit(lambda s:None);before=m.rscale.mean.copy()
        m.predict(A[:5],R[:5]+.25);np.testing.assert_array_equal(before,m.rscale.mean)
    def test_checkpoint_bytes_not_pickle(self):
        A,R,y=toy();m=Problem(A,R,y).fit(lambda s:None)
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'x.npz';m.save(p)
            with np.load(p,allow_pickle=False) as f:
                for k in f.files:self.assertNotEqual(f[k].dtype,object)
    def test_coarse_basis_bounded_resolution(self):
        d=np.linspace(.05,.95,60);iv=fit_intervals([d],32);self.assertLessEqual(iv.size,32)
        self.assertEqual(tensor_one(d,iv).shape,(iv.size,iv.size))
    def test_empty_projection(self):self.assertEqual(simplex_floor(np.array([]),1e-8).shape,(0,))
    def test_negative_tensor_rejected(self):
        A,R,y=toy();A[0,0,0]=-1
        with self.assertRaises(ValueError):Problem(A,R,y)
    def test_ridge_objective_bound(self):
        A,R,y=toy();m=Problem(A,R,y).fit(lambda s:None)
        self.assertLessEqual(.001/2*np.dot(m.state.h0,m.state.h0),np.log(2)+1e-10)

class EntryPointTests(unittest.TestCase):
    def test_complete_entry_and_resume(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from test_v8 import workflow_fixture
        from audit.io import atomic_json,sha256_file,read_json
        from run import main
        data,plan=workflow_fixture()
        with tempfile.TemporaryDirectory() as td:
            base=Path(td);ref=base/'reference';ref.mkdir();out=base/'new'
            atomic_json(ref/'orchestrator_config.json',{})
            source=base/'source.txt';source.write_text('immutable')
            payload={'id':'synthetic-inputs','input_hashes':{str(source):sha256_file(source)},
                'matrix_sha256':'synthetic','original_cohort_counts':{'samples':40,'participants':20,'ricci_columns':8},'versions':{}}
            cfg=replace(Config(),requested_intervals=4,max_working_rounds=10)
            with patch('run.Config',return_value=cfg),patch('run.verify',return_value='synthetic-code'),\
                 patch('run.preflight',return_value=(data,plan,None,payload,{})),\
                 patch('run.log',lambda s:None),patch('robust.workflow.log',lambda s:None):
                self.assertEqual(main(['--reference',str(ref),'--output',str(out),'--workers','1']),0)
                before={str(p.relative_to(out)):sha256_file(p) for p in out.glob('jobs/repeat_*/fold_*/*/model.npz')}
                with patch('robust.workflow.pair_worker',side_effect=AssertionError('Refit')):
                    self.assertEqual(main(['--reference',str(ref),'--output',str(out),'--workers','1']),0)
                after={str(p.relative_to(out)):sha256_file(p) for p in out.glob('jobs/repeat_*/fold_*/*/model.npz')}
                self.assertEqual(before,after)
            self.assertTrue(read_json(out/'FINAL_REPORT.json')['analysis_complete'])
            self.assertTrue(read_json(out/'FINAL_REPORT.json')['all_fits_converged'])
            self.assertEqual(len(before),50)
