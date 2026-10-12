"""Regularised source-weighted binary joint model, in original feature units.

A small, explicit H0 ridge controls the free-coefficient instability. The Ricci
L1 coefficient stays 1/(C*sum_w). All Ricci gradients are scanned. Iteration/time
ceilings yield a labelled approximate finite-budget estimator, not a fictitious
convergence certificate. Invalid data or nonfinite states are never scored.
"""
from __future__ import annotations
from dataclasses import dataclass
import json,time
from pathlib import Path
from typing import Callable
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from audit.basis import Scale
from audit.io import atomic_npz, canonical_json
from .config import Config


class NumericalInputError(ValueError):
    pass

class TimeBudget(Exception):
    pass


def simplex_floor(a: np.ndarray, floor: float) -> np.ndarray:
    """Euclidean projection onto sum(a)=m and a>=floor, including m=0."""
    a=np.asarray(a,float);m=len(a)
    if not m:return a.copy()
    if not np.isfinite(a).all():raise FloatingPointError('Nonfinite alpha trial.')
    z=m*(1-floor);v=a-floor;u=np.sort(v)[::-1];cs=np.cumsum(u)-z
    j=np.flatnonzero(u-cs/np.arange(1,m+1)>0)
    theta=cs[j[-1]]/(j[-1]+1)
    out=np.maximum(v-theta,0)+floor
    out[np.argmax(out)]+=m-float(out.sum())
    if out.min()<floor-1e-12:raise FloatingPointError('Alpha projection failed.')
    return out


def l1_residual(gradient, coefficients, penalty):
    return np.where(coefficients==0,np.maximum(np.abs(gradient)-penalty,0),
                    np.abs(gradient+penalty*np.sign(coefficients)))


@dataclass
class State:
    intercept:float
    h0:np.ndarray
    alpha:np.ndarray
    ricci:np.ndarray
    def copy(self):return State(float(self.intercept),self.h0.copy(),self.alpha.copy(),self.ricci.copy())


@dataclass
class FittedModel:
    state:State
    hscale:Scale
    rscale:Scale
    config:Config
    diagnostics:dict
    def scores(self,A,raw_R,extended=False):
        A=np.asarray(A);raw_R=np.asarray(raw_R)
        m=len(self.state.alpha);p=len(self.state.ricci)
        if A.ndim!=3 or A.shape[1:]!=(m,m) or raw_R.shape!=(len(A),p):
            raise NumericalInputError('Prediction dimensions differ from the saved model.')
        if not np.isfinite(A).all() or not np.isfinite(raw_R).all():
            raise NumericalInputError('Nonfinite prediction observations.')
        dtype=np.longdouble if extended else np.float64
        a=self.state.alpha.astype(dtype);b=self.state.h0.astype(dtype)
        h=(A.astype(dtype)@a-self.hscale.mean.astype(dtype))/self.hscale.sd.astype(dtype)
        # Batch callers limit the extended-precision workspace.
        r=(raw_R.astype(dtype)-self.rscale.mean.astype(dtype))/self.rscale.sd.astype(dtype)
        z=h@b+r@self.state.ricci.astype(dtype)+dtype(self.state.intercept)
        if not np.isfinite(z).all():raise FloatingPointError('Nonfinite predictor score.')
        return z
    def predict(self,A,raw_R,batch=64):
        probs=[];scores=[];err=0.;maxscale=1.
        for i in range(0,len(A),batch):
            zs=self.scores(A[i:i+batch],raw_R[i:i+batch])
            ze=self.scores(A[i:i+batch],raw_R[i:i+batch],extended=True)
            err=max(err,float(np.max(np.abs(zs.astype(np.longdouble)-ze),initial=0.)))
            maxscale=max(maxscale,float(np.max(np.abs(ze),initial=0.)))
            # Use the extended accumulation for returned scores in all fits.
            z=np.asarray(ze,float)
            if not np.isfinite(z).all():raise FloatingPointError('Score exceeds deployment range.')
            pp=expit(z);probs.append(np.column_stack([1-pp,pp]));scores.append(z)
        P=np.concatenate(probs);Z=np.concatenate(scores)
        if not np.isfinite(P).all() or not np.allclose(P.sum(1),1,rtol=0,atol=1e-12):
            raise FloatingPointError('Invalid probabilities.')
        return P,Z,{'max_float64_extended_score_difference':err,'max_score_scale':maxscale,
            'extended_mantissa_bits':int(np.finfo(np.longdouble).nmant+1),
            'returned_scores':'extended accumulation rounded to float64',
            'training_transformations_refitted':False}
    def save(self,path,identity=''):
        atomic_npz(Path(path),intercept=np.array(self.state.intercept),h0=self.state.h0,
            alpha=self.state.alpha,ricci=self.state.ricci,hmean=self.hscale.mean,hsd=self.hscale.sd,
            rmean=self.rscale.mean,rsd=self.rscale.sd,
            metadata=np.array(canonical_json({'config':self.config.payload(),
              'diagnostics':self.diagnostics,'identity':identity,'classes':['nonIBD','IBD']})))
    @classmethod
    def load(cls,path,identity=None):
        with np.load(path,allow_pickle=False) as f:
            meta=json.loads(str(f['metadata'].item()))
            if identity is not None and meta['identity']!=identity:
                raise ValueError('Saved model identity mismatch.')
            s=State(float(f['intercept']),f['h0'].copy(),f['alpha'].copy(),f['ricci'].copy())
            out=cls(s,Scale(f['hmean'].copy(),f['hsd'].copy()),Scale(f['rmean'].copy(),f['rsd'].copy()),
                    Config(**meta['config']),meta['diagnostics'])
        if meta['classes']!=['nonIBD','IBD'] or not all(np.isfinite(v).all() for v in
             [s.h0,s.alpha,s.ricci,out.hscale.mean,out.hscale.sd,out.rscale.mean,out.rscale.sd]):
            raise ValueError('Invalid saved model.')
        if np.any(out.hscale.sd<=0) or np.any(out.rscale.sd<=0) or not np.isfinite(s.intercept):
            raise ValueError('Invalid saved model scale/intercept.')
        if len(s.alpha) and (s.alpha.min()<out.config.alpha_floor-1e-10 or abs(s.alpha.sum()-len(s.alpha))>1e-8):
            raise ValueError('Saved alpha constraints violated.')
        return out


class Problem:
    def __init__(self,A,raw_R,y,config=Config()):
        config.validate();A=np.asarray(A,float);raw_R=np.asarray(raw_R,float);y=np.asarray(y)
        if A.ndim!=3 or A.shape[1]!=A.shape[2] or raw_R.ndim!=2 or len(A)!=len(raw_R) or y.shape!=(len(A),):
            raise NumericalInputError('Training dimensions are inconsistent.')
        if not np.isfinite(A).all() or not np.isfinite(raw_R).all() or np.any(A<0):
            raise NumericalInputError('Training data must be finite, with a nonnegative H0 tensor.')
        if set(y.tolist())!={0,1}:raise NumericalInputError('Both binary training classes required.')
        self.y=y.astype(np.int64);self.n=len(y);self.m=A.shape[1];self.p=raw_R.shape[1];self.config=config
        self.hscale=Scale.fit(A.sum(axis=2));self.rscale=Scale.fit(raw_R)
        self.Q=np.ascontiguousarray(A/self.hscale.sd[None,:,None]);self.qm=self.hscale.mean/self.hscale.sd
        self.R=np.ascontiguousarray(self.rscale.transform(raw_R))
        if not np.isfinite(self.Q).all() or not np.isfinite(self.R).all():
            raise NumericalInputError('Training standardisation overflowed.')
        counts=np.bincount(self.y,minlength=2)
        self.weights=self.n/(2*counts[self.y]);self.W=float(self.weights.sum())
        self.w=self.weights/self.W;self.penalty_R=1/(config.C*self.W)
        self.elapsed_before=0.;self.started=0.;self.last_print=0.;self.evaluations=0
    def initial(self):return State(0.,np.zeros(self.m),np.ones(self.m),np.zeros(self.p))
    def features(self,a):return self.Q@a-self.qm
    def score(self,s):return self.features(s.alpha)@s.h0+self.R@s.ricci+s.intercept
    def objective(self,s):
        z=self.score(s)
        return float(np.dot(self.w,np.logaddexp(0,z)-self.y*z)+
            .5*self.config.h0_ridge*np.dot(s.h0,s.h0)+self.penalty_R*np.abs(s.ricci).sum())
    def certificate(self,s):
        H=self.features(s.alpha);z=H@s.h0+self.R@s.ricci+s.intercept
        residual=self.w*(expit(z)-self.y)
        gb=H.T@residual+self.config.h0_ridge*s.h0
        ga=np.einsum('njk,j,n->k',self.Q,s.h0,residual,optimize=True)
        gr=self.R.T@residual
        kr=l1_residual(gr,s.ricci,self.penalty_R)
        gap=float(np.dot(ga,s.alpha)-self.config.alpha_floor*ga.sum()-
                  self.m*(1-self.config.alpha_floor)*np.min(ga,initial=np.inf)) if self.m else 0.
        j=self.objective(s)
        if not np.isfinite(j) or not np.isfinite(gr).all() or not np.isfinite(ga).all():
            raise FloatingPointError('Nonfinite training objective or derivative.')
        return {'objective':j,'cross_entropy':float(np.dot(self.w,np.logaddexp(0,z)-self.y*z)),
          'ridge_penalty':float(.5*self.config.h0_ridge*np.dot(s.h0,s.h0)),
          'ricci_penalty':float(self.penalty_R*np.abs(s.ricci).sum()),
          'head_kkt':float(max(abs(residual.sum()),np.max(np.abs(gb),initial=0),np.max(kr,initial=0))),
          'alpha_conditional_gap':max(0.,gap),'full_ricci_columns_checked':self.p,
          'max_abs_h0_coefficient':float(np.max(np.abs(s.h0),initial=0)),
          'h0_l2_norm':float(np.linalg.norm(s.h0)),
          'alpha_min':float(np.min(s.alpha)) if self.m else None,
          'alpha_max':float(np.max(s.alpha)) if self.m else None,
          'alpha_sum_error':float(abs(s.alpha.sum()-self.m)),
          'ricci_nonzero':int(np.count_nonzero(s.ricci)),
          'gr':gr,'ga':ga,'scores':z}
    def pack(self,s,active):
        return np.r_[s.intercept,s.h0,s.alpha,np.maximum(s.ricci[active],0),np.maximum(-s.ricci[active],0)]
    def unpack(self,x,active,canonical=True):
        m=self.m;k=len(active);r=np.zeros(self.p)
        r[active]=x[1+2*m:1+2*m+k]-x[1+2*m+k:]
        if canonical:r[np.abs(r)<self.config.zero_tolerance]=0.
        a=simplex_floor(x[1+m:1+2*m],self.config.alpha_floor) if canonical else x[1+m:1+2*m].copy()
        return State(float(x[0]),x[1:1+m].copy(),a,r)
    def check_time(self):
        self.evaluations+=1
        if time.monotonic()-self.started+self.elapsed_before>=self.config.max_fit_seconds:
            raise TimeBudget('Declared finite training budget reached.')
    def restricted(self,s,active,log):
        cfg=self.config;m=self.m;k=len(active);RX=np.ascontiguousarray(self.R[:,active]);x=self.pack(s,active)
        def fun(x):
            self.check_time(); b=x[1:1+m];a=x[1+m:1+2*m];r=x[1+2*m:1+2*m+k]-x[1+2*m+k:]
            H=self.features(a);z=H@b+RX@r+x[0];res=self.w*(expit(z)-self.y)
            val=np.dot(self.w,np.logaddexp(0,z)-self.y*z)+.5*cfg.h0_ridge*np.dot(b,b)+self.penalty_R*x[1+2*m:].sum()
            gb=H.T@res+cfg.h0_ridge*b
            ga=np.einsum('njk,j,n->k',self.Q,b,res,optimize=True)
            gr=RX.T@res
            return float(val),np.r_[res.sum(),gb,ga,gr+self.penalty_R,-gr+self.penalty_R]
        def eq(v):return float(v[1+m:1+2*m].sum()-m)
        ev=np.zeros(len(x));ev[1+m:1+2*m]=1.
        def callback(v):
            now=time.monotonic()
            if now-self.last_print>15:
                log(f'joint optimiser active, working Ricci columns={k}, elapsed={now-self.started+self.elapsed_before:.0f}s')
                self.last_print=now
        bounds=[(None,None)]*(1+m)+[(cfg.alpha_floor,float(m))]*m+[(0.,None)]*(2*k)
        if m:
            res=minimize(fun,x,jac=True,bounds=bounds,
              constraints=[{'type':'eq','fun':eq,'jac':lambda x:ev}],method='SLSQP',callback=callback,
              options={'maxiter':cfg.joint_max_iterations,'ftol':1e-12})
        else:
            res=minimize(fun,x,jac=True,bounds=bounds,method='L-BFGS-B',callback=callback,
              options={'maxiter':cfg.head_max_iterations,'ftol':1e-15,'gtol':1e-9,'maxls':50,'maxcor':30})
        if not np.isfinite(res.x).all():return s.copy(),{'method':'joint','accepted':False,'message':'Nonfinite trial rejected'}
        candidate=self.unpack(res.x,active)
        accepted=self.objective(candidate)<=self.objective(s)+cfg.objective_acceptance_slack
        return candidate if accepted else s.copy(),{'method':'joint','accepted':accepted,
           'library_success':bool(res.success),'iterations':int(res.nit),'message':str(res.message)}
    def coordinate_rescue(self,s,active,log):
        """Bounded convex head and alpha substeps, same objective, no new penalty."""
        cfg=self.config;m=self.m;k=len(active);RX=np.ascontiguousarray(self.R[:,active]);H=self.features(s.alpha)
        x=np.r_[s.intercept,s.h0,np.maximum(s.ricci[active],0),np.maximum(-s.ricci[active],0)]
        def head(v):
            self.check_time();b=v[1:1+m];r=v[1+m:1+m+k]-v[1+m+k:]
            z=v[0]+H@b+RX@r;rr=self.w*(expit(z)-self.y)
            val=np.dot(self.w,np.logaddexp(0,z)-self.y*z)+.5*cfg.h0_ridge*np.dot(b,b)+self.penalty_R*v[1+m:].sum()
            gr=RX.T@rr
            return float(val),np.r_[rr.sum(),H.T@rr+cfg.h0_ridge*b,gr+self.penalty_R,-gr+self.penalty_R]
        r=minimize(head,x,jac=True,bounds=[(None,None)]*(1+m)+[(0.,None)]*(2*k),method='L-BFGS-B',
          options={'maxiter':cfg.head_max_iterations,'ftol':1e-15,'gtol':1e-9,'maxls':50,'maxcor':30})
        trial=s.copy()
        if np.isfinite(r.x).all():
            trial.intercept=float(r.x[0]);trial.h0=r.x[1:1+m].copy();trial.ricci[:]=0
            trial.ricci[active]=r.x[1+m:1+m+k]-r.x[1+m+k:]
            trial.ricci[np.abs(trial.ricci)<cfg.zero_tolerance]=0
            if self.objective(trial)<=self.objective(s)+cfg.objective_acceptance_slack:s=trial
        if not m:return s
        D=np.einsum('njk,j->nk',self.Q,s.h0,optimize=True)
        offset=self.R@s.ricci+s.intercept-self.qm@s.h0
        def af(a):
            self.check_time();z=D@a+offset;res=self.w*(expit(z)-self.y)
            return float(np.dot(self.w,np.logaddexp(0,z)-self.y*z)),D.T@res
        aa=minimize(af,s.alpha,jac=True,bounds=[(cfg.alpha_floor,float(m))]*m,
          constraints=[{'type':'eq','fun':lambda a:a.sum()-m,'jac':lambda a:np.ones(m)}],
          method='SLSQP',options={'maxiter':cfg.alpha_max_iterations,'ftol':1e-13})
        if np.isfinite(aa.x).all():
            trial=s.copy();trial.alpha=simplex_floor(aa.x,cfg.alpha_floor)
            if self.objective(trial)<=self.objective(s)+cfg.objective_acceptance_slack:s=trial
        return s
    def fit(self,log:Callable[[str],None]=print,checkpoint=None,resume=None):
        cfg=self.config;self.started=time.monotonic();history=[];active=np.empty(0,int);state=self.initial();start_round=0
        if resume is not None:
            state,old_meta=resume
            if len(state.h0)!=self.m or len(state.ricci)!=self.p:raise ValueError('Checkpoint shape differs.')
            state=state.copy();self.elapsed_before=float(old_meta['elapsed_seconds'])
            active=np.asarray(old_meta['active'],int);start_round=int(old_meta['next_round']);history=list(old_meta['history'])
        best=state.copy();best_obj=self.objective(best);stop='ROUND_LIMIT'
        try:
            for round_number in range(start_round,cfg.max_working_rounds):
                self.check_time();d=self.certificate(state)
                v=np.maximum(abs(d['gr'])-self.penalty_R,0)
                # Retain nonzero coefficients; inactive columns are NOT permanently excluded.
                existing=np.flatnonzero(state.ricci)
                candidates=np.flatnonzero(v>cfg.head_tolerance*.1)
                candidates=candidates[~np.isin(candidates,existing)]
                candidates=candidates[np.lexsort((candidates,-v[candidates]))]
                available=max(0,cfg.max_working_columns-len(existing))
                # Exchange zero coordinates rather than grow an enormous dense subproblem.
                active=np.unique(np.r_[existing,candidates[:min(cfg.working_batch,available)]])
                state,info=self.restricted(state,active,log)
                d=self.certificate(state)
                if d['objective']<best_obj:best_obj=d['objective'];best=state.copy()
                if d['head_kkt']>cfg.head_tolerance or d['alpha_conditional_gap']>cfg.alpha_gap_tolerance:
                    # Rescue is most useful once all meaningful candidates have entered.
                    missing=np.max(np.maximum(abs(d['gr'])-self.penalty_R,0)[np.setdiff1d(np.arange(self.p),active)],initial=0.)
                    if missing<=cfg.head_tolerance:
                        for _ in range(min(3,cfg.rescue_cycles)):
                            state=self.coordinate_rescue(state,active,log);d=self.certificate(state)
                            if d['objective']<best_obj:best_obj=d['objective'];best=state.copy()
                            if d['head_kkt']<=cfg.head_tolerance and d['alpha_conditional_gap']<=cfg.alpha_gap_tolerance:break
                elapsed=time.monotonic()-self.started+self.elapsed_before
                entry={k:val for k,val in d.items() if k not in ('gr','ga','scores')}
                entry.update(round=round_number+1,elapsed_seconds=elapsed,working_columns=len(active),optimiser=info)
                history.append(entry)
                log(f'round {round_number+1}: J={d["objective"]:.12g}, head={d["head_kkt"]:.3g}, alpha gap={d["alpha_conditional_gap"]:.3g}, selected Ricci={d["ricci_nonzero"]}')
                if checkpoint:
                    checkpoint(best,{'next_round':round_number+1,'active':active.tolist(),'history':history,
                                     'elapsed_seconds':elapsed})
                if d['head_kkt']<=cfg.head_tolerance and d['alpha_conditional_gap']<=cfg.alpha_gap_tolerance:
                    # Prefer the checked state; its objective can differ from the recorded best only by numerical slack.
                    best=state.copy();stop='LOCAL_STATIONARITY_REACHED';break
        except TimeBudget:
            stop='TIME_BUDGET_REACHED'
            log('Finite training budget reached. Retaining best finite state with explicit warning.')
        final=self.certificate(best)
        converged=final['head_kkt']<=cfg.head_tolerance and final['alpha_conditional_gap']<=cfg.alpha_gap_tolerance
        # Objective bound also bounds H0 coefficients, independent of conditioning.
        baseline=float(np.log(2.))
        if not history and not converged:
            raise FloatingPointError('No completed optimisation step is available. No baseline-only substitute is scored.')
        if final['objective']>baseline+1e-8 or not np.isfinite(final['objective']):
            raise FloatingPointError('Best fit failed the finite objective safety check.')
        if self.m and (best.alpha.min()<cfg.alpha_floor-1e-10 or abs(best.alpha.sum()-self.m)>1e-8):
            raise FloatingPointError('Best fit violates alpha constraints.')
        diagnostic={k:val for k,val in final.items() if k not in ('gr','ga','scores')}
        diagnostic.update(status='CONVERGED' if converged else 'FINITE_BUDGET_APPROXIMATION',
            converged=bool(converged),stop_reason=stop,elapsed_seconds=time.monotonic()-self.started+self.elapsed_before,
            evaluations=self.evaluations,history=history,global_optimum_certified=False,
            coefficient_norm_upper_bound=float(np.sqrt(2*baseline/cfg.h0_ridge)),
            old_100_refit_stability_protocol_used=False,initialisation='uniform alpha, zero head; or same-fit checkpoint resume',
            explicit_h0_ridge=cfg.h0_ridge,approximation_warning=None if converged else
             'This is a predefined finite-budget fit, not a certified stationary solution. No failed fold was omitted.')
        return FittedModel(best,self.hscale,self.rscale,cfg,diagnostic)
