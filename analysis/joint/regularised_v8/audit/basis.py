"""Standalone-style H0: raw death-axis quadrature, source-death alpha weights.

All arithmetic here is float64, rather than the historical Torch float32.
The integration rule, kernel, bandwidth rule and minimum boundary gap match the
standalone model. M specifies quantiles 100*j/M, j=1,...,M-1. Ties are merged;
requested and realised counts are distinct, recorded quantities.
"""
from dataclasses import dataclass
import numpy as np

EPS=1e-12

@dataclass(frozen=True)
class Intervals:
    bounds: np.ndarray
    sigma: float
    requested: int
    @property
    def size(self): return len(self.bounds)-1
    def payload(self):
        return {'bounds':self.bounds.tolist(),'sigma':self.sigma,'requested':self.requested,
                'realised':self.size,'merged_boundaries':self.requested-self.size,
                'axis':'original_death','arithmetic':'float64'}

def filter_deaths(d):
    d=np.asarray(d,dtype=np.float64).reshape(-1)
    return np.sort(d[np.isfinite(d)&(d>0)&(d<1-1e-12)])

def fit_intervals(train_diagrams, requested, min_width=1e-8):
    if requested<2: raise ValueError('At least two requested intervals required.')
    pieces=[filter_deaths(d) for d in train_diagrams]
    if not pieces or sum(map(len,pieces))<10:
        raise ValueError('Insufficient eligible training deaths; no fallback basis.')
    pooled=np.concatenate(pieces)
    q=np.arange(1,requested,dtype=float)*100.0/requested
    raw=np.percentile(pooled,q,method='linear')
    bounds=[0.0]
    for v in np.sort(raw):
        if 0<v<1 and v>bounds[-1]+min_width: bounds.append(float(v))
    if 1>bounds[-1]+min_width: bounds.append(1.0)
    else: bounds[-1]=1.0
    b=np.asarray(bounds,float)
    if len(b)<3 or np.any(np.diff(b)<=0): raise ValueError('Degenerate train-only intervals.')
    centers=(b[:-1]+b[1:])/2
    spacing=np.diff(np.log(centers+EPS)); spacing=spacing[spacing>0]
    sigma=max(2*float(np.median(spacing)),1e-4) if len(spacing) else 0.1
    return Intervals(b,sigma,int(requested))

def source_regions(deaths, intervals):
    d=filter_deaths(deaths)
    return d,np.clip(np.searchsorted(intervals.bounds,d,side='right')-1,0,intervals.size-1)

def tensor_one(deaths, intervals, points=5, chunk=1024):
    """A[output_bin,source_region]; no kernel tails truncated or counts normalised."""
    m=intervals.size; out=np.zeros((m,m),float)
    d,k=source_regions(deaths,intervals)
    if not len(d): return out
    nodes,weights=np.polynomial.legendre.leggauss(points)
    a,b=intervals.bounds[:-1],intervals.bounds[1:]; half=(b-a)/2
    x=((a+b)[:,None]/2+half[:,None]*nodes).reshape(-1)
    log_x=np.log(x+EPS); log_d=np.log(d+EPS)
    for start in range(0,len(d),chunk):
        stop=min(start+chunk,len(d))
        diff=(log_x[:,None]-log_d[None,start:stop])/intervals.sigma
        kernels=np.exp(-0.5*diff*diff).reshape(m,points,-1)
        masses=half[:,None]*np.sum(kernels*weights[None,:,None],axis=1)
        region=k[start:stop]; starts=np.r_[0,1+np.flatnonzero(np.diff(region))]
        out[:,region[starts]]+=np.add.reduceat(masses,starts,axis=1)
    if not np.isfinite(out).all() or np.any(out<0): raise FloatingPointError('Invalid H0 tensor.')
    return out

def tensor_many(diagrams, intervals, points=5, progress=None):
    out=np.empty((len(diagrams),intervals.size,intervals.size),float)
    for i,d in enumerate(diagrams):
        out[i]=tensor_one(d,intervals,points)
        if progress and ((i+1)%100==0 or i+1==len(diagrams)):
            progress(f'basis requested {intervals.requested}, realised {intervals.size}: {i+1}/{len(diagrams)}')
    return out

def direct_oracle(deaths, intervals, alpha, points=5):
    """Independent loop implementation of weighting each death before quadrature."""
    d,k=source_regions(deaths,intervals)
    q,w=np.polynomial.legendre.leggauss(points); result=[]
    for a,b in zip(intervals.bounds[:-1],intervals.bounds[1:]):
        x=(a+b)/2+(b-a)/2*q
        rho=np.zeros_like(x)
        for v,r in zip(d,k):
            rho+=alpha[r]*np.exp(-((np.log(x+EPS)-np.log(v+EPS))/intervals.sigma)**2/2)
        result.append((b-a)/2*np.dot(w,rho))
    return np.asarray(result)

@dataclass(frozen=True)
class Scale:
    mean: np.ndarray
    sd: np.ndarray
    def transform(self,x): return (x-self.mean)/self.sd
    @classmethod
    def fit(cls,x):
        mean=np.mean(x,axis=0); sd=np.std(x,axis=0,ddof=0)
        return cls(mean,np.where(sd>1e-12,sd,1.0))

def features(A,alpha,scale):
    # A is output-bin x source-region. Never perform post-bin alpha multiplication.
    return np.ascontiguousarray(scale.transform(A@alpha))
