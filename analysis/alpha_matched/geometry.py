"""Selected-RHS first hits and certified, full-support directed transport."""
from collections import OrderedDict
from dataclasses import dataclass
from fractions import Fraction as F
import os

# POT is used only as a CPU transport proposal.
for backend in ('PYTORCH','JAX','TENSORFLOW','CUPY'):
    os.environ.setdefault('POT_BACKEND_DISABLE_'+backend,'1')

import numpy as np
import ot
from scipy.sparse import eye
from scipy.sparse.linalg import splu

from . import certified_reference as ref
from .log_kernel import construct, parameter_value, ratio
from .integer_transport import solve as integer_transport

EXT=np.longdouble
U=np.finfo(EXT).eps
def up(x):
    return 0. if x==0 else np.nextafter(float(x),np.inf).item()


LOG2=EXT("0.693147180559945309417232121458176568075500134360255254120680")


@dataclass
class Geometry:
    pairs: np.ndarray
    curvature: np.ndarray
    error: dict
    n_vertices: int


def neglog(x):
    """Return -log(x) and a conservative absolute floating arithmetic bound.

    frexp is exact. For m in [1/2,1), t=(m-1)/(m+1) has |t|<=1/3.
    The omitted tail in 2*atanh(t) after degree 47 is <2e-25.
    """
    m,e=np.frexp(x)
    t=(m-1)/(m+1)
    t2=t*t
    series=np.full_like(t,EXT(1)/47)
    for k in range(45,0,-2):
        series=EXT(1)/k+t2*series
    y=-2*t*series-e*LOG2
    err=128*U*(1+np.abs(e))+EXT("2e-25")
    return y,err


class Costs:
    """Factor once. Retain bounded target columns and generate missing ones in batches."""
    def __init__(self,p,rho,eps,limit,row_tv):
        self.p=p
        self.rho=EXT(rho)
        self.eps=EXT(eps)
        self.n=p.shape[0]
        self.cache=OrderedDict()
        # Budget column tables and target solve scratch. Sparse LU fill is data-dependent.
        minimum=self.n*256+4096
        if limit<minimum:
            raise MemoryError("budget smaller than one selected-RHS column and its scratch")
        self.limit=max(1,min(self.n,(limit//2)//max(1,16*self.n)))
        self.batch=max(1,min(16,(limit//2)//max(1,256*self.n)))
        self.factor=splu(eye(self.n,format='csc')-float(rho)*p.astype(float).tocsc())
        degree=np.diff(p.indptr).max(initial=0)
        # h lies in [0,1], hence row TV times its oscillation bounds kernel error.
        # Three roundings on a positive rational split, sparse positive dot product,
        # subtraction and interval endpoint arithmetic are covered by this gamma bound.
        if (degree+1)*U>EXT("0.0001"):
            raise ArithmeticError("sparse residual degree exceeds rounding proof")
        self.guard=EXT(ref.upper_float(max(row_tv,default=F(0))))+32*U*(degree+2)
        self.max_hit=0.

    def _solve(self,targets):
        rhs=np.zeros((self.n,len(targets)))
        ix=np.arange(len(targets))
        rhs[targets,ix]=1
        z=self.factor.solve(rhs)
        if not np.isfinite(z).all() or np.any(z[targets,ix]<=0):
            raise ArithmeticError("invalid selected resolvent column")
        h=z/z[targets,ix]
        if not np.isfinite(h).all() or h.min() < -1e-10 or h.max()>1+1e-10:
            raise ArithmeticError("invalid first-hit approximation")
        h=np.clip(h,0.,float(self.rho)).astype(EXT)
        h[targets,ix]=1
        residual=np.abs(h-self.rho*(self.p@h))
        residual[targets,ix]=0
        eta=(residual.max(axis=0)+self.guard)/(1-self.rho)*(1+4*U)
        self.max_hit=max(self.max_hit,up(eta.max()))
        lowq=np.maximum(0,h-eta-4*U)
        highq=np.minimum(self.rho,h+eta+4*U)
        dl,el=neglog(self.eps+(1-self.eps)*highq)
        dh,eh=neglog(self.eps+(1-self.eps)*lowq)
        lo=dl-el
        hi=dh+eh
        # Convert outwards to float64 bounds, centre/radius in column-major work.
        lo=np.nextafter(lo.astype(float),-np.inf)
        hi=np.nextafter(hi.astype(float),np.inf)
        lo[targets,ix]=hi[targets,ix]=0
        return [(lo[:,j].copy(),hi[:,j].copy()) for j in ix]

    def subset(self,x,y):
        low=np.empty((len(x),len(y)))
        high=np.empty_like(low)
        missing=[]
        for j,target in enumerate(y):
            if target in self.cache:
                lo,hi=self.cache[target]
                self.cache.move_to_end(target)
                low[:,j],high[:,j]=lo[x],hi[x]
            else:
                missing.append((j,target))
        for start in range(0,len(missing),self.batch):
            items=missing[start:start+self.batch]
            for (j,t),(lo,hi) in zip(items,self._solve([t for j,t in items])):
                low[:,j],high[:,j]=lo[x],hi[x]
                self.cache[t]=(lo,hi)
                if len(self.cache)>self.limit:
                    self.cache.popitem(last=False)
        return low,high


def _marginals(p,degree,int_rows,row_tv):
    out=[]
    for i,k in enumerate(degree):
        js,nums,den=int_rows[i]
        if not k:
            out.append((np.array([i]),np.array([1],dtype=EXT),F(0)))
            continue
        row=slice(p.indptr[i],p.indptr[i+1])
        weights=np.r_[EXT(1)/(k+1),p.data[row]*k/(k+1)]
        out.append((np.array([i,*js]),weights,row_tv[i]*F(int(k),int(k)+1)))
    return out


def _certify_fast(a,b,ta,tb,lo,hi,M,marginal_tolerance):
    c=np.ascontiguousarray((lo+hi)*0.5)
    plan,info=ot.emd(a.astype(float),b.astype(float),c,log=True,numItermax=1000000,numThreads=1)
    if (info['warning'] or not np.isfinite(plan).all() or np.min(plan)<0 or
        not np.isfinite(info['u']).all() or not np.isfinite(info['v']).all()):
        raise ArithmeticError("transport solver returned no finite feasible proposal")
    plan=plan.astype(EXT)
    nr,nc=plan.shape
    if (1+nr*nc)*U>EXT("0.0001"):
        raise ArithmeticError("transport size exceeds summation rounding proof")
    fa=info['u'].astype(EXT)
    fb=info['v'].astype(EXT)
    scale=1+M+np.abs(fa).max()+np.abs(fb).max()
    # All primal entries and costs are nonnegative. The L1 adjustment required
    # to strip excess and refill deficits costs <=M times total marginal L1 error.
    ga=32*U*(nr+nc+2)
    l1a=2*EXT(ref.upper_float(ta))+ga
    l1b=2*EXT(ref.upper_float(tb))+ga
    ma=np.abs(plan.sum(axis=1)-a).sum()+l1a
    mb=np.abs(plan.sum(axis=0)-b).sum()+l1b
    marginal=ma+mb
    if marginal>marginal_tolerance:
        raise ArithmeticError("transport marginal tolerance exceeded")
    # With masses near one, sum roundoff on objective <=O(nr*nc*u)*M.
    rounding=64*U*(1+nr*nc)*scale
    violation=max(0,np.max(fa[:,None]+fb-lo))
    lower=a@fa+b@fb-violation-l1a*np.abs(fa).max()-l1b*np.abs(fb).max()-rounding
    upper=np.sum(plan*hi)+M*marginal+rounding
    return max(EXT(0),lower),min(M,upper),up(marginal)


def _features(n,pairs,support,opts):
    family,parameter,rho,eps,budget,tol,mtol=opts
    p,degree,int_rows,row_tv=construct(n,pairs,support,family,parameter)
    costs=Costs(p,rho,eps,budget,row_tv)
    marginals=_marginals(p,degree,int_rows,row_tv)
    M=EXT(ref.upper_float(ref.distance(F(0),ratio(eps)).hi))
    result=np.empty(len(pairs))
    errors=dict.fromkeys(('first_hit_error','transport_error','marginal_error','curvature_error'),0.)
    repaired=0
    reference=None
    for idx,(u,v) in enumerate(pairs):
        x,a,ta=marginals[u]
        y,b,tb=marginals[v]
        try:
            # x[0] is u and y[0] is v for both sinks and nonsinks.
            lo,hi=costs.subset(x,y)
            low,high,marg=_certify_fast(a,b,ta,tb,lo,hi,M,mtol)
            dl,dh=EXT(lo[0,0]),EXT(hi[0,0])
            if not (dl>0 and low<=high and np.isfinite([dl,dh,low,high]).all()):
                raise ArithmeticError("invalid transport bracket")
            # Outward padding covers elementary ratio arithmetic, independent of the LP.
            pad=16*U*(1+M/dl)
            kl,kh=1-high/dl-pad,1-low/dh+pad
            center=0. if kl<=0<=kh else float((kl+kh)/2)
            error=up(max(EXT(center)-kl,kh-EXT(center)))
            if error>tol:
                raise ArithmeticError("fast interval too wide")
            gap=up(high-low)
        except ArithmeticError:
            if reference is None:
                reference=RationalRepair(int_rows,row_tv,degree,rho,eps,budget)
            center,error,gap,marg=reference.edge(int(u),int(v),x,y,tol)
            if marg>mtol:
                raise ArithmeticError("repaired endpoint marginal tolerance unavailable")
            repaired+=1
        result[idx]=center
        errors['transport_error']=max(errors['transport_error'],gap)
        errors['marginal_error']=max(errors['marginal_error'],marg)
        errors['curvature_error']=max(errors['curvature_error'],error)
    errors['first_hit_error']=max(costs.max_hit,reference.max_hit if reference else 0.)
    errors['rational_repair_edges']=repaired
    if not np.isfinite(result).all():
        raise ArithmeticError("nonfinite active curvature")
    return Geometry(pairs,result,errors,n)


class RationalRepair:
    """Certify selected columns and transport using exact rational arithmetic."""
    def __init__(self, int_rows,tv,degrees,rho,eps,budget):
        self.rho=ratio(rho)
        self.eps=ratio(eps)
        self.rows=[{j:F(x,den) for j,x in zip(js,nums)} for js,nums,den in int_rows]
        self.tv=tv
        self.mus=[]
        self.mu_tv=[]
        for i,k in enumerate(map(int,degrees)):
            self.mus.append(({i:F(1,k+1),**{j:p*F(k,k+1) for j,p in self.rows[i].items()}}
                             if k else {i:F(1)}))
            self.mu_tv.append(F(k,k+1)*tv[i])
        self.factor=ref.sparse_factor(self.rows,self.rho)
        self.columns=OrderedDict()
        self.limit=max(1,budget//(4096*max(1,len(self.rows))))
        self.max_hit=0.
        self.M=ref.distance(F(0),self.eps).hi

    def column(self,t,exact):
        cache_key=(t,exact)
        if cache_key in self.columns:
            self.columns.move_to_end(cache_key)
            return self.columns[cache_key]
        n=len(self.rows)
        if exact:
            # Arbitrary-precision fixed point avoids dense n^2 rational systems on large graphs.
            h=(ref.exact_column(self.rows,t,self.rho) if n<=24 else self.iterate(t))
        else:
            unit=np.zeros(n)
            unit[t]=1
            z=self.factor.solve(unit)
            if not np.isfinite(z).all() or z[t]<=0:
                raise ArithmeticError("invalid rational-repair proposal")
            h=[min(self.rho,max(F(0),F(float(x)))) for x in z/z[t]]
            h[t]=F(1)
        r=max((abs(h[i]-self.rho*sum(p*h[j] for j,p in row.items()))+self.rho*self.tv[i]
               for i,row in enumerate(self.rows) if i!=t),default=F(0))
        err=r/(1-self.rho)
        self.max_hit=max(self.max_hit,ref.upper_float(err))
        col=[(ref.Band(F(0),F(0)) if i==t else
              ref.Band(ref.distance(min(self.rho,h[i]+err),self.eps).lo,
                       ref.distance(max(F(0),h[i]-err),self.eps).hi)) for i in range(n)]
        self.columns[cache_key]=col
        if len(self.columns)>self.limit:
            self.columns.popitem(last=False)
        return col

    def iterate(self,t):
        # Round iterates to a positive 144-bit grid. Compute exact residual afterwards.
        n=len(self.rows)
        scale=1 << 144
        old=[0]*n
        old[t]=scale
        stop=max(1,int(F(1,10**24)*(1-self.rho)*scale))
        while True:
            new=[(scale if i==t else int(self.rho*sum(p*old[j] for j,p in row.items())))
                 for i,row in enumerate(self.rows)]
            if max(abs(a-b) for a,b in zip(new,old))<=stop:
                return [F(z,scale) for z in new]
            old=new

    def edge(self,u,v,x,y,tol):
        for exact in (False,True):
            columns=[self.column(int(t),exact) for t in y]
            c=[[col[i] for col in columns] for i in x]
            ma=[self.mus[u][int(i)] for i in x]
            mb=[self.mus[v][int(j)] for j in y]
            if exact:
                w=integer_transport(ma,mb,c)
            else:
                try:
                    w,_=ref.transport(ma,mb,c)
                except ArithmeticError:
                    continue
            drift=self.M*(self.mu_tv[u]+self.mu_tv[v])
            low,high=max(F(0),w.lo-drift),min(self.M,w.hi+drift)
            d=c[0][0]
            kl,kh=1-high/d.lo,1-low/d.hi
            center=0. if kl<=0<=kh else float((kl+kh)/2)
            err=ref.upper_float(max(F(center)-kl,kh-F(center)))
            if err<=tol:
                return center,err,ref.upper_float(high-low),ref.upper_float(2*(self.mu_tv[u]+self.mu_tv[v]))
        raise ArithmeticError(f"certified curvature tolerance unavailable at active edge {(u,v)}")


def compute(n,pairs,log_support,*,family,parameter,rho,epsilon=1e-4,
            memory_bytes=512*1024**2,curvature_tolerance=1e-7,marginal_tolerance=1e-8):
    raw=np.asarray(pairs)
    if raw.size and (raw.ndim!=2 or raw.shape[1]!=2 or not np.issubdtype(raw.dtype,np.integer)):
        raise ValueError("expected integer active edge pairs with shape (edges,2)")
    pairs=np.asarray(raw,dtype=np.int64).reshape(-1,2)
    support=np.asarray(log_support,dtype=EXT)
    if (not isinstance(n,(int,np.integer)) or not (0<rho<1 and 0<epsilon<1 and n>=0 and
        np.isfinite([curvature_tolerance,marginal_tolerance]).all() and
        min(curvature_tolerance,marginal_tolerance,memory_bytes)>0 and support.shape==(len(pairs),))):
        raise ValueError("invalid geometry inputs")
    parameter_value(family,parameter)
    if (not np.isfinite(support).all() or np.any(pairs<0) or np.any(pairs>=n) or
        np.any(pairs[:,0]==pairs[:,1]) or len(np.unique(pairs,axis=0))!=len(pairs)):
        raise ValueError("invalid active edge")
    if not len(pairs):
        errors=dict.fromkeys(('first_hit_error','transport_error','marginal_error','curvature_error','rational_repair_edges'),0.)
        return Geometry(pairs,np.zeros(0),errors,n)
    return _features(n,pairs,support,(family,parameter,rho,epsilon,memory_bytes,curvature_tolerance,marginal_tolerance))
