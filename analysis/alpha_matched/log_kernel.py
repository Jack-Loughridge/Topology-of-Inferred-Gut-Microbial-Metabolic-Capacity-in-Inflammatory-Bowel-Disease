"""Interval-qualified stochastic kernel for a frozen array of input log supports."""
from fractions import Fraction as F
from numbers import Integral

from mpmath.libmp import to_int, round_floor, round_ceiling
import numpy as np
from scipy.sparse import csr_matrix

from .certified_reference import IV, exact_interval, upper_float

BITS = 96
SCALE = 1 << BITS


def ratio(x):
    if isinstance(x,Integral):
        return F(int(x))
    if isinstance(x,(str,F)):
        return F(x)
    n,d=x.as_integer_ratio()
    return F(n,d)


def parameter_value(family, parameter):
    if family in ("uniform","linear"):
        return None
    if family not in ("power","saturation") or parameter is None or ratio(parameter)<=0:
        raise ValueError("Invalid transition family or parameter")
    return ratio(parameter)


def _response(logr, family, param):
    if family == "uniform":
        return IV.mpf(0)
    if family == "linear":
        return logr
    if family == "power":
        return param*logr
    # Stable even for large |log r|. param is already log K for saturation.
    z=logr-param
    if z.b<=0:
        return z-IV.log1p(IV.exp(z))
    return -IV.log1p(IV.exp(-z))


def construct(n,pairs,log_support,family,parameter):
    param=parameter_value(family,parameter)
    if param is not None:
        param=exact_interval(param)
        if family=="saturation":
            param=IV.ln(param)
    order=np.argsort(pairs[:,0],kind="stable")
    counts=np.bincount(pairs[:,0],minlength=n)
    positions=np.r_[0,counts.cumsum()]
    indptr=[0]
    columns=[]
    data=[]
    int_rows=[]
    row_tv=[]
    ext=np.longdouble
    for i in range(n):
        ids=order[positions[i]:positions[i+1]]
        if len(ids)==0:
            nums=[1]
            js=[i]
            tv=F(0)
        elif len(ids)==1:
            nums=[1]
            js=[int(pairs[ids[0],1])]
            tv=F(0)
        else:
            js=[int(x) for x in pairs[ids,1]]
            lw=[_response(exact_interval(ratio(log_support[e])),family,param) for e in ids]
            peak=max(x.b for x in lw)
            weights=[IV.exp(x-peak) for x in lw]
            denom=sum(weights)
            # Keep every structural entry positive, including probabilities below
            # binary64 or the lattice resolution. Bound the absolute TV change.
            brackets=[]
            for w in weights:
                lo,hi=(SCALE*w/denom)._mpi_
                brackets.append((to_int(lo,round_floor),to_int(hi,round_ceiling)))
            nums=[max(1,(lo+hi)//2) for lo,hi in brackets]
            total=sum(nums)
            err=sum(max(abs(z*SCALE-lo*total),abs(hi*total-z*SCALE))
                    for z,(lo,hi) in zip(nums,brackets))
            tv=F(err,2*SCALE*total)
        denom=sum(nums)
        columns.extend(js)
        # Both integers fit the range of binary80. Track the two conversions and
        # division when comparing this matrix to the exact rational centre.
        data.extend([ext(z)/ext(denom) for z in nums])
        int_rows.append((js,nums,denom))
        row_tv.append(tv)
        indptr.append(len(columns))
    p=csr_matrix((np.array(data,dtype=ext),columns,indptr),shape=(n,n))
    return p,counts,int_rows,row_tv
