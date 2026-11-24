"""Matched baseline fits on the same visits and participant/family partitions."""
import numpy as np
from .h0 import merge_deaths
from .nesting import fit_head, score, weights
from .partitions import digest,canonical_json,RelativeReference


class Baseline:
    def __init__(self,graphs,name,train):
        self.graphs=graphs
        self.name=name
        self.reference=None
        self.species_impute=None
        self.species_axis=None
        if name=='species':
            rows=[graphs.lookup[s] for s in train.index]
            observed=graphs.species_abundance[rows].copy()
            observed.eliminate_zeros()
            self.species_axis=np.unique(observed.indices)
            mask=getattr(graphs,'species_observed',None)
            if mask is not None:
                native=mask[np.ix_(rows,self.species_axis)]
                w=weights(train.person)
                den=w@native
                if np.any(den<=0):
                    raise ValueError('Unobserved fitted species coordinate')
                self.species_impute=(w@observed[:,self.species_axis].toarray())/den
        if name=='direct-relative':
            idx=[graphs.lookup[x] for x in train.index]
            self.reference=RelativeReference.fit(graphs.raw[idx],list(train.person),list(train.index),graphs.axis)
        t=self.values(train.index)
        w=weights(train.person)
        self.mean=np.average(t,axis=0,weights=w)
        self.scale=np.sqrt(np.average((t-self.mean)**2,axis=0,weights=w))
        self.scale[self.scale==0]=1.
        self.partition=digest(canonical_json(sorted(zip(train.index,train.person))))

    def values(self,ids):
        g=self.graphs
        rows=[g.lookup[s] for s in ids]
        if self.name=='species':
            values=g.species_abundance[rows][:,self.species_axis].toarray()
            if self.species_impute is None:
                return values
            if g.species_observed is None:
                if not g.species_nondetection_zero:
                    raise ValueError('No qualified native measurement roster')
                measured=np.ones(values.shape,dtype=bool)
            else:
                measured=g.species_observed[np.ix_(rows,self.species_axis)]
            return np.concatenate([np.where(measured,values,self.species_impute),~measured],axis=1)
        if self.name=='graph-count':
            result=[]
            for i in rows:
                e,logs,v,pairs=g.graph(i,None)
                deaths,essential=merge_deaths(len(v),pairs,logs)
                result.append([len(v),len(e),len(deaths),essential])
            return np.array(result,dtype=float).reshape(-1,4)
        x=g.raw[rows].copy()
        if self.name=='activity':
            x.data[:]=1.
        elif self.name=='direct-relative':
            # Exponentiate only after subtracting the fitted log reference.
            # Values outside the model dtype require repair, never saturation.
            logs=np.log(x.data)-self.reference.log_reference[x.indices]
            x.data=np.exp(logs)
            if not np.isfinite(x.data).all() or np.any(x.data<=0):
                raise ArithmeticError('Direct relative comparator exceeded numerical range')
        elif self.name!='direct-raw':
            raise ValueError('Unknown baseline')
        return x.toarray()

    def transform(self,ids):
        return (self.values(ids)-self.mean)/self.scale


def fit_baseline(procedure,graphs,name,key):
    train,valid=procedure.frames(key)
    trials={h:[] for h in procedure.heads}
    for child,a,b in procedure.children(key,train):
        transform=Baseline(graphs,name,a)
        tr=transform.transform(a.index)
        va=transform.transform(b.index)
        for head in trials:
            clf=fit_head(head,tr,a)
            trials[head].append((b,clf.predict_proba(va)))
    scores={h:score(parts) for h,parts in trials.items()}
    for h,s in scores.items():
        procedure.log(key,name,None,h,s)
    head=max(scores,key=scores.get)
    transform=Baseline(graphs,name,train)
    clf=fit_head(head,transform.transform(train.index),train)
    return transform,clf,clf.predict_proba(transform.transform(valid.index))


def historical_reference(procedure,factory,key,settings):
    """Inner training/validation independently fit every alpha and original logit head."""
    from .h0 import fit_historical
    def fit(a,b,setting):
        idx=[factory.graphs.lookup[s] for s in a.index]
        ref=None
        if factory.arm=='RELATIVE':
            ref=RelativeReference.fit(factory.graphs.raw[idx],list(a.person),list(a.index),factory.graphs.axis)
        def diagrams(frame):
            result=[]
            for sample in frame.index:
                _,log,v,p=factory.graphs.graph(factory.graphs.lookup[sample],ref)
                result.append(merge_deaths(len(v),p,log)[0])
            return result
        model=fit_historical(diagrams(a),a.y,a.person,**dict(setting))
        return model,ref,model.probabilities(diagrams(b))
    train,valid=procedure.frames(key)
    results={}
    for setting in settings:
        parts=[]
        for _,a,b in procedure.children(key,train):
            _,_,probs=fit(a,b,setting)
            parts.append((b,probs))
        results[setting]=score(parts)
        procedure.log(key,'historical-corrected-alpha-PI',setting,'original-logits',results[setting])
    return fit(train,valid,max(results,key=results.get))


def curvature_only(procedure,key):
    """Refit only a downstream head on the selected curvature coordinates.

    Each inner validation first selects the complete standalone Ricci pipeline
    inside its training portion. The canonical Ricci block records K0 and B.
    """
    train,valid=procedure.frames(key)
    trials={h:[] for h in procedure.heads}
    def select_and_transform(node,a,b):
        fit,v=procedure.fit_standalone('Ricci',node,a,b)
        keep=np.asarray([col.startswith('K0:') for col in v.features])
        if np.count_nonzero(keep)*2!=len(v.features):
            raise ValueError('Invalid canonical Ricci coordinate policy')
        return fit,keep,fit.canonical_train.values[:,keep],v.values[:,keep]
    for child,a,b in procedure.children(key,train):
        _,_,t,v=select_and_transform(child,a,b)
        for head in trials:
            model=fit_head(head,t,a)
            trials[head].append((b,model.predict_proba(v)))
    scores={h:score(parts) for h,parts in trials.items()}
    for h,val in scores.items():
        procedure.log(key,'curvature-only',None,h,val)
    head=max(scores,key=scores.get)
    fit,mask,t,v=select_and_transform(key,train,valid)
    model=fit_head(head,t,train)
    return fit,mask,model,model.predict_proba(v)


def fixed_alpha(procedure,key):
    train,valid=procedure.frames(key)
    fixed=[s for s in procedure.candidates['H0'] if not dict(s)['learned']]
    if not fixed:
        raise ValueError('Fixed-alpha reference is required in both arms')
    from .nesting import Procedure
    proc=Procedure(procedure.meta,procedure.splits,procedure.factory,{'H0':fixed},procedure.heads,
                   lambda node,block,setting,head,s:procedure.log(node,'fixed-alpha',setting,head,s))
    fit,v=proc.fit_standalone('H0',key,train,valid)
    return fit,v,fit.model.predict_proba(v.values)
