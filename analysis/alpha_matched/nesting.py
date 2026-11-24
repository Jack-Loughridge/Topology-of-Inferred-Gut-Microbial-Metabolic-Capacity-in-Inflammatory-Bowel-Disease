"""Matched nested selection of standalone feature generators and a downstream head."""
from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from .partitions import canonical_json,digest,fusion_input


def weights(person):
    _,index,count=np.unique(person,return_inverse=True,return_counts=True)
    return 1/count[index]


def score(parts):
    """Average eligible whole-cohort participant AUROC over held-out predictions."""
    people={}
    for frame,prob in parts:
        for p,c,y,v in zip(frame.person,frame.cohort,frame.y,prob):
            key=(c,p)
            item=people.setdefault(key,[y,[]])
            if y!=item[0]:
                raise ValueError('Panel endpoint changes within person')
            item[1].append(v)
    cohorts={}
    for (c,_),(y,p) in people.items():
        cohorts.setdefault(c,[]).append((y,np.mean(p,axis=0)))
    auc=[]
    for rows in cohorts.values():
        labels=np.array([r[0] for r in rows])
        proba=np.stack([r[1] for r in rows])
        if len(np.unique(labels))!=proba.shape[1]:
            continue
        auc.append(roc_auc_score(labels,proba[:,1]) if proba.shape[1]==2 else
                   roc_auc_score(labels,proba,multi_class='ovr',average='macro'))
    if not auc:
        raise ValueError('No eligible cohort AUROC in this set of validation folds')
    return float(np.mean(auc))


def fit_head(setting,x,frame):
    family,parameter=setting
    if family=='linear':
        model=LogisticRegression(C=parameter,max_iter=1000,random_state=42)
    elif family=='forest':
        model=ExtraTreesClassifier(n_estimators=96,min_samples_leaf=parameter,max_features='sqrt',
                                   n_jobs=1,random_state=42)
    else:
        raise ValueError(f'Unknown head family: {family}')
    model.fit(x,frame.y,sample_weight=weights(frame.person))
    return model


@dataclass
class Fit:
    generator: object
    canonical_train: object
    model: object
    representation: object
    head: tuple


class Procedure:
    """Factory receives only current training rows and their labels. Its transform
    accepts identity lists only. Fold-local heads share exact standalone vectors.
    """
    def __init__(self,meta,splits,fit_generator,candidates,heads,log):
        if not meta.index.is_unique:
            raise ValueError('Duplicate sample ID')
        self.meta=meta
        self.splits=splits
        self.factory=fit_generator
        self.candidates=candidates
        self.heads=heads
        self.log=log

    def frames(self,key):
        spec=self.splits[key]
        a=self.meta[self.meta.person.isin(spec['train'])]
        b=self.meta[self.meta.person.isin(spec['validate'])]
        if (set(a.group)&set(b.group) or set(a.person)&set(b.person)
                or set(a.person)!=set(spec['train']) or set(b.person)!=set(spec['validate'])):
            raise ValueError('Manifest group or participant mismatch')
        return a,b

    def children(self,key,parent):
        prefix=key+'/'
        keys=sorted(k for k in self.splits if k.startswith(prefix) and '/' not in k[len(prefix):])
        result=[]
        seen=set()
        roster=set(parent.person)
        for child in keys:
            a,b=self.frames(child)
            if set(a.person)|set(b.person)!=roster or seen&set(b.person):
                raise ValueError('Invalid nested training partition')
            seen.update(b.person)
            result.append((child,a,b))
        if seen!=roster:
            raise ValueError('Validation coverage differs from parent')
        return result

    def build(self,block,setting,train,valid):
        gen=self.factory(block,setting,train.copy())
        t=gen.transform(tuple(train.index))
        v=gen.transform(tuple(valid.index))
        expected=digest(canonical_json(sorted(zip(train.index,train.person))))
        if (t.partition!=expected or t.block!=block or v.block!=block or t.arm!=v.arm
                or t.ids!=tuple(train.index) or v.ids!=tuple(valid.index) or t.features!=v.features
                or t.partition!=v.partition or t.generator!=v.generator):
            raise ValueError('Generator provenance or axes disagree')
        return gen,t,v

    def select(self,block,key,train):
        best=None
        folds=self.children(key,train)
        for setting in self.candidates[block]:
            scores={head:[] for head in self.heads}
            for child,a,b in folds:
                _,t,v=self.build(block,setting,a,b)
                for head in self.heads:
                    model=fit_head(head,t.values,a)
                    scores[head].append((b,model.predict_proba(v.values)))
            for head,parts in scores.items():
                value=score(parts)
                self.log(key,block,setting,head,value)
                if best is None or value>best[0]:
                    best=(value,setting,head)
        return best[1:]

    def fit_standalone(self,block,key,train,valid):
        setting,head=self.select(block,key,train)
        gen,t,v=self.build(block,setting,train,valid)
        model=fit_head(head,t.values,train)
        return Fit(gen,t,model,setting,head),v

    def outer(self,key):
        train,valid=self.frames(key)
        fusion_trials={head:[] for head in self.heads}
        for child,a,b in self.children(key,train):
            blocks=[self.fit_standalone(name,child,a,b) for name in ('H0','Ricci')]
            tr=fusion_input(*(fit.canonical_train for fit,_ in blocks))
            va=fusion_input(*(v for _,v in blocks))
            for head in self.heads:
                model=fit_head(head,tr,a)
                fusion_trials[head].append((b,model.predict_proba(va)))
        scores={head:score(parts) for head,parts in fusion_trials.items()}
        for head,value in scores.items():
            self.log(key,'joint',None,head,value)
        joint_head=max(scores,key=scores.get)
        blocks=[self.fit_standalone(name,key,train,valid) for name in ('H0','Ricci')]
        hashes=[(f.canonical_train.identity(),v.identity()) for f,v in blocks]
        joint=fit_head(joint_head,fusion_input(*(f.canonical_train for f,_ in blocks)),train)
        predictions=[f.model.predict_proba(v.values) for f,v in blocks]
        predictions.append(joint.predict_proba(fusion_input(*(v for _,v in blocks))))
        if hashes!=[(f.canonical_train.identity(),v.identity()) for f,v in blocks]:
            raise RuntimeError('Joint training mutated standalone vectors')
        return blocks,joint,predictions,hashes
