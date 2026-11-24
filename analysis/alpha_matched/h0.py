"""Kruskal H0 and training-local source-death weighted persistence images."""
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
import torch
from torch.nn import functional as F

from .partitions import immutable_array


def merge_deaths(n_vertices,pairs,log_support):
    pairs=np.asarray(pairs,dtype=np.int64).reshape(-1,2)
    logs=np.asarray(log_support,dtype=np.longdouble)
    if len(logs)!=len(pairs) or np.any(pairs<0) or np.any(pairs>=n_vertices) or not np.isfinite(logs).all():
        raise ValueError("Invalid H0 input")
    parent=np.arange(n_vertices)
    def find(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]]
            i=parent[i]
        return i
    retained=[]
    for k in np.argsort(-logs,kind='stable'):
        a,b=pairs[k]
        a,b=find(a),find(b)
        if a!=b:
            parent[b]=a
            retained.append(logs[k])
    # Keep log of minus log(death); even when exp(-r) rounds to 0 or 1 the
    # finite merge is retained at full precision in this monotone coordinate.
    return np.asarray(retained,dtype=np.longdouble), n_vertices-len(retained)


def death_logs(log_strength):
    with np.errstate(over='ignore',under='ignore'):
        return -np.exp(np.asarray(log_strength,dtype=np.longdouble))


@dataclass(frozen=True)
class Image:
    bounds: np.ndarray
    bandwidth: float
    alpha: np.ndarray
    source_cuts: tuple

    @classmethod
    def initial(cls,diagrams,quantile_step):
        if not 0<quantile_step<=100:
            raise ValueError('Invalid quantile spacing')
        # The recovered bin estimator pools finite training merges, one count per
        # death. Participant and class weights apply to the supervised objective.
        # Negative log-strength increases monotonically with the death time.
        # Preserve the exact source coordinate when a percentile lands on a
        # training death, including repeated values.
        source=np.sort(-np.concatenate([np.asarray(d,dtype=np.longdouble) for d in diagrams]))
        bounds=[0.]
        cuts=[]
        step=Fraction(str(quantile_step))
        n_inside=(100*step.denominator-1)//step.numerator
        for index in range(1,n_inside+1) if len(source) else []:
            position=index*step*(len(source)-1)/100
            lo=position.numerator//position.denominator
            hi=-(-position.numerator//position.denominator)
            rem=position-lo
            w=np.longdouble(rem.numerator)/rem.denominator
            endpoints=np.exp(death_logs(-source[[lo,hi]]))
            val=(1-w)*endpoints[0]+w*endpoints[1]
            if val>bounds[-1]+1e-8 and val<1.-1e-8:
                cut=source[lo] if source[lo]==source[hi] else -np.log(-np.log(val))
                cuts.append(np.format_float_scientific(cut,unique=True,trim='k'))
                bounds.append(float(val))
        bounds.append(1.)
        bounds=np.array(bounds)
        centers=(bounds[:-1]+bounds[1:])/2
        delta=np.diff(np.log(centers+1e-12))
        sigma=max(1e-4,2*float(np.median(delta))) if len(delta) else .1
        return cls(immutable_array(bounds),sigma,immutable_array(np.ones(len(centers))),tuple(cuts))

    def basis(self,diagram):
        # Historical five-node Gauss-Legendre quadrature in linear death units.
        nodes,weights=np.polynomial.legendre.leggauss(5)
        half=np.diff(self.bounds)/2
        center=self.bounds[:-1]+half
        xp=center[:,None]+half[:,None]*nodes
        wp=half[:,None]*weights
        ld=death_logs(diagram)
        binid=np.searchsorted(np.asarray(self.source_cuts,dtype=np.longdouble),
                              -np.asarray(diagram,dtype=np.longdouble),side='right')
        logs=np.logaddexp(ld,np.log(np.longdouble(1e-12))).astype(np.float64)
        table=np.zeros((len(half),len(half)))
        for i in range(0,len(ld),64):
            dif=(np.log(xp+1e-12)[None,:,:]-logs[i:i+64,None,None])/self.bandwidth
            contributions=np.sum(np.exp(-.5*dif*dif)*wp,axis=-1)
            np.add.at(table,binid[i:i+64],contributions)
        if not np.isfinite(table).all():
            raise ArithmeticError('Nonfinite persistence image')
        return table

    def transform(self,diagrams):
        if len(diagrams)==0:
            return np.empty((0,len(self.alpha)))
        return np.stack([self.alpha@self.basis(d) for d in diagrams])


def _objective_weights(labels, people, dtype):
    """Within/radius use equal participant mass; CE also gives each class equal mass."""
    classes, y = np.unique(labels, return_inverse=True)
    if len(classes) < 2 or len(people) != len(y):
        raise ValueError("Need aligned training rows and at least two classes")
    _, person, visit_counts = np.unique(people, return_inverse=True, return_counts=True)
    w = 1.0 / visit_counts[person]
    w /= w.sum()
    class_masses = np.bincount(y, weights=w)
    wc = w / (len(classes) * class_masses[y])
    return (classes, torch.tensor(y, dtype=torch.long), torch.tensor(w, dtype=dtype),
            torch.tensor(wc, dtype=dtype))


def _alpha_loss(z, y, w, wc, centers, head, batch_count):
    """Averaging an epoch's batch losses at fixed parameters gives the full loss."""
    own = ((z - centers[y])**2).sum(dim=1)
    distance = torch.sqrt(((centers[:,None]-centers[None,:])**2).sum(dim=2) + 1e-12)
    n_classes = len(centers)
    nearest = (distance + torch.eye(n_classes, dtype=z.dtype)*1e9).min(dim=1).values
    radius = F.relu(torch.sqrt(own + 1e-12) - .5*nearest[y]).square()
    ce = (wc * F.cross_entropy(head(z), y, reduction="none")).sum()
    ii, jj = torch.triu_indices(n_classes, n_classes, offset=1)
    between = F.relu(10.0 - distance[ii,jj]).pow(4).mean()
    return batch_count * (1000*ce + (w*(own + radius)).sum()) + 200*between


def learn_image(diagrams,labels,people,*,quantile_step,epochs,seed):
    """Corrected centroid/radius/separation/CE standalone alpha objective.

    Full training partition gradient, participant-normalized weights, float64.
    Epoch count and grid must have been selected inside this partition.
    """
    if len(diagrams)!=len(labels) or len(people)!=len(labels):
        raise ValueError("Training rows do not align")
    image=Image.initial(diagrams,quantile_step)
    bases=torch.tensor(np.stack([image.basis(d) for d in diagrams]),dtype=torch.float64)
    classes, y, w, wc = _objective_weights(labels, people, torch.float64)
    k=len(image.alpha)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        raw=torch.nn.Parameter(torch.full((k,),float(np.log(np.expm1(1.))),dtype=torch.float64))
        centers=torch.nn.Parameter(torch.zeros((len(classes),k),dtype=torch.float64))
        head=torch.nn.Linear(k,len(classes),dtype=torch.float64)
        opt=torch.optim.Adam([raw,centers,*head.parameters()],lr=4e-4,weight_decay=1e-4)
        for _ in range(epochs):
            opt.zero_grad()
            z=torch.einsum('k,skj->sj',F.softplus(raw)+1e-8,bases)
            loss = _alpha_loss(z, y, w, wc, centers, head, batch_count=1)
            if not torch.isfinite(loss):
                raise ArithmeticError('Invalid alpha objective')
            loss.backward()
            opt.step()
        alpha=(F.softplus(raw)+1e-8).detach().numpy()
    return Image(image.bounds,image.bandwidth,immutable_array(alpha),image.source_cuts)


@dataclass(frozen=True)
class HistoricalAlphaFit:
    image: Image
    classes: np.ndarray
    weight: np.ndarray
    intercept: np.ndarray
    centers: np.ndarray

    def probabilities(self,diagrams):
        logits=self.image.transform(diagrams)@self.weight.T+self.intercept
        z=np.exp(logits-np.max(logits,axis=1,keepdims=True))
        return z/z.sum(axis=1,keepdims=True)


def fit_historical(diagrams,labels,people,*,quantile_step,epochs,seed):
    """Corrected historical reference: mini-batch Adam, original logits and losses.

    Training-local bins and kernels retain all finite deaths. Participant
    weights modify the historical equal-visit loss as required by the contract.
    """
    image=Image.initial(diagrams,quantile_step)
    bases=torch.tensor(np.stack([image.basis(d) for d in diagrams]),dtype=torch.float32)
    if len(diagrams) != len(labels) or epochs < 1:
        raise ValueError('Ineligible historical training partition')
    classes, y, w, wc = _objective_weights(labels, people, torch.float32)
    n=len(image.alpha)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        raw=torch.nn.Parameter(torch.full((n,),float(np.log(np.expm1(1.)))))
        centers=torch.nn.Parameter(torch.zeros((len(classes),n)))
        head=torch.nn.Linear(n,len(classes))
        optimizer=torch.optim.Adam([raw,centers,*head.parameters()],lr=4e-4,weight_decay=1e-4)
        shuffle=torch.Generator().manual_seed(seed)
        for _ in range(epochs):
            batches = torch.randperm(len(y), generator=shuffle).split(32)
            for idx in batches:
                optimizer.zero_grad()
                z=torch.einsum('k,skj->sj',F.softplus(raw)+1e-8,bases[idx])
                loss = _alpha_loss(z, y[idx], w[idx], wc[idx], centers, head, len(batches))
                if not torch.isfinite(loss):
                    raise ArithmeticError('Historical objective failed')
                loss.backward()
                optimizer.step()
        alpha=(F.softplus(raw)+1e-8).detach().numpy()
        return HistoricalAlphaFit(Image(image.bounds,image.bandwidth,immutable_array(alpha),image.source_cuts),
                immutable_array(classes),immutable_array(head.weight.detach().numpy()),
                immutable_array(head.bias.detach().numpy()),immutable_array(centers.detach().numpy()))
