"""Common graph and canonical-vector construction for both support arms."""
from dataclasses import dataclass
import hashlib
from pathlib import Path
import numpy as np

from .cache import CurvatureCache,record_sample
from .geometry import compute
from .h0 import Image,learn_image,merge_deaths
from .partitions import FrozenVectors,RelativeReference,array_id,canonical_json,digest,immutable_array,fitted_location_scale


@dataclass
class Graphs:
    """E has global directed-pair columns. Vertices includes all retained isolates."""
    raw: object
    pairs: np.ndarray
    vertices: tuple
    sample_ids: tuple
    species_abundance: object
    catalogue_digest: str
    mapping_digest: str
    species_nondetection_zero: bool = False
    species_ids: tuple = ()
    species_observed: object = None

    def __post_init__(self):
        if self.species_observed is not None:
            mask=np.asarray(self.species_observed)
            if mask.dtype!=np.bool_ or mask.shape!=self.species_abundance.shape:
                raise ValueError('Invalid species measurement mask')
            a,b=self.species_abundance.nonzero()
            if not mask[a,b].all():
                raise ValueError('Positive abundance outside the native measurement axis')
        if self.species_ids and (len(self.species_ids)!=self.species_abundance.shape[1] or len(set(self.species_ids))!=len(self.species_ids)):
            raise ValueError('Invalid native abundance axis')
        self.lookup={name:i for i,name in enumerate(self.sample_ids)}
        if len(self.lookup)!=len(self.sample_ids) or self.raw.shape!=(len(self.lookup),len(self.pairs)):
            raise ValueError('Unaligned graph panel')
        if len(self.vertices)!=len(self.lookup) or self.species_abundance.shape[0]!=len(self.lookup):
            raise ValueError('Unaligned species or vertex panel')
        self.axis=tuple(map(tuple,self.pairs.astype(int).tolist()))
        if len(set(self.axis))!=len(self.axis) or any(u==v for u,v in self.axis):
            raise ValueError('Invalid catalogue edge axis')

    def graph(self,i,reference):
        row=self.raw.getrow(i)
        if not np.isfinite(row.data).all() or np.any(row.data<=0):
            raise ValueError('Invalid active support')
        edge=row.indices
        log=np.log(row.data.astype(np.longdouble))
        if reference is not None:
            log=log-reference.log_reference[edge]
        v=np.asarray(self.vertices[i],dtype=np.int64)
        if np.any(v[:-1]>=v[1:]):
            raise ValueError('Sample vertices must be ordered and unique')
        pairs=np.searchsorted(v,self.pairs[edge])
        if not np.array_equal(v[pairs],self.pairs[edge]):
            raise ValueError('Active pair missing vertex')
        return edge,log,v,pairs


class Factory:
    def __init__(self,graphs,arm,cache_dir,tolerances):
        if arm not in ('RAW','RELATIVE'):
            raise ValueError('Unknown support arm')
        self.graphs=graphs
        self.arm=arm
        self.cache_dir=Path(cache_dir)
        self.tolerances=tolerances
        self.references={}
        folder=Path(__file__).parent
        self.version=digest(*[p.read_bytes() for p in sorted(folder.glob('*.py'))])

    def __call__(self,block,setting,train):
        samples=list(train.index)
        people=list(train.person)
        rows=[self.graphs.lookup[s] for s in samples]
        partition=digest(canonical_json(sorted(zip(samples,people))))
        ref=None
        if self.arm=='RELATIVE':
            if partition not in self.references:
                self.references[partition]=RelativeReference.fit(self.graphs.raw[rows],people,samples,self.graphs.axis)
            ref=self.references[partition]
        return Generator(self,block,setting,train,partition,ref)


class Generator:
    def __init__(self,factory,block,setting,train,partition,reference):
        self.factory=factory
        self.block=block
        self.setting=setting
        self.partition=partition
        self.reference=reference
        self.image=None
        self.config=dict(setting)
        if block=='H0':
            diagrams=[self.diagram(k) for k in train.index]
            opts=dict(quantile_step=self.config['q'])
            if self.config['learned']:
                self.image=learn_image(diagrams,train.y,train.person,epochs=self.config['epochs'],
                                       seed=self.config['seed'],**opts)
            else:
                self.image=Image.initial(diagrams,**opts)
        raw=self.unscaled(tuple(train.index))
        _,p,count=np.unique(train.person,return_inverse=True,return_counts=True)
        w=1/count[p]
        mean,sd=fitted_location_scale(raw,w)
        self.mean=immutable_array(mean)
        self.scale=immutable_array(sd)
        self.names=self.feature_names()
        refhash=array_id(reference.log_reference) if reference is not None else 'identity'
        extra=[] if self.image is None else [array_id(self.image.alpha),array_id(self.image.bounds),self.image.bandwidth,self.image.source_cuts]
        self.generator=digest(canonical_json([factory.arm,partition,block,setting,extra,refhash,
                              array_id(self.mean),array_id(self.scale),factory.version,factory.graphs.catalogue_digest,
                              factory.graphs.mapping_digest,self.names]))

    def diagram(self,key):
        g=self.factory.graphs
        _,log,v,pairs=g.graph(g.lookup[key],self.reference)
        return merge_deaths(len(v),pairs,log)[0]

    def feature_names(self):
        edges=self.factory.graphs.axis
        if self.block=='H0':
            return tuple('PI:'+str(i) for i in range(len(self.image.alpha)))
        if self.block=='Ricci':
            return tuple(f'{kind}:{a}>{b}' for kind in ('K0','B') for a,b in edges)
        raise ValueError('Unknown standalone representation')

    def curvature(self,i):
        f=self.factory
        g=f.graphs
        e,logs,v,pairs=g.graph(i,self.reference)
        row=g.raw.getrow(i)
        # Preserve source dtype without hashing any longdouble padding bytes.
        inp=digest(array_id(v).encode(),array_id(e).encode(),array_id(row.data).encode())
        ref='identity' if self.reference is None else array_id(self.reference.log_reference)
        cfg=self.config
        identity=dict(input_digest=inp,catalogue_digest=g.catalogue_digest,mapping_digest=g.mapping_digest,
                      arm=f.arm,reference_digest=ref,training_people_digest=self.partition,graph_rules='bound-aware-nonself-v1',
                      transition=[cfg['family'],cfg['parameter']],rho=cfg['rho'],epsilon=1e-4,
                      endpoint_measure='degree-plus-one',feature_definition='full-pairwise-K0+B-catalogue-axis',
                      scaler_digest='raw-before-classifier-scaling',solver='deterministic-certified',tolerance=f.tolerances,
                      implementation_digest=f.version,active_pairs_digest=hashlib.sha256(pairs.astype('<i8').tobytes()).hexdigest())
        cache=CurvatureCache(f.cache_dir,identity)
        k=cache.read(pairs)
        if k is None:
            primitive=cache
            if f.arm=='RAW':
                primitive=CurvatureCache(f.cache_dir/'.raw-numerics',
                             {**identity,'training_people_digest':'fixed-RAW-numerical-operator'})
                k=primitive.read(pairs)
            if k is None:
                result=compute(len(v),pairs,logs,family=cfg['family'],parameter=cfg['parameter'],rho=cfg['rho'],
                               curvature_tolerance=f.tolerances['curvature_error'])
                primitive.save(result.curvature,pairs,result.error)
                k=result.curvature
            if primitive is not cache:
                k=cache.bind_raw_calculation(primitive,pairs)
        record_sample(cache,g.sample_ids[i],g.pairs[e],f.cache_dir)
        out=np.zeros((2,len(g.pairs)))
        out[0,e]=k
        out[1,e]=1
        return out.ravel()

    def unscaled(self,keys):
        if self.block=='H0':
            return self.image.transform([self.diagram(k) for k in keys])
        if self.block=='Ricci':
            return np.stack([self.curvature(self.factory.graphs.lookup[k]) for k in keys])
        raise ValueError('Unknown standalone')

    def transform(self,keys):
        values=(self.unscaled(keys)-self.mean)/self.scale if len(keys) else np.empty((0,len(self.names)))
        return FrozenVectors(values,tuple(keys),self.names,self.factory.arm,
                             self.partition,self.generator,self.block)
