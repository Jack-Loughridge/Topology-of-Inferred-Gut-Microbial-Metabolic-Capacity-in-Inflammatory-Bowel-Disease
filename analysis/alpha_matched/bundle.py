"""Build a graph panel from checked species profiles and a complete source ledger."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, load_npz, save_npz, vstack

from .features import Graphs
from .support import project,sample_support


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024**2),b''):
            h.update(chunk)
    return h.hexdigest()


def build(ledger,profile,measurement,out,expected_models):
    if out.exists():
        raise FileExistsError(out)
    prof=pd.read_csv(profile,sep='\t',index_col=0)
    spec=json.loads(measurement.read_text())
    if spec['profile_sha256']!=sha(profile) or spec['unit']!='fraction' or not spec['coverage_rule']:
        raise ValueError('Profile measurement contract mismatch')
    if not prof.index.is_unique or not prof.columns.is_unique or not np.isfinite(prof.values).all():
        raise ValueError('Missing or ambiguous abundance cell requires source adjudication')
    if prof.index.hasnans or (prof.values<0).any() or (prof.values.sum(axis=0)>1.00001).any():
        raise ValueError('Invalid abundance profile')
    species,verts,pairs,inc,vertex_inc,catalogue=project(ledger,expected_models)
    # All nonmatching source taxa continue to count towards the denominator.
    # prof is keyed by adjudicated taxon names (unmapped taxa can keep native IDs).
    matched=[i for i,taxon in enumerate(species) if taxon in prof.index]
    mapped=prof.loc[[species[i] for i in matched]].T
    inc=inc[matched]
    vertex_inc=vertex_inc[matched]
    all_support=[]
    vertex_rows=[]
    for abundance in mapped.to_numpy():
        e,v=sample_support(abundance,inc,vertex_inc)
        all_support.append(e)
        vertex_rows.append(csr_matrix((np.ones(len(v),dtype='bool'),(np.zeros(len(v),int),v)),shape=(1,len(verts))))
    out.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent) as tmp:
        folder=Path(tmp)/'bundle'
        folder.mkdir()
        save_npz(folder/'raw.npz',vstack(all_support,format='csr'))
        save_npz(folder/'vertices.npz',vstack(vertex_rows,format='csr'))
        save_npz(folder/'abundance.npz',csr_matrix(prof.T))
        np.save(folder/'pairs.npy',pairs,allow_pickle=False)
        axes=dict(vertices=verts,samples=list(prof.columns),matched_species=list(mapped.columns),source_species=list(prof.index))
        (folder/'axes.json').write_text(json.dumps(axes))
        receipt=dict(input_profile_sha256=sha(profile),catalogue_digest=catalogue,
                     mapping_digest=sha(measurement),code_sha256=[sha(Path(__file__)),sha(Path(__file__).with_name('support.py'))],
                     coverage_rule=spec['coverage_rule'],expected_models=expected_models,
                     outputs={p.name:sha(p) for p in sorted(folder.iterdir())})
        (folder/'receipt.json').write_text(json.dumps(receipt,sort_keys=True))
        os.rename(folder,out)


def load(path):
    rec=json.loads((path/'receipt.json').read_text())
    for name,expected in rec['outputs'].items():
        if sha(path/name)!=expected:
            raise ValueError('Corrupt graph bundle '+name)
    axes=json.loads((path/'axes.json').read_text())
    raw=load_npz(path/'raw.npz')
    v=load_npz(path/'vertices.npz')
    mask=None
    if 'source_rosters' in axes:
        native=axes['source_rosters']
        releases=axes['sample_releases']
        if len(releases)!=len(axes['samples']):
            raise ValueError('Invalid native assay release roster')
        masks={name:np.isin(axes['source_species'],roster) for name,roster in native.items()}
        mask=np.stack([masks[r] for r in releases])
    return Graphs(raw,np.load(path/'pairs.npy',allow_pickle=False),tuple(
        v.indices[v.indptr[i]:v.indptr[i+1]] for i in range(v.shape[0])),
        tuple(axes['samples']),load_npz(path/'abundance.npz'),rec['catalogue_digest'],rec['mapping_digest'],rec.get('species_nondetection_zero',False),tuple(axes['source_species']),mask)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('ledger','profile','measurement','out'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--expected-models',type=int,default=7302)
    build(**vars(parser.parse_args()))
