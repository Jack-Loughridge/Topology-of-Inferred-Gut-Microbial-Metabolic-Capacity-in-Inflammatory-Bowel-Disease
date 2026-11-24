"""Project the bound-resolved ledger to species incidence with no structural gate."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sqlite3
import zlib

import numpy as np
from scipy.sparse import csr_matrix, save_npz


def accumulate(record, pairs, vertices):
    metabolites=record['species']
    for rid,_,_,_,_,direction,_,left,right in record['reactions']:
        if direction=='blocked':
            continue
        if direction not in ('f','r','rf'):
            raise ValueError(f"Unresolved direction for {record['id']} {rid}")
        lverts=[u for u,_ in left if not metabolites[u][3]]
        rverts=[v for v,_ in right if not metabolites[v][3]]
        vertices.update(lverts)
        vertices.update(rverts)
        for a in lverts:
            for b in rverts:
                if a==b:
                    continue
                if 'f' in direction:
                    pairs.add((a,b))
                if 'r' in direction:
                    pairs.add((b,a))


def matrix(by_species,species,axis):
    order={key:i for i,key in enumerate(axis)}
    indptr=[0]
    indices=[]
    for taxon in species:
        indices.extend(sorted(order[x] for x in by_species[taxon]))
        indptr.append(len(indices))
    return csr_matrix((np.ones(len(indices),dtype=np.int8),indices,indptr),shape=(len(species),len(axis)))


def project(db_path,expected_models,selected_taxa=None):
    if not db_path.is_file():
        raise FileNotFoundError(db_path)
    conn=sqlite3.connect(db_path.resolve().as_uri()+'?mode=ro',uri=True)
    pairs=defaultdict(set)
    vertices=defaultdict(set)
    identity=hashlib.sha256()
    count=0
    with conn:
        cursor=conn.execute('select model,species,source_sha,payload_sha,payload from models order by model')
        for model,species,source_sha,payload_sha,blob in cursor:
            count+=1
            identity.update(json.dumps([model,species,source_sha,payload_sha]).encode()+b'\n')
            if selected_taxa is not None and species not in selected_taxa:
                continue
            content=zlib.decompress(blob)
            if hashlib.sha256(content).hexdigest()!=payload_sha:
                raise ValueError('Catalogue payload digest failed')
            accumulate(json.loads(content),pairs[species],vertices[species])
    conn.close()
    if count!=expected_models:
        raise ValueError(f'Incomplete catalogue: {count} of {expected_models} model records')
    species=sorted(vertices)
    verts=sorted(set().union(*vertices.values()))
    edges=sorted(set().union(*pairs.values()))
    coords={x:i for i,x in enumerate(verts)}
    edge_index=np.array([(coords[x],coords[y]) for x,y in edges],dtype=np.int64).reshape(-1,2)
    return (species,verts,edge_index,matrix(pairs,species,edges),matrix(vertices,species,verts),
            identity.hexdigest())


def sample_support(abundance,edge_incidence,vertex_incidence):
    a=np.asarray(abundance,dtype=np.float64)
    if len(a)!=edge_incidence.shape[0] or not np.isfinite(a).all() or np.any(a<0) or a.sum()>1.00001:
        raise ValueError('Unqualified abundance fractions')
    # Biological activity is defined by positively supported taxa even at subnormal levels.
    present=csr_matrix((a>0).astype(np.float64))
    active=(present@edge_incidence).indices
    vertices=np.sort((present@vertex_incidence).indices)
    e=csr_matrix(a)@edge_incidence
    e.sort_indices()
    if not np.array_equal(e.indices,np.sort(active)) or np.any(e.data<=0):
        raise ArithmeticError('Raw support lost a biologically active edge')
    return e,vertices
