"""Project a receipt-verified development GMrepo matrix into the frozen catalogue."""
import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix,load_npz,save_npz,vstack

from .bundle import sha
from .partitions import canonical_json,digest
from .support import project,sample_support


def measurement_for_panel(receipt,axes,contract):
    if contract is not None:
        required={'panel_key','mapping_sha256','coverage_rule','unlisted_is_measured_nondetection','source_species_key'}
        if set(contract)-{'preserve_unreported_mask'}!=required or contract['panel_key']!=receipt['key']:
            raise ValueError('Measurement registration is not bound to this panel')
        if (not isinstance(contract['unlisted_is_measured_nondetection'],bool)
                or not contract['coverage_rule'] or len(contract['mapping_sha256'])!=64):
            raise ValueError('Incomplete measurement registration')
        # An omitted native row in a combined report is unknown unless the
        # publisher provides a released non-detection interpretation.
        if (any(receipt.get('unreported_lineages_by_release',{}).values())
                and not contract['unlisted_is_measured_nondetection'] and not contract.get('preserve_unreported_mask')):
            raise ValueError('Unresolved native lineage absence across releases')
        return (axes[contract['source_species_key']],contract['mapping_sha256'],contract['coverage_rule'],
                contract['unlisted_is_measured_nondetection'])
    spec=receipt['specification']
    mapping=digest(canonical_json(dict(algorithm=spec['code']['mapping.py'],taxonomy=spec['taxonomy'],endpoint=spec['endpoint'])))
    return axes['taxid'],mapping,spec['endpoint'],True


def build(panel,ledger,out,expected_models=7302,projection=None,measurement=None):
    receipt=json.loads((panel/'receipt.json').read_text())
    for name,expected in receipt['artifacts'].items():
        if sha(panel/name)!=expected:
            raise ValueError('Source panel changed: '+name)
    if (out.parent/'STOP').exists():
        raise RuntimeError('STOP')
    axes=json.loads((panel/'axes.json').read_text())
    registration=json.loads(measurement.read_bytes()) if measurement else None
    source_species,mapper,policy,nondetection=measurement_for_panel(receipt,axes,registration)
    if projection is None:
        projected=project(ledger,expected_models)
    else:
        from .catalogue_projection import load
        projected=load(projection,ledger,expected_models)
    cat,verts,pairs,edge_inc,vertex_inc,cat_id=projected
    mapped=load_npz(panel/'mapped_species.npz').tocsr()
    order={k:i for i,k in enumerate(cat)}
    indices=[order[name] for name in axes['agora_species']]
    if len(set(indices))!=len(indices) or mapped.shape!=(len(axes['sample']),len(indices)):
        raise ValueError('Misaligned mapped species panel')
    edge_inc=edge_inc[indices]
    vertex_inc=vertex_inc[indices]
    es=[]
    vs=[]
    for idx in range(mapped.shape[0]):
        row,vertices=sample_support(mapped.getrow(idx).toarray().ravel(),edge_inc,vertex_inc)
        es.append(row)
        vs.append(csr_matrix((np.ones(len(vertices),dtype='bool'),(np.zeros(len(vertices),int),vertices)),
                             shape=(1,len(verts))))
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent) as tmp:
        folder=Path(tmp)/'panel'
        folder.mkdir()
        save_npz(folder/'raw.npz',vstack(es,format='csr'))
        save_npz(folder/'vertices.npz',vstack(vs,format='csr'))
        shutil.copyfile(panel/'reported_taxa.npz',folder/'abundance.npz')
        np.save(folder/'pairs.npy',pairs,allow_pickle=False)
        dest_axes=dict(vertices=verts,samples=axes['sample'],matched_species=axes['agora_species'],
                       source_species=source_species)
        if registration and registration.get('preserve_unreported_mask'):
            native=pd.read_csv(panel/'visits.tsv.gz',sep='\t',usecols=['sample','release']).set_index('sample',verify_integrity=True)
            dest_axes['source_rosters']=axes['observed_roster_by_release']
            dest_axes['sample_releases']=native.loc[axes['sample'],'release'].tolist()
        (folder/'axes.json').write_text(json.dumps(dest_axes))
        meta=dict(input_profile_sha256=sha(panel/'receipt.json'),catalogue_digest=cat_id,
                  mapping_digest=mapper,measurement_registration_sha256=sha(measurement) if measurement else None,
                  native_axis_sha256=sha(panel/'axes.json'),species_nondetection_zero=nondetection,code_sha256=[sha(Path(__file__)),sha(Path(__file__).with_name('support.py'))],
                  coverage_rule=policy,expected_models=expected_models,
                  outputs={p.name:sha(p) for p in sorted(folder.iterdir())})
        (folder/'receipt.json').write_text(json.dumps(meta,sort_keys=True))
        os.rename(folder,out)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('panel','ledger','out'):
        ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--projection',type=Path)
    ap.add_argument('--measurement',type=Path)
    build(**vars(ap.parse_args()))
