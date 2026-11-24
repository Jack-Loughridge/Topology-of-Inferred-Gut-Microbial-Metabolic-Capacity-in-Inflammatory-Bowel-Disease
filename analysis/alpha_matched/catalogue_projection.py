"""Content-bound shared catalogue projection, independent of participants and arms."""
import json
import os
from pathlib import Path
import tempfile

import numpy as np
from scipy.sparse import load_npz,save_npz

from .bundle import sha
from .support import project


def build(ledger,expected_models,directory):
    if directory.exists():
        raise FileExistsError(directory)
    species,verts,pairs,inc,vinc,ledger_id=project(ledger,expected_models)
    directory.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=directory.parent) as tmp:
        path=Path(tmp)/'projected'
        path.mkdir(mode=0o700)
        (path/'axes.json').write_text(json.dumps({'species':species,'vertices':verts}))
        np.save(path/'pairs.npy',pairs,allow_pickle=False)
        save_npz(path/'edge_incidence.npz',inc)
        save_npz(path/'vertex_incidence.npz',vinc)
        receipt={'ledger_id':ledger_id,'models':expected_models,'ledger_sha256':sha(ledger),
                 'projector_sha256':sha(Path(__file__).with_name('support.py')),
                 'species':len(species),'vertices':len(verts),'directed_pairs':len(pairs),
                 'species_edge_presences':inc.nnz,'outputs':{f.name:sha(f) for f in path.iterdir()}}
        (path/'receipt.json').write_text(json.dumps(receipt,sort_keys=True))
        os.rename(path,directory)


def load(directory,ledger,expected_models):
    info=json.loads((directory/'receipt.json').read_bytes())
    if (info['models']!=expected_models or info['ledger_sha256']!=sha(ledger)
            or info['projector_sha256']!=sha(Path(__file__).with_name('support.py'))):
        raise ValueError('Catalogue release changed')
    for name,hashed in info['outputs'].items():
        if hashed!=sha(directory/name):
            raise ValueError('Projection digest mismatch '+name)
    axes=json.loads((directory/'axes.json').read_bytes())
    return (axes['species'],axes['vertices'],np.load(directory/'pairs.npy',allow_pickle=False),
            load_npz(directory/'edge_incidence.npz'),load_npz(directory/'vertex_incidence.npz'),info['ledger_id'])
