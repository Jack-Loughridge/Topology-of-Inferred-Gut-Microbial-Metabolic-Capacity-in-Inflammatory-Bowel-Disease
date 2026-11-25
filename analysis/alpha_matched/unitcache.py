"""Completed-unit receipts inside the release-specific, locked job directory."""
import hashlib
import json
import os
import platform
import re
import sqlite3
import sys
from contextlib import closing
from importlib.metadata import version

import numpy as np
from threadpoolctl import threadpool_info
import torch

from .artifacts import sha
from .cache import CurvatureCache
from .partitions import array_id, canonical_json, digest


def environment():
    packages = ('numpy', 'scipy', 'pandas', 'scikit-learn', 'torch', 'POT', 'networkx', 'mpmath')
    return [sys.version, platform.platform(), torch.get_num_threads(), threadpool_info(),
            {p: version(p) for p in packages},
            {p: os.environ.get(p) for p in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                                            'MKL_NUM_THREADS', 'CUDA_VISIBLE_DEVICES')}]


def object_closure(store, value):
    """Verify reachable Store objects while preserving the schema's opaque labels."""
    seen = set()
    expanded = set()
    literal_fields = {'sample', 'samples', 'participants', 'classes', 'train_groups', 'endpoint',
                      'name', 'method', 'split', 'setting', 'tolerances'}
    axis_fields = {'ids', 'features', 'species_taxa', 'training_visits'}

    def visit(v, axis=False):
        if isinstance(v, dict):
            for key, child in v.items():
                if key not in literal_fields:
                    visit(child, axis=key in axis_fields)
        elif isinstance(v, (tuple, list)):
            if not axis:
                for child in v:
                    visit(child)
        elif isinstance(v, str) and re.fullmatch(r'[0-9a-f]{64}\.(jsonz?|npyz?)', v):
            if (v, axis) in expanded:
                return
            expanded.add((v, axis))
            seen.add(v)
            if v.endswith(('.npy', '.npyz')):
                if not np.isfinite(store.load_array(v)).all():
                    raise ValueError('Nonfinite unit dependency')
            else:
                record = json.loads(store.read(v))
                visit(record, axis=axis)
    visit(value)
    return seen


def dependencies(store, value):
    """Re-register the verified dependency graph when resuming a unit."""
    store.objects.update(object_closure(store, value))


class Units:
    def __init__(self, store, context):
        self.store = store
        self.context = sha(canonical_json(context))
        self.sealed = (store.root / 'receipt.json').exists()

    def run(self, identity, compute):
        key = sha(canonical_json([self.context, identity]))
        marker = self.store.root / ('unit-' + key + '.json')
        if marker.exists():
            ref = json.loads(marker.read_bytes())['record']
            rec = json.loads(self.store.read(ref))
            if rec['key'] != key:
                raise ValueError('Wrong completed unit')
            dependencies(self.store, rec['result'])
            self.store.objects.add(ref)
            return rec['result']
        if self.sealed:
            raise ValueError('Missing completed unit')
        result = compute()
        dependencies(self.store, result)
        ref = self.store.record(dict(key=key, result=result))
        self.store.publish(marker.name, canonical_json(dict(record=ref)))
        return result


def database(directory):
    path = directory / 'sample-certificates.sqlite'
    return closing(sqlite3.connect(path.as_uri() + '?mode=ro', uri=True))


def geometry_proof(factory, partition, setting, ref, ids):
    """Resolve the exact shard and SQL attestation expected for each visit."""
    g = factory.graphs
    rows = set()
    with database(factory.cache_dir) as db:
        for sample in ids:
            i = g.lookup[sample]
            e, _, v, pairs = g.graph(i, None)
            inp = digest(array_id(v).encode(), array_id(e).encode(), array_id(g.raw.getrow(i).data).encode())
            identity = dict(input_digest=inp, catalogue_digest=g.catalogue_digest, mapping_digest=g.mapping_digest,
                            arm=factory.arm, reference_digest=ref, training_people_digest=partition,
                            graph_rules='bound-aware-nonself-v1', transition=[setting['family'], setting['parameter']],
                            rho=setting['rho'], epsilon=1e-4, endpoint_measure='degree-plus-one',
                            feature_definition='full-pairwise-K0+B-catalogue-axis', scaler_digest='raw-before-classifier-scaling',
                            solver='deterministic-certified', tolerance=factory.tolerances,
                            implementation_digest=factory.version, active_pairs_digest=sha(pairs.astype('<i8').tobytes()))
            shard = sha(canonical_json(identity))
            folder = factory.cache_dir / shard
            if not folder.is_dir():
                raise ValueError('Missing curvature shard')
            cache = CurvatureCache(factory.cache_dir, identity)
            if cache.read(pairs) is None:
                raise ValueError('Missing curvature receipt')
            receipt = json.loads((folder / 'receipt.json').read_bytes())
            if factory.arm == 'RAW':
                numerical_key = dict(identity, training_people_digest='fixed-RAW-numerical-operator')
                numerical_id = sha(canonical_json(numerical_key))
                base = factory.cache_dir / '.raw-numerics'
                if (receipt['record'].get('raw_calculation') != numerical_id
                        or not (base / numerical_id).is_dir()):
                    raise ValueError('Missing RAW numerical dependency')
                primitive = CurvatureCache(base, numerical_key)
                if primitive.read(pairs) is None:
                    raise ValueError('Missing RAW numerical receipt')
                source = json.loads((primitive.folder / 'receipt.json').read_bytes())['record']
                if receipt['record'] != dict(source, key=identity, raw_calculation=numerical_id):
                    raise ValueError('Inconsistent partition-specific RAW curvature')
            expected = dict(active_nonself_edges=len(e), finite_edges=len(e), failed_nonfinite=0,
                            solver_receipt=receipt['digest'], numerical_errors=receipt['record']['certificate'],
                            global_activity=sha(g.pairs[e].astype('<i8').tobytes()))
            row = db.execute('SELECT body FROM receipts WHERE sample=? AND geometry=?', (sample, shard)).fetchone()
            if row != (canonical_json(expected),):
                raise ValueError('Invalid sample curvature attestation')
            rows.add((sample, shard, sha(row[0])))
    return rows


def certificate_digest(directory, expected):
    digestor = hashlib.sha256()
    count = 0
    with database(directory) as db:
        for sample, shard, body in db.execute('SELECT sample,geometry,body FROM receipts ORDER BY sample,geometry'):
            row = (sample, shard, sha(body))
            if row not in expected:
                raise ValueError('Unregistered sample certificate')
            digestor.update(canonical_json(row) + b'\n')
            count += 1
    if count != len(expected):
        raise ValueError('Missing sample certificates')
    return digestor.hexdigest()
