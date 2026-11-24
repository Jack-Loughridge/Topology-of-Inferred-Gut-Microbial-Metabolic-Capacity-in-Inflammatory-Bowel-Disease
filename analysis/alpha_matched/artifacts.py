"""Write-once fold artifacts. JSON records refer to NumPy arrays by SHA-256."""
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
import zlib

import numpy as np
from scipy.special import expit, softmax

from .partitions import FrozenVectors, canonical_json

RESERVE = 5 * 1024**3


def sha(data):
    return hashlib.sha256(data).hexdigest()


def space(path, size):
    if shutil.disk_usage(path).free < RESERVE + size + 65536:
        raise OSError('Alpha disk reserve reached')


def aligned_species(graph, ids, native, taxa, impute):
    """Restore fitted taxon order and apply the qualified per-visit measurement roster."""
    if len(native) != graph.species_abundance.shape[1] or len(set(native)) != len(native):
        raise ValueError('Incompatible abundance coordinates')
    if graph.species_ids and tuple(native) != graph.species_ids:
        raise ValueError('Supplied abundance coordinates differ from the verified bundle')
    columns = {taxon: i for i, taxon in enumerate(native)}
    target = [j for j, taxon in enumerate(taxa) if taxon in columns]
    source = [columns[taxa[j]] for j in target]
    rows = [graph.lookup[key] for key in ids]
    values = np.zeros((len(ids), len(taxa)))
    values[:, target] = graph.species_abundance[rows][:, source].toarray()
    if graph.species_observed is None:
        if graph.species_nondetection_zero is not True and (impute is not None or len(target) != len(taxa)):
            raise ValueError('Missing qualified measurement roster or verified nondetection semantics')
        measured = np.ones(values.shape, dtype=bool)
    else:
        measured = np.zeros(values.shape, dtype=bool)
        measured[:, target] = graph.species_observed[np.ix_(rows, source)]
    if impute is None:
        if not measured.all():
            raise ValueError('Unreported coordinate without a training-fitted imputer')
        return values
    if impute.shape != (len(taxa),):
        raise ValueError('Imputation state and frozen species axis disagree')
    return np.concatenate((np.where(measured, values, impute), ~measured), axis=1)


class Store:
    def __init__(self, directory):
        self.root = Path(directory)
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.objects = set()

    def publish(self, name, data):
        target = self.root / name
        if target.exists():
            if target.read_bytes() != data:
                raise ValueError('Conflicting or corrupt artifact: ' + name)
            return
        space(self.root, len(data))
        with tempfile.NamedTemporaryFile(dir=self.root) as tmp:
            tmp.write(data)
            tmp.flush()
            os.fsync(tmp.fileno())
            os.chmod(tmp.name, 0o400)
            try:
                os.link(tmp.name, target)
            except FileExistsError:
                if target.read_bytes() != data:
                    raise ValueError('Conflicting concurrent artifact')

    def put(self, data, extension):
        name = sha(data) + extension
        self.publish(name, data)
        self.objects.add(name)
        return name

    def record(self, obj):
        data = canonical_json(obj)
        if len(data) > 4096:
            return self.put(zlib.compress(data), '.jsonz')
        return self.put(data, '.json')

    def array(self, arr):
        arr = np.asarray(arr)
        if arr.dtype.kind not in 'bifu' or not np.isfinite(arr).all():
            raise ValueError('Invalid numeric artifact')
        buf = io.BytesIO()
        np.save(buf, arr, allow_pickle=False)
        return self.put(zlib.compress(buf.getbuffer()), '.npyz')

    def read(self, name):
        if len(name) != 64 + len(Path(name).suffix) or Path(name).name != name:
            raise ValueError('Invalid artifact name')
        data = (self.root / name).read_bytes()
        if sha(data) != name.split('.')[0]:
            raise ValueError('Artifact digest mismatch')
        if name.endswith(('.npyz', '.jsonz')):
            return zlib.decompress(data)
        return data

    def load_array(self, name):
        return np.load(io.BytesIO(self.read(name)), allow_pickle=False)

    def vector(self, vector):
        return self.record(dict(values=self.array(vector.values), ids=self.record(vector.ids),
                                features=self.record(vector.features), arm=vector.arm, partition=vector.partition,
                                generator=vector.generator, block=vector.block, identity=vector.identity()))

    def load_vector(self, name):
        record = json.loads(self.read(name))
        identity = record.pop('identity')
        record['values'] = self.load_array(record['values'])
        for axis in ('ids', 'features'):
            record[axis] = json.loads(self.read(record[axis]))
        vector = FrozenVectors(**record)
        if vector.identity() != identity:
            raise ValueError('Vector identity mismatch')
        return vector

    def reference(self, ref):
        if ref is None:
            return None
        return self.record(dict(log_reference=self.array(ref.log_reference), unseen=self.array(ref.unseen),
                                partition=ref.partition, axis=ref.axis))

    def image(self, image):
        return dict(alpha=self.array(image.alpha), bounds=self.array(image.bounds),
                    bandwidth=image.bandwidth, source_cuts=image.source_cuts)

    def head(self, model):
        state = dict(classes=model.classes_.tolist(), n_features=model.n_features_in_)
        if hasattr(model, 'coef_'):
            state.update(family='linear', coef=self.array(model.coef_), intercept=self.array(model.intercept_))
        else:
            state.update(family='forest', trees=[])
            for estimator in model.estimators_:
                t = estimator.tree_
                state['trees'].append({key: self.array(getattr(t, key)) for key in
                                       ('children_left', 'children_right', 'feature', 'threshold', 'value')})
        return self.record(state)

    def predict(self, name, x):
        state = json.loads(self.read(name))
        if x.shape[1] != state['n_features'] or state['classes'] != list(range(len(state['classes']))):
            raise ValueError('Head schema mismatch')
        if state['family'] == 'linear':
            logits = x @ self.load_array(state['coef']).T + self.load_array(state['intercept'])
            if logits.shape[1] > 1:
                return softmax(logits, axis=1)
            p = expit(logits)
            return np.column_stack((1 - p, p))
        votes = np.zeros((len(x), len(state['classes'])))
        for tree in state['trees']:
            t = {key: self.load_array(value) for key, value in tree.items()}
            for i, row in enumerate(x.astype(np.float32, copy=False)):
                n = 0
                while t['children_left'][n] != -1:
                    left = row[t['feature'][n]] <= t['threshold'][n]
                    n = t['children_left' if left else 'children_right'][n]
                counts = t['value'][n, 0]
                votes[i] += counts / counts.sum()
        return votes / len(state['trees'])

    def probabilities(self, method, fitted, x):
        if method == 'historical':
            state = json.loads(self.read(fitted))['state']
            logits = x @ self.load_array(state['weight']).T + self.load_array(state['intercept'])
            return softmax(logits, axis=1)
        return self.predict(fitted, x)

    def deployment_files(self, model):
        """Enumerate only inference objects for transport to the evaluator."""
        files = {model}

        def read(name):
            files.add(name)
            return json.loads(self.read(name))

        def generator(name):
            spec = read(name)
            if spec['name'] == 'K0':
                generator(spec['source'])
                return
            files.add(spec['features'])
            if spec['reference']:
                ref = read(spec['reference'])
                files.update((ref['unseen'], ref['log_reference']))
            state = spec['state']
            for field in ('mean', 'scale', 'species_axis', 'species_taxa', 'species_impute', 'weight', 'intercept', 'centers'):
                if state.get(field) is not None:
                    files.add(state[field])
            if state.get('image'):
                files.update((state['image']['alpha'], state['image']['bounds']))

        deployment = read(model)
        manifest = read(deployment['generators'])
        files.add(manifest['training_visits'])
        for block in manifest['blocks']:
            generator(block)
        if deployment['method'] != 'historical':
            head = read(deployment['head'])
            if head['family'] == 'forest':
                for tree in head['trees']:
                    files.update(tree.values())
            else:
                files.update((head['intercept'], head['coef']))
        return sorted(files)

    def transform(self, model, factory, ids, species, *, curvature=None):
        """Apply saved numeric state to an aligned graph bundle without fitting."""
        from .comparators import Baseline
        from .features import Generator
        from .h0 import Image
        from .partitions import RelativeReference, digest

        spec = json.loads(self.read(model))
        if spec['name'] == 'K0':
            vec = self.transform(spec['source'], factory, ids, species, curvature=curvature)
            start, stop = spec['columns']
            return FrozenVectors(vec.values[:, start:stop], vec.ids, vec.features[start:stop],
                                 vec.arm, vec.partition, model, 'K0')
        graph = factory.graphs
        if ((spec['arm'], spec['catalogue'], spec['mapping'], spec['axis']) !=
            (factory.arm, graph.catalogue_digest, graph.mapping_digest, digest(canonical_json(graph.axis)))):
            raise ValueError('Incompatible graph or support arm')
        state = spec['state']
        names = json.loads(self.read(spec['features']))
        ref = None
        if spec['reference']:
            r = json.loads(self.read(spec['reference']))
            ref = RelativeReference(self.load_array(r['log_reference']), self.load_array(r['unseen']),
                                    r['partition'], r['axis'])
        name = spec['name']
        if name in ('H0', 'fixed-alpha', 'Ricci', 'historical'):
            gen = Generator.__new__(Generator)
            gen.factory, gen.reference, gen.partition = factory, ref, spec['partition']
            gen.block = state.get('block', 'H0')
            gen.config = state['setting']
            if gen.block == 'Ricci' and factory.tolerances != state['tolerances']:
                raise ValueError('Changed geometry tolerances')
            if state.get('image'):
                im = state['image']
                gen.image = Image(self.load_array(im['bounds']), im['bandwidth'],
                                  self.load_array(im['alpha']), tuple(im['source_cuts']))
            values = gen.unscaled(ids) if len(ids) else np.empty((0, len(names)))
            if curvature is not None and gen.block == 'Ricci':
                size = len(graph.axis)
                for sample, row in zip(ids, values):
                    edge = graph.raw.getrow(graph.lookup[sample]).indices
                    if not np.array_equal(np.flatnonzero(row[size:]), np.sort(edge)) or not np.all(row[size:][edge] == 1):
                        raise ValueError('Incomplete active-edge curvature')
                    curvature(sample, graph.pairs[edge], row[:size][edge])
        else:
            baseline = Baseline.__new__(Baseline)
            baseline.graphs, baseline.reference, baseline.name = graph, ref, name
            if name == 'species':
                taxa = json.loads(self.read(state['species_taxa'])) if 'species_taxa' in state else names
                impute = self.load_array(state['species_impute']) if state.get('species_impute') else None
                values = aligned_species(graph, ids, species, taxa, impute)
            else:
                values=baseline.values(ids)
        if name != 'historical':
            values = (values - self.load_array(state['mean'])) / self.load_array(state['scale'])
        return FrozenVectors(values, tuple(ids), tuple(names), spec['arm'], spec['partition'],
                             model, name)
