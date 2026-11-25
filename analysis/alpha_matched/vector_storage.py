"""Qualified active Ricci representation and producer-bound split evidence."""
import io
import json
import os

import numpy as np

from . import unitcache
from .cache import payload_name
from analysis.alpha_matched.artifacts import Store, sha, space
from analysis.alpha_matched.partitions import FrozenVectors, array_id, canonical_json, digest, fitted_location_scale


def record(store, key):
    return json.loads(store.read(key))


def visit_inputs(graph, ids):
    """Capture identities from the graph in operator order, independently of any receipt."""
    rows = []
    for sample in ids:
        i = graph.lookup[sample]
        edge, _, vertices, local_pairs = graph.graph(i, None)
        rows.append(dict(sample=sample, vertices=array_id(vertices), indices=array_id(edge),
                         support=array_id(graph.raw.getrow(i).data),
                         local_pairs=sha(local_pairs.astype('<i8').tobytes()),
                         global_pairs=sha(graph.pairs[edge].astype('<i8').tobytes())))
    return rows


def frozen_key(store, model, version, names, mean, scale):
    state = model['state']
    cfg = state['setting']
    arm = model['arm']
    refhash = 'identity'
    if arm == 'RELATIVE':
        ref = record(store, model['reference'])
        logref, unseen = (store.load_array(ref[key]) for key in ('log_reference', 'unseen'))
        if (ref['axis'] != model['axis'] or logref.shape != (len(names)//2,) or
                unseen.shape != logref.shape or unseen.dtype != np.bool_ or not np.isfinite(logref).all()):
            raise ValueError('Invalid frozen RELATIVE reference')
        refhash = array_id(logref)
    elif arm != 'RAW' or model['reference'] is not None:
        raise ValueError('Unexpected support reference')
    generator = digest(canonical_json([arm, model['partition'], 'Ricci', sorted(cfg.items()), [],
                                       refhash, array_id(mean), array_id(scale), version,
                                       model['catalogue'], model['mapping'], names]))
    if generator != state['generator'] or state['block'] != 'Ricci' or state['image'] is not None:
        raise ValueError('Frozen generator identity mismatch')
    return dict(catalogue_digest=model['catalogue'], mapping_digest=model['mapping'], arm=arm,
                reference_digest=refhash, training_people_digest=model['partition'],
                graph_rules='bound-aware-nonself-v1', transition=[cfg['family'], cfg['parameter']],
                rho=cfg['rho'], epsilon=1e-4, endpoint_measure='degree-plus-one',
                feature_definition='full-pairwise-K0+B-catalogue-axis',
                scaler_digest='raw-before-classifier-scaling', solver='deterministic-certified',
                tolerance=state['tolerances'], implementation_digest=version)


class RecipeStore(Store):
    def link_npy(self, source):
        """Pin the already qualified physical payload without changing its encoding."""
        if sha(source.read_bytes()) != source.stem or source.suffix not in ('.npy', '.npyz'):
            raise ValueError('Wrong geometry payload digest')
        name = source.name
        space(self.root, 4096)
        try:
            os.link(source, self.root / name)
        except FileExistsError:
            self.read(name)
        self.objects.add(name)
        return name

    def capture(self, vector, factory):
        model = record(self, vector.generator)
        graph = factory.graphs
        if ((model['name'], model['arm'], model['partition'], model['catalogue'], model['mapping'],
             model['axis']) != ('Ricci', vector.arm, vector.partition, graph.catalogue_digest,
                                graph.mapping_digest, digest(canonical_json(graph.axis)))):
            raise ValueError('Wrong Ricci source')
        if (factory.arm != vector.arm or model['state']['tolerances'] != factory.tolerances):
            raise ValueError('Wrong capture factory')
        ref = model['reference']
        refhash = 'identity' if ref is None else array_id(self.load_array(record(self, ref)['log_reference']))
        proof = unitcache.geometry_proof(factory, vector.partition, model['state']['setting'],
                                         refhash, vector.ids)
        by_sample = {sample: (shard, body) for sample, shard, body in proof}
        rows = []
        with unitcache.database(factory.cache_dir) as db:
            for sample in vector.ids:
                shard, body_hash = by_sample[sample]
                folder = factory.cache_dir / shard
                receipt = json.loads((folder / 'receipt.json').read_bytes())
                info = receipt['record']
                body, = db.execute('SELECT body FROM receipts WHERE sample=? AND geometry=?',
                                   (sample, shard)).fetchone()
                if sha(body) != body_hash:
                    raise ValueError('Changed sample certificate')
                edge = graph.raw.getrow(graph.lookup[sample]).indices
                item = dict(sample=sample, indices=self.array(edge), shard=shard, receipt=receipt,
                            body=json.loads(body), payload=self.link_npy(folder / payload_name(info)))
                if 'raw_calculation' in info:
                    primitive = factory.cache_dir / '.raw-numerics' / info['raw_calculation']
                    item['primitive'] = json.loads((primitive / 'receipt.json').read_bytes())
                rows.append(item)
        spec = dict(format='active-ricci-v2', model=vector.generator, ids=self.record(vector.ids),
                    version=factory.version, inputs=self.record(visit_inputs(graph, vector.ids)),
                    rows=rows, identity=vector.identity())
        self.restore(spec)  # Verify original canonical bytes before publishing the root.
        return self.record(spec)

    def restore(self, spec, train_weights=None):
        model = record(self, spec['model'])
        if spec['format'] == 'k0-view-v2':
            parent = self.load_vector(spec['source'])
            width = len(parent.features) // 2
            if (parent.block != 'Ricci' or model != dict(name='K0', source=parent.generator,
                                                        columns=[0, width])):
                raise ValueError('Wrong K0 parent')
            vector = FrozenVectors(parent.values[:, :width], parent.ids, parent.features[:width],
                                   parent.arm, parent.partition, spec['model'], 'K0')
        elif spec['format'] == 'active-ricci-v2' and model['name'] == 'Ricci':
            ids = record(self, spec['ids'])
            names = record(self, model['features'])
            width = len(names) // 2
            axis = np.array([tuple(map(int, f.removeprefix('K0:').split('>')))
                             for f in names[:width]], dtype='<i8').reshape(width, 2)
            expected = [f'{kind}:{a}>{b}' for kind in ('K0', 'B') for a, b in axis]
            if (names != expected or model['axis'] != digest(canonical_json(axis.tolist())) or
                    ids != [r['sample'] for r in spec['rows']]):
                raise ValueError('Wrong ordered catalogue or visits')
            state = model['state']
            mean, scale = (self.load_array(state[k]) for k in ('mean', 'scale'))
            if any(a.dtype != np.float64 or a.shape != (2 * width,) or
                   not np.isfinite(a).all() for a in (mean, scale)) or (scale <= 0).any():
                raise ValueError('Invalid frozen scaler')
            key_base = frozen_key(self, model, spec['version'], names, mean, scale)
            inputs = record(self, spec['inputs'])
            if ids != [x['sample'] for x in inputs]:
                raise ValueError('Wrong visit input roster')
            raw = np.zeros((len(ids), 2 * width), dtype=np.float64)
            for i, row in enumerate(spec['rows']):
                edge = self.load_array(row['indices'])
                if (edge.ndim != 1 or edge.dtype.kind not in 'iu' or (edge < 0).any() or
                        (edge >= width).any() or len(np.unique(edge)) != len(edge)):
                    raise ValueError('Invalid global indices')
                info = row['receipt']['record']
                key = info['key']
                body = row['body']
                witness = inputs[i]
                expected_key = dict(key_base, active_pairs_digest=witness['local_pairs'],
                                    input_digest=digest(witness['vertices'].encode(), array_id(edge).encode(),
                                                        witness['support'].encode()))
                if (canonical_json(key) != canonical_json(expected_key) or
                        witness['indices'] != array_id(edge) or
                        witness['global_pairs'] != sha(axis[edge].tobytes()) or
                        sha(canonical_json(expected_key)) != row['shard'] or
                        row['receipt']['digest'] != sha(canonical_json(info)) or
                        row['payload'] != payload_name(info) or
                        (info['failed'], info['finite_edges'], info['active_edges']) != (0, len(edge), len(edge)) or
                        body != dict(active_nonself_edges=len(edge), finite_edges=len(edge), failed_nonfinite=0,
                                     solver_receipt=row['receipt']['digest'], numerical_errors=info['certificate'],
                                     global_activity=sha(axis[edge].tobytes()))):
                    raise ValueError('Invalid sample evidence')
                errors = info['certificate']
                tolerance = key_base['tolerance']
                # The certified production solver additionally reports a count, not an error.
                repaired = errors.get('rational_repair_edges', 0)
                if (set(errors) - {'rational_repair_edges'} != set(tolerance) or
                        not np.isfinite(repaired) or not 0 <= repaired <= len(edge) or repaired != int(repaired) or
                        any(not 0 <= errors[k] <= limit or not np.isfinite([errors[k], limit]).all()
                            for k, limit in tolerance.items())):
                    raise ValueError('Invalid numerical error bounds')
                if model['arm'] == 'RAW':
                    prim = row['primitive']['record']
                    if (row['primitive']['digest'] != sha(canonical_json(prim)) or
                            prim['key'] != dict(key, training_people_digest='fixed-RAW-numerical-operator') or
                            info != dict(prim, key=key, raw_calculation=sha(canonical_json(prim['key'])))):
                        raise ValueError('Invalid RAW primitive')
                data = self.read(row['payload'])
                if sha(data) != info['sha256']:
                    raise ValueError('Wrong logical geometry digest')
                values = np.load(io.BytesIO(data), allow_pickle=False)
                if values.dtype.kind != 'f' or values.shape != edge.shape or not np.isfinite(values).all():
                    raise ValueError('Invalid finite floating-point geometry')
                # Generator.curvature scatters the original payload into native float64.
                raw[i, edge] = values
                raw[i, width + edge.astype(np.intp)] = 1.
            if train_weights is not None:
                rebuilt = fitted_location_scale(raw, train_weights)
                if any(array_id(a) != array_id(b) for a, b in zip(rebuilt, (mean, scale))):
                    raise ValueError('Scaler differs from certified training rows')
            vector = FrozenVectors((raw - mean) / scale, ids, names, model['arm'], model['partition'],
                                   spec['model'], 'Ricci')
        else:
            raise ValueError('Unsupported vector recipe')
        if vector.identity() != spec['identity']:
            raise ValueError('Canonical vector identity mismatch')
        return vector

    def closure(self, roots):
        """Read every reachable object from disk, including proof NPYs and reference state."""
        return unitcache.object_closure(self, roots)

    def verify_block(self, block, expected, train_weights=None):
        if set(block) != {'model', 'train', 'validate'} or block['model'] != expected['model']:
            raise ValueError('Wrong finalized block model')
        model = record(self, block['model'])
        base = record(self, model['source']) if model['name'] == 'K0' else model
        if (model['name'], base['partition'], base['arm']) != (
                expected['block'], expected['partition'], expected['arm']):
            raise ValueError('Wrong finalized block scope')
        for side in ('train', 'validate'):
            spec = record(self, block[side])
            generator = spec.get('model', spec.get('generator'))
            if generator != block['model']:
                raise ValueError('Split generator differs from finalized block model')
            if side == 'train' and train_weights is not None:
                vec = self.restore(spec, train_weights)
            else:
                vec = self.load_vector(block[side])
            want = expected[side]
            if ((vec.generator, vec.block, vec.arm, vec.partition, list(vec.ids), sha(canonical_json(vec.features)), vec.identity())
                    != (block['model'], model['name'], expected['arm'], expected['partition'],
                        want['ids'], want['layout'], want['identity'])):
                raise ValueError('Wrong ordered supervised split or layout')
            if model['name'] in ('Ricci', 'K0'):
                ricci = record(self, spec['source']) if model['name'] == 'K0' else spec
                if sha(canonical_json(record(self, ricci['inputs']))) != want['inputs']:
                    raise ValueError('Wrong original visit inputs')
                proof = [(r['sample'], r['shard'], sha(canonical_json(r['body']))) for r in ricci['rows']]
                if sha(canonical_json(sorted(proof))) != want['proof']:
                    raise ValueError('Wrong original sample attestation')
        return block


def representation(store, name):
    return record(store, name).get('format')


def split_witness(factory, model, frame, store):
    ref = model['reference']
    refhash = 'identity' if ref is None else array_id(store.load_array(record(store, ref)['log_reference']))
    proof = unitcache.geometry_proof(factory, model['partition'], model['state']['setting'], refhash, frame.index)
    return dict(inputs=sha(canonical_json(visit_inputs(factory.graphs, frame.index))),
                proof=sha(canonical_json(sorted(proof))))


def capture_block(store, model_key, vectors, frames, factory):
    """Capture the contract from the original in-memory standalone results before encoding."""
    reader = RecipeStore(store.root)
    model = record(store, model_key)
    block = dict(model=model_key)
    contract = dict(model=model_key, arm=factory.arm, partition=model['partition'], block='Ricci')
    for side, frame in frames.items():
        vec = vectors[side]
        if (vec.generator, vec.ids, vec.partition, vec.arm, vec.block) != (
                model_key, tuple(frame.index), model['partition'], factory.arm, 'Ricci'):
            raise ValueError('Standalone vector differs from producing split')
        contract[side] = dict(ids=list(frame.index), layout=sha(canonical_json(vec.features)),
                              identity=vec.identity(), **split_witness(factory, model, frame, store))
        block[side] = reader.capture(vec, factory)
    reader.verify_block(block, contract)
    block['contract'] = store.record(contract)
    store.objects.update(reader.objects)
    return block


def projected_block(store, parent, model, width):
    contract = record(store, parent['contract'])
    result = dict(model=model)
    projected = dict(contract, model=model, block='K0')
    for side in ('train', 'validate'):
        vec = store.load_vector(parent[side])
        want = contract[side]
        if vec.identity() != want['identity'] or vec.generator != parent['model']:
            raise ValueError('K0 source differs from standalone contract')
        view = FrozenVectors(vec.values[:, :width], vec.ids, vec.features[:width],
                             vec.arm, vec.partition, model, 'K0')
        projected[side] = dict(want, identity=view.identity(), layout=sha(canonical_json(view.features)))
        result[side] = store.record(dict(format='k0-view-v2', source=parent[side],
                                         model=model, identity=view.identity()))
    RecipeStore(store.root).verify_block(result, projected)
    result['contract'] = store.record(projected)
    return result


def verify_evidence(store, block, train, valid, arm, *, objects=None, factory=None):
    """Check the anchored fit block, ordered split rosters and all physical dependencies."""
    if 'contract' not in block:
        if any(representation(store, block[side]) for side in ('train', 'validate')):
            raise ValueError('Missing compact vector contract')
        return
    contract = record(store, block['contract'])
    partition = digest(canonical_json(sorted(zip(train.index, train.person))))
    if (contract['arm'], contract['partition']) != (arm, partition):
        raise ValueError('Wrong compact evidence partition')
    reader = RecipeStore(store.root)
    base = {field: block[field] for field in ('model', 'train', 'validate')}
    model = record(store, block['model'])
    for side, frame in [('train', train), ('validate', valid)]:
        if contract[side]['ids'] != list(frame.index):
            raise ValueError('Wrong compact evidence roster')
        if factory is not None:
            spec = record(store, block[side])
            if (spec['version'] != factory.version or
                    any(contract[side][k] != v for k, v in split_witness(factory, model, frame, store).items())):
                raise ValueError('Compact witness differs from released graph or source')
    weights = None
    if factory is not None:
        _, index, counts = np.unique(train.person, return_inverse=True, return_counts=True)
        weights = 1 / counts[index]
    reader.verify_block(base, contract, weights)
    if objects is not None and not reader.closure(block) <= objects:
        raise ValueError('Missing compact evidence from anchored inventory')
