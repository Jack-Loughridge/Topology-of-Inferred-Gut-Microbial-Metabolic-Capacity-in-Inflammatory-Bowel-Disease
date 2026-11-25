"""Experimental Ricci storage reader. No production imports from this directory."""
import json
import os
import re

import numpy as np

from analysis.alpha_matched import unitcache
from analysis.alpha_matched.artifacts import Store, sha, space
from analysis.alpha_matched.partitions import FrozenVectors, array_id, canonical_json, digest


def record(store, key):
    return json.loads(store.read(key))


def ordinary_closure(store, roots):
    reader = Store(store.root)
    unitcache.dependencies(reader, roots)
    return reader.objects


def copy_objects(source, dest, names):
    for name in sorted(names - dest.objects):
        source.read(name)  # Verify the content address, including external NPY.
        dest.publish(name, (source.root / name).read_bytes())
        dest.objects.add(name)


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


def block_expectation(dense, original, train, valid, arm, factory):
    """Trusted caller input from the original dense build, graph and supervised split."""
    model = record(dense, original['model'])
    partition = digest(canonical_json(sorted(zip(train.index, train.person))))
    if factory.arm != arm:
        raise ValueError('Unexpected source arm')
    result = dict(model=original['model'], block=model['name'], partition=partition, arm=arm)
    for side, frame in [('train', train), ('validate', valid)]:
        v = record(dense, original[side])
        if ((v['generator'], v['block'], v['arm'], v['partition'], record(dense, v['ids'])) !=
                (original['model'], model['name'], arm, partition, list(frame.index))):
            raise ValueError('Wrong original supervised split')
        result[side] = dict(ids=list(frame.index), layout=sha(dense.read(v['features'])), identity=v['identity'])
        if model['name'] in ('Ricci', 'K0'):
            result[side]['inputs'] = sha(canonical_json(visit_inputs(factory.graphs, frame.index)))
            base = record(dense, model['source']) if model['name'] == 'K0' else model
            ref = base['reference']
            refhash = 'identity' if ref is None else array_id(dense.load_array(record(dense, ref)['log_reference']))
            proof = unitcache.geometry_proof(factory, partition, base['state']['setting'], refhash, frame.index)
            result[side]['proof'] = sha(canonical_json(sorted(proof)))
    return result


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
        if sha(source.read_bytes()) + '.npy' != source.name:
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
                            body=json.loads(body), payload=self.link_npy(folder / (info['sha256'] + '.npy')))
                if 'raw_calculation' in info:
                    primitive = factory.cache_dir / '.raw-numerics' / info['raw_calculation']
                    item['primitive'] = json.loads((primitive / 'receipt.json').read_bytes())
                rows.append(item)
        spec = dict(format='active-ricci-v2', model=vector.generator, ids=self.record(vector.ids),
                    version=factory.version, inputs=self.record(visit_inputs(graph, vector.ids)),
                    rows=rows, identity=vector.identity())
        self.restore(spec)  # Verify original canonical bytes before publishing the root.
        return self.record(spec)

    def project(self, source, frozen):
        spec = dict(format='k0-view-v2', source=source, model=frozen.generator, identity=frozen.identity())
        self.restore(spec)
        return self.record(spec)

    def load_vector(self, name):
        spec = record(self, name)
        return self.restore(spec) if 'format' in spec else super().load_vector(name)

    def restore(self, spec):
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
                        row['payload'] != info['sha256'] + '.npy' or
                        (info['failed'], info['finite_edges'], info['active_edges']) != (0, len(edge), len(edge)) or
                        body != dict(active_nonself_edges=len(edge), finite_edges=len(edge), failed_nonfinite=0,
                                     solver_receipt=row['receipt']['digest'], numerical_errors=info['certificate'],
                                     global_activity=sha(axis[edge].tobytes()))):
                    raise ValueError('Invalid sample evidence')
                errors = info['certificate']
                tolerance = key_base['tolerance']
                if (set(errors) != set(tolerance) or any(not 0 <= errors[k] <= limit
                        or not np.isfinite([errors[k], limit]).all() for k, limit in tolerance.items())):
                    raise ValueError('Invalid numerical error bounds')
                if model['arm'] == 'RAW':
                    prim = row['primitive']['record']
                    if (row['primitive']['digest'] != sha(canonical_json(prim)) or
                            prim['key'] != dict(key, training_people_digest='fixed-RAW-numerical-operator') or
                            info != dict(prim, key=key, raw_calculation=sha(canonical_json(prim['key'])))):
                        raise ValueError('Invalid RAW primitive')
                values = self.load_array(row['payload'])
                if values.dtype != np.float64 or values.shape != edge.shape or not np.isfinite(values).all():
                    raise ValueError('Invalid finite float64 geometry')
                raw[i, edge] = values
                raw[i, width + edge.astype(np.intp)] = 1.
            vector = FrozenVectors((raw - mean) / scale, ids, names, model['arm'], model['partition'],
                                   spec['model'], 'Ricci')
        else:
            raise ValueError('Unsupported vector recipe')
        if vector.identity() != spec['identity']:
            raise ValueError('Canonical vector identity mismatch')
        return vector

    def closure(self, roots):
        """Read every reachable object from disk, including proof NPYs and reference state."""
        names = set()

        expanded = set()

        def visit(value, labels=False):
            if isinstance(value, dict):
                for field, child in value.items():
                    if field != 'sample':
                        visit(child, labels=field in ('ids', 'features'))
            elif isinstance(value, (list, tuple)):
                for child in value:
                    visit(child)
            elif isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}\.(jsonz?|npyz?)', value):
                if (value, labels) in expanded:
                    return
                expanded.add((value, labels))
                names.add(value)
                if value.endswith(('.npy', '.npyz')):
                    if not np.isfinite(self.load_array(value)).all():
                        raise ValueError('Nonfinite portable dependency')
                else:
                    obj = record(self, value)
                    if not labels:
                        visit(obj)
        visit(roots)
        return names

    def verify_block(self, block, expected):
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

    def verify_set(self, roots, expected):
        if not roots or len(roots) != len(expected):
            raise ValueError('Wrong portable block roster')
        names = self.closure(roots)
        for block, want in zip(roots, expected):
            self.verify_block(block, want)
        return names

    @staticmethod
    def seal_record(roots, contract, names):
        return dict(format='portable-ricci-v2', roots=roots, contract=contract,
                    closure=sha(canonical_json(sorted(names))))

    def export(self, dest, roots, expected):
        names = self.verify_set(roots, expected)
        copy_objects(self, dest, names)
        received = dest.verify_set(roots, expected)
        if names != received:
            raise ValueError('Incomplete portable dependencies')
        contract = dest.record(expected)
        if record(dest, contract) != expected:
            raise ValueError('Invalid portable expectation document')
        received.add(contract)
        dest.publish('complete.json', canonical_json(self.seal_record(roots, contract, received)))
        return received

    def restart(self, expected):
        """Cold restart with the trusted expectation document or its external SHA-256."""
        marker = json.loads((self.root / 'complete.json').read_bytes())
        contract = record(self, marker['contract'])
        anchor = expected if isinstance(expected, str) else sha(canonical_json(expected))
        if sha(canonical_json(contract)) != anchor:
            raise ValueError('Wrong portable expectation document')
        roots = marker['roots']
        names = self.verify_set(roots, contract)
        names.add(marker['contract'])
        if marker != self.seal_record(roots, marker['contract'], names):
            raise ValueError('Invalid portable seal')
        return roots

    def run_block(self, context, key, expected, compute):
        def verified(block):
            self.verify_set([block], [expected])
            return block
        return verified(unitcache.Units(self, [context, expected]).run(key, lambda: verified(compute())))

    def dense_block(self, dest, block, expected):
        """Compatibility boundary for the existing parent/comparator reader."""
        self.verify_set([block], [expected])
        copy_objects(self, dest, self.closure(block['model']))
        return dict(model=block['model'], **{
            side: dest.vector(self.load_vector(block[side])) for side in ('train', 'validate')})
