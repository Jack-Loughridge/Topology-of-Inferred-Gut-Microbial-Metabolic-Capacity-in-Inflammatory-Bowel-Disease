"""Run one registered development outer fold, including its nested trial ledger."""
import argparse
from datetime import datetime
import fcntl
import json
import io
import os
from pathlib import Path
import re

import numpy as np
import pandas as pd

from . import unitcache, vector_storage, bundle, release as clinical
from .cache import payload_name
from .artifacts import Store, sha, space
from .comparators import Baseline
from .features import Factory
from .h0 import fit_historical, merge_deaths
from .nesting import Procedure, fit_head, score
from .thresholding import select_thresholds
from .partitions import FrozenVectors, RelativeReference, array_id, canonical_json, digest, fusion_input

FINAL = 'final_refit'
ROOT = Path(__file__).resolve().parents[3]
BUNDLE_FILES = {'raw.npz', 'vertices.npz', 'abundance.npz', 'pairs.npy', 'axes.json'}


def source_hashes():
    modules = ('artifacts bundle cache certified_reference comparators features geometry h0 '
               'integer_transport log_kernel nesting partitions release run_fold statistics support thresholding unitcache vector_storage').split()
    return {name + '.py': bundle.sha(Path(__file__).with_name(name + '.py')) for name in modules}


def active(root):
    if os.path.lexists(root / 'STOP'):
        raise RuntimeError('Alpha STOP')
    status = json.loads((root / 'status.json').read_bytes())
    if status['status'] != 'ACTIVE':
        raise RuntimeError('Inactive Alpha run')
    return status


def inside(base, name):
    p = base / name
    if p.resolve() != p or not p.is_relative_to(base) or p == base:
        raise ValueError('Path outside registered scope')
    return p


def release(root, fold):
    status = active(root)
    final = fold == FINAL
    if final and status['stage'] not in ('G5', 'G6'):
        raise RuntimeError('Final refit requires completed G5')
    if not final and status['stage'] not in ('G4', 'G5'):
        raise RuntimeError('Development fitting requires G4 or G5')
    entry = status['final_refit_release' if final else 'development_fold_release']
    path = inside(root / 'reviews', entry['file'])
    data = path.read_bytes()
    if sha(data) != entry['sha256']:
        raise ValueError('Changed release')
    rec = json.loads(data)
    roles = json.loads((root / 'roles.json').read_bytes())
    reviewer = roles['independent reviewer'].split()[0]
    if final and rec.get('gate') != 'G5':
        raise ValueError('Expected independent G5 completion decision')
    scope = 'development-final-refit' if final else 'development-only-fold'
    if (rec['decision'] != 'PASS' or rec['scope'] != scope
            or rec['stage'] != status['stage'] or rec['reviewer'] != reviewer
            or rec['blocking_findings'] != [] or fold not in rec['folds']
            or datetime.fromisoformat(rec['timestamp']).utcoffset() is None):
        raise ValueError('Missing applicable signed scientific release')
    if rec['source_sha256'] != source_hashes():
        raise ValueError('Implementation changed since review')
    return rec, entry['sha256']


def read_inputs(root, rec):
    # The signed release designates a development-only export under this root.
    base = root / 'manifests' / 'development-fold-inputs'
    def checked(key):
        spec = rec['inputs'][key]
        path = inside(base, spec['path'])
        data = path.read_bytes()
        if sha(data) != spec['sha256']:
            raise ValueError('Released input changed: ' + key)
        return data
    config = json.loads(checked('config'))
    splits = json.loads(checked('splits'))
    meta = pd.read_csv(io.BytesIO(checked('meta')), sep='\t', dtype=str, keep_default_na=False)
    meta = meta.set_index('sample')
    classes = config['classes']
    if (not isinstance(classes, list) or len(classes) not in (2, 3) or
            len(set(classes)) != len(classes) or any(not isinstance(c, str) or not c.strip() for c in classes)
            or not config['endpoint'] or rec['endpoint'] != config['endpoint']):
        raise ValueError('Invalid released endpoint or class order')
    fit_thresholds = config.get('fit_thresholds', False)
    if type(fit_thresholds) is not bool:
        raise ValueError('fit_thresholds must be boolean')
    if len(classes) == 2 and (rec['stage'] == 'G5' or rec['scope'] == 'development-final-refit') and not fit_thresholds:
        raise ValueError('Binary G5 and final refits require fit_thresholds=true')
    if (not meta.index.is_unique or '' in meta.index or
            (meta[['person', 'group', 'cohort', 'family', 'y']] == '').any().any()
            or not meta.role.eq('development').all()
            or not meta.y.isin(list(map(str, range(len(classes))))).all()):
        raise ValueError('Invalid endpoint or non-development metadata')
    meta['y'] = meta.y.astype(int)
    if (meta.groupby('person')[['cohort', 'family', 'group', 'y']].nunique() != 1).any().any():
        raise ValueError('Inconsistent participant metadata')
    if (meta.groupby('cohort').family.nunique() != 1).any():
        raise ValueError('Cohort spans multiple recruitment families')
    for key, spec in splits.items():
        for side in ('train', 'validate'):
            roster = spec[side]
            if key == FINAL and side == 'validate':
                if roster != []:
                    raise ValueError('Final refit requires an empty application roster')
                continue
            if not isinstance(roster, list) or not roster or len(set(roster)) != len(roster):
                raise ValueError('Invalid participant roster')
    receipt = json.loads(checked('bundle'))
    path = inside(base, rec['inputs']['bundle']['path'])
    if path.name != 'receipt.json' or set(receipt['outputs']) != BUNDLE_FILES:
        raise ValueError('Incomplete graph bundle')
    for name in BUNDLE_FILES:
        inside(base, path.parent / name)
    graphs = bundle.load(path.parent)
    checked('bundle')
    if set(meta.index) != set(graphs.sample_ids):
        raise ValueError('Development bundle and metadata rosters differ')
    axes = json.loads((path.parent / 'axes.json').read_bytes())
    species = axes['source_species']
    if len(species) != graphs.species_abundance.shape[1] or len(set(species)) != len(species):
        raise ValueError('Misaligned source species catalogue')
    if any(bundle.sha(path.parent / n) != h for n, h in receipt['outputs'].items()):
        raise ValueError('Graph bundle changed during read')
    return config, splits, meta, graphs, axes['source_species']


def validate_probabilities(p, nrows, classes):
    if (not np.isfinite(p).all() or p.shape != (nrows, len(classes)) or (p < 0).any()
            or (p > 1).any() or not np.allclose(p.sum(axis=1), 1, rtol=0, atol=1e-10)):
        raise ValueError('Invalid class probabilities')


def species_model_key(store, model, arm):
    descriptor = json.loads(store.read(model))
    if descriptor['name'] != 'species' or descriptor.pop('arm') != arm or descriptor['reference'] is not None:
        raise ValueError('Invalid arm-independent species generator')
    return sha(canonical_json(descriptor))


def species_prediction_key(store, arm, spec, gen, x):
    model = species_model_key(store, gen['blocks'][0], arm)
    return (model, spec['head'], spec['endpoint'], tuple(spec['classes']), gen['implementation'], array_id(x))


class Fold:
    def __init__(self, proc, factory, store, config, species, check, release_id=None, species_fits=None):
        self.proc, self.factory, self.store = proc, factory, store
        self.config, self.species, self.check = config, species, check
        self.references = {}
        self.axis = digest(canonical_json(factory.graphs.axis))
        self.species_fits = species_fits
        self.records = []
        self.trials = []
        self.heads = [tuple(h) for h in config['heads']]
        self.ricci_storage = config.get('ricci_storage', 'active-v2')
        if self.ricci_storage not in ('active-v2', 'dense'):
            raise ValueError('Unknown Ricci storage format')

        self.units = None if release_id is None else unitcache.Units(store, [
            release_id, factory.arm, factory.version, config, unitcache.environment()])
        self.certificates = set()

    def unit(self, operation, name, key, parameters, compute):
        if self.units is None:
            return compute()
        rosters = [list(zip(frame.index, frame.person)) for frame in self.proc.frames(key)]
        return self.units.run([operation, name, key, rosters, parameters], compute)

    def build(self, name, setting, key):
        self.check()
        train, valid = self.proc.frames(key)
        partition = digest(canonical_json(sorted(zip(train.index, train.person))))
        def proof(block):
            model = json.loads(self.store.read(block['model']))
            ref = model['reference']
            required = name == 'direct-relative' or self.factory.arm == 'RELATIVE' and name in (
                'Ricci', 'H0', 'fixed-alpha', 'historical')
            if ((model['name'], model['arm'], model['partition'], model['axis']) !=
                    (name, self.factory.arm, partition, self.axis) or
                    required != (ref is not None) or
                    required and self.references.setdefault(partition, ref) != ref):
                raise ValueError('Inconsistent unit generator or relative reference')
            graph = self.factory.graphs
            if (model['catalogue'] != graph.catalogue_digest or model['mapping'] != graph.mapping_digest
                    or model['training_people_sha256'] != sha(canonical_json(sorted(set(train.person))))
                    or model['state'].get('setting', {}) != setting):
                raise ValueError('Build model differs from reviewed inputs')
            if required:
                references = self.factory.references
                if partition not in references:
                    rows = [graph.lookup[i] for i in train.index]
                    references[partition] = RelativeReference.fit(graph.raw[rows], train.person, train.index, graph.axis)
                if ref != self.store.reference(references[partition]):
                    raise ValueError('Wrong training reference')
            for side, frame in [('train', train), ('validate', valid)]:
                vec = self.store.load_vector(block[side])
                if ((vec.generator, vec.arm, vec.partition, vec.block, vec.ids) !=
                        (block['model'], self.factory.arm, partition, name, tuple(frame.index)) or
                        list(vec.features) != json.loads(self.store.read(model['features']))):
                    raise ValueError('Wrong unit vector layout')
            if name != 'Ricci':
                return None
            if self.ricci_storage == 'active-v2':
                if 'contract' not in block:
                    raise ValueError('Unbound Ricci producer evidence')
                vector_storage.verify_evidence(self.store, block, train, valid, self.factory.arm, factory=self.factory)
            refhash = 'identity' if ref is None else array_id(self.store.load_array(
                json.loads(self.store.read(ref))['log_reference']))
            rows = unitcache.geometry_proof(self.factory, partition, setting, refhash, [*train.index, *valid.index])
            self.certificates.update(rows)
            return sha(canonical_json(sorted(rows)))

        computed = False
        def compute():
            nonlocal computed
            block = self._build(name, setting, key)
            computed = True
            return dict(block=block, proof=proof(block))

        result = self.unit('build', name, key, setting, compute)
        if not computed and proof(result['block']) != result['proof']:
            raise ValueError('Changed build evidence')
        return result['block']

    def _build(self, name, setting, key):
        self.check()
        train, valid = self.proc.frames(key)
        f, s = self.factory, self.store
        graph = f.graphs
        partition = digest(canonical_json(sorted(zip(train.index, train.person))))
        # Allow for uncached per-sample curvature shards before Generator writes.
        space(s.root, (graph.raw.nnz * 16 + len(graph.sample_ids) * 32768)
              if name in ('Ricci', 'K0') else 0)
        if name in ('H0', 'fixed-alpha', 'Ricci'):
            block = 'H0' if name in ('H0', 'fixed-alpha') else 'Ricci'
            gen, t, v = self.proc.build(block, tuple(sorted(setting.items())), train, valid)
            ref = gen.reference
            state = dict(block=block, mean=s.array(gen.mean), scale=s.array(gen.scale),
                         image=s.image(gen.image) if gen.image else None, setting=setting,
                         generator=gen.generator, tolerances=f.tolerances)
            names = t.features
            tr, va = t.values, v.values
        elif name == 'historical':
            ref = f.references.get(partition)
            if f.arm == 'RELATIVE' and ref is None:
                ref = Baseline(graph, 'direct-relative', train).reference
            def diagrams(frame):
                result = []
                for sample in frame.index:
                    _, log, vertices, pairs = graph.graph(graph.lookup[sample], ref)
                    result.append(merge_deaths(len(vertices), pairs, log)[0])
                return result
            td, vd = diagrams(train), diagrams(valid)
            historical = fit_historical(td, train.y, train.person, **setting)
            tr, va = historical.image.transform(td), historical.image.transform(vd)
            names = tuple('PI:' + str(i) for i in range(tr.shape[1]))
            state = dict(image=s.image(historical.image), weight=s.array(historical.weight),
                         intercept=s.array(historical.intercept), centers=s.array(historical.centers),
                         classes=historical.classes.tolist(), setting=setting)
        else:
            gen = Baseline(graph, name, train)
            ref = gen.reference
            tr, va = gen.transform(train.index), gen.transform(valid.index)
            names = ([self.species[i] for i in gen.species_axis] if name == 'species' else
                     ('vertices', 'edges', 'deaths', 'components') if name == 'graph-count' else
                     tuple(f'{a}>{b}' for a, b in graph.axis))
            state = dict(mean=s.array(gen.mean), scale=s.array(gen.scale))
            if name == 'species':
                state['species_axis'] = s.array(gen.species_axis)
                state['species_taxa'] = s.record(names)
                state['species_impute'] = None
                if gen.species_impute is not None:
                    state['species_impute'] = s.array(gen.species_impute)
                    names = tuple(kind + ':' + canonical_json(taxon).decode()
                                  for kind in ('abundance', 'unreported') for taxon in names)
        reference = s.reference(ref)
        if name == 'direct-relative' or f.arm == 'RELATIVE' and name not in (
                'species', 'activity', 'direct-raw', 'graph-count'):
            if reference is None or self.references.setdefault(partition, reference) != reference:
                raise ValueError('Mismatched fold-local relative references')
        model = s.record(dict(state=state, name=name, arm=f.arm, partition=partition,
                              reference=reference, catalogue=graph.catalogue_digest,
                              mapping=graph.mapping_digest, features=s.record(list(names)), axis=self.axis,
                              training_people_sha256=sha(canonical_json(sorted(set(train.person))))))
        if name == 'Ricci' and self.ricci_storage == 'active-v2':
            frames = dict(train=train, validate=valid)
            vectors = {side: FrozenVectors(values, tuple(frames[side].index), tuple(names),
                                           f.arm, partition, model, name)
                       for side, values in [('train', tr), ('validate', va)]}
            return vector_storage.capture_block(s, model, vectors, frames, f)
        def freeze(values, frame):
            vec = FrozenVectors(values, tuple(frame.index), tuple(names), f.arm, partition, model, name)
            return s.vector(vec)
        return dict(model=model, train=freeze(tr, train), validate=freeze(va, valid))

    def species_fit_key(self, key, block, head):
        # This map lives inside one execute() call on immutable released metadata.
        # Bind all fitting inputs and the application roster, including participant weights.
        model = species_model_key(self.store, block['model'], self.factory.arm)
        axes = []
        for side in ('train', 'validate'):
            v = json.loads(self.store.read(block[side]))
            if (v['generator'], v['arm'], v['block']) != (block['model'], self.factory.arm, 'species'):
                raise ValueError('Invalid species vector provenance')
            axes.append([v[field] for field in ('partition', 'ids', 'features', 'values')])
        train, valid = self.proc.frames(key)
        training = list(zip(train.index, train.person, train.group, train.y.map(int)))
        application = list(zip(valid.index, valid.person, valid.group))
        return sha(canonical_json([self.config['endpoint'], self.config['classes'], self.factory.version,
                                   key, head, model, axes, training, application]))

    def reuse_species_fit(self, source, blocks):
        info = json.loads(self.store.read(source))
        info.update(arm=self.factory.arm, blocks=blocks)
        if info['split'] == FINAL:
            generators = json.loads(self.store.read(info['generators']))
            generators['blocks'] = [b['model'] for b in blocks]
            info['generators'] = self.store.record(generators)
            deployment = json.loads(self.store.read(info['deployment']))
            deployment.update(arm=self.factory.arm, generators=info['generators'])
            info['deployment'] = self.store.record(deployment)
        return self.store.record(info)

    def fit(self, name, key, blocks, head):
        self.check()
        s = self.store
        share = self.species_fit_key(key, blocks[0], head) if name == 'species' and self.species_fits is not None else None
        def compute():
            if share is not None and share in self.species_fits:
                return self.reuse_species_fit(self.species_fits[share], blocks)
            return self._fit(name, key, blocks, head)
        record = self.unit('fit', name, key, [blocks, head], compute)
        info = json.loads(s.read(record))
        train, valid = self.proc.frames(key)
        expected = dict(name=name, arm=self.factory.arm, split=key, blocks=blocks, head=head,
                        classes=self.config['classes'], endpoint=self.config['endpoint'],
                        train_groups=sorted(set(train.group)))
        if any(canonical_json(info[k]) != canonical_json(v) for k, v in expected.items()):
            raise ValueError('Wrong fit unit')
        p = None
        if name == 'historical' and info['fitted'] != blocks[0]['model']:
            raise ValueError('Historical head differs from its generator')
        if key == FINAL:
            gen = json.loads(s.read(info['generators']))
            deployment = json.loads(s.read(info['deployment']))
            if (deployment != dict(method=name, arm=self.factory.arm, endpoint=info['endpoint'],
                                   classes=info['classes'], head=info['fitted'], generators=info['generators'])
                    or gen['blocks'] != [b['model'] for b in blocks]
                    or gen['implementation'] != self.factory.version
                    or gen['tolerances'] != self.factory.tolerances
                    or json.loads(s.read(gen['training_visits'])) != sorted(sha(canonical_json(i)) for i in train.index)):
                raise ValueError('Inconsistent final training deployment')
        else:
            p = s.load_array(info['probabilities'])
            validate_probabilities(p, len(valid), self.config['classes'])
            vectors = [s.load_vector(b['validate']) for b in blocks]
            x = vectors[0].values if len(vectors) == 1 else fusion_input(*vectors)
            np.testing.assert_allclose(p, s.probabilities(name, info['fitted'], x), rtol=1e-12, atol=1e-15)
            means = pd.DataFrame(p, index=valid.index).groupby(valid.person, sort=True).mean()
            if list(valid.index) != info['samples'] or list(means.index) != info['participants']:
                raise ValueError('Wrong fitted prediction roster')
            np.testing.assert_array_equal(means.to_numpy(), s.load_array(info['participant_means']))
        if share is not None:
            self.species_fits.setdefault(share, record)
        self.records.append(record)
        return p, record

    def _fit(self, name, key, blocks, head):
        self.check()
        s = self.store
        train, valid = self.proc.frames(key)
        def matrix(split):
            vectors = [s.load_vector(b[split]) for b in blocks]
            return vectors[0].values if len(vectors) == 1 else fusion_input(*vectors)
        tr, va = matrix('train'), matrix('validate')
        classes = list(range(len(self.config['classes'])))
        if sorted(train.y.unique()) != classes:
            raise ValueError('Incomplete training class roster')
        if name == 'historical':
            fitted = blocks[0]['model']
        else:
            clf = fit_head(head, tr, train)
            fitted = s.head(clf)
        info = dict(name=name, arm=self.factory.arm, split=key, blocks=blocks, head=head,
                    fitted=fitted, classes=self.config['classes'], endpoint=self.config['endpoint'],
                    train_groups=sorted(set(train.group)))
        p = None
        if key == FINAL:
            if len(valid):
                raise ValueError('Unexpected application rows during final refit')
            dev_ids = s.record(sorted(sha(canonical_json(i)) for i in train.index))
            generators = s.record(dict(training_visits=dev_ids, blocks=[b['model'] for b in blocks],
                                       implementation=self.factory.version,
                                       tolerances=self.factory.tolerances))
            info['deployment'] = s.record(dict(method=name, arm=self.factory.arm, endpoint=info['endpoint'],
                                               classes=info['classes'], head=fitted, generators=generators))
            info['generators'] = generators
        else:
            p = s.probabilities(name, fitted, va)
            if name != 'historical':
                observed = clf.predict_proba(va)
                np.testing.assert_allclose(p, observed, rtol=1e-12, atol=1e-15)
                p = observed
            validate_probabilities(p, len(valid), classes)
            participant = pd.DataFrame(p, index=valid.index).groupby(valid.person, sort=True).mean()
            info.update(samples=list(valid.index), probabilities=s.array(p),
                        participants=list(participant.index), participant_means=s.array(participant.to_numpy()))
        return s.record(info)

    def select(self, name, key, settings):
        train, _ = self.proc.frames(key)
        folds = self.proc.children(key, train)
        heads = [None] if name == 'historical' else self.heads
        best = None
        for setting in settings:
            parts = {h: [] for h in heads}
            evidence = {h: [] for h in heads}
            for child, _, valid in folds:
                block = self.build(name, setting, child)
                for head in heads:
                    p, record = self.fit(name, child, [block], head)
                    parts[head].append((valid, p))
                    evidence[head].append(record)
            for head in heads:
                value = self.trial(name, key, setting, head, parts[head], evidence[head])
                if best is None or value > best[0]:
                    best = value, setting, head
        if best is None:
            raise ValueError('Missing registered settings or heads for ' + name)
        block = self.build(name, best[1], key)
        _, record = self.fit(name, key, [block], best[2])
        return block, record

    def trial(self, name, key, setting, head, parts, evidence):
        value = score(parts)
        self.trials.append(self.store.record(dict(name=name, arm=self.factory.arm, split=key,
                           setting=setting, head=head, score=value, fold_fits=evidence)))
        return value

    def curvature_block(self, ricci):
        s = self.store
        n = len(self.factory.graphs.axis)
        model = s.record(dict(name='K0', source=ricci['model'], columns=[0, n]))
        if 'contract' in ricci:
            return vector_storage.projected_block(s, ricci, model, n)
        result = dict(model=model)
        for split in ('train', 'validate'):
            v = s.load_vector(ricci[split])
            if not all(f.startswith('K0:') for f in v.features[:n]) or len(v.features) != 2*n:
                raise ValueError('Unexpected Ricci coordinates')
            result[split] = s.vector(FrozenVectors(v.values[:, :n], v.ids, v.features[:n],
                                                   v.arm, v.partition, model, 'K0'))
        return result

    def run(self, key):
        settings = self.config['candidates']
        train, _ = self.proc.frames(key)
        derived = ('joint', 'K0')
        parts = {m: {h: [] for h in self.heads} for m in derived}
        evidence = {m: {h: [] for h in self.heads} for m in derived}
        primary = ('H0', 'Ricci')
        for child, _, valid in self.proc.children(key, train):
            blocks = [self.select(name, child, settings[name])[0] for name in primary]
            for name, inputs in [('joint', blocks), ('K0', [self.curvature_block(blocks[1])])]:
                for head in self.heads:
                    p, rec = self.fit(name, child, inputs, head)
                    parts[name][head].append((valid, p))
                    evidence[name][head].append(rec)
        fits = [self.select(name, key, settings[name]) for name in primary]
        outputs = {name: fit[1] for name, fit in zip(primary, fits)}
        blocks = [fit[0] for fit in fits]
        for name, inputs in [('joint', blocks), ('K0', [self.curvature_block(blocks[1])])]:
            scores = {h: self.trial(name, key, None, h, parts[name][h], evidence[name][h]) for h in self.heads}
            _, outputs[name] = self.fit(name, key, inputs, max(scores, key=scores.get))
        baselines = {'species': [{}], 'activity': [{}], 'graph-count': [{}],
                     'direct-' + self.factory.arm.lower(): [{}],
                     'fixed-alpha': [x for x in settings['H0'] if not x['learned']],
                     'historical': self.config['historical']}
        for name, options in baselines.items():
            _, outputs[name] = self.select(name, key, options)
        return dict(outputs=outputs, fits=self.records, trials=self.trials)


def geometry_files(destination, expected):
    files = {}
    attestations = destination / 'geometry/sample-certificates.sqlite'
    if attestations.is_file():
        files[str(attestations.relative_to(destination))] = unitcache.certificate_digest(attestations.parent, expected)
    if not attestations.is_file() and expected:
        raise ValueError('Missing sample certificate database')
    shards = {shard for _, shard, _ in expected}
    geometry = destination / 'geometry'
    primitives = set()
    for path in sorted(geometry.glob('*/receipt.json')):
        if path.parent.name not in shards:
            raise ValueError('Unexpected geometry shard')
        shards.remove(path.parent.name)
        data = path.read_bytes()
        entry = json.loads(data)
        record = entry['record']
        if sha(canonical_json(record)) != entry['digest'] or record['failed'] != 0:
            raise ValueError('Invalid geometry receipt')
        if record['key']['arm'] == 'RAW':
            primitives.add(record['raw_calculation'])
        arr = path.with_name(payload_name(record))
        if bundle.sha(arr) != record.get('zlib_sha256', record['sha256']):
            raise ValueError('Changed curvature array')
        files[str(path.relative_to(destination))] = sha(data)
        files[str(arr.relative_to(destination))] = record.get('zlib_sha256', record['sha256'])
    if shards:
        raise ValueError('Missing geometry receipts')
    actual = {p.parent.name for p in (geometry / '.raw-numerics').glob('*/receipt.json')}
    if actual != primitives:
        raise ValueError('Incomplete RAW numerical inventory')
    for primitive in sorted(primitives):
        path = geometry / '.raw-numerics' / primitive / 'receipt.json'
        data = path.read_bytes()
        record = json.loads(data)['record']
        array = path.with_name(payload_name(record))
        if bundle.sha(array) != record.get('zlib_sha256', record['sha256']):
            raise ValueError('Changed RAW numerical array')
        files[str(path.relative_to(destination))] = sha(data)
        files[str(array.relative_to(destination))] = record.get('zlib_sha256', record['sha256'])
    return files


def execute(root, fold):
    rec, release_id = release(root, fold)
    config, splits, meta, graph, species = read_inputs(root, rec)
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', fold) or fold not in splits:
        raise ValueError('Invalid outer fold')
    job_type = 'development-final-refits' if fold == FINAL else 'development-folds'
    destination = inside(root / 'jobs', Path(job_type) / release_id / fold)
    store = Store(destination)
    def check():
        if release(root, fold)[1] != release_id:
            raise RuntimeError('Release changed during job')
        space(destination, 0)
    with (destination / 'job.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        check()
        # Shards may survive an interruption before their parent build is published.
        store.publish('runtime.json', canonical_json(unitcache.environment()))
        marker = destination / 'receipt.json'
        if marker.exists():
            envelope = json.loads(marker.read_bytes())
            receipt = json.loads(store.read(envelope['record']))
            if receipt['release'] != release_id or receipt['fold'] != fold:
                raise ValueError('Completion identity mismatch')
            for name in receipt['objects']:
                store.read(name)

        try:
            arms = {}
            certificates = set()
            species_fits = {}
            for arm in ('RAW', 'RELATIVE'):
                factory = Factory(graph, arm, destination / 'geometry', config['tolerances'])
                factory.version = sha(canonical_json(rec['source_sha256']))
                proc = Procedure(meta, splits, factory, {}, [], None)
                train, valid = proc.frames(fold)
                if (set(train.person) | set(valid.person) != set(meta.person)
                        or set(train.family) & set(valid.family) or train.empty or valid.empty != (fold == FINAL)):
                    raise ValueError('Invalid outer cohort partition')
                worker = Fold(proc, factory, store, config, species, check, release_id, species_fits)
                arms[arm] = worker.run(fold)
                if config.get('fit_thresholds', False):
                    arms[arm]['decision_rules'] = select_thresholds(worker, fold)
                certificates.update(worker.certificates)
            check()
            receipt = store.record(dict(fold=fold, release=release_id, arms=arms,
                                        objects=sorted(store.objects), inputs=rec['inputs'],
                                        geometry=geometry_files(destination, certificates)))
            for name in store.objects:
                store.read(name)
            check()
            store.publish('receipt.json', canonical_json(dict(record=receipt)))
            return sha(marker.read_bytes())
        except Exception as exc:
            store.record(dict(status='failed', fold=fold, release=release_id,
                              error=type(exc).__name__, detail=str(exc), objects=sorted(store.objects)))
            raise


def application_release(root, submission):
    status = active(root)
    if status['stage'] not in ('G6', 'G7'):
        raise RuntimeError('Application requires independent G6 release')
    entry = status['final_prediction_release']
    review = inside(root / 'reviews', entry['file'])
    roles = json.loads((root / 'roles.json').read_bytes())
    reviewer = roles['independent reviewer'].split()[0]
    evaluator = roles['reserved evaluator'].split()[0]
    if bundle.sha(review) != entry['sha256']:
        raise ValueError('Changed G6 review')
    sid, sub = clinical.verify_release(submission, review, reviewer, evaluator)
    if bundle.sha(review) != entry['sha256']:
        raise ValueError('Changed G6 review during verification')
    return sid, sub, evaluator


def released_models(submission, sub, items):
    """Resolve the portable descriptors and all transitive numeric objects from G6."""
    indexed = {sub['files'][p]: p for p in sub['roles']['models']}
    implementation = sha(canonical_json(source_hashes()))
    methods = {}
    for item in items:
        model = clinical.resolve(submission, indexed[item['model_sha256']])
        store = Store(model.parent)
        spec = json.loads(store.read(model.name))
        generators = spec['generators']
        if (model.name.split('.')[0] != item['model_sha256'] or
                generators.split('.')[0] != item['generator_sha256'] or
                (spec['arm'], spec['method'], spec['endpoint'], spec['classes']) !=
                (item['arm'], item['method'], item['endpoint'], item['class_order'])):
            raise ValueError('Frozen prediction identity mismatch')
        for name in store.deployment_files(model.name):
            relative = str((model.parent / name).relative_to(submission))
            if sub['files'].get(relative) != name.split('.')[0]:
                raise ValueError('Inference dependency missing from G6: ' + name)
        gen = json.loads(store.read(generators))
        if gen['implementation'] != implementation:
            raise ValueError('Runtime differs from frozen implementation')
        methods[item['arm'], item['method']] = (store, spec, gen)
    for (arm, method), (store, _, gen) in methods.items():
        blocks = gen['blocks']
        expected = ['H0', 'Ricci'] if method == 'joint' else [method]
        if [json.loads(store.read(b))['name'] for b in blocks] != expected:
            raise ValueError('Wrong generator layout')
        if method == 'joint' and blocks != [methods[arm, name][2]['blocks'][0] for name in expected]:
            raise ValueError('Joint deployment changed the standalone generators')
        if method == 'K0' and json.loads(store.read(blocks[0]))['source'] != methods[arm, 'Ricci'][2]['blocks'][0]:
            raise ValueError('K0 deployment changed the selected Ricci generator')
    return methods


def curvature_artifact(output, pairs, values):
    if len(values) != len(pairs) or not np.isfinite(values).all():
        raise ValueError('Incomplete application curvature')
    buf = io.BytesIO()
    np.save(buf, np.asarray(pairs, dtype='<i8'), allow_pickle=False)
    edges = output.put(buf.getvalue(), '.npy')
    buf = io.BytesIO()
    np.savez_compressed(buf, pairs=pairs, curvature=values)
    evidence = output.put(buf.getvalue(), '.npz')
    return dict(active_pairs=edges, active_pairs_sha256=edges.split('.')[0],
                values=evidence, values_sha256=evidence.split('.')[0])


def apply_final(root, submission, panel, cohort, endpoint, prediction_root):
    """Evaluator entry point. Apply one G6 cohort/endpoint plan to unlabeled graphs.

    Writes private predictions in the verify_and_seal format. The evaluator
    independently supplies the eligible graph roster and seals the complete plan.
    """
    submission = Path(submission).resolve(strict=True)
    sid, sub, evaluator = application_release(root, submission)
    panel, prediction_root = (Path(p).resolve(strict=True) for p in (panel, prediction_root))
    if any(p == prediction_root or p in prediction_root.parents or prediction_root in p.parents
           for p in (submission, panel)):
        raise ValueError('Prediction workspace overlaps input or frozen submission')
    items = [i for i in sub['planned_predictions'] if (i['cohort'], i['endpoint']) == (cohort, endpoint)]
    if not items or any(not re.fullmatch(r'[A-Za-z0-9_-][A-Za-z0-9_.-]*', i['id']) for i in items):
        raise ValueError('Missing or invalid cohort prediction plan')
    methods = released_models(submission, sub, items)
    eligibility = clinical.resolve(submission, sub['roles']['test_eligibility'][0])
    ids = json.loads(eligibility.read_bytes())['rosters'][cohort][endpoint]
    eligible_ids = {sha(canonical_json(i)) for i in ids}
    checked_rosters = set()
    for store, _, gen in methods.values():
        roster_id = gen['training_visits']
        if roster_id not in checked_rosters:
            if eligible_ids.intersection(json.loads(store.read(roster_id))):
                raise ValueError('Application roster contains a development visit')
            checked_rosters.add(roster_id)
    roster_path = prediction_root / 'graph_roster.json'
    registry_bytes = roster_path.read_bytes()
    registry = json.loads(registry_bytes)
    receipt = panel / 'receipt.json'
    bundle_id = bundle.sha(receipt)
    if (registry['submission_sha256'] != sid or registry['evaluator'] != evaluator or
            registry['bundle_receipts'][cohort] != bundle_id):
        raise ValueError('Evaluator graph bundle binding mismatch')
    native = json.loads(receipt.read_bytes())
    if set(native['outputs']) != BUNDLE_FILES or type(native.get('species_nondetection_zero', False)) is not bool:
        raise ValueError('Invalid application graph receipt')
    for name in BUNDLE_FILES:
        inside(panel, name)
    graph = bundle.load(panel)
    roster = registry['cohorts'][cohort]
    if (not set(ids) <= set(graph.sample_ids) or not set(ids) <= set(roster)
            or bundle.sha(receipt) != bundle_id or any(bundle.sha(panel / n) != h for n, h in native['outputs'].items())):
        raise ValueError('Application bundle or roster changed')
    output = Store(prediction_root)
    buffers = {i['id']: io.BytesIO() for i in items}
    counts = {i['id']: 0 for i in items}
    factories = {}
    for sample in ids:
        active(root)
        evidence = {}
        vectors = {}
        species_predictions = {}
        for item in items:
            arm, method = item['arm'], item['method']
            store, spec, gen = methods[arm, method]
            if arm not in factories:
                factories[arm] = Factory(graph, arm, prediction_root / 'geometry', gen['tolerances'])
                factories[arm].version = gen['implementation']
            factory = factories[arm]
            if factory.tolerances != gen['tolerances']:
                raise ValueError('Inconsistent registered geometry tolerances')
            expected = roster[sample]
            if expected['sample'] != sample or expected['graph_rules_sha256'] != item['graph_rules_sha256']:
                raise ValueError('Wrong application graph rules')
            def vector(key):
                if key not in vectors:
                    block = json.loads(store.read(key))
                    if block['name'] == 'K0':
                        base = vector(block['source'])
                        lo, hi = block['columns']
                        vectors[key] = FrozenVectors(base.values[:, lo:hi], base.ids, base.features[lo:hi],
                                                     base.arm, base.partition, key, 'K0')
                    else:
                        def capture(sample_id, pairs, k):
                            proof = curvature_artifact(output, pairs, k)
                            if sample_id != sample or proof['active_pairs_sha256'] != expected['active_pairs_sha256']:
                                raise ValueError('Curvature and evaluator edge rosters disagree')
                            evidence[arm] = proof
                        space(prediction_root, 65536 + 64 * graph.raw.getrow(graph.lookup[sample]).nnz)
                        vectors[key] = store.transform(key, factory, (sample,), graph.species_ids, curvature=capture)
                return vectors[key]
            blocks = [vector(key) for key in gen['blocks']]
            x = fusion_input(*blocks) if method == 'joint' else blocks[0].values
            if method == 'species':
                fit = species_prediction_key(store, arm, spec, gen, x)
                if fit not in species_predictions:
                    species_predictions[fit] = store.probabilities(method, spec['head'], x)
                p = species_predictions[fit]
            else:
                p = store.probabilities(method, spec['head'], x)
            validate_probabilities(p, 1, spec['classes'])
            row = dict(sample=sample, probabilities=p[0].tolist(), submission_sha256=sid,
                       prediction_id=item['id'], class_order=spec['classes'],
                       model_sha256=item['model_sha256'], generator_sha256=item['generator_sha256'])
            if item['requires_curvature']:
                row['geometry'] = dict(**evidence[arm], sample=sample, profile_sha256=expected['profile_sha256'],
                                       graph_rules_sha256=item['graph_rules_sha256'], prediction_id=item['id'],
                                       generator_sha256=item['generator_sha256'])
                counts[item['id']] += graph.raw.getrow(graph.lookup[sample]).nnz
            buffers[item['id']].write(canonical_json(row) + b'\n')
    if application_release(root, submission)[0] != sid or roster_path.read_bytes() != registry_bytes:
        raise RuntimeError('Application release or evaluator roster changed')
    records = {}
    for item in items:
        active(root)
        key = item['id']
        artifact = output.put(buffers[key].getvalue(), '.jsonl')
        count = clinical.verify_prediction(prediction_root, prediction_root / artifact, item, sid, registry['cohorts'])
        if count != counts[key]:
            raise ValueError('Incomplete application geometry')
        result = dict(submission_sha256=sid, prediction_id=key, eligible_row_hash=item['eligible_row_hash'],
                      row_count=len(ids), artifact=artifact, sha256=artifact.split('.')[0], expected_edges=count,
                      computed_finite_edges=count, failed_edges=0, schema_validation_passed=True,
                      bundle_receipt_sha256=bundle_id)
        output.publish(key + '.json', canonical_json(result))
        records[key] = artifact
    return records


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument('--fold', help='Registered outer fold ID')
    mode.add_argument('--final-refit', action='store_true', help='Fit on the full released development roster after G5')
    args = ap.parse_args()
    print(execute(ROOT, FINAL if args.final_refit else args.fold))


if __name__ == '__main__':
    main()
