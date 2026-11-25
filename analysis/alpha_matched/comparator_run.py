"""Separately released comparator jobs on the parent development roster and splits."""
import argparse
from datetime import datetime
import fcntl
from importlib import import_module
from importlib.metadata import version
import io
import json
from pathlib import Path
import re
from types import SimpleNamespace

import numpy as np
import pandas as pd

from . import comparator_heads as heads
from . import run_fold as core, unitcache
from .artifacts import Store, sha, space
from .nesting import Procedure, score
from .partitions import canonical_json, digest, fusion_input
from .statistics import select_inner_oof_threshold

ARMS = ('RAW', 'RELATIVE')
BASE = ('logistic', 'random_forest', 'xgboost')
DIRECT = {'species': 'SHARED', 'direct-raw': 'RAW', 'direct-relative': 'RELATIVE'}


def source_hashes():
    sources = core.source_hashes()
    for module in (Path(__file__), Path(heads.__file__)):
        sources[module.name] = core.bundle.sha(module)
    return sources


def packages():
    return {p: version(p) for p in ('numpy', 'scipy', 'scikit-learn', 'torch', 'tabm', 'xgboost', 'rtdl-num-embeddings')}


def document(base, ref):
    data = core.inside(base, ref['file']).read_bytes()
    if sha(data) != ref['sha256']:
        raise ValueError('Changed comparator registration ' + ref['file'])
    return json.loads(data)


def registry(budget):
    families = budget['candidates']
    neural = budget['neural_family']
    if neural not in (None, 'tabm') or set(families) != set(BASE) | ({neural} if neural else set()):
        raise ValueError('Unregistered or unsupported family')
    for family, choices in families.items():
        if (not isinstance(choices, list) or not choices or
                any(set(c) != {'family', 'params', 'seed'} or c['family'] != family for c in choices)
                or len(set(map(canonical_json, choices))) != len(choices)):
            raise ValueError('Invalid finite candidate budget')
    if not budget['budget_rationale'] or (neural is None and not budget['unmet_reason']):
        raise ValueError('Budget or neural feasibility decision missing')
    entries = {}
    def add(arm, representation, family):
        key = '.'.join(('cmp', arm.lower(), representation, family))
        entries[key] = dict(representation=representation, family=family, arm=arm,
                            applies_to=list(ARMS) if arm == 'SHARED' else [arm])
    qualified = budget['neural_inputs']
    if (not isinstance(qualified, list) or len(set(qualified)) != len(qualified) or not set(qualified) <= DIRECT.keys()
            or bool(qualified) != bool(neural) or set(qualified) != DIRECT.keys() and not budget.get('unmet_reason')):
        raise ValueError('Unaccounted neural representation')
    for name, arm in DIRECT.items():
        for family in families:
            if family != neural or name in qualified:
                add(arm, name, family)
    optional = budget['frozen_vectors']
    if (not isinstance(optional, list) or len(optional) != len(set(optional)) or
            not set(optional) <= {'H0', 'Ricci', 'joint'} or optional and not neural):
        raise ValueError('Invalid optional frozen-vector registration')
    for name in optional:
        for arm in ARMS:
            add(arm, name, neural)
    return entries


def release(root, fold):
    parent, parent_id = core.release(root, fold)
    status = core.active(root)
    ref = status['comparator_final_refit_release' if fold == core.FINAL else 'comparator_release']
    rec = document(root / 'reviews', ref)
    if (rec['reviewer'] != parent['reviewer'] or rec['decision'] != 'PASS' or rec['blocking_findings'] != []
            or rec['scope'] != 'development-comparators' or rec['parent_sha256'] != parent_id
            or rec['stage'] != status['stage'] or fold not in rec['folds']
            or datetime.fromisoformat(rec['timestamp']).utcoffset() is None
            or rec['source_sha256'] != source_hashes()):
        raise ValueError('Invalid comparator release or parent binding')
    budget = document(root / 'manifests', rec['budget'])
    registry(budget)
    if budget['packages'] != packages():
        raise ValueError('Comparator package versions changed')
    if budget['neural_family']:
        approval = document(root / 'reviews', rec['neural_review'])
        if (approval['decision'] != 'PASS' or approval['scope'] != 'local-operational-feasibility'
                or approval['blocking_findings'] != [] or approval['reviewer'] != parent['reviewer']
                or approval['family'] != budget['neural_family'] or approval['budget_sha256'] != rec['budget']['sha256']
                or not approval['evidence'] or datetime.fromisoformat(approval['timestamp']).utcoffset() is None):
            raise ValueError('Missing applicable local neural feasibility approval')
        if approval['selection_basis'] != 'operational' or not approval['fallback_reason']:
            raise ValueError('Missing operational fallback rationale')
        access = document(root / 'jobs/neural-feasibility', approval['preferred_family_probe'])
        if access.get('model') != 'TabPFN-3.5':
            raise ValueError('Missing preferred-family feasibility assessment')
        probes = approval['evidence']
        if set(probes) != set(budget['neural_inputs'] + budget['frozen_vectors']):
            raise ValueError('Neural approval does not cover registered inputs')
        for name, item in probes.items():
            probe = document(root / 'jobs/neural-feasibility', item)
            if (probe['family'].lower() != budget['neural_family'] or probe['version'] != budget['packages']['tabm']
                    or probe['score_inspected'] is not False or probe['replay'] != 'exact'
                    or probe['torch_threads'] != 1 or not probe['checkpoint_sha256']
                    or not all(np.isfinite(probe[k]) and probe[k] > 0 for k in
                               ('width', 'fit_rows', 'application_rows', 'wall_seconds', 'peak_rss_kib'))):
                raise ValueError('Unqualified operational feasibility receipt')
            if not 0 < budget['max_features'][name] <= probe['width']:
                raise ValueError('Registered width exceeds operational evidence')
    return parent, rec, budget, ref['sha256']


def read(store, key):
    return json.loads(store.read(key))


def references(store, value):
    """Collect core Store object references, including frozen generator state."""
    tracking = Store(store.root)
    unitcache.dependencies(tracking, value)
    return tracking.objects


class FrozenPrimary:
    """Selected parent supervision at each exact nested node, attested by joint fits."""
    def __init__(self, root, fold, parent, rec):
        category = 'development-final-refits' if fold == core.FINAL else 'development-folds'
        directory = core.inside(root / 'jobs', Path(category) / rec['parent_sha256'] / fold)
        marker = document(directory, dict(file='receipt.json', sha256=rec['primary_receipts'][fold]))
        self.store = Store(directory)
        receipt = read(self.store, marker['record'])
        if (receipt['release'], receipt['fold'], receipt['inputs']) != (rec['parent_sha256'], fold, parent['inputs']):
            raise ValueError('Frozen primary receipt differs from parent')
        self.objects = set(receipt['objects'])
        self.selected = {}
        for arm, result in receipt['arms'].items():
            for key in set(result['fits']):
                if key not in self.objects:
                    raise ValueError('Missing parent fit')
                fit = read(self.store, key)
                if fit['name'] != 'joint':
                    continue
                index = arm, fit['split']
                if fit['arm'] != arm or self.selected.setdefault(index, fit['blocks']) != fit['blocks']:
                    raise ValueError('Inconsistent standalone selection for frozen neural input')

    def blocks(self, dest, arm, name, key, train, valid):
        h0, ricci = self.selected[arm, key]
        blocks = {'H0': [h0], 'Ricci': [ricci], 'joint': [h0, ricci]}[name]
        deps = references(self.store, blocks)
        if not deps <= self.objects:
            raise ValueError('Unsigned frozen-vector dependency')
        for obj in deps - dest.objects:
            dest.publish(obj, (self.store.root / obj).read_bytes())
            dest.objects.add(obj)
        partition = digest(canonical_json(sorted(zip(train.index, train.person))))
        for block in blocks:
            core.vector_storage.verify_evidence(dest, block, train, valid, arm, objects=self.objects)
            for side, frame in [('train', train), ('validate', valid)]:
                v = dest.load_vector(block[side])
                expected_name = 'H0' if block == h0 else 'Ricci'
                if (v.partition, v.ids, v.arm, v.block, v.generator) != (
                        partition, tuple(frame.index), arm, expected_name, block['model']):
                    raise ValueError('Frozen vectors do not match comparator partition')
        return blocks


class Driver:
    def __init__(self, store, config, splits, meta, graph, species, budget, context, check, primary):
        self.store, self.config, self.budget = store, config, budget
        self.proc = Procedure(meta, splits, None, {}, [], None)
        self.check, self.primary = check, primary
        self.units = unitcache.Units(store, context)
        self.workers = {}
        for arm in ('SHARED', *ARMS):
            factory = SimpleNamespace(arm=arm, graphs=graph, references={}, tolerances=config['tolerances'])
            self.workers[arm] = core.Fold(self.proc, factory, store, config, species, check)
        self.trials = {}
        self.block_cache = {}

    def blocks(self, entry, key):
        arm, name = entry['arm'], entry['representation']
        identity = (arm, name, key)
        if identity not in self.block_cache:
            def make():
                if name in DIRECT:
                    return [self.workers[arm].build(name, {}, key)]
                return self.primary.blocks(self.store, arm, name, key, *self.proc.frames(key))
            self.block_cache[identity] = self.units.run(['representation', *identity], make)
        return self.block_cache[identity]

    def fit(self, method, entry, key, candidate):
        self.check()
        train, valid = self.proc.frames(key)
        blocks = self.blocks(entry, key)
        s = self.store
        def matrix(side):
            vectors = [s.load_vector(b[side]) for b in blocks]
            return fusion_input(*vectors) if len(vectors) == 2 else vectors[0].values
        classes = self.config['classes']
        def make():
            if sorted(train.y.unique()) != list(range(len(classes))):
                raise ValueError('Missing comparator training class')
            x = matrix('train')
            if (entry['family'] == self.budget['neural_family'] and
                    x.shape[1] > self.budget['max_features'][entry['representation']]):
                raise ValueError('Neural input exceeds approved operational width')
            model = heads.fit(s, x, train.y, train.person, candidate)
            info = dict(name=method, arm=entry['arm'], split=key, blocks=blocks, fitted=model, candidate=candidate,
                        classes=classes, endpoint=self.config['endpoint'], train_groups=sorted(set(train.group)))
            if len(valid):
                p = heads.predict(s, model, matrix('validate'))
                core.validate_probabilities(p, len(valid), classes)
                means = pd.DataFrame(p, index=valid.index).groupby(valid.person, sort=True).mean()
                info.update(samples=list(valid.index), probabilities=s.array(p), participants=list(means.index),
                            participant_means=s.array(means.to_numpy()))
            return s.record(info)
        fit = self.units.run(['fit', method, key, blocks, candidate], make)
        info = read(s, fit)
        if (info['name'], info['arm'], info['split'], info['candidate'], info['blocks'], info['classes'],
                info['train_groups']) != (method, entry['arm'], key, candidate, blocks, classes, sorted(set(train.group))):
            raise ValueError('Stale comparator fit')
        s.objects.update(heads.dependencies(s, info['fitted']))
        head = read(s, info['fitted'])
        if head['config'] != candidate or head['classes'] != list(range(len(classes))):
            raise ValueError('Comparator head or class axis differs from selected fit')
        if len(valid):
            probability = s.load_array(info['probabilities'])
            core.validate_probabilities(probability, len(valid), classes)
            means = pd.DataFrame(probability, index=valid.index).groupby(valid.person, sort=True).mean()
            if info['samples'] != list(valid.index) or info['participants'] != list(means.index):
                raise ValueError('Comparator probability roster mismatch')
            np.testing.assert_array_equal(s.load_array(info['participant_means']), means.to_numpy())
        return fit

    def select(self, method, entry, key):
        train, _ = self.proc.frames(key)
        children = self.proc.children(key, train)
        best = None
        for candidate in self.budget['candidates'][entry['family']]:
            parts, ids = [], []
            for node, _, valid in children:
                fit = self.fit(method, entry, node, candidate)
                ids.append(fit)
                parts.append((valid, self.store.load_array(read(self.store, fit)['probabilities'])))
            value = score(parts)
            self.trials.setdefault(method, []).append(self.store.record(dict(method=method, parent=key,
                                           candidate=candidate, child_fits=ids, participant_cohort_auc=value)))
            if best is None or value > best[0]:
                best = value, candidate
        return self.fit(method, entry, key, best[1])

    def procedure(self, method, entry, fold):
        classes = self.config['classes']
        train, _ = self.proc.frames(fold)
        if len(classes) == 2:
            parts, ids = [], []
            for child, a, b in self.proc.children(fold, train):
                fit = self.select(method, entry, child)
                ids.append(fit)
                p = self.store.load_array(read(self.store, fit)['probabilities'])
                parts.append((set(a.group), pd.DataFrame(p, index=b.index, columns=classes)))
            rule = dict(rule='max-participant-balanced-accuracy', threshold=select_inner_oof_threshold(
                        train, classes, parts), heldout_fits=ids)
        else:
            rule = dict(rule='argmax', threshold=None, heldout_fits=[])
        fit = self.select(method, entry, fold)
        return dict(fit=fit, decision=self.store.record(dict(**rule, classes=classes, method=method, parent=fold)),
                    trials=self.trials[method])


def report_rows(store, receipt, arm):
    """Yield one linked fit/rule per applicable method without expanding shared arrays."""
    for method, entry in receipt['registry'].items():
        if arm in entry['applies_to']:
            result = receipt['results'][method]
            yield dict(method=method, family=entry['family'], representation=entry['representation'],
                       arm=arm, fitted_arm=entry['arm'], fit=result['fit'], decision=result['decision'],
                       record=read(store, result['fit']), rule=read(store, result['decision']))


def report_fold(directory, expected_release, expected_parent, fold, marker_sha256):
    """Read the pinned fold for a reporter without fitting, applying or opening input data."""
    directory = Path(directory)
    marker = document(directory, dict(file='receipt.json', sha256=marker_sha256))
    s = Store(directory)
    receipt = read(s, marker['record'])
    reg = read(s, receipt['registration'])
    if ((receipt['release'], receipt['parent'], receipt['fold']) != (expected_release, expected_parent, fold)
            or reg['entries'] != receipt['registry'] or reg['release'] != expected_release
            or reg['parent'] != expected_parent or set(reg['entries']) != set(receipt['results'])):
        raise ValueError('Comparator reporting registry mismatch')
    for name in receipt['objects']:
        s.read(name)
    return s, receipt


def execute(root, fold):
    parent, rec, budget, rid = release(root, fold)
    if not re.fullmatch('[A-Za-z0-9_-][A-Za-z0-9_.-]*', fold):
        raise ValueError('Invalid split name')
    config, splits, meta, graph, species = core.read_inputs(root, parent)
    directory = core.inside(root / 'jobs', Path('comparator-folds') / rid / fold)
    s = Store(directory)
    def check():
        if release(root, fold)[3] != rid:
            raise ValueError('Comparator or parent release changed')
        space(directory, 0)
    with (directory / 'job.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        check()
        # Capture the loaded numerical runtimes before the first fit, also on a cold restart.
        import_module('xgboost')
        if budget['neural_family']:
            import_module(budget['neural_family'])
        environment = unitcache.environment()
        environment[3] = sorted(environment[3], key=lambda lib: lib['filepath'])
        runtime = [environment, packages()]
        s.publish('runtime.json', canonical_json(runtime))
        primary = FrozenPrimary(root, fold, parent, rec) if budget['frozen_vectors'] else None
        worker = Driver(s, config, splits, meta, graph, species, budget, [rid, fold, runtime], check, primary)
        train, valid = worker.proc.frames(fold)
        if (train.empty or valid.empty != (fold == core.FINAL) or set(train.family) & set(valid.family)
                or set(train.person) | set(valid.person) != set(meta.person)):
            raise ValueError('Invalid comparator outer partition')
        entries = registry(budget)
        registry_id = s.record(dict(entries=entries, implementation=source_hashes(), parent=rec['parent_sha256'],
                                    release=rid, packages=budget['packages'], neural_review=rec.get('neural_review'),
                                    mandatory_complete=set(budget['neural_inputs']) == DIRECT.keys(),
                                    budget=rec['budget']))
        results = {}
        for method, entry in entries.items():
            result = worker.procedure(method, entry, fold)
            if fold == core.FINAL:
                fit = read(s, result['fit'])
                gen = s.record(dict(blocks=[b['model'] for b in fit['blocks']], tolerances=config['tolerances'],
                                    core_version=sha(canonical_json(parent['source_sha256'])),
                                    training_visits=s.record(sorted(sha(canonical_json(i)) for i in train.index))))
                rule = read(s, result['decision'])
                deploy = dict(method=method, endpoint=config['endpoint'], classes=config['classes'], registry=registry_id,
                              generators=gen, head=fit['fitted'], decision=dict(rule=rule['rule'], threshold=rule['threshold']))
                result['deployment'] = s.record(deploy)
            results[method] = result
        if release(root, fold)[3] != rid:
            raise ValueError('Comparator review changed')
        body = dict(fold=fold, release=rid, parent=rec['parent_sha256'], registry=entries, registration=registry_id,
                    results=results, status='complete' if set(budget['neural_inputs']) == DIRECT.keys() else 'partial-neural-unmet',
                    inputs=parent['inputs'], objects=sorted(s.objects))
        for name in body['objects']:
            s.read(name)
        record = s.record(body)
        s.publish('receipt.json', canonical_json(dict(record=record)))
        return record


def deployment_files(store, key):
    spec = read(store, key)
    return sorted({key, spec['registry']} | references(store, spec['generators']) |
                  set(heads.dependencies(store, spec['head'])))


def load_deployments(submission, sub, items):
    """Validate the comparator registry and reuse the signed core reader for core methods."""
    core_items = [i for i in items if not i['method'].startswith('cmp.')]
    methods = core.released_models(submission, sub, core_items)
    models = {sub['files'][p]: p for p in sub['roles']['models']}
    registrations, selected = {}, {}
    for item in items:
        name, arm = item['method'], item['arm']
        if not name.startswith('cmp.'):
            continue
        path = core.clinical.resolve(submission, models[item['model_sha256']])
        s = Store(path.parent)
        spec = read(s, path.name)
        reg = read(s, spec['registry'])
        gen = read(s, spec['generators'])
        entry = reg['entries'][name]
        if (reg.get('mandatory_complete') is not True or not reg.get('neural_review')
                or path.name.split('.')[0] != item['model_sha256'] or reg['implementation'] != source_hashes()
                or reg['packages'] != packages() or spec['method'] != name or arm not in entry['applies_to']
                or spec['endpoint'] != item['endpoint'] or spec['classes'] != item['class_order']
                or spec['generators'].split('.')[0] != item['generator_sha256']
                or read(s, spec['head'])['config']['family'] != entry['family']
                or read(s, spec['head'])['classes'] != list(range(len(spec['classes'])))
                or item['requires_curvature'] != (entry['representation'] in ('Ricci', 'joint'))):
            raise ValueError('Comparator deployment disagrees with G6 plan')
        for obj in deployment_files(s, path.name):
            if sub['files'].get(str((path.parent / obj).relative_to(submission))) != obj.split('.')[0]:
                raise ValueError('Comparator dependency outside reviewed submission')
        if registrations.setdefault(spec['endpoint'], spec['registry']) != spec['registry']:
            raise ValueError('Mixed comparator registrations')
        if selected.setdefault(name, path.name) != path.name:
            raise ValueError('Species aliases must use the identical fitted procedure')
        rep = entry['representation']
        if rep in DIRECT:
            if len(gen['blocks']) != 1:
                raise ValueError('Invalid direct representation')
            block = read(s, gen['blocks'][0])
            if (block['name'], block['arm']) != (rep, entry['arm']):
                raise ValueError('Incorrect comparator generator')
        elif methods[arm, rep][2]['blocks'] != gen['blocks']:
            raise ValueError('Optional neural model changed primary supervision')
        methods[arm, name] = s, spec, gen
    if not registrations:
        raise ValueError('G6 plan contains no comparator registry')
    expected = {(a, n) for n, e in reg['entries'].items() for a in e['applies_to']}
    if expected != {(i['arm'], i['method']) for i in items if i['method'].startswith('cmp.')}:
        raise ValueError('G6 plan omits a registered comparator or species alias')
    return methods


def apply_final(root, submission, panel, cohort, endpoint, prediction_root):
    """Mixed-plan evaluator dispatch. Emit seal-compatible rows for core and comparator items."""
    submission = Path(submission).resolve(strict=True)
    sid, sub, evaluator = core.application_release(root, submission)
    panel, prediction_root = (Path(p).resolve(strict=True) for p in (panel, prediction_root))
    if any(p.is_relative_to(prediction_root) or prediction_root.is_relative_to(p) for p in (submission, panel)):
        raise ValueError('Prediction workspace overlaps frozen inputs')
    items = [i for i in sub['planned_predictions'] if (i['cohort'], i['endpoint']) == (cohort, endpoint)]
    if any(not re.fullmatch(r'[A-Za-z0-9_-][A-Za-z0-9_.-]*', i['id']) for i in items):
        raise ValueError('Unsafe prediction ID')
    methods = load_deployments(submission, sub, items)
    roster_file = prediction_root / 'graph_roster.json'
    registry_bytes = roster_file.read_bytes()
    roster = json.loads(registry_bytes)
    bundle_id = core.bundle.sha(panel / 'receipt.json')
    if (roster['submission_sha256'] != sid or roster['evaluator'] != evaluator
            or roster['bundle_receipts'][cohort] != bundle_id):
        raise ValueError('Incorrect evaluator bundle')
    native = json.loads((panel / 'receipt.json').read_bytes())
    if set(native['outputs']) != core.BUNDLE_FILES or type(native.get('species_nondetection_zero', False)) is not bool:
        raise ValueError('Invalid application bundle')
    for name in core.BUNDLE_FILES:
        core.inside(panel, name)
    eligible = core.clinical.resolve(submission, sub['roles']['test_eligibility'][0])
    ids = json.loads(eligible.read_bytes())['rosters'][cohort][endpoint]
    hashed = {sha(canonical_json(i)) for i in ids}
    for s, _, gen in methods.values():
        if hashed.intersection(read(s, gen['training_visits'])):
            raise ValueError('Training visit in evaluation')
    graph = core.bundle.load(panel)
    if (not set(ids) <= set(graph.sample_ids) or core.bundle.sha(panel / 'receipt.json') != bundle_id
            or any(core.bundle.sha(panel / n) != h for n, h in native['outputs'].items())):
        raise ValueError('Changed application bundle or visit roster')
    out = Store(prediction_root)
    streams = {i['id']: io.BytesIO() for i in items}
    edge_total = sum(graph.raw.getrow(graph.lookup[i]).nnz for i in ids)
    for sample in ids:
        core.active(root)
        expected = roster['cohorts'][cohort][sample]
        vectors, curves, probabilities = {}, {}, {}
        for item in items:
            arm, method = item['arm'], item['method']
            s, spec, gen = methods[arm, method]
            comparator = method.startswith('cmp.')
            if (expected['sample'], expected['graph_rules_sha256']) != (sample, item['graph_rules_sha256']):
                raise ValueError('Evaluator graph identity mismatch')
            blocks = []
            for key in gen['blocks']:
                block = read(s, key)
                if key not in vectors:
                    def proof(sample_id, pairs, values):
                        space(prediction_root, 65536 + 64 * len(pairs))
                        geom = core.curvature_artifact(out, pairs, values)
                        if sample_id != sample or geom['active_pairs_sha256'] != expected['active_pairs_sha256']:
                            raise ValueError('Active-edge roster mismatch')
                        curves[key] = geom
                    factory = SimpleNamespace(graphs=graph, arm=block.get('arm', arm), tolerances=gen['tolerances'],
                                              cache_dir=prediction_root / 'geometry',
                                              version=gen['core_version'] if comparator else gen['implementation'])
                    vectors[key] = s.transform(key, factory, (sample,), graph.species_ids, curvature=proof)
                blocks.append(vectors[key])
            identity = item['model_sha256']
            if identity not in probabilities:
                x = blocks[0].values if len(blocks) == 1 else fusion_input(*blocks)
                p = heads.predict(s, spec['head'], x) if comparator else s.probabilities(method, spec['head'], x)
                core.validate_probabilities(p, 1, item['class_order'])
                probabilities[identity] = p[0].tolist()
            row = dict(sample=sample, probabilities=probabilities[identity], submission_sha256=sid,
                       prediction_id=item['id'], class_order=item['class_order'], model_sha256=identity,
                       generator_sha256=item['generator_sha256'])
            if item['requires_curvature']:
                row['geometry'] = dict(**curves[gen['blocks'][-1]], sample=sample,
                    profile_sha256=expected['profile_sha256'], graph_rules_sha256=item['graph_rules_sha256'],
                    prediction_id=item['id'], generator_sha256=item['generator_sha256'])
            streams[item['id']].write(canonical_json(row) + b'\n')
    if core.application_release(root, submission)[0] != sid or roster_file.read_bytes() != registry_bytes:
        raise ValueError('G6 release or evaluator roster changed')
    outputs = {}
    for item in items:
        core.active(root)
        key = item['id']
        artifact = out.put(streams[key].getvalue(), '.jsonl')
        count = core.clinical.verify_prediction(prediction_root, prediction_root / artifact, item, sid, roster['cohorts'])
        if count != (edge_total if item['requires_curvature'] else 0):
            raise ValueError('Invalid curvature coverage')
        out.publish(key + '.json', canonical_json(dict(submission_sha256=sid, prediction_id=key,
                    eligible_row_hash=item['eligible_row_hash'], row_count=len(ids), artifact=artifact,
                    sha256=artifact.split('.')[0], expected_edges=count, computed_finite_edges=count,
                    failed_edges=0, schema_validation_passed=True, bundle_receipt_sha256=bundle_id)))
        outputs[key] = artifact
    return outputs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--fold')
    mode.add_argument('--final-refit', action='store_true')
    args = parser.parse_args()
    print(execute(core.ROOT, core.FINAL if args.final_refit else args.fold))
