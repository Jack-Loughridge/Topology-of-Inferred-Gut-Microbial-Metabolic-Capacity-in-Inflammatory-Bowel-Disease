"""One bounded, synthetic vector-storage measurement. Run from the repair checkout."""
import io
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from analysis.alpha_matched import unitcache
from analysis.alpha_matched.artifacts import Store, sha, space
from analysis.alpha_matched.cache import CurvatureCache, record_sample
from analysis.alpha_matched.comparator_run import FrozenPrimary
from analysis.alpha_matched.features import Factory, Graphs
from analysis.alpha_matched.nesting import Procedure
from analysis.alpha_matched.partitions import FrozenVectors, array_id, canonical_json, digest, fusion_input
from analysis.alpha_matched.run_fold import Fold, active
from qualification.vector_recipe import RecipeStore, block_expectation, copy_objects, ordinary_closure, record

BASE = Path(__file__).resolve().parents[1]
RUN = BASE.parents[1]
WIDTH = 13525
SETTINGS = [dict(family='synthetic', parameter=1., rho=x) for x in (0.3, 0.6)]
TOL = dict.fromkeys(('first_hit_error', 'transport_error', 'marginal_error', 'curvature_error'), 1e-8)
ERROR = dict.fromkeys(TOL, 0.0)


def check():
    active(RUN)
    space(BASE, 2**20)


def usage(*roots):
    inodes = {}
    for root in roots:
        for path in [root, *root.rglob('*')]:
            st = path.stat()
            inodes[st.st_dev, st.st_ino] = st.st_blocks * 512
    return sum(inodes.values())


def panel(count, stress=()):
    rng = np.random.default_rng(93681)
    pairs = np.array([(a, b) for a in range(120) for b in range(120) if a != b][:WIDTH], dtype='<i8')
    edges = []
    for i in range(count):
        n = len(stress[i]) if stress else (0 if i == 0 else 6 if i == 1 else int(WIDTH * .71))
        e = np.sort(rng.choice(WIDTH, n, replace=False)).astype('i')
        edges.append(e[::-1] if i % 3 == 1 else e)
    indices = np.concatenate(edges)
    support = csr_matrix((rng.uniform(.01, 1., len(indices)), indices,
                          np.cumsum([0] + list(map(len, edges)))), shape=(count, WIDTH))
    ids = tuple(f'invented-{i:02}' for i in range(count))
    meta = pd.DataFrame(dict(person=[f'person-{i//4}-{max(i%4-1, 0)}' for i in range(count)]), index=ids)
    graph = Graphs(support, pairs, tuple(np.arange(120) for _ in ids), ids, csr_matrix((count, 120)),
                   sha(b'invented-catalogue'), sha(b'invented-mapping'))
    return graph, meta


def seed(factory, train, samples, setting, seed_index, stress):
    """Seed authorized fabricated geometry, or verified numerical values with invented positions."""
    g = factory.graphs
    part = digest(canonical_json(sorted(zip(train.index, train.person))))
    ref = None
    if factory.arm == 'RELATIVE':
        from analysis.alpha_matched.partitions import RelativeReference
        rows = [g.lookup[s] for s in train.index]
        ref = RelativeReference.fit(g.raw[rows], train.person, train.index, g.axis)
        factory.references[part] = ref
    for sample in samples:
        i = g.lookup[sample]
        edge, _, vertices, pairs = g.graph(i, ref)
        inp = digest(array_id(vertices).encode(), array_id(edge).encode(), array_id(g.raw.getrow(i).data).encode())
        identity = dict(input_digest=inp, catalogue_digest=g.catalogue_digest, mapping_digest=g.mapping_digest,
                        arm=factory.arm, reference_digest=array_id(ref.log_reference) if ref else 'identity',
                        training_people_digest=part, graph_rules='bound-aware-nonself-v1',
                        transition=[setting['family'], setting['parameter']], rho=setting['rho'], epsilon=1e-4,
                        endpoint_measure='degree-plus-one', feature_definition='full-pairwise-K0+B-catalogue-axis',
                        scaler_digest='raw-before-classifier-scaling', solver='deterministic-certified',
                        tolerance=TOL, implementation_digest=factory.version,
                        active_pairs_digest=sha(pairs.astype('<i8').tobytes()))
        shard = CurvatureCache(factory.cache_dir, identity)
        geom = shard
        if not ref:
            geom = CurvatureCache(factory.cache_dir / '.raw-numerics',
                                  dict(identity, training_people_digest='fixed-RAW-numerical-operator'))
        if stress:
            values = stress[i]
        else:
            rng = np.random.default_rng([i, seed_index, int(part[:8], 16) if ref else 0])
            values = rng.normal(size=len(edge))
            values[::3] = np.round(values[::3], 1)
            values[:4] = [0., -0., -2., 1e-100][:min(4, len(values))]
        geom.save(values, pairs, ERROR)
        if geom is not shard:
            shard.bind_raw_calculation(geom, pairs)
        record_sample(shard, sample, g.pairs[edge], factory.cache_dir)


def expect_error(fn, kinds):
    try:
        fn()
    except kinds:
        return
    raise AssertionError('Expected rejection')


def equal(left, right):
    assert left.identity() == right.identity()
    assert left.values.dtype == right.values.dtype == np.float64
    assert left.values.shape == right.values.shape and left.values.tobytes() == right.values.tobytes()


def verified_stress():
    probe = json.loads((RUN / 'jobs/g4-lossless-storage-probe.json').read_bytes())
    approved = {r['array_sha256']: r for r in probe['rows']}
    rows = {'RAW': [], 'RELATIVE': []}
    for path in sorted((RUN / 'jobs/panel-numerics').glob('*/*.npy')):
        data = path.read_bytes()
        key = sha(data)
        if key not in approved:
            continue
        arr = np.load(io.BytesIO(data), allow_pickle=False)
        assert arr.dtype == np.float64 and arr.ndim == 1 and arr.nbytes == approved[key]['uncompressed_payload']
        rows[approved[key]['arm']].append(arr)
    assert [len(r) for r in rows.values()] == [8, 8]
    return rows


def exercise(root, stress=None):
    check()
    dense = Store(root / 'dense')
    compact = RecipeStore(root / 'compact')
    portable = RecipeStore(root / 'portable')
    bridge = Store(root / 'bridge')
    comparator = Store(root / 'comparator')
    geometries = root / 'geometry'
    expected = []
    contracts = []
    saved = []
    selectors = []
    scalers = set()
    indices = set()
    matrix_files = set()
    frozen_scalers = {}
    physical_inputs = set()
    large_recipes = set()
    sample_occurrences = 0
    for arm in ('RAW', 'RELATIVE'):
        inputs = stress[arm] if stress else []
        graph, meta = panel(len(inputs) if inputs else 20, inputs)
        fac = Factory(graph, arm, geometries / arm, TOL)
        proc = Procedure(meta, {}, fac, {}, [], None)
        fold = Fold(proc, fac, dense, {'heads': []}, (), check)
        splits = [('outer', 16, 20), ('inner', 12, 16), ('inner/child', 8, 12),
                  ('inner/child/leaf', 4, 8), ('final_refit', 20, 20)]
        if stress:
            splits = [('outer', 4, 8), ('final_refit', 8, 8)]
        for label, stop, end in splits:
            check()
            train, valid = meta.iloc[:stop], meta.iloc[stop:end]
            for setting_i, setting in enumerate(SETTINGS[:1] if stress else SETTINGS):
                seed(fac, train, meta.index[:end], setting, setting_i, inputs)
                gen, t, v = proc.build('Ricci', tuple(sorted(setting.items())), train, valid)
                state = dict(block='Ricci', mean=dense.array(gen.mean), scale=dense.array(gen.scale),
                             image=None, setting=setting, generator=gen.generator, tolerances=TOL)
                scalers.update([state['mean'], state['scale']])
                frozen_scalers[arm, label, setting_i] = state['mean']
                model = dense.record(dict(state=state, name='Ricci', arm=arm, partition=gen.partition,
                                          reference=dense.reference(gen.reference), catalogue=graph.catalogue_digest,
                                          mapping=graph.mapping_digest, features=dense.record(gen.names),
                                          axis=fold.axis, training_people_sha256=sha(canonical_json(sorted(set(train.person))))))
                copy_objects(dense, compact, ordinary_closure(dense, model))
                old = dict(model=model)
                new = dict(model=model)
                if not stress and stop >= 8:
                    assert (t.values[0, WIDTH:] != 0.).any()  # Invented visit zero has no active edges.
                for side, vec in [('train', t), ('validate', v)]:
                    frozen = FrozenVectors(vec.values, vec.ids, vec.features, arm, gen.partition, model, 'Ricci')
                    old[side] = dense.vector(frozen)
                    new[side] = compact.capture(frozen, fac)
                    equal(RecipeStore(compact.root).load_vector(new[side]), frozen)
                    large_recipes.add(new[side])
                    sample_occurrences += len(vec.ids)
                    for r in record(compact, new[side])['rows']:
                        indices.add(r['indices'])
                        physical_inputs.add(r['payload'])
                expected.append(old)
                saved.append(new)
                contracts.append(block_expectation(dense, old, train, valid, arm, fac))
                if setting_i == 0:
                    k0 = fold.curvature_block(old)
                    copy_objects(dense, compact, ordinary_closure(dense, k0['model']))
                    view = dict(model=k0['model'])
                    hmodel = dense.record(dict(name='H0', partition=gen.partition, arm=arm))
                    h0 = dict(model=hmodel)
                    for side in ('train', 'validate'):
                        baseline = dense.load_vector(k0[side])
                        view[side] = compact.project(new[side], baseline)
                        dim = len(baseline.ids)
                        h = FrozenVectors(np.arange(dim * 3, dtype=float).reshape(dim, 3), baseline.ids,
                                          ('PI:0', 'PI:1', 'PI:2'), arm, gen.partition, hmodel, 'H0')
                        h0[side] = dense.vector(h)
                        assert array_id(fusion_input(h, dense.load_vector(old[side]))) == array_id(
                            fusion_input(h, compact.load_vector(new[side])))
                    copy_objects(dense, compact, ordinary_closure(dense, h0))
                    saved.extend([view, h0])
                    contracts.extend(block_expectation(dense, b, train, valid, arm, fac) for b in (k0, h0))
                    expected.extend([k0, h0])
                    selectors.append((arm, label, train, valid, h0, old))
    assert frozen_scalers['RAW', 'outer', 0] != frozen_scalers['RAW', 'final_refit', 0]
    for block in expected:
        if record(dense, block['model'])['name'] in ('Ricci', 'K0'):
            matrix_files.update(record(dense, block[side])['values'] for side in ('train', 'validate'))
    # The compact reader must not accept existing dense matrices as an implicit fallback.
    assert not (matrix_files & compact.objects) or all(
        dense.load_array(name).size == 0 for name in matrix_files & compact.objects)
    def allocated(store, names):
        return sum((store.root / name).stat().st_blocks * 512 for name in names)
    geometry = usage(geometries)
    local = usage(compact.root, geometries)
    metrics = dict(geometry=geometry, baseline_total=usage(dense.root, geometries), shared_total=local,
                   dense_store=usage(dense.root), compact_standalone=usage(compact.root),
                   compact_increment=local - geometry, geometry_payload=allocated(compact, physical_inputs),
                   unique_payloads=len(physical_inputs), global_indices=allocated(compact, indices),
                   unique_indices=len(indices), scalers=allocated(compact, scalers), unique_scalers=len(scalers),
                   dense_matrices=allocated(dense, matrix_files), unique_matrices=len(matrix_files),
                   blocks=len(saved), train_application_vectors=2 * len(saved),
                   ricci_recipes=allocated(compact, large_recipes), sample_occurrences=sample_occurrences)
    for name in physical_inputs:
        assert (compact.root / name).stat().st_nlink >= 2
    # Simulate losing the copying process before the closure is complete.
    publish = portable.publish
    calls = 0
    def interrupt(name, data):
        nonlocal calls
        calls += 1
        if calls == 8:
            raise InterruptedError
        return publish(name, data)
    with patch.object(portable, 'publish', interrupt):
        expect_error(lambda: compact.export(portable, saved, contracts), InterruptedError)
    assert not (portable.root / 'complete.json').exists()
    portable = RecipeStore(portable.root)
    names = compact.export(portable, saved, contracts)
    assert all((compact.root / n).stat().st_ino != (portable.root / n).stat().st_ino for n in physical_inputs)
    assert RecipeStore(portable.root).restart(contracts) == saved
    # Check read-time loss and corruption only in the disposable portable copy.
    first = next(b for b in saved if record(compact, b['model'])['name'] == 'Ricci')
    source = record(compact, first['train'])
    row = next(r for r in source['rows'] if len(compact.load_array(r['indices'])) > 1)
    missing = portable.root / row['payload']
    missing.unlink()
    expect_error(lambda: portable.load_vector(first['train']), FileNotFoundError)
    copy_objects(compact, RecipeStore(portable.root), {missing.name})
    bad = portable.root / row['indices']
    original = bad.read_bytes()
    bad.chmod(0o600)
    bad.write_bytes(original[:-1])
    expect_error(lambda: portable.load_vector(first['train']), ValueError)
    bad.unlink()
    copy_objects(compact, RecipeStore(portable.root), {bad.name})
    # Even a well-formed object with shuffled indices must fail its attestation.
    row['indices'] = portable.array(portable.load_array(row['indices'])[::-1])
    expect_error(lambda: portable.restore(source), ValueError)
    # Ordinary comparator blocks must work with no recipe-aware production caller.
    for want, compact_block, contract in zip(expected, saved, contracts):
        assert portable.dense_block(bridge, compact_block, contract) == want
    primary = FrozenPrimary.__new__(FrozenPrimary)
    primary.store, primary.objects = bridge, bridge.objects
    primary.selected = {(arm, key): [h, r] for arm, key, _, _, h, r in selectors}
    for arm, key, tr, va, h, r in selectors:
        for rep in ('Ricci', 'joint', 'H0'):
            primary.blocks(comparator, arm, rep, key, tr, va)
        for side in ('train', 'validate'):
            assert array_id(fusion_input(comparator.load_vector(h[side]), comparator.load_vector(r[side]))) == array_id(
                fusion_input(dense.load_vector(h[side]), dense.load_vector(r[side])))
    metrics.update(portable_objects=len(names), portable_payload=allocated(portable, names),
                   portable_directory_and_marker=(portable.root.stat().st_blocks +
                                                  (portable.root / 'complete.json').stat().st_blocks) * 512,
                   dense_bridge=usage(bridge.root), comparator_selected=usage(comparator.root))
    # Deny opening anything outside the exported Store during each cold vector read.
    cold = RecipeStore(portable.root)
    original_open = io.open
    def portable_only(path, *args, **kwargs):
        if Path(path).parent != portable.root:
            raise AssertionError('Replay accessed external dependency')
        return original_open(path, *args, **kwargs)
    digests = []
    for original, recipe in zip(expected, saved):
        for side in ('train', 'validate'):
            baseline = dense.load_vector(original[side])
            with patch('io.open', portable_only):
                vec = cold.load_vector(recipe[side])
            equal(vec, baseline)
            digests.append(vec.identity())
    metrics['identity_digest'] = sha(canonical_json(digests))
    return metrics


if __name__ == '__main__':
    check()
    with patch('analysis.alpha_matched.features.compute', side_effect=AssertionError('Unexpected solve')):
        for label, inputs in [('fabricated', None), ('stress', verified_stress())]:
            with tempfile.TemporaryDirectory(dir=BASE, prefix='.vector-') as tmp:
                print(json.dumps({label: exercise(Path(tmp), inputs)}, sort_keys=True), flush=True)
