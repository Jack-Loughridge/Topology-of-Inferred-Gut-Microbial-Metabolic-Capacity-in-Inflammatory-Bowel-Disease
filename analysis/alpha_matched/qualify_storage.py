"""Bounded storage-only comparison. Invented panels and previously qualified stress NPYs."""
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import time
from unittest.mock import patch

import numpy as np

from .artifacts import Store, space
from .cache import CurvatureCache, FIELDS, encode_payload, payload_name, record_sample
from .partitions import FrozenVectors, canonical_json, fitted_location_scale, fusion_input
from .run_fold import active

CHECKOUT = Path(__file__).resolve().parents[2]
RUN = CHECKOUT.parents[1]
WIDTH = 13525
ERRORS = dict.fromkeys(('first_hit_error', 'transport_error', 'marginal_error', 'curvature_error'), 0.)


def sizes(directory):
    result = dict(files=0, directories=0, regular_bytes=0, allocated=0, physical_files=0)
    seen = set()
    for path in [directory, *directory.rglob('*')]:
        st = path.stat()
        if path.is_dir():
            result['directories'] += 1
        else:
            result['files'] += 1
            result['regular_bytes'] += st.st_size
        if st.st_ino not in seen:
            result['allocated'] += st.st_blocks * 512
            result['physical_files'] += int(path.is_file())
            seen.add(st.st_ino)
    return result


def compare_stress(tmp):
    probe = json.loads((RUN / 'jobs/g4-lossless-storage-probe.json').read_bytes())
    arms = {r['array_sha256']: r['arm'] for r in probe['rows']}
    summary = {}
    for path in sorted((RUN / 'jobs/panel-numerics').glob('*/*.npy')):
        active(RUN)
        raw = path.read_bytes()
        arm = arms[hashlib.sha256(raw).hexdigest()]
        arr = np.load(io.BytesIO(raw), allow_pickle=False)
        stats = summary.setdefault(arm, dict(count=0, nbytes=0, old_file=0, new_file=0, old_disk=0, new_disk=0,
                                            encode_ms=0., decode_ms=0.))
        start = time.process_time_ns()
        rec, encoded = encode_payload(arr, os.statvfs(tmp).f_frsize)
        elapsed = time.process_time_ns() - start
        if rec['sha256'] != hashlib.sha256(raw).hexdigest():
            raise AssertionError('Original NPY mismatch')
        for label, data in [('old', raw), ('new', encoded)]:
            dest = tmp / (label + '-stress')
            dest.write_bytes(data)
            stats[label + '_file'] += len(data)
            stats[label + '_disk'] += dest.stat().st_blocks * 512
        import zlib
        start = time.process_time_ns()
        restored = zlib.decompress(encoded) if 'zlib_sha256' in rec else encoded
        stats['decode_ms'] += (time.process_time_ns() - start) / 1e6
        if restored != raw:
            raise AssertionError('Restore mismatch')
        stats['count'] += 1
        stats['nbytes'] += arr.nbytes
        stats['encode_ms'] += elapsed / 1e6
    return summary


def fabricated(tmp):
    geom = tmp / 'geometry'
    store = Store(tmp / 'store')
    rng = np.random.default_rng(60031)
    # Heterogeneous per-edge activity frequencies, averaging about 71%.
    masks = rng.random((32, WIDTH)) < rng.beta(2.5, 1., WIDTH)
    catalogue = np.column_stack((np.arange(WIDTH), np.arange(1, WIDTH + 1))).astype('<i8')
    names = tuple(f'{kind}:{a}>{b}' for kind in ('K0', 'B') for a, b in catalogue)
    summaries = []
    totals = dict(stored_matrices=0, matrix_payload=0, logical_active_values=0, primitive_active_values=0)
    for node, n, train in [('outer', 32, 24), ('inner', 24, 16)]:
        for arm in ('RAW', 'RELATIVE'):
            for setting in range(2):
                active(RUN)
                # Bound temporary fixture writes without relying on compression.
                space(tmp, 40 * 1024**2)
                raw = np.zeros((n, 2 * WIDTH))
                for i in range(n):
                    edge = np.flatnonzero(masks[i])
                    pairs = catalogue[edge]
                    seed = 600031 + 200 * setting + i + (500 * n if arm == 'RELATIVE' else 0)
                    r = np.random.default_rng(seed)
                    vals = r.normal(-.2, .5, len(edge))
                    vals[::3] = np.round(vals[::3], 1)
                    vals[::17] = -0.
                    key = dict.fromkeys(FIELDS, 'fabricated')
                    key.update(input_digest=str(i), active_pairs_digest=hashlib.sha256(pairs.tobytes()).hexdigest(),
                               arm=arm, transition=setting, training_people_digest=node, tolerance=ERRORS,
                               reference_digest='identity' if arm == 'RAW' else node)
                    shard = CurvatureCache(geom, key)
                    primitive = shard
                    if arm == 'RAW':
                        primitive = CurvatureCache(geom / '.raw-numerics', dict(key, training_people_digest='fixed-RAW-numerical-operator'))
                    if primitive.read(pairs) is None:
                        primitive.save(vals, pairs, ERRORS)
                        totals['primitive_active_values'] += len(vals)
                    if primitive is not shard:
                        shard.bind_raw_calculation(primitive, pairs)
                    restored = shard.read(pairs)
                    if restored.tobytes() != vals.tobytes():
                        raise AssertionError('Lost edge values')
                    record_sample(shard, f'v{i}', pairs, geom)
                    totals['logical_active_values'] += len(vals)
                    raw[i, edge] = restored
                    raw[i, WIDTH + edge] = 1
                mean, scale = fitted_location_scale(raw[:train], np.ones(train))
                for value in (mean, scale):
                    store.array(value)
                for side in (slice(0, train), slice(train, n)):
                    ids = tuple(f'v{i}' for i in range(n)[side])
                    full = (raw[side] - mean) / scale
                    for label, x, coords in [('Ricci', full, names), ('K0', full[:, :WIDTH], names[:WIDTH])]:
                        if label == 'K0' and setting:
                            continue
                        v = FrozenVectors(x, ids, coords, arm, node, str(setting), label)
                        restored = store.load_vector(store.vector(v))
                        if restored.identity() != v.identity():
                            raise AssertionError('Canonical vector mismatch')
                        if label == 'Ricci':
                            h0 = FrozenVectors(np.zeros((len(ids), 1)), ids, ('H0',), arm, node, 'h', 'H0')
                            if fusion_input(h0, restored)[:, 1:].tobytes() != x.tobytes():
                                raise AssertionError('Changed fusion bytes')
                        summaries.append(v.identity())
                        totals['stored_matrices'] += 1
                        totals['matrix_payload'] += x.nbytes
    return dict(geometry=sizes(geom), canonical_and_scalers=sizes(store.root), totals=totals,
                canonical_digest=hashlib.sha256(canonical_json(summaries)).hexdigest())


def main():
    active(RUN)
    print('Filesystem allocation unit:', os.statvfs(CHECKOUT).f_frsize)
    with tempfile.TemporaryDirectory(dir=CHECKOUT) as folder:
        print('Verified stress arrays:', json.dumps(compare_stress(Path(folder)), sort_keys=True), flush=True)
    for label, encoder in [('original', lambda arr, unit: encode_payload(arr, arr.nbytes + 4096)),
                           ('adaptive', encode_payload)]:
        with tempfile.TemporaryDirectory(dir=CHECKOUT) as folder, patch(
                'analysis.alpha_matched.cache.encode_payload', encoder):
            start = time.perf_counter()
            result = fabricated(Path(folder))
            result['wall_seconds'] = round(time.perf_counter() - start, 3)
            print(label, json.dumps(result, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
