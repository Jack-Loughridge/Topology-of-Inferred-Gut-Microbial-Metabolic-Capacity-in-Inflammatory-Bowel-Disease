"""Validated per-shard numerical cache. Scientific identity is supplied by the caller."""
from contextlib import contextmanager
from pathlib import Path
import fcntl
import hashlib
import io
import json
import os
import tempfile
import zlib

import numpy as np

from .artifacts import space
from .partitions import canonical_json

FIELDS = frozenset("""
input_digest catalogue_digest mapping_digest arm reference_digest training_people_digest
graph_rules transition rho epsilon endpoint_measure feature_definition scaler_digest
solver tolerance implementation_digest active_pairs_digest
""".split())


def payload_name(record):
    """The logical checksum always identifies the complete original NPY."""
    if 'zlib_sha256' in record:
        return record['zlib_sha256'] + '.npyz'
    return record['sha256'] + '.npy'


def encode_payload(arr, allocation_unit):
    stream = io.BytesIO()
    np.save(stream, arr, allow_pickle=False)
    raw = stream.getvalue()
    record = {'sha256': hashlib.sha256(raw).hexdigest()}
    # Saving bytes inside the same allocation unit saves no physical blocks.
    if len(raw) > allocation_unit:
        compressed = zlib.compress(raw)
        if (len(compressed) - 1) // allocation_unit < (len(raw) - 1) // allocation_unit:
            record['zlib_sha256'] = hashlib.sha256(compressed).hexdigest()
            return record, compressed
    return record, raw


class CurvatureCache:
    def __init__(self, directory, identity):
        if set(identity) != FIELDS or any(v is None for v in identity.values()):
            raise ValueError("Incomplete or unknown scientific cache key")
        self.identity = canonical_json(identity)
        self.folder = Path(directory)/hashlib.sha256(self.identity).hexdigest()
        self.folder.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def locked(self):
        with (self.folder/'lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            yield

    def read(self, active_pairs):
        receipt = self.folder/'receipt.json'
        if not receipt.exists():
            return None
        envelope = json.loads(receipt.read_bytes())
        info = envelope['record']
        if hashlib.sha256(canonical_json(info)).hexdigest() != envelope['digest']:
            raise ValueError('Corrupt scientific receipt')
        self.validate_certificate(info['certificate'])
        if info['key'] != json.loads(self.identity):
            raise ValueError('Unexpected cached configuration')
        path = self.folder/payload_name(info)
        content = path.read_bytes()
        if 'zlib_sha256' in info:
            if hashlib.sha256(content).hexdigest() != info['zlib_sha256']:
                raise ValueError('Compressed curvature cache failed digest verification')
            content = zlib.decompress(content)
        if hashlib.sha256(content).hexdigest() != info['sha256']:
            raise ValueError('Curvature cache failed digest verification')
        arr = np.load(io.BytesIO(content),allow_pickle=False)
        self.validate(arr,active_pairs)
        if info['active_edges'] != len(arr) or info['finite_edges'] != len(arr) or info['failed'] != 0:
            raise ValueError('Invalid completeness receipt')
        return arr

    def bind_raw_calculation(self, primitive, active_pairs):
        """Attest byte-identical RAW curvature under this fitted partition.

        The shared calculation accepts a numerical kernel and biological graph.
        It has no access to training labels or to a fitted reference.
        """
        target_key=json.loads(self.identity)
        primitive_key=json.loads(primitive.identity)
        if (target_key['arm']!='RAW' or target_key['reference_digest']!='identity'
                or primitive_key!={**target_key,'training_people_digest':'fixed-RAW-numerical-operator'}):
            raise ValueError('Ineligible curvature reuse between configurations')
        values=primitive.read(active_pairs)
        if values is None:
            raise ValueError('Missing qualified numerical calculation')
        with self.locked():
            present=self.read(active_pairs)
            if present is None:
                evidence=json.loads((primitive.folder/'receipt.json').read_bytes())['record']
                name=payload_name(evidence)
                dest=self.folder/name
                if dest.exists():
                    if hashlib.sha256(dest.read_bytes()).hexdigest()!=evidence.get('zlib_sha256',evidence['sha256']):
                        raise ValueError('Conflicting RAW numerical object')
                else:
                    os.link(primitive.folder/name,dest)
                record={**evidence,'key':target_key,'raw_calculation':primitive.folder.name}
                tmp=self.folder/'pending.json'
                tmp.write_bytes(canonical_json({'record':record,'digest':hashlib.sha256(canonical_json(record)).hexdigest()}))
                os.replace(tmp,self.folder/'receipt.json')
            elif present.dtype != values.dtype or present.tobytes() != values.tobytes():
                raise ValueError('Partition RAW value differs from numerical primitive')
        return values

    def validate(self, arr, active_pairs):
        pairs = np.asarray(active_pairs,dtype='<i8').reshape(-1,2)
        if hashlib.sha256(pairs.tobytes()).hexdigest() != json.loads(self.identity)['active_pairs_digest']:
            raise ValueError('Active pair identity mismatch')
        if len(set(map(tuple,pairs))) != len(pairs) or np.any(pairs[:,0]==pairs[:,1]):
            raise ValueError('Invalid active pair roster')
        if arr.dtype.kind != 'f' or arr.shape != (len(pairs),) or not np.isfinite(arr).all():
            raise ValueError('Incomplete curvature output')

    def validate_certificate(self, certificate):
        for name in ('first_hit_error','transport_error','marginal_error','curvature_error'):
            bound = certificate[name]
            threshold = json.loads(self.identity)['tolerance'][name]
            if not np.isfinite([bound,threshold]).all() or not 0 <= bound <= threshold:
                raise ValueError('Accuracy check failed for '+name)

    def save(self, arr, active_pairs, certificate):
        arr = np.asarray(arr)
        self.validate(arr,active_pairs)
        self.validate_certificate(certificate)
        with self.locked():
            prior = self.read(active_pairs)
            if prior is not None:
                if prior.dtype != arr.dtype or prior.tobytes() != arr.tobytes():
                    raise ValueError('Conflicting results for same cache identity')
                return
            checksums, payload = encode_payload(arr, os.statvfs(self.folder).f_frsize)
            space(self.folder, len(payload))
            # Publish payload first. A crash here can leave an unreferenced object,
            # never a receipt pointing at a partially written array.
            with tempfile.NamedTemporaryFile(dir=self.folder) as tmp:
                tmp.write(payload)
                tmp.flush()
                os.fsync(tmp.fileno())
                path = self.folder/payload_name(checksums)
                try:
                    os.link(tmp.name, path)
                except FileExistsError:
                    if path.read_bytes() != payload:
                        raise ValueError('Conflicting curvature object on resume')
            record = dict(key=json.loads(self.identity), **checksums, active_edges=len(arr),
                          finite_edges=len(arr), failed=0, certificate=certificate)
            marker = self.folder/'pending.json'
            marker.write_bytes(canonical_json({'record':record,'digest':hashlib.sha256(canonical_json(record)).hexdigest()}))
            os.replace(marker,self.folder/'receipt.json')


def record_sample(cache, sample_id, global_active_pairs, certificates_dir):
    """Map every completed sample/configuration to its numerical receipt."""
    import sqlite3
    record=json.loads((cache.folder/'receipt.json').read_bytes())
    if hashlib.sha256(canonical_json(record['record'])).hexdigest()!=record['digest']:
        raise ValueError('Cannot attest a corrupt curvature receipt')
    validated=record['record']
    roster=np.asarray(global_active_pairs,dtype='<i8').reshape(-1,2)
    if len(roster)!=validated['active_edges']:
        raise ValueError('Sample curvature and global axis count disagree')
    active=hashlib.sha256(roster.tobytes()).hexdigest()
    body=canonical_json(dict(active_nonself_edges=len(roster),finite_edges=validated['finite_edges'],
                             failed_nonfinite=validated['failed'],solver_receipt=record['digest'],
                             numerical_errors=validated['certificate'],global_activity=active))
    path=Path(certificates_dir)/'sample-certificates.sqlite'
    with sqlite3.connect(path,timeout=30) as db:
        db.execute('CREATE TABLE IF NOT EXISTS receipts(sample TEXT, geometry TEXT, body BLOB NOT NULL, PRIMARY KEY(sample,geometry))')
        db.execute('INSERT OR IGNORE INTO receipts VALUES(?,?,?)',(str(sample_id),cache.folder.name,body))
        previous=db.execute('SELECT body FROM receipts WHERE sample=? AND geometry=?',(str(sample_id),cache.folder.name)).fetchone()
        if previous[0]!=body:
            raise ValueError('Conflicting sample-level curvature attestation')
