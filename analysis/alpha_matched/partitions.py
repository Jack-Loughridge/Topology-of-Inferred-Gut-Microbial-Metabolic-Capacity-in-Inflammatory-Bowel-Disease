"""Partition-scoped relative support and immutable classifier inputs."""
from dataclasses import dataclass
import hashlib
import json

import numpy as np
from scipy.sparse import csr_matrix


def digest(*parts):
    h = hashlib.sha256()
    for part in parts:
        h.update(len(part).to_bytes(8, 'big'))
        h.update(part)
    return h.hexdigest()


def canonical_json(obj):
    return json.dumps(obj, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def array_id(arr):
    return digest(canonical_json(arr.shape), arr.dtype.str.encode(), arr.tobytes(order='C'))


def fitted_location_scale(values, weights):
    """Keep exactly constant training coordinates at their native value and unit scale."""
    values = np.asarray(values, dtype=np.float64)
    # Use wider arithmetic for differences of finite doubles and variance at either
    # end of their exponent range. Center relative to a training observation so
    # the rounding of the serialized mean cannot inflate near-constant variance.
    mean = np.empty(values.shape[1])
    scale = np.empty_like(mean)
    for begin in range(0, len(mean), 128):
        columns = slice(begin, begin + 128)
        block = values[:, columns].astype(np.longdouble)
        origin = block[0]
        offsets = block - origin
        center = np.average(offsets, axis=0, weights=weights)
        mean[columns] = origin + center
        variance = np.average((offsets - center) ** 2, axis=0, weights=weights)
        constant = (block.min(axis=0) == block.max(axis=0))
        scale[columns] = np.where(constant, 1, np.sqrt(variance))
    if not np.all(np.isfinite(mean) & np.isfinite(scale) & (scale > 0)):
        raise FloatingPointError("Training location or scale outside representable range")
    return mean, scale


def immutable_array(values):
    arr = np.asarray(values)
    return np.frombuffer(arr.tobytes(order='C'), dtype=arr.dtype).reshape(arr.shape)


@dataclass(frozen=True)
class RelativeReference:
    log_reference: np.ndarray
    unseen: np.ndarray
    partition: str
    axis: str

    @classmethod
    def fit(cls, raw_train, people, sample_ids, axis):
        x = csr_matrix(raw_train, dtype=np.float64, copy=True)
        x.sum_duplicates()
        x.eliminate_zeros()
        if x.shape != (len(people), len(axis)) or len(sample_ids) != len(people):
            raise ValueError("Incorrect training axes")
        if len(set(sample_ids)) != len(sample_ids) or len(set(axis)) != len(axis) or not len(people):
            raise ValueError("Duplicate axes or empty training partition")
        if not np.isfinite(x.data).all() or (x.data <= 0).any():
            raise ValueError("Invalid fraction support")
        _, person, visits = np.unique(people, return_inverse=True, return_counts=True)
        weights = np.repeat(1.0 / visits[person], np.diff(x.indptr))
        den = np.bincount(x.indices, weights=weights, minlength=x.shape[1])
        num = np.bincount(x.indices, weights=weights * np.log(x.data), minlength=x.shape[1])
        seen = den > 0
        logref = np.zeros_like(num)
        logref[seen] = num[seen] / den[seen]
        if seen.any():
            # Median in support units (average the two middle values for even n).
            ordered = np.sort(logref[seen])
            mid = len(ordered)//2
            logref[~seen] = np.logaddexp(ordered[(len(ordered)-1)//2], ordered[mid]) - np.log(2.)
        partition = digest(canonical_json(sorted(zip(map(str, sample_ids), map(str, people)))),
                           array_id(x.indptr).encode(), array_id(x.indices).encode(), array_id(x.data).encode(),
                           canonical_json(list(map(str,sample_ids))))
        return cls(immutable_array(logref), immutable_array(~seen), partition, digest(canonical_json(list(axis))))

    def log_active_support(self, raw_row, axis):
        if digest(canonical_json(list(axis))) != self.axis:
            raise ValueError("Relative support axis changed")
        row = csr_matrix(raw_row, dtype=np.float64, copy=True)
        if row.shape != (1,len(axis)):
            raise ValueError("Expected one aligned sample")
        row.sum_duplicates()
        row.eliminate_zeros()
        if np.any(row.data <= 0) or not np.isfinite(row.data).all():
            raise ValueError("Invalid support")
        return row.indices, np.log(row.data) - self.log_reference[row.indices]


@dataclass(frozen=True)
class FrozenVectors:
    values: np.ndarray
    ids: tuple
    features: tuple
    arm: str
    partition: str
    generator: str
    block: str

    def __post_init__(self):
        arr = np.asarray(self.values)
        if arr.dtype.kind != 'f' or arr.ndim != 2 or not np.isfinite(arr).all():
            raise ValueError("Expected finite canonical floating-point vectors")
        if arr.shape != (len(self.ids), len(self.features)):
            raise ValueError("Vector axes disagree")
        if len(set(self.ids)) != len(self.ids) or len(set(self.features)) != len(self.features):
            raise ValueError("Duplicate labels")
        object.__setattr__(self, 'values', immutable_array(arr))
        object.__setattr__(self, 'ids', tuple(self.ids))
        object.__setattr__(self, 'features', tuple(self.features))

    def identity(self):
        return digest(canonical_json([self.ids,self.features,self.arm,self.partition,self.generator,self.block]),
                      array_id(self.values).encode())


def fusion_input(h0, ricci):
    if (h0.block,ricci.block) != ('H0','Ricci'):
        raise ValueError("Incorrect frozen blocks")
    if (h0.ids,h0.partition,h0.arm,h0.values.dtype) != (ricci.ids,ricci.partition,ricci.arm,ricci.values.dtype):
        raise ValueError("Incompatible frozen vectors")
    joined = np.concatenate((h0.values,ricci.values),axis=1)
    offset = h0.values.shape[1]
    if array_id(joined[:,:offset]) != array_id(h0.values) or array_id(joined[:,offset:]) != array_id(ricci.values):
        raise RuntimeError("Fusion changed canonical vector bytes")
    return joined
