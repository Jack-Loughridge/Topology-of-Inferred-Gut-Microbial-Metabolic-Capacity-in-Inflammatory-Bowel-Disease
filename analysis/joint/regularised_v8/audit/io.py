from __future__ import annotations
import contextlib
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any
import numpy as np


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(2**20), b''):
            h.update(b)
    return h.hexdigest()


def json_default(x: Any):
    if isinstance(x, Path): return str(x)
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, np.generic): return x.item()
    raise TypeError(type(x).__name__)


def canonical_json(x: Any) -> str:
    return json.dumps(x, sort_keys=True, separators=(',', ':'), default=json_default,
                      allow_nan=False)


def fingerprint(x: Any) -> str:
    return hashlib.sha256(canonical_json(x).encode()).hexdigest()


def atomic_text(path: Path, text: str) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name+'.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            f.write(text); f.flush(); os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name): os.unlink(name)


def atomic_json(path: Path, obj: Any) -> None:
    atomic_text(path, json.dumps(obj, indent=2, sort_keys=True, default=json_default,
                                 allow_nan=False)+'\n')


def atomic_csv(path: Path, frame) -> None:
    atomic_text(path, frame.to_csv(index=False))


def atomic_npz(path: Path, **arrays) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.stem+'.', suffix='.npz', dir=path.parent)
    os.close(fd)
    try:
        np.savez_compressed(name, **arrays)
        with open(name, 'rb') as f: os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name): os.unlink(name)


def read_json(path: Path):
    return json.loads(Path(path).read_text())


def stat_record(path: Path) -> dict:
    s = Path(path).stat()
    return {'size': s.st_size, 'mtime_ns': s.st_mtime_ns, 'inode': s.st_ino}


def completed(directory: Path, expected_id: str) -> bool:
    """A completion marker is trusted only with its exact ID and file checksums."""
    mark = directory/'COMPLETE.json'
    if not mark.exists(): return False
    obj = read_json(mark)
    if obj.get('id') != expected_id:
        raise RuntimeError(f'Incompatible completed artifact: {mark}')
    for name, digest in obj['files'].items():
        path = directory/name
        if not path.is_file() or sha256_file(path) != digest:
            raise RuntimeError(f'Missing or corrupted completed artifact: {path}')
    return True


def finish(directory: Path, identity: str, names: list[str]) -> None:
    atomic_json(directory/'COMPLETE.json', {'id': identity,
        'files': {n: sha256_file(directory/n) for n in names}})


@contextlib.contextmanager
def exclusive_lock(path: Path):
    """Kernel releases flock after normal exit, crash, or a killed process."""
    import fcntl
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+') as f:
        try: fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f'Another process holds the run lock: {path}') from exc
        f.seek(0); f.truncate(); f.write(str(os.getpid())+'\n'); f.flush()
        try: yield
        finally: fcntl.flock(f, fcntl.LOCK_UN)
