from __future__ import annotations

import hashlib
import json
import os
import random
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch


def now_iso() -> str:
    return pd.Timestamp.now(tz="UTC").isoformat()


def normalise_sample_id(value: object) -> str:
    text = os.path.basename(str(value).strip())
    text = re.sub(r"\.npy$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"_H0$", "", text, flags=re.IGNORECASE)
    return text


def find_col(frame: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    lookup = {str(column).strip().lower(): column for column in frame.columns}
    for candidate in candidates:
        key = candidate.strip().lower()
        if key in lookup:
            return str(lookup[key])
    return None


def set_all_seeds(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n")
    temporary.replace(path)


def atomic_csv(frame: pd.DataFrame, path: Path, **kwargs: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, **kwargs)
    temporary.replace(path)


def atomic_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, value)
    temporary.replace(path)


def atomic_npz(path: Path, **values: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **values)
    temporary.replace(path)


def atomic_torch(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def canonical_frame_sha256(frame: pd.DataFrame, columns: list[str]) -> str:
    work = frame[columns].copy().sort_values(columns).reset_index(drop=True)
    payload = work.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def describe(values: Iterable[float], prefix: str) -> dict[str, float]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return {f"{prefix}_mean": np.nan, f"{prefix}_sd": np.nan}
    return {
        f"{prefix}_mean": float(np.mean(array)),
        f"{prefix}_sd": float(np.std(array, ddof=1)) if len(array) > 1 else 0.0,
    }
