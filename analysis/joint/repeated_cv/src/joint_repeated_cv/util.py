from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd


def now_iso() -> str:
    return pd.Timestamp.now(tz="UTC").isoformat()


def normalise_sample_id(value: Any) -> str:
    text = os.path.basename(str(value).strip())
    text = re.sub(r"\.npy$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"_H0$", "", text, flags=re.IGNORECASE)
    return text


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_csv(frame: pd.DataFrame, path: Path, **kwargs: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, **kwargs)
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_frame_sha256(frame: pd.DataFrame, columns: list[str]) -> str:
    ordered = frame[columns].copy().sort_values(columns, kind="mergesort").reset_index(drop=True)
    return hashlib.sha256(ordered.to_csv(index=False).encode("utf-8")).hexdigest()
