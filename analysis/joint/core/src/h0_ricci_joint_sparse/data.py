from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import sparse

from .tasks import normalise_label


SAMPLE_COL_CANDIDATES = (
    "sample_id", "sample", "external_id", "External ID", "Sample ID", "sampleid"
)
PARTICIPANT_COL_CANDIDATES = (
    "participant_id", "participant", "Participant ID", "subject_id", "subject"
)
LABEL_COL_CANDIDATES = (
    "diagnosis", "label", "condition", "cond", "Disease", "disease", "diagnosis_simplified"
)


@dataclass
class LoadedInputs:
    ricci: sparse.csr_matrix
    metadata: pd.DataFrame
    sample_ids: np.ndarray
    participant_ids: np.ndarray
    labels3: np.ndarray
    h0_deaths: list[np.ndarray]
    h0_paths: list[str]
    feature_metadata: pd.DataFrame
    source_columns: dict[str, str]
    h0_pd_dir: Path


def _canonical(text: str) -> str:
    return text.lower().strip().replace(" ", "").replace("_", "").replace("-", "")


def find_first_col(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    exact = {str(c): str(c) for c in df.columns}
    canonical = {_canonical(str(c)): str(c) for c in df.columns}
    for cand in candidates:
        if cand in exact:
            return exact[cand]
        key = _canonical(cand)
        if key in canonical:
            return canonical[key]
    return None


def load_ricci_matrix(feature_dir: Path) -> tuple[sparse.csr_matrix, pd.DataFrame, dict[str, str]]:
    matrix_path = feature_dir / "feature_matrix_B_K0.npz"
    metadata_path = feature_dir / "matched_metadata.csv"
    if not matrix_path.exists():
        raise FileNotFoundError(f"Missing faithful Ricci matrix: {matrix_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing faithful Ricci metadata: {metadata_path}")

    matrix = sparse.load_npz(matrix_path).tocsr().astype(np.float64)
    metadata = pd.read_csv(metadata_path, low_memory=False)
    if matrix.shape[0] != len(metadata):
        raise ValueError(
            f"Ricci matrix rows ({matrix.shape[0]}) do not match metadata rows ({len(metadata)})."
        )

    sample_col = find_first_col(metadata, SAMPLE_COL_CANDIDATES)
    participant_col = find_first_col(metadata, PARTICIPANT_COL_CANDIDATES)
    label_col = find_first_col(metadata, LABEL_COL_CANDIDATES)
    missing = [name for name, value in {
        "sample": sample_col, "participant": participant_col, "label": label_col
    }.items() if value is None]
    if missing:
        raise ValueError(
            f"Could not identify {missing} column(s) in {metadata_path}. "
            f"Columns were: {list(metadata.columns)}"
        )

    metadata = metadata.copy()
    if metadata[sample_col].isna().any() or metadata[sample_col].astype(str).str.strip().isin({"", "nan", "None"}).any():
        raise ValueError("Faithful Ricci metadata contains empty sample IDs.")
    if metadata[participant_col].isna().any() or metadata[participant_col].astype(str).str.strip().isin({"", "nan", "None"}).any():
        raise ValueError("Faithful Ricci metadata contains empty participant IDs.")
    metadata["_sample_id"] = metadata[sample_col].astype(str).str.strip()
    metadata["_participant_id"] = metadata[participant_col].astype(str).str.strip()
    metadata["_label3"] = metadata[label_col].map(normalise_label)
    unknown_labels = sorted(set(metadata["_label3"]) - {"nonIBD", "UC", "CD"})
    if unknown_labels:
        raise ValueError(f"Unexpected diagnosis labels in faithful Ricci metadata: {unknown_labels}")

    if metadata["_sample_id"].duplicated().any():
        duplicated = metadata.loc[metadata["_sample_id"].duplicated(), "_sample_id"].head().tolist()
        raise ValueError(f"Duplicate sample IDs in faithful Ricci metadata, e.g. {duplicated}")

    # Sparse matrices cannot safely carry NaN/Inf into sklearn. Zero is the established
    # absent-edge value for both the B and K0 blocks.
    if matrix.data.size:
        bad = ~np.isfinite(matrix.data)
        if bad.any():
            matrix.data[bad] = 0.0
            matrix.eliminate_zeros()

    return matrix, metadata, {
        "sample_col": sample_col,
        "participant_col": participant_col,
        "label_col": label_col,
    }


def _paths_mentioned_in_text(root: Path) -> list[Path]:
    found: list[Path] = []
    pattern = re.compile(r"(?:/home/[^\s\"']+|~/[^\s\"']+)?out_pds(?:/[^\s\"']*)?")
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".txt", ".json", ".md", ".yaml", ".yml", ".log"}:
            continue
        try:
            text = path.read_text(errors="ignore")[:2_000_000]
        except OSError:
            continue
        for match in pattern.findall(text):
            cleaned = match.rstrip(".,;:)]}")
            candidate = Path(cleaned).expanduser()
            if candidate.suffix:
                candidate = candidate.parent
            found.append(candidate)
    return found


def resolve_h0_pd_dir(h0_results_dir: Path, explicit: Path | None = None) -> Path:
    if explicit is not None:
        explicit = explicit.expanduser().resolve()
        if explicit.is_dir() and any(explicit.glob("*.npy")):
            return explicit
        raise FileNotFoundError(
            f"Explicit --h0-pd-dir does not contain NPY diagrams: {explicit}"
        )

    candidates: list[Path] = []
    candidates.extend(_paths_mentioned_in_text(h0_results_dir))
    candidates.extend([
        h0_results_dir / "out_pds",
        h0_results_dir.parent / "out_pds",
        Path.home() / "Real_Data" / "out_pds",
    ])

    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.expanduser().resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.is_dir() and any(candidate.glob("*.npy")):
            return candidate

    rendered = "\n".join(f"  - {p}" for p in candidates)
    raise FileNotFoundError(
        "Could not resolve the raw H0 persistence-diagram directory. Checked:\n"
        f"{rendered}\nPass --h0-pd-dir explicitly. The canonical project location is "
        "~/Real_Data/out_pds."
    )


def _sample_id_from_h0_path(path: Path) -> str:
    stem = path.stem
    for suffix in ("_PD_H0", "_pd_h0", "_H0", "_h0", "-H0", "-h0"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def index_h0_files(pd_dir: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    duplicates: dict[str, list[Path]] = {}
    for path in sorted(pd_dir.rglob("*.npy")):
        sample_id = _sample_id_from_h0_path(path)
        if sample_id in index:
            duplicates.setdefault(sample_id, [index[sample_id]]).append(path)
        else:
            index[sample_id] = path
    if duplicates:
        preview = {key: [str(x) for x in value] for key, value in list(duplicates.items())[:5]}
        raise ValueError(f"Ambiguous H0 NPY files for sample IDs: {json.dumps(preview, indent=2)}")
    return index


def extract_h0_deaths(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array)
    if arr.ndim == 0:
        deaths = arr.reshape(1)
    elif arr.ndim == 1:
        deaths = arr
    elif arr.ndim == 2 and arr.shape[1] == 2:
        deaths = arr[:, 1]
    elif arr.ndim == 2 and arr.shape[0] == 2:
        deaths = arr[1, :]
    elif arr.ndim == 2 and arr.shape[1] > 2:
        deaths = arr[:, 1]
    else:
        deaths = arr.reshape(-1)

    deaths = np.asarray(deaths, dtype=np.float64)
    deaths = deaths[np.isfinite(deaths)]
    deaths = deaths[deaths > 0]
    deaths.sort()
    if deaths.size == 0:
        raise ValueError("H0 persistence diagram contains no positive finite death values.")
    return deaths


def load_h0_for_samples(pd_dir: Path, sample_ids: Iterable[str]) -> tuple[list[np.ndarray], list[str]]:
    index = index_h0_files(pd_dir)
    deaths: list[np.ndarray] = []
    paths: list[str] = []
    missing: list[str] = []
    for sample_id in map(str, sample_ids):
        path = index.get(sample_id)
        if path is None:
            missing.append(sample_id)
            continue
        try:
            values = extract_h0_deaths(np.load(path, allow_pickle=False))
        except Exception as exc:  # pragma: no cover - message is the important behaviour
            raise ValueError(f"Failed to read H0 diagram {path}: {exc}") from exc
        deaths.append(values)
        paths.append(str(path))

    if missing:
        raise FileNotFoundError(
            f"Missing H0 diagrams for {len(missing)} faithful Ricci samples. "
            f"Examples: {missing[:10]}. H0 directory: {pd_dir}"
        )
    return deaths, paths


def _read_feature_metadata_candidate(path: Path, n_features: int) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception:
        return None
    feature_index_col = find_first_col(df, ("feature_index", "index", "column_index"))
    if feature_index_col is None:
        return None
    values = pd.to_numeric(df[feature_index_col], errors="coerce")
    if values.notna().sum() == 0:
        return None
    df = df.copy()
    df["feature_index"] = values.astype("Int64")
    df = df[df["feature_index"].between(0, n_features - 1, inclusive="both")]
    if df.empty:
        return None
    return df.drop_duplicates("feature_index", keep="first")


def load_feature_metadata(
    faithful_dir: Path,
    original_style_dir: Path,
    n_features: int,
) -> pd.DataFrame:
    preferred = (
        "feature_metadata_B_K0.csv",
        "feature_names_B_K0.csv",
        "feature_index_B_K0.csv",
        "feature_manifest.csv",
    )
    candidates: list[Path] = [faithful_dir / name for name in preferred]
    candidates.extend(sorted(faithful_dir.glob("*.csv")))
    # The original-style outputs contain biological annotations for selected coefficients.
    candidates.extend(sorted(original_style_dir.rglob("selected_coefficients_final_trainval_model.csv")))
    candidates.extend(sorted(original_style_dir.rglob("top_absolute_features.csv")))

    pieces: list[pd.DataFrame] = []
    for path in candidates:
        if not path.exists():
            continue
        candidate = _read_feature_metadata_candidate(path, n_features)
        if candidate is not None:
            candidate["annotation_source"] = str(path)
            pieces.append(candidate)

    base = pd.DataFrame({"feature_index": np.arange(n_features, dtype=int)})
    if n_features % 2 == 0:
        half = n_features // 2
        base["feature_type"] = np.where(base["feature_index"] < half, "B", "K0")
        base["edge_index"] = base["feature_index"] % half
        base["feature"] = base["feature_type"] + "_edge_" + base["edge_index"].astype(str)
    else:
        base["feature_type"] = "unknown"
        base["edge_index"] = base["feature_index"]
        base["feature"] = "feature_" + base["feature_index"].astype(str)

    if not pieces:
        return base

    annotations = pd.concat(pieces, ignore_index=True, sort=False)
    annotations = annotations.sort_values("feature_index").drop_duplicates("feature_index", keep="first")
    overlapping = [c for c in annotations.columns if c in base.columns and c != "feature_index"]
    annotations = annotations.drop(columns=overlapping, errors="ignore")
    return base.merge(annotations, on="feature_index", how="left")


def load_all_inputs(
    h0_results_dir: Path,
    ricci_original_dir: Path,
    ricci_feature_dir: Path,
    h0_pd_dir: Path | None,
) -> LoadedInputs:
    if not h0_results_dir.is_dir():
        raise FileNotFoundError(f"Missing canonical H0 results directory: {h0_results_dir}")
    if not ricci_original_dir.is_dir():
        raise FileNotFoundError(f"Missing original-style Ricci results directory: {ricci_original_dir}")
    ricci, metadata, source_columns = load_ricci_matrix(ricci_feature_dir)
    resolved_pd_dir = resolve_h0_pd_dir(h0_results_dir, h0_pd_dir)
    sample_ids = metadata["_sample_id"].astype(str).to_numpy()
    participant_ids = metadata["_participant_id"].astype(str).to_numpy()
    labels3 = metadata["_label3"].astype(str).to_numpy()
    h0_deaths, h0_paths = load_h0_for_samples(resolved_pd_dir, sample_ids)
    feature_metadata = load_feature_metadata(
        faithful_dir=ricci_feature_dir,
        original_style_dir=ricci_original_dir,
        n_features=ricci.shape[1],
    )
    return LoadedInputs(
        ricci=ricci,
        metadata=metadata,
        sample_ids=sample_ids,
        participant_ids=participant_ids,
        labels3=labels3,
        h0_deaths=h0_deaths,
        h0_paths=h0_paths,
        feature_metadata=feature_metadata,
        source_columns=source_columns,
        h0_pd_dir=resolved_pd_dir,
    )
