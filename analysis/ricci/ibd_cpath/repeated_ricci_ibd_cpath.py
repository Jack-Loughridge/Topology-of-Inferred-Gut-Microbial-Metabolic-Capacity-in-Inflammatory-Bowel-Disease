#!/usr/bin/env python3
"""
Repeated participant-grouped IBD-vs-non-IBD Ricci classification across an L1 C path.

This script replaces a single five-fold interpretation run with a repeated,
participant-grouped analysis designed for robust reaction- and process-level
interpretation.

Primary design
--------------
* Task: IBD (UC + CD) versus non-IBD.
* Features: [B | K0]
    B(e)  = 1 if directed reaction edge e is active in the sample, else 0.
    K0(e) = Ricci curvature of e if active, else 0.
* Model: fold-internal StandardScaler + L1 LogisticRegression(solver="saga",
  class_weight="balanced").
* Repeated CV: by default 20 repetitions x 5 participant-grouped folds.
* C path: by default 0.005, 0.01, 0.02, 0.05, 0.1.
* The exact participant splits are generated once, saved, and reused for every C.
* Performance distributions are reported primarily across 20 pooled out-of-fold
  repetition estimates, not by treating all 100 folds as independent.
* Reaction tables contain coefficient distributions across all fitted folds,
  including zeros, plus distributions conditional on selection.
* Process tables are constructed fold-by-fold and use only held-out samples for
  realised contribution summaries.
* Fold model parameters (training-fold scaler and logistic coefficients) are
  saved so the exact source-trained models can later be applied to an external
  cohort without refitting or external rescaling.
* A full-source model is also saved for each C for later locked external use.

Required input files in --feature-dir
-------------------------------------
    feature_matrix_B_K0.npz
    matched_metadata.csv
    edge_metadata.csv

matched_metadata.csv must contain:
    sample_id, participant_id, cond

edge_metadata.csv must contain:
    edge
and may contain:
    process

Optional edge annotations can be supplied via --edge-annotation-csv.

Resume behaviour
----------------
Each completed C/repetition/fold fit is written immediately to disk. Re-running
with the same output directory resumes from completed fold artifacts unless
--no-resume is supplied.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn import __version__ as sklearn_version
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler


SCHEMA_VERSION = 1
LABEL_ORDER = ["non-IBD", "IBD"]
LABEL_MAP = {"nonibd": 0, "uc": 1, "cd": 1}

PROCESS_CANDIDATES = [
    "process",
    "process_class",
    "process_category",
    "biological_process",
    "category",
    "edge_process",
    "reaction_process",
]
EDGE_CANDIDATES = ["edge", "edge_id", "edge_key", "reaction_edge", "directed_edge"]
SOURCE_CANDIDATES = ["source", "src", "u", "a", "from", "tail"]
TARGET_CANDIDATES = ["target", "dst", "v", "b", "to", "head"]


@dataclass(frozen=True)
class FoldKey:
    repeat: int
    fold: int
    split_seed: int
    model_seed: int


@dataclass
class FoldBundle:
    key: FoldKey
    C: float
    coef: np.ndarray
    intercept: float
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    abs_push_participant_mean: np.ndarray
    signed_push_participant_mean: np.ndarray
    abs_push_nonibd_participant_mean: np.ndarray
    abs_push_ibd_participant_mean: np.ndarray
    signed_push_nonibd_participant_mean: np.ndarray
    signed_push_ibd_participant_mean: np.ndarray
    result: dict[str, Any]
    predictions: pd.DataFrame


def now_iso() -> str:
    return pd.Timestamp.now(tz="UTC").isoformat()


def c_tag(C: float) -> str:
    text = f"{float(C):.12g}"
    return "C_" + text.replace("-", "m").replace(".", "p")


def fmt_float(x: Any, digits: int = 6) -> str:
    try:
        value = float(x)
    except Exception:
        return "NA"
    if not math.isfinite(value):
        return "NA"
    return f"{value:.{digits}f}"


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str) + "\n")
    tmp.replace(path)


def atomic_to_csv(df: pd.DataFrame, path: Path, **kwargs: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, **kwargs)
    tmp.replace(path)


def atomic_savez_compressed(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    np.savez_compressed(tmp, **arrays)
    generated = Path(str(tmp) + ".npz") if not str(tmp).endswith(".npz") else tmp
    generated.replace(path)


def pick_col(columns: Iterable[str], candidates: list[str], required: bool, label: str) -> str | None:
    cols = list(columns)
    lower = {str(c).lower(): c for c in cols}
    for candidate in candidates:
        if candidate in cols:
            return candidate
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    if required:
        raise ValueError(f"Could not find {label}. Tried {candidates}. Columns: {cols}")
    return None


def decode_edge_for_display(edge: str) -> str:
    text = str(edge)
    text = text.replace("__91__", "[")
    text = text.replace("__93__", "]")
    text = text.replace("M_", "")
    text = text.replace("_DASH_", "-")
    return text


def load_edge_annotation(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"Annotation CSV does not exist: {path}")

    df = pd.read_csv(path, low_memory=False)
    process_col = pick_col(df.columns, PROCESS_CANDIDATES, required=True, label="process column")
    edge_col = pick_col(df.columns, EDGE_CANDIDATES, required=False, label="edge column")
    display_col = pick_col(
        df.columns,
        ["edge_display", "display_edge", "edge_name", "reaction_name", "label"],
        required=False,
        label="display edge column",
    )

    if edge_col is None:
        source_col = pick_col(df.columns, SOURCE_CANDIDATES, required=True, label="source column")
        target_col = pick_col(df.columns, TARGET_CANDIDATES, required=True, label="target column")
        edge_series = df[source_col].astype(str) + " -> " + df[target_col].astype(str)
    else:
        edge_series = df[edge_col].astype(str)

    mapping: dict[str, dict[str, str]] = {}
    for row_idx, edge in enumerate(edge_series):
        proc_value = df.iloc[row_idx][process_col]
        display_value = df.iloc[row_idx][display_col] if display_col is not None else edge
        process = str(proc_value) if pd.notna(proc_value) else "Unannotated"
        display = str(display_value) if pd.notna(display_value) else decode_edge_for_display(edge)
        item = {"process": process, "edge_display": display}
        mapping[str(edge)] = item
        mapping[decode_edge_for_display(str(edge))] = item

    print(f"[annotation] Loaded {len(mapping):,} annotation keys from {path}", flush=True)
    return mapping


def load_inputs(
    feature_dir: Path,
    annotation_csv: Path | None,
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    matrix_path = feature_dir / "feature_matrix_B_K0.npz"
    metadata_path = feature_dir / "matched_metadata.csv"
    edge_path = feature_dir / "edge_metadata.csv"

    for path in [matrix_path, metadata_path, edge_path]:
        if not path.exists():
            raise FileNotFoundError(f"Missing required input: {path}")

    X_sparse = load_npz(matrix_path).tocsr()
    metadata = pd.read_csv(metadata_path, low_memory=False)
    edge_meta = pd.read_csv(edge_path, low_memory=False)

    for column in ["sample_id", "participant_id", "cond"]:
        if column not in metadata.columns:
            raise ValueError(f"matched_metadata.csv missing required column: {column}")
    if "edge" not in edge_meta.columns:
        raise ValueError("edge_metadata.csv must contain an 'edge' column")
    if "process" not in edge_meta.columns:
        edge_meta["process"] = "Unannotated"

    metadata = metadata.copy()
    metadata["sample_id"] = metadata["sample_id"].astype(str)
    metadata["participant_id"] = metadata["participant_id"].astype(str)
    metadata["cond"] = metadata["cond"].astype(str).str.strip().str.lower().str.replace(r"[\s_-]+", "", regex=True)

    if metadata["sample_id"].duplicated().any():
        duplicates = metadata.loc[metadata["sample_id"].duplicated(), "sample_id"].head(10).tolist()
        raise ValueError(f"sample_id must be unique. Example duplicates: {duplicates}")

    edge_meta = edge_meta.copy()
    edge_meta["edge"] = edge_meta["edge"].astype(str)
    edge_meta["process"] = edge_meta["process"].fillna("Unannotated").astype(str)

    if X_sparse.shape[0] != len(metadata):
        raise ValueError(
            f"Feature rows ({X_sparse.shape[0]}) do not match metadata rows ({len(metadata)})"
        )
    n_edges = len(edge_meta)
    if X_sparse.shape[1] != 2 * n_edges:
        raise ValueError(
            f"Feature columns ({X_sparse.shape[1]}) do not equal 2*n_edges ({2*n_edges})"
        )

    annotation_map = load_edge_annotation(annotation_csv)
    edge_display: list[str] = []
    process: list[str] = []
    for edge, process0 in zip(edge_meta["edge"], edge_meta["process"]):
        decoded = decode_edge_for_display(edge)
        item = annotation_map.get(edge) or annotation_map.get(decoded)
        if item is None:
            edge_display.append(decoded)
            process.append(process0 if process0 else "Unannotated")
        else:
            edge_display.append(item["edge_display"])
            process.append(item["process"])

    edge_meta["edge_display"] = edge_display
    edge_meta["process"] = pd.Series(process).fillna("Unannotated").astype(str)

    feature_meta = pd.DataFrame(
        {
            "feature_index": np.arange(2 * n_edges, dtype=int),
            "feature": [f"B__{e}" for e in edge_meta["edge"]]
            + [f"K__{e}" for e in edge_meta["edge"]],
            "feature_type": ["B"] * n_edges + ["K0"] * n_edges,
            "edge_index": list(range(n_edges)) + list(range(n_edges)),
            "edge": edge_meta["edge"].tolist() + edge_meta["edge"].tolist(),
            "edge_display": edge_meta["edge_display"].tolist()
            + edge_meta["edge_display"].tolist(),
            "process": edge_meta["process"].tolist() + edge_meta["process"].tolist(),
        }
    )

    valid_rows = metadata["cond"].isin(LABEL_MAP)
    if not valid_rows.any():
        raise ValueError(
            "No rows matched conditions nonibd/uc/cd after lower-casing metadata['cond']"
        )

    row_indices = np.flatnonzero(valid_rows.to_numpy())
    task_meta = metadata.iloc[row_indices].reset_index(drop=True)
    task_meta["label"] = task_meta["cond"].map(LABEL_MAP).astype(int)
    task_meta["label_name"] = task_meta["label"].map({0: "non-IBD", 1: "IBD"})

    participant_label_counts = task_meta.groupby("participant_id")["label"].nunique()
    bad = participant_label_counts[participant_label_counts > 1]
    if len(bad):
        raise ValueError(
            "At least one participant has both labels in the IBD-vs-non-IBD task: "
            + ", ".join(bad.index.astype(str).tolist()[:10])
        )

    print("[load] Converting selected sparse rows to dense float32...", flush=True)
    X_dense = X_sparse[row_indices].toarray().astype(np.float32, copy=False)

    return X_dense, task_meta, edge_meta, feature_meta


def generate_or_load_splits(
    task_meta: pd.DataFrame,
    output_dir: Path,
    n_repeats: int,
    n_splits: int,
    split_seed: int,
    resume: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_dir = output_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    participant_path = split_dir / "participant_split_manifest.csv"
    sample_path = split_dir / "sample_split_manifest.csv"
    config_path = split_dir / "split_config.json"

    if resume and participant_path.exists() and sample_path.exists() and config_path.exists():
        config = json.loads(config_path.read_text())
        expected = {
            "schema_version": SCHEMA_VERSION,
            "n_repeats": int(n_repeats),
            "n_splits": int(n_splits),
            "split_seed": int(split_seed),
        }
        for key, value in expected.items():
            if config.get(key) != value:
                raise ValueError(
                    f"Existing split manifest has {key}={config.get(key)!r}, requested {value!r}. "
                    "Use a new output directory or remove the split directory."
                )
        sample_manifest = pd.read_csv(sample_path, dtype={"sample_id": str, "participant_id": str})
        participant_manifest = pd.read_csv(
            participant_path, dtype={"participant_id": str}
        )
        current_ids = set(task_meta["sample_id"].astype(str))
        manifest_ids = set(sample_manifest["sample_id"].astype(str))
        if current_ids != manifest_ids:
            missing = sorted(current_ids - manifest_ids)[:10]
            extra = sorted(manifest_ids - current_ids)[:10]
            raise ValueError(
                "Existing split manifest sample IDs do not match current inputs. "
                f"Missing examples={missing}; extra examples={extra}"
            )
        print(f"[splits] Reusing saved split manifests from {split_dir}", flush=True)
        return sample_manifest, participant_manifest

    y = task_meta["label"].to_numpy(dtype=int)
    groups = task_meta["participant_id"].astype(str).to_numpy()
    participant_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []

    for repeat_idx in range(1, n_repeats + 1):
        seed = split_seed + (repeat_idx - 1) * 1009
        cv = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=seed,
        )
        seen_test_samples: set[int] = set()
        for fold_idx, (train_idx, test_idx) in enumerate(
            cv.split(np.zeros(len(y), dtype=np.uint8), y, groups=groups), start=1
        ):
            train_participants = set(groups[train_idx])
            test_participants = set(groups[test_idx])
            overlap = train_participants & test_participants
            if overlap:
                raise RuntimeError(
                    f"Participant leakage in repeat {repeat_idx}, fold {fold_idx}: {sorted(overlap)[:10]}"
                )
            if set(np.unique(y[train_idx])) != {0, 1}:
                raise RuntimeError(
                    f"Training fold lacks a class in repeat {repeat_idx}, fold {fold_idx}"
                )
            if set(np.unique(y[test_idx])) != {0, 1}:
                raise RuntimeError(
                    f"Test fold lacks a class in repeat {repeat_idx}, fold {fold_idx}"
                )
            duplicate_test = seen_test_samples.intersection(set(test_idx.tolist()))
            if duplicate_test:
                raise RuntimeError(
                    f"Samples repeated across test folds in repeat {repeat_idx}: {sorted(duplicate_test)[:10]}"
                )
            seen_test_samples.update(test_idx.tolist())

            for role, indices in [("train", train_idx), ("test", test_idx)]:
                subset = task_meta.iloc[indices]
                for row in subset.itertuples(index=False):
                    sample_rows.append(
                        {
                            "repeat": repeat_idx,
                            "fold": fold_idx,
                            "split_seed": seed,
                            "role": role,
                            "sample_id": str(row.sample_id),
                            "participant_id": str(row.participant_id),
                            "label": int(row.label),
                            "label_name": str(row.label_name),
                            "condition": str(row.cond),
                        }
                    )

            participant_labels = (
                task_meta[["participant_id", "label", "label_name"]]
                .drop_duplicates("participant_id")
                .set_index("participant_id")
            )
            for role, participants in [
                ("train", sorted(train_participants)),
                ("test", sorted(test_participants)),
            ]:
                for participant_id in participants:
                    participant_rows.append(
                        {
                            "repeat": repeat_idx,
                            "fold": fold_idx,
                            "split_seed": seed,
                            "role": role,
                            "participant_id": participant_id,
                            "label": int(participant_labels.loc[participant_id, "label"]),
                            "label_name": str(
                                participant_labels.loc[participant_id, "label_name"]
                            ),
                        }
                    )

        if len(seen_test_samples) != len(task_meta):
            raise RuntimeError(
                f"Repeat {repeat_idx} test folds cover {len(seen_test_samples)} of {len(task_meta)} samples"
            )

    sample_manifest = pd.DataFrame(sample_rows)
    participant_manifest = pd.DataFrame(participant_rows)
    atomic_to_csv(sample_manifest, sample_path, index=False)
    atomic_to_csv(participant_manifest, participant_path, index=False)
    write_json(
        config_path,
        {
            "schema_version": SCHEMA_VERSION,
            "created_utc": now_iso(),
            "n_repeats": int(n_repeats),
            "n_splits": int(n_splits),
            "split_seed": int(split_seed),
            "splitter": "StratifiedGroupKFold on sample rows with participant_id as groups",
            "task": "IBD (UC+CD) versus non-IBD",
            "sample_count": int(len(task_meta)),
            "participant_count": int(task_meta["participant_id"].nunique()),
        },
    )
    print(f"[splits] Saved split manifests to {split_dir}", flush=True)
    return sample_manifest, participant_manifest


def model_seed_for(base_seed: int, repeat_idx: int, fold_idx: int) -> int:
    return int(base_seed + repeat_idx * 1009 + fold_idx * 97)


def indices_for_fold(
    sample_manifest: pd.DataFrame,
    sample_to_index: dict[str, int],
    repeat_idx: int,
    fold_idx: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    fold_rows = sample_manifest[
        (sample_manifest["repeat"] == repeat_idx)
        & (sample_manifest["fold"] == fold_idx)
    ]
    if len(fold_rows) == 0:
        raise KeyError(f"No split rows for repeat={repeat_idx}, fold={fold_idx}")
    train_ids = fold_rows.loc[fold_rows["role"] == "train", "sample_id"].astype(str)
    test_ids = fold_rows.loc[fold_rows["role"] == "test", "sample_id"].astype(str)
    train_idx = np.fromiter((sample_to_index[x] for x in train_ids), dtype=int)
    test_idx = np.fromiter((sample_to_index[x] for x in test_ids), dtype=int)
    split_seed = int(fold_rows["split_seed"].iloc[0])
    return train_idx, test_idx, split_seed


def safe_binary_metrics(y_true: np.ndarray, proba_ibd: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    proba_ibd = np.asarray(proba_ibd, dtype=float)
    pred = (proba_ibd >= 0.5).astype(int)
    result = {
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "positive_f1": float(f1_score(y_true, pred, pos_label=1, zero_division=0)),
        "brier": float(brier_score_loss(y_true, proba_ibd)),
    }
    try:
        result["roc_auc"] = float(roc_auc_score(y_true, proba_ibd))
    except Exception:
        result["roc_auc"] = np.nan
    try:
        proba = np.column_stack([1.0 - proba_ibd, proba_ibd])
        result["log_loss"] = float(log_loss(y_true, proba, labels=[0, 1]))
    except Exception:
        result["log_loss"] = np.nan
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    result.update(
        {
            "tn": int(cm[0, 0]),
            "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]),
            "tp": int(cm[1, 1]),
        }
    )
    return result


def participant_prediction_table(predictions: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        predictions.groupby("participant_id", as_index=False)
        .agg(
            true_label=("true_label", "first"),
            true_label_name=("true_label_name", "first"),
            proba_ibd=("proba_ibd", "mean"),
            n_samples=("sample_id", "count"),
        )
        .sort_values("participant_id")
        .reset_index(drop=True)
    )
    grouped["pred_label"] = (grouped["proba_ibd"] >= 0.5).astype(int)
    grouped["pred_label_name"] = grouped["pred_label"].map({0: "non-IBD", 1: "IBD"})
    return grouped


def participant_balanced_feature_means(
    contributions: np.ndarray,
    test_meta: pd.DataFrame,
) -> dict[str, np.ndarray]:
    """Return participant-balanced featurewise means on held-out samples.

    contributions is X_test_standardised * beta and has shape n_test_samples x n_features.
    Absolute push is calculated featurewise before summation, matching the intended
    reaction/process realised-contribution definition.
    """
    participant_ids = test_meta["participant_id"].astype(str).to_numpy()
    labels = test_meta["label"].to_numpy(dtype=int)
    unique_participants = pd.unique(participant_ids)

    signed_participant: list[np.ndarray] = []
    abs_participant: list[np.ndarray] = []
    participant_labels: list[int] = []

    for participant_id in unique_participants:
        mask = participant_ids == participant_id
        participant_label_values = np.unique(labels[mask])
        if len(participant_label_values) != 1:
            raise RuntimeError(f"Participant {participant_id} has inconsistent held-out labels")
        participant_labels.append(int(participant_label_values[0]))
        block = contributions[mask]
        signed_participant.append(np.mean(block, axis=0, dtype=np.float64))
        abs_participant.append(np.mean(np.abs(block), axis=0, dtype=np.float64))

    signed = np.vstack(signed_participant)
    absolute = np.vstack(abs_participant)
    participant_labels_array = np.asarray(participant_labels, dtype=int)

    def mean_or_nan(array: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if not np.any(mask):
            return np.full(array.shape[1], np.nan, dtype=np.float32)
        return np.mean(array[mask], axis=0, dtype=np.float64).astype(np.float32)

    return {
        "signed_all": np.mean(signed, axis=0, dtype=np.float64).astype(np.float32),
        "abs_all": np.mean(absolute, axis=0, dtype=np.float64).astype(np.float32),
        "signed_nonibd": mean_or_nan(signed, participant_labels_array == 0),
        "signed_ibd": mean_or_nan(signed, participant_labels_array == 1),
        "abs_nonibd": mean_or_nan(absolute, participant_labels_array == 0),
        "abs_ibd": mean_or_nan(absolute, participant_labels_array == 1),
    }


def fold_paths(c_dir: Path, repeat_idx: int, fold_idx: int) -> tuple[Path, Path, Path]:
    stem = f"repeat_{repeat_idx:02d}_fold_{fold_idx:02d}"
    artifact_dir = c_dir / "fold_artifacts"
    prediction_dir = c_dir / "fold_predictions"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir.mkdir(parents=True, exist_ok=True)
    return (
        artifact_dir / f"{stem}.npz",
        artifact_dir / f"{stem}.json",
        prediction_dir / f"{stem}.csv",
    )


def load_fold_bundle(
    artifact_path: Path,
    result_path: Path,
    prediction_path: Path,
) -> FoldBundle:
    result = json.loads(result_path.read_text())
    if result.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported fold artifact schema in {result_path}")
    with np.load(artifact_path, allow_pickle=False) as arrays:
        coef = arrays["coef"].astype(np.float32, copy=True)
        intercept = float(arrays["intercept"][0])
        scaler_mean = arrays["scaler_mean"].astype(np.float32, copy=True)
        scaler_scale = arrays["scaler_scale"].astype(np.float32, copy=True)
        abs_push_participant_mean = arrays["abs_push_participant_mean"].astype(
            np.float32, copy=True
        )
        signed_push_participant_mean = arrays["signed_push_participant_mean"].astype(
            np.float32, copy=True
        )
        abs_push_nonibd_participant_mean = arrays[
            "abs_push_nonibd_participant_mean"
        ].astype(np.float32, copy=True)
        abs_push_ibd_participant_mean = arrays[
            "abs_push_ibd_participant_mean"
        ].astype(np.float32, copy=True)
        signed_push_nonibd_participant_mean = arrays[
            "signed_push_nonibd_participant_mean"
        ].astype(np.float32, copy=True)
        signed_push_ibd_participant_mean = arrays[
            "signed_push_ibd_participant_mean"
        ].astype(np.float32, copy=True)

    predictions = pd.read_csv(
        prediction_path,
        dtype={"sample_id": str, "participant_id": str},
    )
    key = FoldKey(
        repeat=int(result["repeat"]),
        fold=int(result["fold"]),
        split_seed=int(result["split_seed"]),
        model_seed=int(result["model_seed"]),
    )
    return FoldBundle(
        key=key,
        C=float(result["C"]),
        coef=coef,
        intercept=intercept,
        scaler_mean=scaler_mean,
        scaler_scale=scaler_scale,
        abs_push_participant_mean=abs_push_participant_mean,
        signed_push_participant_mean=signed_push_participant_mean,
        abs_push_nonibd_participant_mean=abs_push_nonibd_participant_mean,
        abs_push_ibd_participant_mean=abs_push_ibd_participant_mean,
        signed_push_nonibd_participant_mean=signed_push_nonibd_participant_mean,
        signed_push_ibd_participant_mean=signed_push_ibd_participant_mean,
        result=result,
        predictions=predictions,
    )


def fit_or_load_fold(
    X: np.ndarray,
    task_meta: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    key: FoldKey,
    C: float,
    c_dir: Path,
    max_iter: int,
    tol: float,
    n_jobs: int,
    coef_eps: float,
    resume: bool,
) -> FoldBundle:
    artifact_path, result_path, prediction_path = fold_paths(c_dir, key.repeat, key.fold)
    if resume and artifact_path.exists() and result_path.exists() and prediction_path.exists():
        bundle = load_fold_bundle(artifact_path, result_path, prediction_path)
        if not math.isclose(bundle.C, C, rel_tol=0.0, abs_tol=1e-15):
            raise ValueError(f"C mismatch in resumed artifact {result_path}")
        if bundle.key != key:
            raise ValueError(f"Fold key mismatch in resumed artifact {result_path}")
        print(
            f"[resume] {c_tag(C)} repeat={key.repeat:02d} fold={key.fold:02d}",
            flush=True,
        )
        return bundle

    started = time.time()
    X_train = X[train_idx]
    X_test = X[test_idx]
    y_train = task_meta.iloc[train_idx]["label"].to_numpy(dtype=int)
    y_test = task_meta.iloc[test_idx]["label"].to_numpy(dtype=int)
    train_groups = task_meta.iloc[train_idx]["participant_id"].astype(str).to_numpy()
    test_groups = task_meta.iloc[test_idx]["participant_id"].astype(str).to_numpy()
    overlap = set(train_groups) & set(test_groups)
    if overlap:
        raise RuntimeError(
            f"Participant leakage before fit repeat={key.repeat}, fold={key.fold}: {sorted(overlap)[:10]}"
        )

    scaler = StandardScaler(with_mean=True, with_std=True, copy=True)
    X_train_z = scaler.fit_transform(X_train).astype(np.float32, copy=False)
    X_test_z = scaler.transform(X_test).astype(np.float32, copy=False)

    classifier = LogisticRegression(
        penalty="l1",
        solver="saga",
        class_weight="balanced",
        C=float(C),
        max_iter=int(max_iter),
        tol=float(tol),
        random_state=int(key.model_seed),
        n_jobs=int(n_jobs),
    )

    convergence_warning = False
    warning_messages: list[str] = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        classifier.fit(X_train_z, y_train)
        for warning in caught:
            if isinstance(warning.message, ConvergenceWarning):
                convergence_warning = True
                warning_messages.append(str(warning.message))

    proba_ibd = classifier.predict_proba(X_test_z)[:, 1]
    pred_label = (proba_ibd >= 0.5).astype(int)
    sample_metrics = safe_binary_metrics(y_test, proba_ibd)

    test_meta = task_meta.iloc[test_idx].reset_index(drop=True)
    predictions = pd.DataFrame(
        {
            "schema_version": SCHEMA_VERSION,
            "C": float(C),
            "repeat": key.repeat,
            "fold": key.fold,
            "split_seed": key.split_seed,
            "model_seed": key.model_seed,
            "sample_id": test_meta["sample_id"].astype(str),
            "participant_id": test_meta["participant_id"].astype(str),
            "condition": test_meta["cond"].astype(str),
            "true_label": y_test,
            "true_label_name": test_meta["label_name"].astype(str),
            "pred_label": pred_label,
            "pred_label_name": pd.Series(pred_label).map({0: "non-IBD", 1: "IBD"}),
            "proba_nonibd": 1.0 - proba_ibd,
            "proba_ibd": proba_ibd,
        }
    )
    participant_predictions = participant_prediction_table(predictions)
    participant_metrics = safe_binary_metrics(
        participant_predictions["true_label"].to_numpy(dtype=int),
        participant_predictions["proba_ibd"].to_numpy(dtype=float),
    )

    beta = classifier.coef_[0].astype(np.float32, copy=False)
    contributions = X_test_z * beta.reshape(1, -1)
    feature_push = participant_balanced_feature_means(contributions, test_meta)

    n_edges = X.shape[1] // 2
    selected = np.abs(beta) > coef_eps
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": now_iso(),
        "C": float(C),
        "repeat": key.repeat,
        "fold": key.fold,
        "split_seed": key.split_seed,
        "model_seed": key.model_seed,
        "train_samples": int(len(train_idx)),
        "test_samples": int(len(test_idx)),
        "train_participants": int(len(set(train_groups))),
        "test_participants": int(len(set(test_groups))),
        "participant_overlap": 0,
        "train_nonibd_samples": int(np.sum(y_train == 0)),
        "train_ibd_samples": int(np.sum(y_train == 1)),
        "test_nonibd_samples": int(np.sum(y_test == 0)),
        "test_ibd_samples": int(np.sum(y_test == 1)),
        "test_nonibd_participants": int(
            participant_predictions.loc[
                participant_predictions["true_label"] == 0, "participant_id"
            ].nunique()
        ),
        "test_ibd_participants": int(
            participant_predictions.loc[
                participant_predictions["true_label"] == 1, "participant_id"
            ].nunique()
        ),
        "selected_total": int(np.sum(selected)),
        "selected_B": int(np.sum(selected[:n_edges])),
        "selected_K0": int(np.sum(selected[n_edges:])),
        "abs_coef_mass_total": float(np.sum(np.abs(beta), dtype=np.float64)),
        "abs_coef_mass_B": float(np.sum(np.abs(beta[:n_edges]), dtype=np.float64)),
        "abs_coef_mass_K0": float(np.sum(np.abs(beta[n_edges:]), dtype=np.float64)),
        "n_iter": int(classifier.n_iter_[0]),
        "convergence_warning": bool(convergence_warning),
        "convergence_warning_messages": warning_messages,
        "elapsed_seconds": float(time.time() - started),
    }
    for prefix, metrics in [
        ("sample", sample_metrics),
        ("participant", participant_metrics),
    ]:
        for metric_name, value in metrics.items():
            result[f"{prefix}_{metric_name}"] = value

    atomic_savez_compressed(
        artifact_path,
        schema_version=np.asarray([SCHEMA_VERSION], dtype=np.int32),
        coef=beta,
        intercept=np.asarray([classifier.intercept_[0]], dtype=np.float32),
        scaler_mean=scaler.mean_.astype(np.float32),
        scaler_scale=scaler.scale_.astype(np.float32),
        abs_push_participant_mean=feature_push["abs_all"],
        signed_push_participant_mean=feature_push["signed_all"],
        abs_push_nonibd_participant_mean=feature_push["abs_nonibd"],
        abs_push_ibd_participant_mean=feature_push["abs_ibd"],
        signed_push_nonibd_participant_mean=feature_push["signed_nonibd"],
        signed_push_ibd_participant_mean=feature_push["signed_ibd"],
    )
    write_json(result_path, result)
    atomic_to_csv(predictions, prediction_path, index=False)

    print(
        f"[fit] {c_tag(C)} repeat={key.repeat:02d} fold={key.fold:02d} "
        f"sample_bacc={sample_metrics['balanced_accuracy']:.3f} "
        f"sample_auc={sample_metrics['roc_auc']:.3f} "
        f"participant_bacc={participant_metrics['balanced_accuracy']:.3f} "
        f"selected={result['selected_total']} (B={result['selected_B']}, K0={result['selected_K0']}) "
        f"n_iter={result['n_iter']} conv_warning={convergence_warning}",
        flush=True,
    )

    return load_fold_bundle(artifact_path, result_path, prediction_path)


def numeric_distribution(values: np.ndarray, prefix: str) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    if len(finite) == 0:
        return {
            f"{prefix}_n": 0,
            f"{prefix}_mean": np.nan,
            f"{prefix}_sd": np.nan,
            f"{prefix}_q025": np.nan,
            f"{prefix}_q25": np.nan,
            f"{prefix}_median": np.nan,
            f"{prefix}_q75": np.nan,
            f"{prefix}_q975": np.nan,
            f"{prefix}_min": np.nan,
            f"{prefix}_max": np.nan,
        }
    q025, q25, median, q75, q975 = np.quantile(
        finite, [0.025, 0.25, 0.5, 0.75, 0.975]
    )
    return {
        f"{prefix}_n": int(len(finite)),
        f"{prefix}_mean": float(np.mean(finite)),
        f"{prefix}_sd": float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0,
        f"{prefix}_q025": float(q025),
        f"{prefix}_q25": float(q25),
        f"{prefix}_median": float(median),
        f"{prefix}_q75": float(q75),
        f"{prefix}_q975": float(q975),
        f"{prefix}_min": float(np.min(finite)),
        f"{prefix}_max": float(np.max(finite)),
    }


def columnwise_stats(array: np.ndarray, prefix: str) -> dict[str, np.ndarray]:
    arr = np.asarray(array, dtype=np.float64)
    return {
        f"{prefix}_mean": np.mean(arr, axis=0),
        f"{prefix}_sd": np.std(arr, axis=0, ddof=1),
        f"{prefix}_q025": np.quantile(arr, 0.025, axis=0),
        f"{prefix}_q25": np.quantile(arr, 0.25, axis=0),
        f"{prefix}_median": np.quantile(arr, 0.5, axis=0),
        f"{prefix}_q75": np.quantile(arr, 0.75, axis=0),
        f"{prefix}_q975": np.quantile(arr, 0.975, axis=0),
        f"{prefix}_min": np.min(arr, axis=0),
        f"{prefix}_max": np.max(arr, axis=0),
    }


def conditional_selected_stats(
    coefficients: np.ndarray,
    selected_mask: np.ndarray,
) -> dict[str, np.ndarray]:
    n_features = coefficients.shape[1]
    result = {
        "coef_selected_mean": np.full(n_features, np.nan, dtype=float),
        "coef_selected_sd": np.full(n_features, np.nan, dtype=float),
        "coef_selected_q025": np.full(n_features, np.nan, dtype=float),
        "coef_selected_q25": np.full(n_features, np.nan, dtype=float),
        "coef_selected_median": np.full(n_features, np.nan, dtype=float),
        "coef_selected_q75": np.full(n_features, np.nan, dtype=float),
        "coef_selected_q975": np.full(n_features, np.nan, dtype=float),
        "abs_coef_selected_mean": np.full(n_features, np.nan, dtype=float),
        "abs_coef_selected_median": np.full(n_features, np.nan, dtype=float),
    }
    selected_counts = np.sum(selected_mask, axis=0)
    for feature_idx in np.flatnonzero(selected_counts > 0):
        values = coefficients[selected_mask[:, feature_idx], feature_idx].astype(float)
        absolute = np.abs(values)
        q025, q25, median, q75, q975 = np.quantile(
            values, [0.025, 0.25, 0.5, 0.75, 0.975]
        )
        result["coef_selected_mean"][feature_idx] = np.mean(values)
        result["coef_selected_sd"][feature_idx] = (
            np.std(values, ddof=1) if len(values) > 1 else 0.0
        )
        result["coef_selected_q025"][feature_idx] = q025
        result["coef_selected_q25"][feature_idx] = q25
        result["coef_selected_median"][feature_idx] = median
        result["coef_selected_q75"][feature_idx] = q75
        result["coef_selected_q975"][feature_idx] = q975
        result["abs_coef_selected_mean"][feature_idx] = np.mean(absolute)
        result["abs_coef_selected_median"][feature_idx] = np.median(absolute)
    return result


def build_reaction_summary(
    C: float,
    feature_meta: pd.DataFrame,
    coefficients: np.ndarray,
    abs_push: np.ndarray,
    signed_push: np.ndarray,
    abs_push_nonibd: np.ndarray,
    abs_push_ibd: np.ndarray,
    signed_push_nonibd: np.ndarray,
    signed_push_ibd: np.ndarray,
    coef_eps: float,
    full_coef: np.ndarray | None,
) -> pd.DataFrame:
    selected = np.abs(coefficients) > coef_eps
    positive = coefficients > coef_eps
    negative = coefficients < -coef_eps
    selected_count = np.sum(selected, axis=0)
    positive_count = np.sum(positive, axis=0)
    negative_count = np.sum(negative, axis=0)
    n_models = coefficients.shape[0]

    with np.errstate(divide="ignore", invalid="ignore"):
        positive_fraction_selected = np.where(
            selected_count > 0, positive_count / selected_count, np.nan
        )
        negative_fraction_selected = np.where(
            selected_count > 0, negative_count / selected_count, np.nan
        )
        sign_consistency = np.where(
            selected_count > 0,
            np.maximum(positive_count, negative_count) / selected_count,
            np.nan,
        )

    dominant_sign = np.where(
        selected_count == 0,
        "never selected",
        np.where(
            positive_count > negative_count,
            "positive",
            np.where(negative_count > positive_count, "negative", "tie"),
        ),
    )

    extra: dict[str, Any] = {
        "C": np.full(coefficients.shape[1], float(C)),
        "n_fold_models": np.full(coefficients.shape[1], n_models, dtype=int),
        "selection_count": selected_count,
        "selection_frequency": selected_count / n_models,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "zero_count": n_models - selected_count,
        "positive_frequency_all": positive_count / n_models,
        "negative_frequency_all": negative_count / n_models,
        "positive_fraction_when_selected": positive_fraction_selected,
        "negative_fraction_when_selected": negative_fraction_selected,
        "sign_consistency_when_selected": sign_consistency,
        "dominant_sign": dominant_sign,
    }

    extra.update(columnwise_stats(coefficients, "coef_all_including_zero"))
    extra.update(columnwise_stats(np.abs(coefficients), "abs_coef_all_including_zero"))
    extra.update(conditional_selected_stats(coefficients, selected))

    contribution_arrays = {
        "heldout_abs_push_participant_balanced": abs_push,
        "heldout_signed_push_participant_balanced": signed_push,
        "heldout_abs_push_nonibd_participant_balanced": abs_push_nonibd,
        "heldout_abs_push_ibd_participant_balanced": abs_push_ibd,
        "heldout_signed_push_nonibd_participant_balanced": signed_push_nonibd,
        "heldout_signed_push_ibd_participant_balanced": signed_push_ibd,
        "heldout_signed_push_ibd_minus_nonibd": signed_push_ibd - signed_push_nonibd,
        "heldout_abs_push_ibd_minus_nonibd": abs_push_ibd - abs_push_nonibd,
    }
    for prefix, values in contribution_arrays.items():
        extra.update(columnwise_stats(values, prefix))

    extra["stability_weighted_abs_coefficient"] = extra[
        "abs_coef_all_including_zero_mean"
    ]
    extra["stable_same_sign_ge_50pct"] = (
        (extra["selection_frequency"] >= 0.50) & (sign_consistency >= 0.90)
    )
    extra["stable_same_sign_ge_70pct"] = (
        (extra["selection_frequency"] >= 0.70) & (sign_consistency >= 0.90)
    )
    extra["stable_same_sign_ge_90pct"] = (
        (extra["selection_frequency"] >= 0.90) & (sign_consistency >= 0.90)
    )
    if full_coef is not None:
        extra["full_source_coefficient_standardised"] = full_coef
        extra["full_source_selected"] = np.abs(full_coef) > coef_eps

    output = pd.concat(
        [feature_meta.reset_index(drop=True), pd.DataFrame(extra)],
        axis=1,
    )
    first_columns = ["C"] + [c for c in output.columns if c != "C"]
    return output.loc[:, first_columns]


def block_indices(feature_meta: pd.DataFrame) -> dict[str, np.ndarray]:
    all_indices = feature_meta["feature_index"].to_numpy(dtype=int)
    return {
        "B": feature_meta.loc[feature_meta["feature_type"] == "B", "feature_index"].to_numpy(dtype=int),
        "K0": feature_meta.loc[feature_meta["feature_type"] == "K0", "feature_index"].to_numpy(dtype=int),
        "Combined": all_indices,
    }


def build_process_fold_distributions(
    C: float,
    fold_results: pd.DataFrame,
    feature_meta: pd.DataFrame,
    coefficients: np.ndarray,
    abs_push: np.ndarray,
    signed_push: np.ndarray,
    abs_push_nonibd: np.ndarray,
    abs_push_ibd: np.ndarray,
    signed_push_nonibd: np.ndarray,
    signed_push_ibd: np.ndarray,
    coef_eps: float,
) -> pd.DataFrame:
    process_values = feature_meta["process"].fillna("Unannotated").astype(str)
    blocks = block_indices(feature_meta)
    rows: list[dict[str, Any]] = []

    for fit_idx, fold_row in fold_results.reset_index(drop=True).iterrows():
        beta = coefficients[fit_idx]
        for block_name, block_idx in blocks.items():
            block_processes = process_values.iloc[block_idx]
            for process_name in sorted(block_processes.unique()):
                local_mask = block_processes.to_numpy() == process_name
                indices = block_idx[local_mask]
                beta_sub = beta[indices]
                selected = np.abs(beta_sub) > coef_eps
                selected_count = int(np.sum(selected))
                abs_mass = float(np.sum(np.abs(beta_sub), dtype=np.float64))
                row = {
                    "C": float(C),
                    "repeat": int(fold_row["repeat"]),
                    "fold": int(fold_row["fold"]),
                    "fit_index": int(fit_idx),
                    "block": block_name,
                    "process": process_name,
                    "eligible_features": int(len(indices)),
                    "selected_features": selected_count,
                    "selected_fraction": selected_count / len(indices) if len(indices) else np.nan,
                    "positive_selected": int(np.sum(beta_sub > coef_eps)),
                    "negative_selected": int(np.sum(beta_sub < -coef_eps)),
                    "fit_has_any_selected": int(selected_count > 0),
                    "abs_coefficient_mass": abs_mass,
                    "signed_coefficient_mass": float(np.sum(beta_sub, dtype=np.float64)),
                    "mean_abs_coefficient_per_eligible": abs_mass / len(indices) if len(indices) else np.nan,
                    "mean_abs_coefficient_per_selected": (
                        float(np.mean(np.abs(beta_sub[selected]))) if selected_count else np.nan
                    ),
                    "heldout_abs_push_participant_balanced": float(
                        np.sum(abs_push[fit_idx, indices], dtype=np.float64)
                    ),
                    "heldout_signed_push_participant_balanced": float(
                        np.sum(signed_push[fit_idx, indices], dtype=np.float64)
                    ),
                    "heldout_abs_push_nonibd_participant_balanced": float(
                        np.sum(abs_push_nonibd[fit_idx, indices], dtype=np.float64)
                    ),
                    "heldout_abs_push_ibd_participant_balanced": float(
                        np.sum(abs_push_ibd[fit_idx, indices], dtype=np.float64)
                    ),
                    "heldout_signed_push_nonibd_participant_balanced": float(
                        np.sum(signed_push_nonibd[fit_idx, indices], dtype=np.float64)
                    ),
                    "heldout_signed_push_ibd_participant_balanced": float(
                        np.sum(signed_push_ibd[fit_idx, indices], dtype=np.float64)
                    ),
                }
                row["heldout_signed_push_ibd_minus_nonibd"] = (
                    row["heldout_signed_push_ibd_participant_balanced"]
                    - row["heldout_signed_push_nonibd_participant_balanced"]
                )
                row["heldout_abs_push_ibd_minus_nonibd"] = (
                    row["heldout_abs_push_ibd_participant_balanced"]
                    - row["heldout_abs_push_nonibd_participant_balanced"]
                )
                rows.append(row)

    return pd.DataFrame(rows)


def summarise_process_distributions(
    process_fold_df: pd.DataFrame,
    reaction_summary: pd.DataFrame,
) -> pd.DataFrame:
    metric_columns = [
        "selected_features",
        "selected_fraction",
        "fit_has_any_selected",
        "abs_coefficient_mass",
        "signed_coefficient_mass",
        "mean_abs_coefficient_per_eligible",
        "mean_abs_coefficient_per_selected",
        "heldout_abs_push_participant_balanced",
        "heldout_signed_push_participant_balanced",
        "heldout_abs_push_nonibd_participant_balanced",
        "heldout_abs_push_ibd_participant_balanced",
        "heldout_signed_push_nonibd_participant_balanced",
        "heldout_signed_push_ibd_participant_balanced",
        "heldout_signed_push_ibd_minus_nonibd",
        "heldout_abs_push_ibd_minus_nonibd",
    ]

    rows: list[dict[str, Any]] = []
    grouped = process_fold_df.groupby(["C", "block", "process"], sort=True)
    for (C, block, process), sub in grouped:
        row: dict[str, Any] = {
            "C": float(C),
            "block": str(block),
            "process": str(process),
            "n_fold_models": int(len(sub)),
            "eligible_features": int(sub["eligible_features"].iloc[0]),
        }
        for metric in metric_columns:
            row.update(numeric_distribution(sub[metric].to_numpy(dtype=float), metric))

        if block == "Combined":
            reaction_sub = reaction_summary[reaction_summary["process"] == process]
        else:
            reaction_sub = reaction_summary[
                (reaction_summary["process"] == process)
                & (reaction_summary["feature_type"] == block)
            ]
        row["reaction_features_selected_ge_10pct"] = int(
            np.sum(reaction_sub["selection_frequency"] >= 0.10)
        )
        row["reaction_features_selected_ge_50pct"] = int(
            np.sum(reaction_sub["selection_frequency"] >= 0.50)
        )
        row["reaction_features_selected_ge_70pct"] = int(
            np.sum(reaction_sub["selection_frequency"] >= 0.70)
        )
        row["reaction_features_selected_ge_90pct"] = int(
            np.sum(reaction_sub["selection_frequency"] >= 0.90)
        )
        row["reaction_features_stable_same_sign_ge_50pct"] = int(
            np.sum(reaction_sub["stable_same_sign_ge_50pct"])
        )
        row["reaction_features_stable_same_sign_ge_70pct"] = int(
            np.sum(reaction_sub["stable_same_sign_ge_70pct"])
        )
        row["reaction_features_stable_same_sign_ge_90pct"] = int(
            np.sum(reaction_sub["stable_same_sign_ge_90pct"])
        )
        rows.append(row)
    return pd.DataFrame(rows)


def fit_full_source_model(
    X: np.ndarray,
    task_meta: pd.DataFrame,
    feature_meta: pd.DataFrame,
    C: float,
    c_dir: Path,
    max_iter: int,
    tol: float,
    n_jobs: int,
    model_seed: int,
    coef_eps: float,
    resume: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    model_dir = c_dir / "full_source_model"
    model_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = model_dir / "full_source_model.npz"
    metadata_path = model_dir / "full_source_model.json"
    selected_path = model_dir / "full_source_selected_coefficients.csv"

    if resume and artifact_path.exists() and metadata_path.exists() and selected_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if not math.isclose(float(metadata.get("C", np.nan)), float(C), rel_tol=0.0, abs_tol=1e-15):
            raise ValueError(f"C mismatch in resumed full-source model: {metadata_path}")
        with np.load(artifact_path, allow_pickle=False) as arrays:
            coef = arrays["coef"].astype(np.float32, copy=True)
        print(f"[resume] Reusing full-source model for {c_tag(C)}", flush=True)
        return coef, metadata

    y = task_meta["label"].to_numpy(dtype=int)
    scaler = StandardScaler(with_mean=True, with_std=True, copy=True)
    Xz = scaler.fit_transform(X).astype(np.float32, copy=False)
    classifier = LogisticRegression(
        penalty="l1",
        solver="saga",
        class_weight="balanced",
        C=float(C),
        max_iter=int(max_iter),
        tol=float(tol),
        random_state=int(model_seed),
        n_jobs=int(n_jobs),
    )
    convergence_warning = False
    messages: list[str] = []
    started = time.time()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        classifier.fit(Xz, y)
        for warning in caught:
            if isinstance(warning.message, ConvergenceWarning):
                convergence_warning = True
                messages.append(str(warning.message))

    beta = classifier.coef_[0].astype(np.float32, copy=False)
    selected = np.abs(beta) > coef_eps
    n_edges = X.shape[1] // 2
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": now_iso(),
        "purpose": "Full IBDMDB source-cohort fit for later locked external evaluation; not an internal performance estimate",
        "C": float(C),
        "model_seed": int(model_seed),
        "samples": int(len(task_meta)),
        "participants": int(task_meta["participant_id"].nunique()),
        "selected_total": int(np.sum(selected)),
        "selected_B": int(np.sum(selected[:n_edges])),
        "selected_K0": int(np.sum(selected[n_edges:])),
        "abs_coef_mass_total": float(np.sum(np.abs(beta), dtype=np.float64)),
        "abs_coef_mass_B": float(np.sum(np.abs(beta[:n_edges]), dtype=np.float64)),
        "abs_coef_mass_K0": float(np.sum(np.abs(beta[n_edges:]), dtype=np.float64)),
        "n_iter": int(classifier.n_iter_[0]),
        "convergence_warning": bool(convergence_warning),
        "convergence_warning_messages": messages,
        "elapsed_seconds": float(time.time() - started),
    }
    atomic_savez_compressed(
        artifact_path,
        schema_version=np.asarray([SCHEMA_VERSION], dtype=np.int32),
        coef=beta,
        intercept=np.asarray([classifier.intercept_[0]], dtype=np.float32),
        scaler_mean=scaler.mean_.astype(np.float32),
        scaler_scale=scaler.scale_.astype(np.float32),
    )
    write_json(metadata_path, metadata)

    selected_df = feature_meta.loc[selected].copy()
    selected_df.insert(0, "C", float(C))
    selected_df["coefficient_standardised"] = beta[selected]
    selected_df["absolute_coefficient"] = np.abs(beta[selected])
    selected_df = selected_df.sort_values("absolute_coefficient", ascending=False)
    atomic_to_csv(selected_df, selected_path, index=False)
    print(
        f"[full-source] {c_tag(C)} selected={metadata['selected_total']} "
        f"(B={metadata['selected_B']}, K0={metadata['selected_K0']}) "
        f"n_iter={metadata['n_iter']} conv_warning={convergence_warning}",
        flush=True,
    )
    return beta, metadata


def aggregate_repetition_performance(
    C: float,
    fold_results: pd.DataFrame,
    all_predictions: pd.DataFrame,
    n_repeats: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pooled_rows: list[dict[str, Any]] = []
    fold_mean_rows: list[dict[str, Any]] = []
    participant_prediction_rows: list[pd.DataFrame] = []

    sample_metric_names = [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "positive_f1",
        "roc_auc",
        "brier",
        "log_loss",
    ]

    for repeat_idx in range(1, n_repeats + 1):
        rep_predictions = all_predictions[all_predictions["repeat"] == repeat_idx].copy()
        duplicate_samples = rep_predictions["sample_id"].duplicated().sum()
        if duplicate_samples:
            raise RuntimeError(
                f"Repeat {repeat_idx} has {duplicate_samples} duplicate OOF sample predictions"
            )
        sample_metrics = safe_binary_metrics(
            rep_predictions["true_label"].to_numpy(dtype=int),
            rep_predictions["proba_ibd"].to_numpy(dtype=float),
        )
        participant_predictions = participant_prediction_table(rep_predictions)
        participant_predictions.insert(0, "C", float(C))
        participant_predictions.insert(1, "repeat", repeat_idx)
        participant_prediction_rows.append(participant_predictions)
        participant_metrics = safe_binary_metrics(
            participant_predictions["true_label"].to_numpy(dtype=int),
            participant_predictions["proba_ibd"].to_numpy(dtype=float),
        )

        row: dict[str, Any] = {
            "C": float(C),
            "repeat": repeat_idx,
            "oof_samples": int(len(rep_predictions)),
            "oof_participants": int(rep_predictions["participant_id"].nunique()),
        }
        for prefix, metrics in [
            ("sample", sample_metrics),
            ("participant", participant_metrics),
        ]:
            for metric_name, value in metrics.items():
                row[f"{prefix}_{metric_name}"] = value
        pooled_rows.append(row)

        rep_folds = fold_results[fold_results["repeat"] == repeat_idx]
        fold_mean_row: dict[str, Any] = {
            "C": float(C),
            "repeat": repeat_idx,
            "n_folds": int(len(rep_folds)),
        }
        for level in ["sample", "participant"]:
            for metric in sample_metric_names:
                column = f"{level}_{metric}"
                fold_mean_row[f"mean_fold_{column}"] = float(rep_folds[column].mean())
                fold_mean_row[f"sd_fold_{column}"] = float(rep_folds[column].std(ddof=1))
        fold_mean_rows.append(fold_mean_row)

    pooled_df = pd.DataFrame(pooled_rows)
    fold_mean_df = pd.DataFrame(fold_mean_rows)
    participant_predictions_df = pd.concat(participant_prediction_rows, ignore_index=True)

    summary_row: dict[str, Any] = {
        "C": float(C),
        "n_repetitions": int(n_repeats),
        "n_fold_models": int(len(fold_results)),
        "convergence_warning_folds": int(fold_results["convergence_warning"].sum()),
    }
    for column in [
        c
        for c in pooled_df.columns
        if c.startswith("sample_") or c.startswith("participant_")
    ]:
        if column.endswith(("_tn", "_fp", "_fn", "_tp")):
            continue
        summary_row.update(numeric_distribution(pooled_df[column].to_numpy(), column))
    for column in [
        "selected_total",
        "selected_B",
        "selected_K0",
        "abs_coef_mass_total",
        "abs_coef_mass_B",
        "abs_coef_mass_K0",
        "n_iter",
        "elapsed_seconds",
    ]:
        summary_row.update(numeric_distribution(fold_results[column].to_numpy(), column))
    performance_summary = pd.DataFrame([summary_row])
    return pooled_df, fold_mean_df, participant_predictions_df, performance_summary


def write_top_reaction_tables(reaction_summary: pd.DataFrame, c_dir: Path, top_n: int) -> None:
    selected = reaction_summary[reaction_summary["selection_count"] > 0].copy()
    stable_sort = selected.sort_values(
        [
            "selection_frequency",
            "sign_consistency_when_selected",
            "stability_weighted_abs_coefficient",
            "heldout_abs_push_participant_balanced_median",
        ],
        ascending=[False, False, False, False],
    )
    atomic_to_csv(stable_sort.head(top_n), c_dir / "top_stable_reactions.csv", index=False)

    for feature_type in ["B", "K0"]:
        block = selected[selected["feature_type"] == feature_type]
        atomic_to_csv(
            block.sort_values(
                ["selection_frequency", "sign_consistency_when_selected", "stability_weighted_abs_coefficient"],
                ascending=[False, False, False],
            ).head(top_n),
            c_dir / f"top_stable_{feature_type}_reactions.csv",
            index=False,
        )
        atomic_to_csv(
            block.sort_values("coef_all_including_zero_mean", ascending=False).head(top_n),
            c_dir / f"top_positive_mean_{feature_type}_reactions.csv",
            index=False,
        )
        atomic_to_csv(
            block.sort_values("coef_all_including_zero_mean", ascending=True).head(top_n),
            c_dir / f"top_negative_mean_{feature_type}_reactions.csv",
            index=False,
        )


def build_c_summary_tables(
    C: float,
    bundles: list[FoldBundle],
    task_meta: pd.DataFrame,
    feature_meta: pd.DataFrame,
    c_dir: Path,
    n_repeats: int,
    coef_eps: float,
    full_coef: np.ndarray,
    top_n: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    bundles = sorted(bundles, key=lambda b: (b.key.repeat, b.key.fold))
    fold_results = pd.DataFrame([bundle.result for bundle in bundles])
    all_predictions = pd.concat([bundle.predictions for bundle in bundles], ignore_index=True)

    coefficients = np.vstack([bundle.coef for bundle in bundles]).astype(np.float32)
    abs_push = np.vstack([bundle.abs_push_participant_mean for bundle in bundles]).astype(np.float32)
    signed_push = np.vstack([bundle.signed_push_participant_mean for bundle in bundles]).astype(np.float32)
    abs_push_nonibd = np.vstack(
        [bundle.abs_push_nonibd_participant_mean for bundle in bundles]
    ).astype(np.float32)
    abs_push_ibd = np.vstack([bundle.abs_push_ibd_participant_mean for bundle in bundles]).astype(
        np.float32
    )
    signed_push_nonibd = np.vstack(
        [bundle.signed_push_nonibd_participant_mean for bundle in bundles]
    ).astype(np.float32)
    signed_push_ibd = np.vstack(
        [bundle.signed_push_ibd_participant_mean for bundle in bundles]
    ).astype(np.float32)

    expected_models = n_repeats * fold_results["fold"].nunique()
    if len(fold_results) != expected_models:
        raise RuntimeError(f"Expected {expected_models} fold models for C={C}, found {len(fold_results)}")

    atomic_to_csv(fold_results, c_dir / "fold_results_100_models.csv", index=False)
    atomic_to_csv(all_predictions, c_dir / "cross_validated_sample_predictions.csv", index=False)

    pooled_df, fold_mean_df, participant_predictions_df, performance_summary = (
        aggregate_repetition_performance(C, fold_results, all_predictions, n_repeats)
    )
    atomic_to_csv(pooled_df, c_dir / "repetition_pooled_oof_results.csv", index=False)
    atomic_to_csv(fold_mean_df, c_dir / "repetition_mean_fold_results.csv", index=False)
    atomic_to_csv(
        participant_predictions_df,
        c_dir / "cross_validated_participant_predictions.csv",
        index=False,
    )
    atomic_to_csv(performance_summary, c_dir / "performance_summary.csv", index=False)

    reaction_summary = build_reaction_summary(
        C=C,
        feature_meta=feature_meta,
        coefficients=coefficients,
        abs_push=abs_push,
        signed_push=signed_push,
        abs_push_nonibd=abs_push_nonibd,
        abs_push_ibd=abs_push_ibd,
        signed_push_nonibd=signed_push_nonibd,
        signed_push_ibd=signed_push_ibd,
        coef_eps=coef_eps,
        full_coef=full_coef,
    )
    atomic_to_csv(reaction_summary, c_dir / "reaction_coefficient_distributions.csv", index=False)
    atomic_to_csv(
        reaction_summary[reaction_summary["selection_count"] > 0].copy(),
        c_dir / "reaction_coefficient_distributions_selected_at_least_once.csv",
        index=False,
    )
    write_top_reaction_tables(reaction_summary, c_dir, top_n)

    process_fold_df = build_process_fold_distributions(
        C=C,
        fold_results=fold_results,
        feature_meta=feature_meta,
        coefficients=coefficients,
        abs_push=abs_push,
        signed_push=signed_push,
        abs_push_nonibd=abs_push_nonibd,
        abs_push_ibd=abs_push_ibd,
        signed_push_nonibd=signed_push_nonibd,
        signed_push_ibd=signed_push_ibd,
        coef_eps=coef_eps,
    )
    process_summary = summarise_process_distributions(process_fold_df, reaction_summary)
    atomic_to_csv(process_fold_df, c_dir / "process_fold_distributions.csv", index=False)
    atomic_to_csv(process_summary, c_dir / "process_summary_distributions.csv", index=False)

    process_compact_columns = [
        "C",
        "block",
        "process",
        "n_fold_models",
        "eligible_features",
        "selected_features_mean",
        "selected_features_sd",
        "selected_features_q025",
        "selected_features_median",
        "selected_features_q975",
        "fit_has_any_selected_mean",
        "abs_coefficient_mass_mean",
        "abs_coefficient_mass_sd",
        "abs_coefficient_mass_q025",
        "abs_coefficient_mass_median",
        "abs_coefficient_mass_q975",
        "signed_coefficient_mass_median",
        "heldout_abs_push_participant_balanced_mean",
        "heldout_abs_push_participant_balanced_sd",
        "heldout_abs_push_participant_balanced_q025",
        "heldout_abs_push_participant_balanced_median",
        "heldout_abs_push_participant_balanced_q975",
        "heldout_signed_push_nonibd_participant_balanced_median",
        "heldout_signed_push_ibd_participant_balanced_median",
        "heldout_signed_push_ibd_minus_nonibd_median",
        "reaction_features_selected_ge_50pct",
        "reaction_features_selected_ge_70pct",
        "reaction_features_selected_ge_90pct",
        "reaction_features_stable_same_sign_ge_50pct",
        "reaction_features_stable_same_sign_ge_70pct",
        "reaction_features_stable_same_sign_ge_90pct",
    ]
    atomic_to_csv(
        process_summary.loc[:, process_compact_columns],
        c_dir / "process_summary_compact.csv",
        index=False,
    )

    process_ranked = process_summary.sort_values(
        [
            "block",
            "heldout_abs_push_participant_balanced_median",
            "abs_coefficient_mass_median",
        ],
        ascending=[True, False, False],
    )
    atomic_to_csv(process_ranked, c_dir / "process_summary_ranked.csv", index=False)

    compact_columns = [
        "C",
        "feature_index",
        "feature_type",
        "edge",
        "edge_display",
        "process",
        "selection_frequency",
        "sign_consistency_when_selected",
        "dominant_sign",
        "coef_all_including_zero_mean",
        "coef_all_including_zero_median",
        "coef_selected_median",
        "abs_coef_all_including_zero_mean",
        "heldout_abs_push_participant_balanced_median",
        "heldout_signed_push_nonibd_participant_balanced_median",
        "heldout_signed_push_ibd_participant_balanced_median",
        "heldout_signed_push_ibd_minus_nonibd_median",
        "stable_same_sign_ge_50pct",
        "stable_same_sign_ge_70pct",
        "stable_same_sign_ge_90pct",
        "full_source_coefficient_standardised",
        "full_source_selected",
    ]
    atomic_to_csv(
        reaction_summary.loc[:, compact_columns],
        c_dir / "reaction_summary_compact.csv",
        index=False,
    )

    return performance_summary, reaction_summary, process_summary


def paired_c_comparisons(repetition_results: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "sample_accuracy",
        "sample_balanced_accuracy",
        "sample_macro_f1",
        "sample_roc_auc",
        "sample_brier",
        "participant_accuracy",
        "participant_balanced_accuracy",
        "participant_macro_f1",
        "participant_roc_auc",
        "participant_brier",
    ]
    C_values = sorted(repetition_results["C"].unique())
    rows: list[dict[str, Any]] = []
    for i, C_a in enumerate(C_values):
        for C_b in C_values[i + 1 :]:
            a = repetition_results[repetition_results["C"] == C_a].set_index("repeat")
            b = repetition_results[repetition_results["C"] == C_b].set_index("repeat")
            common = a.index.intersection(b.index)
            for metric in metrics:
                delta = b.loc[common, metric].to_numpy(dtype=float) - a.loc[common, metric].to_numpy(
                    dtype=float
                )
                row = {
                    "C_a": float(C_a),
                    "C_b": float(C_b),
                    "metric": metric,
                    "definition": "C_b minus C_a on the same repeated split",
                    "higher_is_better": not metric.endswith("brier"),
                    "n_paired_repetitions": int(len(common)),
                    "C_b_wins": int(np.sum(delta < 0) if metric.endswith("brier") else np.sum(delta > 0)),
                    "ties": int(np.sum(np.isclose(delta, 0.0, atol=1e-12, rtol=0.0))),
                }
                row.update(numeric_distribution(delta, "paired_delta"))
                rows.append(row)
    return pd.DataFrame(rows)


def write_global_outputs(
    output_dir: Path,
    performance_summaries: list[pd.DataFrame],
    repetition_results: list[pd.DataFrame],
    full_model_metadata: list[dict[str, Any]],
) -> None:
    all_performance = pd.concat(performance_summaries, ignore_index=True)
    all_repetitions = pd.concat(repetition_results, ignore_index=True)
    atomic_to_csv(all_performance, output_dir / "all_C_performance_summary.csv", index=False)
    atomic_to_csv(all_repetitions, output_dir / "all_C_repetition_pooled_oof_results.csv", index=False)
    atomic_to_csv(
        paired_c_comparisons(all_repetitions),
        output_dir / "paired_C_performance_comparisons.csv",
        index=False,
    )
    atomic_to_csv(
        pd.DataFrame(full_model_metadata),
        output_dir / "all_C_full_source_model_inventory.csv",
        index=False,
    )

    compact_rows: list[dict[str, Any]] = []
    for _, row in all_performance.iterrows():
        compact_rows.append(
            {
                "C": row["C"],
                "sample_accuracy_mean": row.get("sample_accuracy_mean"),
                "sample_accuracy_sd": row.get("sample_accuracy_sd"),
                "sample_balanced_accuracy_mean": row.get("sample_balanced_accuracy_mean"),
                "sample_balanced_accuracy_sd": row.get("sample_balanced_accuracy_sd"),
                "sample_macro_f1_mean": row.get("sample_macro_f1_mean"),
                "sample_macro_f1_sd": row.get("sample_macro_f1_sd"),
                "sample_roc_auc_mean": row.get("sample_roc_auc_mean"),
                "sample_roc_auc_sd": row.get("sample_roc_auc_sd"),
                "participant_balanced_accuracy_mean": row.get(
                    "participant_balanced_accuracy_mean"
                ),
                "participant_balanced_accuracy_sd": row.get("participant_balanced_accuracy_sd"),
                "participant_roc_auc_mean": row.get("participant_roc_auc_mean"),
                "participant_roc_auc_sd": row.get("participant_roc_auc_sd"),
                "selected_total_median": row.get("selected_total_median"),
                "selected_B_median": row.get("selected_B_median"),
                "selected_K0_median": row.get("selected_K0_median"),
                "convergence_warning_folds": row.get("convergence_warning_folds"),
            }
        )
    atomic_to_csv(
        pd.DataFrame(compact_rows),
        output_dir / "all_C_performance_compact.csv",
        index=False,
    )


def run(args: argparse.Namespace) -> None:
    feature_dir = Path(args.feature_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    annotation_csv = (
        Path(args.edge_annotation_csv).expanduser().resolve()
        if args.edge_annotation_csv
        else None
    )

    C_values = [float(value) for value in args.c_values]
    if len(set(C_values)) != len(C_values):
        raise ValueError("--c-values contains duplicates")
    if any(value <= 0 for value in C_values):
        raise ValueError("All C values must be positive")
    if args.n_repeats < 1 or args.n_splits < 2:
        raise ValueError("n_repeats must be >=1 and n_splits must be >=2")

    config = vars(args).copy()
    config.update(
        {
            "schema_version": SCHEMA_VERSION,
            "created_utc": now_iso(),
            "feature_dir_resolved": str(feature_dir),
            "output_dir_resolved": str(output_dir),
            "C_values": C_values,
            "task": "IBD (UC + CD) versus non-IBD",
            "feature_definition": "[B | K0], B=edge presence and K0=Ricci curvature if active else zero",
            "classifier": "Fold-internal StandardScaler(with_mean=True, with_std=True) followed by L1 LogisticRegression(solver=saga, class_weight=balanced)",
            "primary_performance_distribution": "20 pooled out-of-fold repetition estimates; 100 fold scores retained as descriptive",
            "reaction_interpretation": "Coefficient and held-out participant-balanced realised-push distributions across repeated fold models",
            "process_interpretation": "Foldwise process coefficient summaries and held-out participant-balanced process contributions",
            "python_version": sys.version,
            "platform": platform.platform(),
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "sklearn_version": sklearn_version,
        }
    )
    write_json(output_dir / "run_config.json", config)

    X, task_meta, edge_meta, feature_meta = load_inputs(feature_dir, annotation_csv)
    atomic_to_csv(task_meta, output_dir / "task_metadata_used.csv", index=False)
    atomic_to_csv(edge_meta, output_dir / "edge_metadata_used.csv", index=False)
    atomic_to_csv(feature_meta, output_dir / "feature_metadata_used.csv", index=False)

    print("\n[input summary]", flush=True)
    print(f"  feature matrix shape: {X.shape}", flush=True)
    print(f"  dense memory: {X.nbytes / 1024**2:.1f} MiB", flush=True)
    print(f"  samples: {len(task_meta)}", flush=True)
    print(f"  participants: {task_meta['participant_id'].nunique()}", flush=True)
    participant_counts = (
        task_meta.drop_duplicates("participant_id")["label_name"].value_counts().sort_index()
    )
    sample_counts = task_meta["label_name"].value_counts().sort_index()
    print("  participant labels:")
    print(participant_counts.to_string())
    print("  sample labels:")
    print(sample_counts.to_string())
    annotation_fraction = float(np.mean(feature_meta["process"] != "Unannotated"))
    print(f"  annotated feature fraction: {annotation_fraction:.3f}", flush=True)
    if annotation_fraction == 0.0:
        print(
            "[warning] All process labels are Unannotated. Reaction tables will be valid, but process tables will contain one uninformative category.",
            flush=True,
        )

    sample_manifest, _ = generate_or_load_splits(
        task_meta=task_meta,
        output_dir=output_dir,
        n_repeats=args.n_repeats,
        n_splits=args.n_splits,
        split_seed=args.split_seed,
        resume=args.resume,
    )
    sample_to_index = {
        sample_id: idx for idx, sample_id in enumerate(task_meta["sample_id"].astype(str))
    }

    performance_summaries: list[pd.DataFrame] = []
    repetition_result_tables: list[pd.DataFrame] = []
    full_model_metadata: list[dict[str, Any]] = []

    for C in C_values:
        c_dir = output_dir / c_tag(C)
        c_dir.mkdir(parents=True, exist_ok=True)
        print("\n" + "=" * 100, flush=True)
        print(f"Running C={C} in {c_dir}", flush=True)
        print("=" * 100, flush=True)

        bundles: list[FoldBundle] = []
        for repeat_idx in range(1, args.n_repeats + 1):
            for fold_idx in range(1, args.n_splits + 1):
                train_idx, test_idx, split_seed = indices_for_fold(
                    sample_manifest,
                    sample_to_index,
                    repeat_idx,
                    fold_idx,
                )
                key = FoldKey(
                    repeat=repeat_idx,
                    fold=fold_idx,
                    split_seed=split_seed,
                    model_seed=model_seed_for(args.model_seed, repeat_idx, fold_idx),
                )
                bundle = fit_or_load_fold(
                    X=X,
                    task_meta=task_meta,
                    train_idx=train_idx,
                    test_idx=test_idx,
                    key=key,
                    C=C,
                    c_dir=c_dir,
                    max_iter=args.max_iter,
                    tol=args.tol,
                    n_jobs=args.n_jobs,
                    coef_eps=args.coef_eps,
                    resume=args.resume,
                )
                bundles.append(bundle)

        full_coef, full_metadata = fit_full_source_model(
            X=X,
            task_meta=task_meta,
            feature_meta=feature_meta,
            C=C,
            c_dir=c_dir,
            max_iter=args.max_iter,
            tol=args.tol,
            n_jobs=args.n_jobs,
            model_seed=args.model_seed + int(round(C * 1_000_000)) + 999_983,
            coef_eps=args.coef_eps,
            resume=args.resume,
        )
        full_model_metadata.append(full_metadata)

        performance_summary, _, _ = build_c_summary_tables(
            C=C,
            bundles=bundles,
            task_meta=task_meta,
            feature_meta=feature_meta,
            c_dir=c_dir,
            n_repeats=args.n_repeats,
            coef_eps=args.coef_eps,
            full_coef=full_coef,
            top_n=args.top_n,
        )
        performance_summaries.append(performance_summary)
        repetition_result_tables.append(
            pd.read_csv(c_dir / "repetition_pooled_oof_results.csv")
        )

    write_global_outputs(
        output_dir=output_dir,
        performance_summaries=performance_summaries,
        repetition_results=repetition_result_tables,
        full_model_metadata=full_model_metadata,
    )

    write_json(
        output_dir / "RUN_COMPLETE.json",
        {
            "schema_version": SCHEMA_VERSION,
            "completed_utc": now_iso(),
            "C_values": C_values,
            "n_repeats": args.n_repeats,
            "n_splits": args.n_splits,
            "n_fold_models_total": len(C_values) * args.n_repeats * args.n_splits,
            "output_dir": str(output_dir),
        },
    )
    print("\n[READY] Repeated IBD-vs-non-IBD C-path analysis finished.", flush=True)
    print(f"Output directory: {output_dir}", flush=True)
    print(f"Primary summary: {output_dir / 'all_C_performance_compact.csv'}", flush=True)
    print(
        f"Paired C comparisons: {output_dir / 'paired_C_performance_comparisons.csv'}",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Repeated participant-grouped IBD-vs-non-IBD Ricci L1 C-path analysis",
    )
    parser.add_argument(
        "--feature-dir",
        default=str(Path.home() / "Real_Data" / "Ricci_Classifier_Faithful_Eps0001_n250_v3"),
        help="Directory containing feature_matrix_B_K0.npz, matched_metadata.csv, edge_metadata.csv",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path.home() / "Real_Data" / "Ricci_IBD_RepeatedCV_CPath"),
    )
    parser.add_argument(
        "--edge-annotation-csv",
        default=None,
        help="Optional CSV with edge-to-process annotations",
    )
    parser.add_argument(
        "--c-values",
        nargs="+",
        type=float,
        default=[0.005, 0.01, 0.02, 0.05, 0.1],
    )
    parser.add_argument("--n-repeats", type=int, default=20)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--split-seed", type=int, default=13)
    parser.add_argument("--model-seed", type=int, default=20260717)
    parser.add_argument("--max-iter", type=int, default=10000)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument(
        "--coef-eps",
        type=float,
        default=1e-12,
        help="Absolute coefficient threshold used to define selected/nonzero",
    )
    parser.add_argument("--top-n", type=int, default=100)
    parser.set_defaults(resume=True)
    parser.add_argument("--resume", dest="resume", action="store_true")
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
