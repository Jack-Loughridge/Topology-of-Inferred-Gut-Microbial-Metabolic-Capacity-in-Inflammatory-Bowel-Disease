#!/usr/bin/env python3
"""Deploy the corrected frozen IBDMDB H0 Alpha-Pi outer-model ensemble.

This script performs two deliberately distinct analyses:

1. v1 frozen-support Serrano-Gomez H0 diagrams (strict external validation).
2. v2 expanded-community, external-normalized H0 diagrams (exploratory
   structural replication; not strict external validation).

The five corrected repeat-01 outer-fold models are applied without refitting.
Each model retains its own outer-training-only adaptive bounds.  Probabilities
are computed before any external labels or participant identifiers are loaded.
The primary ensemble probability is the unweighted mean of the five frozen
softmax IBD probabilities.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import shutil
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import sklearn
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    recall_score,
    roc_auc_score,
)


SCRIPT_VERSION = "1.1.0"
EXPECTED_SOURCE_SHA256 = "97c0e15e3e01eb2461c640ac083970780f9a4cf88aa6fdcf92ed092976ac8381"
EXPECTED_COUNTS = {"v1": 90, "v2": 184}
THRESHOLD = 0.5

EXPECTED_MODELS = [
    {
        "fold": 1,
        "sha256": "2b997ede2d01a8a88a0dada0b987569d9f824b405fd139a250f3d7036369815d",
        "relative": "folds/repeat_01_fold_1/final_outer_model/final_outer_model.pt",
    },
    {
        "fold": 2,
        "sha256": "38e4d1268c7b2c66eeff46cea9855832a1d62ffe07aee3493182ebf66fb7ebc2",
        "relative": "folds/repeat_01_fold_2/final_outer_model/final_outer_model.pt",
    },
    {
        "fold": 3,
        "sha256": "d9bf9adf065c6dbbbd10ddd0e53d089dbdc90813d95b8d8ff59591cd3b41523c",
        "relative": "folds/repeat_01_fold_3/final_outer_model/final_outer_model.pt",
    },
    {
        "fold": 4,
        "sha256": "1371c2f6d9397a0fe43eb9938b60d52a392f1bca96192c618d80b4cb082c5db9",
        "relative": "folds/repeat_01_fold_4/final_outer_model/final_outer_model.pt",
    },
    {
        "fold": 5,
        "sha256": "38d9ea048e5a74a467ba6772d8f2f515ac49f7cecd27f8fcfa7bd3bdfd98d7fb",
        "relative": "folds/repeat_01_fold_5/final_outer_model/final_outer_model.pt",
    },
]

MISSING_TOKENS = {
    "", "na", "nan", "none", "null", "missing", "unknown", "not available",
    "not_applicable", "not applicable", "healthy_control", "healthy control",
}


def die(message: str) -> None:
    raise RuntimeError(message)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def require_file(path: Path, label: str, expected_sha: str | None = None) -> None:
    if not path.is_file():
        die(f"Missing {label}: {path}")
    if path.stat().st_size <= 0:
        die(f"Empty {label}: {path}")
    if expected_sha is not None:
        found = sha256(path)
        if found != expected_sha:
            die(
                f"SHA-256 mismatch for {label}: expected {expected_sha}, "
                f"found {found}: {path}"
            )


def torch_load_package(path: Path) -> dict[str, Any]:
    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, dict):
        die(f"Model artifact is not a dictionary: {path}")
    return obj


def import_production_source(path: Path):
    require_file(path, "H0 Alpha-Pi production source", EXPECTED_SOURCE_SHA256)
    module_name = "locked_h0_alpha_pi_repeated_cv_source"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        die(f"Could not construct import specification for {path}")
    module = importlib.util.module_from_spec(spec)
    # Required for dataclasses whose annotations resolve through sys.modules.
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    for symbol in ["H0DatasetBundle", "make_loader", "AlphaPiModel", "TOL"]:
        if not hasattr(module, symbol):
            die(f"Locked production source lacks required symbol {symbol!r}")
    return module


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    label: str
    role: str
    h0_dir: Path
    metadata_path: Path
    metadata_sep: str
    sample_key_candidates: tuple[str, ...]


def discover_diagrams(spec: DatasetSpec, expected_n: int) -> tuple[list[str], dict[str, Path]]:
    if not spec.h0_dir.is_dir():
        die(f"Missing {spec.key} H0 directory: {spec.h0_dir}")
    paths: dict[str, Path] = {}
    for path in sorted(spec.h0_dir.glob("*_H0.npy")):
        sample_id = path.name[: -len("_H0.npy")]
        if not sample_id:
            die(f"Cannot derive sample ID from {path}")
        if sample_id in paths:
            die(f"Duplicate H0 diagram for {sample_id}: {paths[sample_id]} and {path}")
        require_file(path, f"{spec.key} H0 diagram for {sample_id}")
        paths[sample_id] = path
    sample_ids = sorted(paths)
    if len(sample_ids) != expected_n:
        die(f"Expected {expected_n} {spec.key} H0 diagrams, found {len(sample_ids)}")
    return sample_ids, paths


def load_deaths_label_free(
    sample_ids: Sequence[str], paths: dict[str, Path], tol: float
) -> tuple[list[np.ndarray], pd.DataFrame, str]:
    deaths_list: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    manifest_digest = hashlib.sha256()
    for sample_id in sample_ids:
        path = paths[sample_id]
        file_sha = sha256(path)
        manifest_digest.update(f"{sample_id}\t{file_sha}\n".encode("utf-8"))
        arr = np.asarray(np.load(path, allow_pickle=False))
        input_values = int(arr.size)
        if arr.size == 0:
            deaths_all = np.zeros(0, dtype=np.float32)
        elif arr.ndim == 1:
            deaths_all = arr.astype(np.float32, copy=False)
        elif arr.ndim == 2 and arr.shape[1] >= 2:
            deaths_all = arr[:, 1].astype(np.float32, copy=False)
        else:
            deaths_all = arr.ravel().astype(np.float32, copy=False)
        n_finite = int(np.isfinite(deaths_all).sum())
        deaths = deaths_all[np.isfinite(deaths_all)]
        deaths = deaths[(deaths > 0.0) & (deaths < (1.0 - tol))]
        deaths = deaths.astype(np.float32, copy=False)
        if not np.isfinite(deaths).all() or np.any(deaths <= 0.0) or np.any(deaths >= 1.0 - tol):
            die(f"Invalid retained H0 death values for {sample_id}")
        deaths_list.append(deaths)
        rows.append(
            {
                "sample_id": sample_id,
                "array_shape": str(tuple(arr.shape)),
                "input_array_values": input_values,
                "finite_death_candidates": n_finite,
                "retained_deaths_0_lt_d_lt_1_minus_tol": int(len(deaths)),
                "h0_sha256": file_sha,
            }
        )
    return deaths_list, pd.DataFrame(rows), manifest_digest.hexdigest()


def make_label_free_bundle(source, sample_ids: Sequence[str], deaths_list: list[np.ndarray]):
    n = len(sample_ids)
    dummy_participants = [f"label-firewall::{sid}" for sid in sample_ids]
    dummy_labels = np.zeros(n, dtype=int)
    return source.H0DatasetBundle(
        sample_ids=list(sample_ids),
        participant_ids=dummy_participants,
        labels=dummy_labels,
        label_names=["withheld"] * n,
        conditions=["withheld"] * n,
        deaths_list=deaths_list,
        pd_paths={},
        metadata=pd.DataFrame({"sample_id": sample_ids}),
        sample_to_index={sid: idx for idx, sid in enumerate(sample_ids)},
    )


def validate_package(package: dict[str, Any], expected_fold: int, path: Path) -> dict[str, Any]:
    required = {
        "schema_version", "state_dict", "bounds", "sigma_log", "quad_points",
        "point_chunk", "label_order", "prediction_rule", "metadata",
    }
    missing = sorted(required - set(package))
    if missing:
        die(f"Model package {path} lacks keys: {missing}")
    if int(package["schema_version"]) != 2:
        die(f"Expected schema_version 2 in {path}, found {package['schema_version']!r}")
    if list(package["label_order"]) != ["non-IBD", "IBD"]:
        die(f"Unexpected label order in {path}: {package['label_order']!r}")
    expected_rule = "softmax(logits), argmax; IBD probability is column 1"
    if str(package["prediction_rule"]) != expected_rule:
        die(f"Unexpected prediction rule in {path}: {package['prediction_rule']!r}")
    metadata = dict(package["metadata"])
    if metadata.get("stage") != "outer_final_refit":
        die(f"Model is not an outer final refit: {path}")
    if int(metadata.get("repeat", -1)) != 1 or int(metadata.get("fold", -1)) != expected_fold:
        die(f"Unexpected repeat/fold metadata in {path}: {metadata}")
    if metadata.get("bounds_source") != "all outer-training H0 deaths only":
        die(f"Bounds are not certified outer-training-only in {path}: {metadata}")
    bounds = np.asarray(package["bounds"], dtype=np.float32)
    if bounds.ndim != 1 or len(bounds) < 2:
        die(f"Invalid bounds in {path}: shape={bounds.shape}")
    if not np.isfinite(bounds).all() or np.any(np.diff(bounds) <= 0):
        die(f"Bounds must be finite and strictly increasing in {path}")
    if float(bounds[0]) != 0.0 or float(bounds[-1]) != 1.0:
        die(f"Bounds must start at 0 and end at 1 in {path}")
    state = package["state_dict"]
    if not isinstance(state, dict):
        die(f"state_dict is not a dictionary in {path}")
    n_intervals = len(bounds) - 1
    expected_shapes = {
        "raw_alpha": (n_intervals,),
        "centers": (2, n_intervals),
        "x_points": (n_intervals, int(package["quad_points"])),
        "w_points": (n_intervals, int(package["quad_points"])),
        "clf_head.weight": (2, n_intervals),
        "clf_head.bias": (2,),
    }
    for key, shape in expected_shapes.items():
        if key not in state or tuple(state[key].shape) != shape:
            found = None if key not in state else tuple(state[key].shape)
            die(f"State shape mismatch for {key} in {path}: expected {shape}, found {found}")
        if torch.is_floating_point(state[key]) and not torch.isfinite(state[key]).all():
            die(f"Non-finite state tensor {key} in {path}")
    return {
        "fold": expected_fold,
        "n_intervals": n_intervals,
        "sigma_log": float(package["sigma_log"]),
        "quad_points": int(package["quad_points"]),
        "point_chunk": int(package["point_chunk"]),
        "metadata": metadata,
    }


def load_model(source, package: dict[str, Any], device: torch.device):
    bounds = np.asarray(package["bounds"], dtype=np.float32)
    model = source.AlphaPiModel(
        n_intervals=len(bounds) - 1,
        n_classes=2,
        bounds=bounds,
        sigma_log=float(package["sigma_log"]),
        quad_points=int(package["quad_points"]),
        point_chunk=int(package["point_chunk"]),
        device=device,
    ).to(device)
    model.load_state_dict(package["state_dict"], strict=True)
    model.eval()
    return model, bounds


def predict_one_model(
    source,
    bundle,
    package: dict[str, Any],
    device: torch.device,
    batch_size: int,
    seed: int,
) -> np.ndarray:
    model, bounds = load_model(source, package, device)
    loader = source.make_loader(
        bundle=bundle,
        indices=np.arange(len(bundle.sample_ids), dtype=int),
        bounds=bounds,
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
        num_workers=0,
    )
    pieces: list[np.ndarray] = []
    with torch.no_grad():
        for deaths, bins, mask, _labels, _sids, _pids, _indices in loader:
            deaths = deaths.to(device)
            bins = bins.to(device)
            mask = mask.to(device)
            _z, logits = model(deaths, bins, mask)
            probabilities = F.softmax(logits, dim=1)
            if probabilities.ndim != 2 or probabilities.shape[1] != 2:
                die(f"Frozen model returned probability shape {tuple(probabilities.shape)}")
            max_simplex_error = float(torch.max(torch.abs(probabilities.sum(dim=1) - 1.0)).item())
            tolerance = 32.0 * float(torch.finfo(probabilities.dtype).eps)
            if max_simplex_error > tolerance:
                die(
                    f"Probability simplex failure: max error {max_simplex_error:.6g}, "
                    f"tolerance {tolerance:.6g}"
                )
            pieces.append(probabilities[:, 1].detach().cpu().numpy().astype(np.float64))
    probability = np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float64)
    if len(probability) != len(bundle.sample_ids):
        die(f"Prediction count mismatch: expected {len(bundle.sample_ids)}, found {len(probability)}")
    if not np.isfinite(probability).all() or np.any(probability < 0) or np.any(probability > 1):
        die("Frozen model emitted invalid IBD probabilities")
    return probability


def predict_ensemble_label_free(
    source,
    model_root: Path,
    sample_ids: Sequence[str],
    deaths_list: list[np.ndarray],
    device: torch.device,
    batch_size: int,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    bundle = make_label_free_bundle(source, sample_ids, deaths_list)
    output = pd.DataFrame({"sample_id": list(sample_ids)})
    model_audit: list[dict[str, Any]] = []
    fold_columns: list[str] = []
    for expected in EXPECTED_MODELS:
        fold = int(expected["fold"])
        path = model_root / expected["relative"]
        require_file(path, f"fold {fold} frozen outer model", str(expected["sha256"]))
        package = torch_load_package(path)
        audit = validate_package(package, fold, path)
        probability = predict_one_model(
            source, bundle, package, device=device, batch_size=batch_size, seed=10_000 + fold
        )
        column = f"probability_IBD_fold_{fold}"
        fold_columns.append(column)
        output[column] = probability
        model_audit.append(
            {
                **audit,
                "path": str(path),
                "sha256": str(expected["sha256"]),
                "probability_min": float(probability.min()),
                "probability_median": float(np.median(probability)),
                "probability_max": float(probability.max()),
            }
        )
        print(
            f"    fold {fold}: intervals={audit['n_intervals']}, "
            f"p_IBD min/median/max={probability.min():.4f}/"
            f"{np.median(probability):.4f}/{probability.max():.4f}",
            flush=True,
        )
    matrix = output[fold_columns].to_numpy(dtype=float)
    output["probability_IBD"] = matrix.mean(axis=1)
    output["fold_probability_sd_population"] = matrix.std(axis=1, ddof=0)
    output["fold_probability_min"] = matrix.min(axis=1)
    output["fold_probability_max"] = matrix.max(axis=1)
    output["fold_probability_range"] = matrix.max(axis=1) - matrix.min(axis=1)
    output["predicted_binary"] = (output["probability_IBD"] >= THRESHOLD).astype(int)
    output["predicted_label"] = output["predicted_binary"].map({0: "nonIBD", 1: "IBD"})
    return output, model_audit


def read_metadata(spec: DatasetSpec) -> pd.DataFrame:
    require_file(spec.metadata_path, f"{spec.key} evaluation metadata")
    return pd.read_csv(spec.metadata_path, sep=spec.metadata_sep, low_memory=False)


def resolve_sample_key(metadata: pd.DataFrame, candidates: Iterable[str], sample_ids: Sequence[str]) -> str:
    wanted = set(map(str, sample_ids))
    scored: list[tuple[int, str]] = []
    for column in candidates:
        if column in metadata.columns:
            overlap = len(wanted & set(metadata[column].dropna().astype(str)))
            scored.append((overlap, column))
    if not scored:
        die(f"No candidate sample-ID column exists in metadata; tried {list(candidates)}")
    overlap, column = max(scored, key=lambda x: (x[0], x[1]))
    if overlap != len(wanted):
        die(f"Best metadata sample key {column!r} covers {overlap}/{len(wanted)} samples")
    return column


def parse_binary_label(value: Any) -> int:
    if pd.isna(value):
        die("Missing external IBD label")
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, (int, np.integer)) and int(value) in (0, 1):
        return int(value)
    if isinstance(value, (float, np.floating)) and np.isfinite(value) and float(value) in (0.0, 1.0):
        return int(value)
    text = str(value).strip().lower().replace("_", "").replace("-", "").replace(" ", "")
    if text in {"0", "nonibd", "control", "healthy", "healthycontrol", "hc"}:
        return 0
    if text in {"1", "ibd", "cd", "uc", "crohns", "crohnsdisease", "ulcerativecolitis"}:
        return 1
    die(f"Unsupported external IBD binary label: {value!r}")


def resolve_label_column(metadata: pd.DataFrame) -> str:
    candidates = [
        "external_label_ibd_binary",
        "PRJEB42155_ge50_clean_metadata_full__external_label_ibd_binary",
        "label",
        "y_binary",
    ]
    found = [column for column in candidates if column in metadata.columns]
    if not found:
        die(f"No external binary IBD label column found; tried {candidates}")
    return found[0]


def clean_identifier(value: Any) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if text.lower() in MISSING_TOKENS:
        return None
    return text


def resolve_participant_ids(frame: pd.DataFrame) -> tuple[pd.Series, list[str]]:
    candidates = [
        "host_subject_id",
        "PRJEB42155_ge50_clean_metadata_full__host_subject_id",
        "patient_id",
        "PRJEB42155_ge50_clean_metadata_full__patient_id",
        "participant_id",
        "subject_id",
    ]
    available = [column for column in candidates if column in frame.columns]
    resolved: list[str] = []
    for row in frame.itertuples(index=False, name=None):
        row_map = dict(zip(frame.columns, row))
        participant = None
        for column in available:
            participant = clean_identifier(row_map[column])
            if participant is not None:
                break
        if participant is None:
            participant = "sample::" + str(row_map["sample_id"])
        resolved.append(participant)
    return pd.Series(resolved, index=frame.index, dtype="object"), available


def binary_metrics(y_true: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=int)
    probability = np.asarray(probability, dtype=float)
    if len(y_true) != len(probability) or len(y_true) == 0:
        die("Invalid inputs to binary metrics")
    if set(np.unique(y_true)) != {0, 1}:
        die(f"Metrics require both classes; found {np.unique(y_true).tolist()}")
    predicted = (probability >= THRESHOLD).astype(int)
    cm = confusion_matrix(y_true, predicted, labels=[0, 1])
    clipped = np.clip(probability, np.finfo(float).eps, 1.0 - np.finfo(float).eps)
    counts = np.bincount(y_true, minlength=2)
    return {
        "n": int(len(y_true)),
        "class_counts": {"nonIBD": int(counts[0]), "IBD": int(counts[1])},
        "threshold": THRESHOLD,
        "accuracy": float(accuracy_score(y_true, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
        "macro_f1": float(f1_score(y_true, predicted, average="macro", zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "average_precision_ibd": float(average_precision_score(y_true, probability)),
        "brier": float(brier_score_loss(y_true, probability)),
        "log_loss": float(log_loss(y_true, clipped, labels=[0, 1])),
        "recall_nonIBD": float(recall_score(y_true, predicted, pos_label=0, zero_division=0)),
        "recall_IBD": float(recall_score(y_true, predicted, pos_label=1, zero_division=0)),
        "confusion_matrix": cm.astype(int).tolist(),
    }


def attach_labels_and_evaluate(
    spec: DatasetSpec, predictions: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any]]:
    # This function is intentionally called only after all model probabilities exist.
    metadata = read_metadata(spec)
    sample_key = resolve_sample_key(metadata, spec.sample_key_candidates, predictions["sample_id"])
    selected = metadata[metadata[sample_key].astype(str).isin(set(predictions["sample_id"]))].copy()
    selected[sample_key] = selected[sample_key].astype(str)
    if selected[sample_key].duplicated().any():
        duplicates = selected.loc[selected[sample_key].duplicated(False), sample_key].tolist()[:10]
        die(f"Duplicate metadata rows for {spec.key}: {duplicates}")
    evaluated = predictions.merge(
        selected, left_on="sample_id", right_on=sample_key, how="left", validate="one_to_one"
    )
    if len(evaluated) != len(predictions):
        die(f"Metadata merge changed row count for {spec.key}")
    label_column = resolve_label_column(evaluated)
    evaluated["y_binary"] = evaluated[label_column].map(parse_binary_label).astype(int)
    evaluated["true_label"] = evaluated["y_binary"].map({0: "nonIBD", 1: "IBD"})
    evaluated["participant_id_external"], participant_sources = resolve_participant_ids(evaluated)
    consistency = evaluated.groupby("participant_id_external")["y_binary"].nunique()
    inconsistent = consistency[consistency > 1]
    if len(inconsistent):
        die(
            f"Participants with inconsistent IBD labels in {spec.key}: "
            f"{inconsistent.index.astype(str).tolist()[:10]}"
        )
    sample_metrics = binary_metrics(evaluated["y_binary"], evaluated["probability_IBD"])
    participant = evaluated.groupby("participant_id_external", as_index=False).agg(
        probability_IBD=("probability_IBD", "mean"),
        y_binary=("y_binary", "first"),
        n_samples=("sample_id", "size"),
        fold_probability_sd_population=("fold_probability_sd_population", "mean"),
    )
    participant["true_label"] = participant["y_binary"].map({0: "nonIBD", 1: "IBD"})
    participant["predicted_binary"] = (participant["probability_IBD"] >= THRESHOLD).astype(int)
    participant["predicted_label"] = participant["predicted_binary"].map({0: "nonIBD", 1: "IBD"})
    participant_metrics = binary_metrics(participant["y_binary"], participant["probability_IBD"])
    audit = {
        "sample_key": sample_key,
        "label_column": label_column,
        "participant_columns_considered_in_order": participant_sources,
        "metadata_sha256": sha256(spec.metadata_path),
    }
    return evaluated, participant, sample_metrics, participant_metrics | {"metadata_audit": audit}


def confusion_frame(metrics: dict[str, Any]) -> pd.DataFrame:
    cm = np.asarray(metrics["confusion_matrix"], dtype=int)
    return pd.DataFrame(cm, index=["true_nonIBD", "true_IBD"], columns=["pred_nonIBD", "pred_IBD"])


def paired_comparison(v1: pd.DataFrame, v2: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    columns = ["sample_id", "y_binary", "probability_IBD", "predicted_binary"]
    paired = v1[columns].merge(
        v2[columns], on="sample_id", how="inner", validate="one_to_one", suffixes=("_v1", "_v2")
    )
    expected = EXPECTED_COUNTS["v1"]
    if len(paired) != expected:
        die(f"Expected {expected} paired v1/v2 samples, found {len(paired)}")
    if not np.array_equal(paired["y_binary_v1"].to_numpy(), paired["y_binary_v2"].to_numpy()):
        die("v1 and v2 labels disagree on paired samples")
    paired["probability_difference_v2_minus_v1"] = (
        paired["probability_IBD_v2"] - paired["probability_IBD_v1"]
    )
    paired["prediction_agrees"] = paired["predicted_binary_v1"] == paired["predicted_binary_v2"]
    y = paired["y_binary_v1"].to_numpy(dtype=int)
    v1_metrics = binary_metrics(y, paired["probability_IBD_v1"])
    v2_metrics = binary_metrics(y, paired["probability_IBD_v2"])
    diff = paired["probability_difference_v2_minus_v1"].to_numpy(dtype=float)
    summary = {
        "n": int(len(paired)),
        "v1_metrics_on_paired_samples": v1_metrics,
        "v2_metrics_on_same_paired_samples": v2_metrics,
        "metric_differences_v2_minus_v1": {
            key: float(v2_metrics[key] - v1_metrics[key])
            for key in ["accuracy", "balanced_accuracy", "macro_f1", "roc_auc", "average_precision_ibd", "brier", "log_loss"]
        },
        "probability_difference_v2_minus_v1": {
            "mean": float(np.mean(diff)),
            "median": float(np.median(diff)),
            "mean_absolute": float(np.mean(np.abs(diff))),
            "min": float(np.min(diff)),
            "max": float(np.max(diff)),
        },
        "prediction_agreement_fraction": float(paired["prediction_agrees"].mean()),
    }
    return paired, summary


def fmt(value: float) -> str:
    return f"{value:.3f}"


def latex_metrics_table(results: dict[str, dict[str, Any]], paired: dict[str, Any]) -> str:
    rows = [
        ("v1 frozen support", "Strict external", "Sample", results["v1"]["sample"]),
        ("v1 frozen support", "Strict external", "Participant", results["v1"]["participant_probability_averaged"]),
        ("v2 expanded community", "Exploratory replication", "Sample", results["v2"]["sample"]),
        ("v2 expanded community", "Exploratory replication", "Participant", results["v2"]["participant_probability_averaged"]),
        ("v2 on paired v1 samples", "Exploratory paired", "Sample", paired["v2_metrics_on_same_paired_samples"]),
    ]
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\caption{External IBD-versus-non-IBD performance of the frozen five-model H$_0$ Alpha--Pi ensemble.}",
        r"\label{tab:external_h0_alphapi}",
        r"\begin{tabular}{lllrrrrrrrr}",
        r"\toprule",
        r"Representation & Role & Unit & $n$ & Acc. & Bal. acc. & Macro F1 & ROC--AUC & AP$_{\mathrm{IBD}}$ & Rec. nonIBD & Rec. IBD \\",
        r"\midrule",
    ]
    for representation, role, unit, metric in rows:
        lines.append(
            f"{representation} & {role} & {unit} & {metric['n']} & "
            f"{fmt(metric['accuracy'])} & {fmt(metric['balanced_accuracy'])} & "
            f"{fmt(metric['macro_f1'])} & {fmt(metric['roc_auc'])} & "
            f"{fmt(metric['average_precision_ibd'])} & {fmt(metric['recall_nonIBD'])} & "
            f"{fmt(metric['recall_IBD'])} \\\\" 
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\begin{minipage}{0.98\linewidth}",
            r"\footnotesize",
            r"Probabilities are the unweighted mean of five corrected outer-training models, each using its own train-only adaptive interval bounds. Participant probabilities are grouped by host subject whenever that field is available. The v1 representation is the strict frozen-support external analysis. The v2 representation uses external-cohort support normalization and is therefore an exploratory structural replication, not a second strict external validation. AP denotes average precision for IBD.",
            r"\end{minipage}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_outputs(
    output_root: Path,
    predictions: dict[str, pd.DataFrame],
    participants: dict[str, pd.DataFrame],
    h0_audits: dict[str, pd.DataFrame],
    results: dict[str, dict[str, Any]],
    paired_frame: pd.DataFrame,
    paired_summary: dict[str, Any],
    provenance: dict[str, Any],
    overwrite: bool,
) -> None:
    output_root.parent.mkdir(parents=True, exist_ok=True)
    if output_root.exists():
        if not overwrite:
            die(f"Output directory exists; use --overwrite to replace via backup: {output_root}")
        backup = output_root.with_name(output_root.name + ".backup_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
        if backup.exists():
            die(f"Backup path already exists: {backup}")
        output_root.rename(backup)
        print(f"Existing output moved to recoverable backup: {backup}")
    temporary = Path(tempfile.mkdtemp(prefix=output_root.name + ".tmp_", dir=output_root.parent))
    try:
        for key in ["v1", "v2"]:
            sub = temporary / ("v1_frozen_support_strict" if key == "v1" else "v2_expanded_community_exploratory")
            sub.mkdir(parents=True)
            predictions[key].to_csv(sub / "sample_predictions.csv", index=False)
            participants[key].to_csv(sub / "participant_predictions_probability_averaged.csv", index=False)
            h0_audits[key].to_csv(sub / "h0_input_audit.csv", index=False)
            write_json(sub / "external_metrics.json", results[key])
            confusion_frame(results[key]["sample"]).to_csv(sub / "sample_confusion_matrix.csv")
            confusion_frame(results[key]["participant_probability_averaged"]).to_csv(
                sub / "participant_confusion_matrix_probability_averaged.csv"
            )
        paired_dir = temporary / "paired_v1_v2"
        paired_dir.mkdir()
        paired_frame.to_csv(paired_dir / "paired_sample_predictions.csv", index=False)
        write_json(paired_dir / "paired_comparison.json", paired_summary)
        write_json(temporary / "DEPLOYMENT_PROVENANCE.json", provenance)
        (temporary / "external_h0_alphapi_results_table.tex").write_text(
            latex_metrics_table(results, paired_summary)
        )
        complete = {
            "status": "complete",
            "completed_utc": utc_now(),
            "script_version": SCRIPT_VERSION,
            "v1_sample_predictions": int(len(predictions["v1"])),
            "v2_sample_predictions": int(len(predictions["v2"])),
            "frozen_models": len(EXPECTED_MODELS),
        }
        write_json(temporary / "RUN_COMPLETE.json", complete)
        os.replace(temporary, output_root)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def parse_args() -> argparse.Namespace:
    home = Path.home()
    external_root = home / "Real_Data/external_validation/serrano_gomez_ibd/ge50_external_validation"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--external-root", type=Path, default=external_root)
    parser.add_argument(
        "--model-root", type=Path,
        default=home / "Real_Data/H0_AlphaPi_RepeatedCV_TrainOnlyBins_IBD",
    )
    parser.add_argument(
        "--production-source", type=Path,
        default=home / "Real_Data/h0_alpha_pi_repeated_cv/h0_alpha_pi_repeated_cv.py",
    )
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        die("--batch-size must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        die("--device cuda requested but CUDA is unavailable")
    device = torch.device(args.device)
    ext = args.external_root.resolve()
    model_root = args.model_root.resolve()
    source_path = args.production_source.resolve()
    output_root = (
        args.output_root.resolve()
        if args.output_root is not None
        else ext / "external_prediction_frozen_h0_alphapi_outer_ensemble_v1"
    )

    specs = {
        "v1": DatasetSpec(
            key="v1",
            label="Frozen-support H0",
            role="strict external validation",
            h0_dir=ext / "structural_frozen_ibdmdb/h0",
            metadata_path=ext / "structural_frozen_ibdmdb/ricci_features_frozen_training_order/matched_metadata.csv",
            metadata_sep=",",
            sample_key_candidates=("sample_id", "run_accession", "sample_accession"),
        ),
        "v2": DatasetSpec(
            key="v2",
            label="Expanded-community H0",
            role="exploratory structural replication",
            h0_dir=ext / "expanded_community_structural_replication_v2/04_h0",
            metadata_path=ext / "expanded_community_structural_replication_v2/01_curated/matched_external_metadata_184.tsv.gz",
            metadata_sep="\t",
            sample_key_candidates=("sample_id_v2", "run_accession", "sample_id", "sample_accession"),
        ),
    }

    print("=" * 108)
    print("FROZEN IBDMDB H0 ALPHA-PI OUTER-MODEL ENSEMBLE — EXTERNAL DEPLOYMENT")
    print("=" * 108)
    print(f"Production source: {source_path}")
    print(f"Model root: {model_root}")
    print(f"Output root: {output_root}")
    print(f"Device: {device}")
    print("v1 role: strict frozen-support external validation")
    print("v2 role: exploratory expanded-community structural replication")
    print("Label firewall: probabilities are computed before external metadata is loaded")

    source = import_production_source(source_path)
    tol = float(source.TOL)
    if not np.isfinite(tol) or tol <= 0 or tol >= 1:
        die(f"Invalid TOL in production source: {tol}")
    print(f"Locked source SHA-256: {EXPECTED_SOURCE_SHA256}")
    print(f"Locked H0 death filter: 0 < death < 1 - {tol:g}")

    all_predictions: dict[str, pd.DataFrame] = {}
    h0_audits: dict[str, pd.DataFrame] = {}
    h0_manifest_hashes: dict[str, str] = {}
    model_audits_by_dataset: dict[str, list[dict[str, Any]]] = {}

    print("\n[PHASE 1] Label-free H0 loading and frozen prediction")
    for key in ["v1", "v2"]:
        spec = specs[key]
        sample_ids, paths = discover_diagrams(spec, EXPECTED_COUNTS[key])
        deaths_list, h0_audit, h0_manifest_sha = load_deaths_label_free(sample_ids, paths, tol)
        retained = h0_audit["retained_deaths_0_lt_d_lt_1_minus_tol"]
        print(
            f"\n{spec.label}: {len(sample_ids)} samples; retained deaths "
            f"min/median/max={retained.min()}/{retained.median():.1f}/{retained.max()}"
        )
        predictions, model_audit = predict_ensemble_label_free(
            source, model_root, sample_ids, deaths_list, device=device, batch_size=args.batch_size
        )
        print(
            f"    ensemble p_IBD min/median/max="
            f"{predictions['probability_IBD'].min():.4f}/"
            f"{predictions['probability_IBD'].median():.4f}/"
            f"{predictions['probability_IBD'].max():.4f}"
        )
        print(
            f"    fold disagreement SD median/max="
            f"{predictions['fold_probability_sd_population'].median():.4f}/"
            f"{predictions['fold_probability_sd_population'].max():.4f}"
        )
        all_predictions[key] = predictions
        h0_audits[key] = h0_audit
        h0_manifest_hashes[key] = h0_manifest_sha
        model_audits_by_dataset[key] = model_audit

    print("\nFrozen probabilities completed for both datasets without external labels: PASS")

    print("\n[PHASE 2] Post-prediction label merge and evaluation")
    evaluated: dict[str, pd.DataFrame] = {}
    participants: dict[str, pd.DataFrame] = {}
    results: dict[str, dict[str, Any]] = {}
    for key in ["v1", "v2"]:
        sample_frame, participant_frame, sample_metrics, participant_metrics = attach_labels_and_evaluate(
            specs[key], all_predictions[key]
        )
        metadata_audit = participant_metrics.pop("metadata_audit")
        evaluated[key] = sample_frame
        participants[key] = participant_frame
        results[key] = {
            "representation": specs[key].label,
            "methodological_role": specs[key].role,
            "sample": sample_metrics,
            "participant_probability_averaged": participant_metrics,
            "metadata_audit": metadata_audit,
            "fold_disagreement": {
                "sample_probability_sd_population_mean": float(
                    sample_frame["fold_probability_sd_population"].mean()
                ),
                "sample_probability_sd_population_median": float(
                    sample_frame["fold_probability_sd_population"].median()
                ),
                "sample_probability_sd_population_max": float(
                    sample_frame["fold_probability_sd_population"].max()
                ),
            },
        }
        print(f"\n{specs[key].label} — {specs[key].role}")
        print("Sample metrics:")
        print(json.dumps(sample_metrics, indent=2))
        print("Participant probability-averaged metrics:")
        print(json.dumps(participant_metrics, indent=2))

    paired_frame, paired_summary = paired_comparison(evaluated["v1"], evaluated["v2"])
    print("\nPaired v1/v2 comparison on the same 90 samples:")
    print(json.dumps(paired_summary, indent=2))

    script_path = Path(__file__).resolve()
    provenance = {
        "script_version": SCRIPT_VERSION,
        "script_path": str(script_path),
        "script_sha256": sha256(script_path),
        "created_utc": utc_now(),
        "method": {
            "ensemble": "unweighted arithmetic mean of five frozen repeat-01 outer-model IBD probabilities",
            "model_status": "corrected outer-training models; not a single full-source refit",
            "model_refitting": False,
            "external_rescaling": False,
            "threshold": THRESHOLD,
            "label_firewall": "all H0 loading, binning and probabilities completed before external metadata was read",
            "h0_death_filter": f"finite deaths satisfying 0 < d < 1 - TOL, TOL={tol}",
            "binning": "each model's own frozen float32 outer-training bounds; clip then searchsorted(side='right') - 1",
            "v1_role": specs["v1"].role,
            "v2_role": specs["v2"].role,
        },
        "production_source": {"path": str(source_path), "sha256": EXPECTED_SOURCE_SHA256},
        "models": model_audits_by_dataset["v1"],
        "model_audit_consistent_across_datasets": (
            [x["sha256"] for x in model_audits_by_dataset["v1"]]
            == [x["sha256"] for x in model_audits_by_dataset["v2"]]
        ),
        "inputs": {
            key: {
                "h0_dir": str(specs[key].h0_dir),
                "h0_manifest_sha256": h0_manifest_hashes[key],
                "n_samples": EXPECTED_COUNTS[key],
                "metadata_path": str(specs[key].metadata_path),
                "metadata_sha256": sha256(specs[key].metadata_path),
            }
            for key in ["v1", "v2"]
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "torch": torch.__version__,
            "sklearn": sklearn.__version__,
            "device": str(device),
        },
    }
    if not provenance["model_audit_consistent_across_datasets"]:
        die("Model audit differs between v1 and v2 prediction passes")

    if args.verify_only:
        print("\nVERIFY-ONLY COMPLETE: no result files written.")
        return

    write_outputs(
        output_root=output_root,
        predictions=evaluated,
        participants=participants,
        h0_audits=h0_audits,
        results=results,
        paired_frame=paired_frame,
        paired_summary=paired_summary,
        provenance=provenance,
        overwrite=args.overwrite,
    )
    print(f"\nRUN COMPLETE: {output_root}")


if __name__ == "__main__":
    main()
