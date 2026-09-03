#!/usr/bin/env python3
"""Apply the frozen IBDMDB Ricci C=0.02 model to expanded external v2.

This is an exploratory robustness transfer, not a second strict external
validation, because expanded-v2 graph weights use positive-support geometric
means estimated from the external cohort.  The frozen model is evaluated on:

* all 184 expanded-v2 samples; and
* the locked 90-sample intersection with frozen-support v1.

The script also compares v1 and v2 probabilities on exactly those 90 samples.
It computes every v2 probability before reading any external label.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import math
import os
import platform
import re
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.sparse import load_npz
from scipy.special import expit
from scipy.stats import pearsonr, spearmanr
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


SCRIPT_VERSION = "1.0.0"
EXPECTED_V1_DEPLOYMENT_SOURCE_SHA256 = (
    "c14166c86677248eeb322ec2646b428ad9726ede73c296d643d50b4492ce2138"
)
EXPECTED_MODEL_SHA256 = "e3152d4006266962c63efdaf11f11cb2ca88ca6cdad1b728183b41e53304638d"
EXPECTED_EDGE_METADATA_SHA256 = "a3cb8e9b0f19677dbffdd5b8d6ca3119b14a06c6643d0781347872e6687936a3"
EXPECTED_ALL_SAMPLES = 184
EXPECTED_PAIRED_SAMPLES = 90
EXPECTED_FROZEN_EDGES = 16932
EXPECTED_FEATURES = 33864
THRESHOLD = 0.5
CLASS_NAMES = ("nonIBD", "IBD")

MISSING_IDENTIFIERS = {
    "", "na", "nan", "none", "null", "missing", "unknown", "not available",
    "not_applicable", "not applicable", "healthy_control", "healthy control",
}


def die(message: str) -> None:
    raise RuntimeError(message)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def require_file(path: Path, label: str, expected_sha: str | None = None) -> None:
    if not path.is_file():
        die(f"Missing {label}: {path}")
    if path.stat().st_size <= 0:
        die(f"Empty {label}: {path}")
    if expected_sha is not None:
        found = sha256_file(path)
        if found != expected_sha:
            die(
                f"SHA-256 mismatch for {label}: expected {expected_sha}, "
                f"found {found}: {path}"
            )


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def import_locked_v1_deployment(path: Path):
    require_file(
        path,
        "audited v1 Ricci deployment source",
        EXPECTED_V1_DEPLOYMENT_SOURCE_SHA256,
    )
    name = "audited_frozen_ricci_v1_deployment"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        die(f"Could not construct import specification for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    for symbol in [
        "verify_required_files",
        "load_and_verify_model",
        "prove_saved_scaler",
    ]:
        if not hasattr(module, symbol):
            die(f"Audited v1 deployment source lacks {symbol!r}")
    return module


def parse_frozen_edge(value: object) -> tuple[str, str]:
    text = str(value).strip()
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (tuple, list)) and len(parsed) == 2:
            return str(parsed[0]), str(parsed[1])
    except Exception:
        pass
    for separator in (" -> ", "->", "\t", "||"):
        if separator in text:
            left, right = text.split(separator, 1)
            return left.strip().strip("'\"()[]"), right.strip().strip("'\"()[]")
    match = re.match(
        r"^\(?\s*['\"]?(.*?)['\"]?\s*,\s*['\"]?(.*?)['\"]?\s*\)?$",
        text,
    )
    if match:
        return match.group(1), match.group(2)
    die(f"Cannot parse frozen edge coordinate: {value!r}")


def binary_label(value: object) -> int:
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, (int, np.integer, float, np.floating)):
        numeric = float(value)
        if math.isfinite(numeric) and numeric in (0.0, 1.0):
            return int(numeric)
    compact = re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())
    if compact in {
        "0", "nonibd", "nonibdcontrol", "healthy", "healthycontrol", "control", "hc",
    }:
        return 0
    if compact in {
        "1", "ibd", "case", "cd", "uc", "crohn", "crohns", "crohnsdisease",
        "ulcerativecolitis",
    }:
        return 1
    die(f"Unsupported external IBD binary label: {value!r}")


def clean_identifier(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if text.lower() in MISSING_IDENTIFIERS:
        return None
    return text


def choose_column(frame: pd.DataFrame, candidates: Iterable[str], label: str) -> str:
    lower = {str(column).lower(): str(column) for column in frame.columns}
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    die(f"Could not find {label}; tried {list(candidates)}")


def resolve_participant_ids(frame: pd.DataFrame) -> tuple[pd.Series, list[str]]:
    # Prefer host_subject_id because that was the prespecified v1 grouping key.
    candidates = [
        "host_subject_id",
        "participant_id",
        "patient_id",
        "subject_id",
        "PRJEB42155_ge50_clean_metadata_full__host_subject_id",
        "PRJEB42155_ge50_clean_metadata_full__patient_id",
    ]
    available = [column for column in candidates if column in frame.columns]
    if not available:
        die("No participant identifier column exists in external metadata")
    resolved: list[str] = []
    for _, row in frame.iterrows():
        participant = None
        for column in available:
            participant = clean_identifier(row[column])
            if participant is not None:
                break
        if participant is None:
            participant = "sample::" + str(row["sample_id"])
        resolved.append(participant)
    return pd.Series(resolved, index=frame.index, dtype="object"), available


def metric_bundle(y_true: np.ndarray, probability: np.ndarray) -> tuple[dict[str, Any], np.ndarray]:
    y_true = np.asarray(y_true, dtype=int)
    probability = np.asarray(probability, dtype=float)
    if len(y_true) == 0 or len(y_true) != len(probability):
        die("Invalid inputs to metric calculation")
    if set(np.unique(y_true)) != {0, 1}:
        die(f"Metrics require both classes; found {np.unique(y_true).tolist()}")
    if not np.isfinite(probability).all() or np.any(probability < 0) or np.any(probability > 1):
        die("Invalid probability supplied to metrics")
    prediction = (probability >= THRESHOLD).astype(int)
    cm = confusion_matrix(y_true, prediction, labels=[0, 1])
    clipped = np.clip(probability, np.finfo(float).eps, 1.0 - np.finfo(float).eps)
    metrics = {
        "n": int(len(y_true)),
        "class_counts": {
            "nonIBD": int(np.sum(y_true == 0)),
            "IBD": int(np.sum(y_true == 1)),
        },
        "threshold": THRESHOLD,
        "accuracy": float(accuracy_score(y_true, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "macro_f1": float(f1_score(y_true, prediction, average="macro", zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "average_precision_ibd": float(average_precision_score(y_true, probability)),
        "brier": float(brier_score_loss(y_true, probability)),
        "log_loss": float(log_loss(y_true, clipped, labels=[0, 1])),
        "recall_nonIBD": float(recall_score(y_true, prediction, pos_label=0, zero_division=0)),
        "recall_IBD": float(recall_score(y_true, prediction, pos_label=1, zero_division=0)),
        "confusion_matrix": cm.astype(int).tolist(),
    }
    return metrics, cm


def participant_average(labelled: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any], np.ndarray]:
    conflicts = labelled.groupby("participant_id_external")["y_true"].nunique()
    if (conflicts > 1).any():
        die(
            "Participants with conflicting labels: "
            + ", ".join(conflicts[conflicts > 1].index.astype(str).tolist()[:10])
        )
    participant = labelled.groupby("participant_id_external", as_index=False).agg(
        sample_count=("sample_id", "size"),
        y_true=("y_true", "first"),
        proba_IBD=("proba_IBD", "mean"),
    )
    participant["true_label"] = participant["y_true"].map({0: "nonIBD", 1: "IBD"})
    participant["pred"] = (participant["proba_IBD"] >= THRESHOLD).astype(int)
    participant["pred_label"] = participant["pred"].map({0: "nonIBD", 1: "IBD"})
    metrics, cm = metric_bundle(participant["y_true"], participant["proba_IBD"])
    return participant, metrics, cm


def verify_v2_feature_set(
    feature_dir: Path,
    vectorization_marker: Path,
    source_axis: list[str],
) -> tuple[Any, pd.DataFrame, dict[str, Any]]:
    matrix_path = feature_dir / "feature_matrix_B_K0.npz"
    order_path = feature_dir / "sample_order.csv"
    edges_path = feature_dir / "edge_metadata.csv"
    provenance_path = feature_dir / "FEATURE_MATRIX_PROVENANCE.json"
    require_file(vectorization_marker, "v2 vectorization completion marker")
    for path, label in [
        (matrix_path, "v2 frozen-edge feature matrix"),
        (order_path, "v2 sample order"),
        (edges_path, "v2 frozen-edge metadata"),
        (provenance_path, "v2 feature provenance"),
    ]:
        require_file(path, label)

    matrix = load_npz(matrix_path).tocsr()
    order = pd.read_csv(order_path).sort_values("sample_index").reset_index(drop=True)
    edges = pd.read_csv(edges_path).sort_values("edge_index").reset_index(drop=True)
    provenance = json.loads(provenance_path.read_text())

    if matrix.shape != (EXPECTED_ALL_SAMPLES, EXPECTED_FEATURES):
        die(f"Expected v2 matrix {(EXPECTED_ALL_SAMPLES, EXPECTED_FEATURES)}, found {matrix.shape}")
    if len(order) != EXPECTED_ALL_SAMPLES or order["sample_id"].astype(str).duplicated().any():
        die("v2 sample order must contain 184 unique sample IDs")
    if not np.array_equal(order["sample_index"].to_numpy(), np.arange(EXPECTED_ALL_SAMPLES)):
        die("v2 sample indices are not exactly 0..183")
    if len(edges) != EXPECTED_FROZEN_EDGES:
        die(f"Expected {EXPECTED_FROZEN_EDGES} v2 frozen edges, found {len(edges)}")
    if not np.array_equal(edges["edge_index"].to_numpy(), np.arange(EXPECTED_FROZEN_EDGES)):
        die("v2 frozen edge indices are not exactly 0..16931")
    if not {"source", "target"}.issubset(edges.columns):
        die("v2 edge metadata lacks source/target columns")
    v2_axis = list(zip(edges["source"].astype(str), edges["target"].astype(str)))
    parsed_source_axis = [parse_frozen_edge(value) for value in source_axis]
    if v2_axis != parsed_source_axis:
        mismatch = next(
            (i for i, (left, right) in enumerate(zip(v2_axis, parsed_source_axis)) if left != right),
            None,
        )
        die(f"v2 frozen projection axis differs from source axis; first mismatch={mismatch}")
    if len(v2_axis) != len(set(v2_axis)):
        die("v2 frozen projection axis contains duplicate edges")
    if not np.isfinite(matrix.data).all():
        die("v2 feature matrix contains nonfinite stored values")

    n_edges = EXPECTED_FROZEN_EDGES
    block_b = matrix[:, :n_edges]
    block_k0 = matrix[:, n_edges:]
    if block_b.nnz and not np.isin(block_b.data, (0.0, 1.0)).all():
        die("v2 B block is not binary")
    orphan = block_k0 - block_k0.multiply(block_b)
    orphan.eliminate_zeros()
    if orphan.nnz:
        die(f"v2 K0 has {orphan.nnz} nonzero coordinates where B=0")

    expected_shape = [EXPECTED_ALL_SAMPLES, EXPECTED_FEATURES]
    if provenance.get("matrix_shape") != expected_shape:
        die(
            f"v2 provenance matrix_shape {provenance.get('matrix_shape')} "
            f"does not equal {expected_shape}"
        )
    if provenance.get("n_samples") != EXPECTED_ALL_SAMPLES:
        die("v2 feature provenance has wrong sample count")
    if provenance.get("n_edges") != EXPECTED_FROZEN_EDGES:
        die("v2 feature provenance has wrong frozen-edge count")
    if provenance.get("frozen_edge_metadata_sha256") != EXPECTED_EDGE_METADATA_SHA256:
        die("v2 provenance does not name the audited source frozen-edge hash")
    if provenance.get("external_normalization") is not True:
        die("v2 provenance does not certify external normalization")
    representation = str(provenance.get("representation", "")).lower()
    if "frozen" not in representation or "projection" not in representation:
        die(f"Unexpected v2 feature representation: {provenance.get('representation')!r}")

    marker_payload = json.loads(vectorization_marker.read_text())
    return matrix, order, {
        "matrix_path": str(matrix_path),
        "matrix_sha256": sha256_file(matrix_path),
        "sample_order_path": str(order_path),
        "sample_order_sha256": sha256_file(order_path),
        "edge_metadata_path": str(edges_path),
        "edge_metadata_sha256": sha256_file(edges_path),
        "feature_provenance_path": str(provenance_path),
        "feature_provenance_sha256": sha256_file(provenance_path),
        "feature_provenance": provenance,
        "vectorization_marker_path": str(vectorization_marker),
        "vectorization_marker_sha256": sha256_file(vectorization_marker),
        "vectorization_marker": marker_payload,
        "B_nnz": int(block_b.nnz),
        "K0_nnz": int(block_k0.nnz),
    }


def predict_label_blind(
    matrix,
    order: pd.DataFrame,
    model: dict[str, Any],
) -> pd.DataFrame:
    if matrix.shape[1] != model["n_features"]:
        die(f"v2/model feature mismatch: matrix={matrix.shape}, model={model['n_features']}")
    dtype = np.dtype(model["training_input_dtype"])
    dense = matrix.toarray().astype(dtype, copy=False)
    standardised = (dense - model["scaler_mean"]) / model["scaler_scale"]
    logits = standardised @ model["coef"].reshape(-1) + float(model["intercept"][0])
    probability = expit(logits)
    if not np.isfinite(logits).all() or not np.isfinite(probability).all():
        die("v2 frozen-model scores or probabilities are nonfinite")
    predictions = pd.DataFrame(
        {
            "sample_id": order["sample_id"].astype(str),
            "logit_IBD": logits,
            "proba_nonIBD": 1.0 - probability,
            "proba_IBD": probability,
            "pred": (probability >= THRESHOLD).astype(int),
        }
    )
    predictions["pred_label"] = predictions["pred"].map({0: "nonIBD", 1: "IBD"})
    return predictions


def attach_metadata_after_prediction(
    predictions: pd.DataFrame,
    metadata_path: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    require_file(metadata_path, "matched v2 metadata")
    metadata = pd.read_csv(metadata_path, sep="\t", low_memory=False)
    sample_column = choose_column(
        metadata,
        ("sample_id_v2", "run_accession", "sample_id", "sample_accession"),
        "v2 metadata sample ID",
    )
    metadata[sample_column] = metadata[sample_column].astype(str)
    wanted = set(predictions["sample_id"])
    selected = metadata[metadata[sample_column].isin(wanted)].copy()
    if len(selected) != len(predictions) or selected[sample_column].duplicated().any():
        die(
            f"v2 metadata must match predictions one-to-one: metadata={len(selected)}, "
            f"predictions={len(predictions)}"
        )
    labelled = predictions.merge(
        selected,
        left_on="sample_id",
        right_on=sample_column,
        how="left",
        validate="one_to_one",
    )
    label_column = choose_column(
        labelled,
        (
            "external_label_ibd_binary",
            "PRJEB42155_ge50_clean_metadata_full__external_label_ibd_binary",
            "y_binary",
            "label",
        ),
        "v2 external IBD label",
    )
    labelled["y_true"] = labelled[label_column].map(binary_label).astype(int)
    labelled["true_label"] = labelled["y_true"].map({0: "nonIBD", 1: "IBD"})
    labelled["participant_id_external"], participant_columns = resolve_participant_ids(labelled)
    audit = {
        "metadata_path": str(metadata_path),
        "metadata_sha256": sha256_file(metadata_path),
        "sample_id_column": sample_column,
        "label_column": label_column,
        "participant_columns_considered_in_order": participant_columns,
    }
    return labelled, audit


def evaluate_subset(labelled: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any], np.ndarray, pd.DataFrame, dict[str, Any], np.ndarray]:
    sample_metrics, sample_cm = metric_bundle(labelled["y_true"], labelled["proba_IBD"])
    participant, participant_metrics, participant_cm = participant_average(labelled)
    return labelled, sample_metrics, sample_cm, participant, participant_metrics, participant_cm


def load_paired_ids(path: Path, all_ids: set[str]) -> list[str]:
    require_file(path, "locked v1/v2 paired sample IDs")
    frame = pd.read_csv(path, dtype=str)
    column = choose_column(frame, ("sample_id", "sample_id_v2"), "paired sample ID")
    ids = frame[column].astype(str).tolist()
    if len(ids) != EXPECTED_PAIRED_SAMPLES or len(ids) != len(set(ids)):
        die(f"Expected 90 unique paired sample IDs, found {len(ids)} rows/{len(set(ids))} unique")
    missing = sorted(set(ids) - all_ids)
    if missing:
        die(f"Paired IDs missing from v2 predictions: {missing[:10]}")
    return ids


def load_and_verify_v1_predictions(
    prediction_path: Path,
    metrics_path: Path,
    completion_path: Path,
    paired_ids: set[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    for path, label in [
        (prediction_path, "strict v1 sample predictions"),
        (metrics_path, "strict v1 external metrics"),
        (completion_path, "strict v1 completion marker"),
    ]:
        require_file(path, label)
    completion = json.loads(completion_path.read_text())
    if completion.get("status") != "complete":
        die("v1 completion marker is not complete")
    if completion.get("model_sha256") != EXPECTED_MODEL_SHA256:
        die("v1 predictions were not produced by the audited frozen model")
    frame = pd.read_csv(prediction_path, low_memory=False)
    sample_column = choose_column(frame, ("sample_id",), "v1 prediction sample ID")
    probability_column = choose_column(
        frame,
        ("proba_IBD", "probability_IBD", "probability_ibd"),
        "v1 IBD probability",
    )
    label_column = choose_column(
        frame,
        ("y_true", "y_binary", "external_label_ibd_binary"),
        "v1 true label",
    )
    frame = frame.rename(
        columns={
            sample_column: "sample_id",
            probability_column: "proba_IBD_v1",
            label_column: "y_true_v1",
        }
    )
    frame["sample_id"] = frame["sample_id"].astype(str)
    frame["y_true_v1"] = frame["y_true_v1"].map(binary_label).astype(int)
    if len(frame) != EXPECTED_PAIRED_SAMPLES or frame["sample_id"].duplicated().any():
        die("Strict v1 predictions do not contain exactly 90 unique samples")
    if set(frame["sample_id"]) != paired_ids:
        die("Locked paired-ID set does not exactly equal strict v1 prediction sample set")
    metrics_saved = json.loads(metrics_path.read_text())
    metrics_recomputed, _ = metric_bundle(frame["y_true_v1"], frame["proba_IBD_v1"])
    for key in [
        "accuracy", "balanced_accuracy", "macro_f1", "roc_auc",
        "average_precision_ibd", "recall_nonIBD", "recall_IBD",
    ]:
        expected = float(metrics_saved["sample"][key])
        observed = float(metrics_recomputed[key])
        if not math.isclose(expected, observed, rel_tol=0.0, abs_tol=1e-12):
            die(f"Recomputed v1 metric {key}={observed} differs from saved value {expected}")
    return frame[["sample_id", "y_true_v1", "proba_IBD_v1"]], {
        "prediction_path": str(prediction_path),
        "prediction_sha256": sha256_file(prediction_path),
        "metrics_path": str(metrics_path),
        "metrics_sha256": sha256_file(metrics_path),
        "completion_path": str(completion_path),
        "completion_sha256": sha256_file(completion_path),
        "v1_sample_metrics_recomputed": metrics_recomputed,
    }


def paired_representation_comparison(
    v2_paired: pd.DataFrame,
    v1: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    comparison = v2_paired[
        ["sample_id", "participant_id_external", "y_true", "proba_IBD", "pred"]
    ].merge(v1, on="sample_id", how="inner", validate="one_to_one")
    if len(comparison) != EXPECTED_PAIRED_SAMPLES:
        die(f"Expected 90 v1/v2 comparison rows, found {len(comparison)}")
    if not np.array_equal(comparison["y_true"], comparison["y_true_v1"]):
        die("v1 and v2 labels disagree on paired samples")
    comparison = comparison.rename(
        columns={"proba_IBD": "proba_IBD_v2", "pred": "pred_v2"}
    )
    comparison["pred_v1"] = (comparison["proba_IBD_v1"] >= THRESHOLD).astype(int)
    comparison["probability_difference_v2_minus_v1"] = (
        comparison["proba_IBD_v2"] - comparison["proba_IBD_v1"]
    )
    comparison["absolute_probability_difference"] = comparison[
        "probability_difference_v2_minus_v1"
    ].abs()
    comparison["prediction_agrees"] = comparison["pred_v1"] == comparison["pred_v2"]

    v1_metrics, _ = metric_bundle(comparison["y_true"], comparison["proba_IBD_v1"])
    v2_metrics, _ = metric_bundle(comparison["y_true"], comparison["proba_IBD_v2"])
    pearson = pearsonr(comparison["proba_IBD_v1"], comparison["proba_IBD_v2"])
    spearman = spearmanr(comparison["proba_IBD_v1"], comparison["proba_IBD_v2"])
    difference = comparison["probability_difference_v2_minus_v1"].to_numpy(float)

    participant = comparison.groupby("participant_id_external", as_index=False).agg(
        sample_count=("sample_id", "size"),
        y_true=("y_true", "first"),
        proba_IBD_v1=("proba_IBD_v1", "mean"),
        proba_IBD_v2=("proba_IBD_v2", "mean"),
    )
    participant["probability_difference_v2_minus_v1"] = (
        participant["proba_IBD_v2"] - participant["proba_IBD_v1"]
    )
    participant["pred_v1"] = (participant["proba_IBD_v1"] >= THRESHOLD).astype(int)
    participant["pred_v2"] = (participant["proba_IBD_v2"] >= THRESHOLD).astype(int)
    participant["prediction_agrees"] = participant["pred_v1"] == participant["pred_v2"]
    v1_participant_metrics, _ = metric_bundle(participant["y_true"], participant["proba_IBD_v1"])
    v2_participant_metrics, _ = metric_bundle(participant["y_true"], participant["proba_IBD_v2"])

    metric_keys = [
        "accuracy", "balanced_accuracy", "macro_f1", "roc_auc",
        "average_precision_ibd", "brier", "log_loss", "recall_nonIBD", "recall_IBD",
    ]
    summary = {
        "interpretation": (
            "paired robustness comparison of strict frozen-support v1 against "
            "external-normalized expanded-community v2 on identical samples; v2 is exploratory"
        ),
        "sample": {
            "n": EXPECTED_PAIRED_SAMPLES,
            "v1_strict_metrics": v1_metrics,
            "v2_exploratory_metrics": v2_metrics,
            "metric_differences_v2_minus_v1": {
                key: float(v2_metrics[key] - v1_metrics[key]) for key in metric_keys
            },
            "probability_pearson_r": float(pearson.statistic),
            "probability_pearson_p_two_sided_descriptive": float(pearson.pvalue),
            "probability_spearman_rho": float(spearman.statistic),
            "probability_spearman_p_two_sided_descriptive": float(spearman.pvalue),
            "prediction_agreement_fraction": float(comparison["prediction_agrees"].mean()),
            "probability_difference_v2_minus_v1": {
                "mean": float(np.mean(difference)),
                "median": float(np.median(difference)),
                "mean_absolute": float(np.mean(np.abs(difference))),
                "minimum": float(np.min(difference)),
                "maximum": float(np.max(difference)),
            },
        },
        "participant_probability_averaged": {
            "n": int(len(participant)),
            "v1_strict_metrics": v1_participant_metrics,
            "v2_exploratory_metrics": v2_participant_metrics,
            "metric_differences_v2_minus_v1": {
                key: float(v2_participant_metrics[key] - v1_participant_metrics[key])
                for key in metric_keys
            },
            "prediction_agreement_fraction": float(participant["prediction_agrees"].mean()),
        },
        "p_value_note": (
            "Correlation p-values are descriptive only; no inferential claim or multiple-testing "
            "interpretation is made. Performance uncertainty should be reported separately."
        ),
    }
    return comparison, participant, summary


def optional_crosscheck_existing_prediction(
    path: Path,
    predictions: pd.DataFrame,
) -> dict[str, Any]:
    if not path.is_file():
        return {"status": "not_available", "path": str(path)}
    existing = pd.read_csv(path)
    sample_column = choose_column(existing, ("sample_id",), "existing v2 sample ID")
    probability_column = choose_column(
        existing,
        ("probability_IBD", "proba_IBD"),
        "existing v2 probability",
    )
    check = predictions[["sample_id", "proba_IBD"]].merge(
        existing[[sample_column, probability_column]].rename(
            columns={sample_column: "sample_id", probability_column: "existing_probability"}
        ),
        on="sample_id",
        how="inner",
        validate="one_to_one",
    )
    if len(check) != EXPECTED_ALL_SAMPLES:
        die("Existing v2 pipeline predictions do not cover all 184 samples")
    maximum = float(np.max(np.abs(check["proba_IBD"] - check["existing_probability"])))
    if maximum > 1e-12:
        die(f"Independent v2 probability cross-check failed: max abs difference={maximum}")
    return {
        "status": "pass",
        "path": str(path),
        "sha256": sha256_file(path),
        "max_abs_probability_difference": maximum,
        "tolerance": 1e-12,
    }


def confusion_frame(cm: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(cm, index=["true_nonIBD", "true_IBD"], columns=["pred_nonIBD", "pred_IBD"])


def fmt(value: float) -> str:
    return f"{value:.3f}"


def latex_table(
    all_metrics: dict[str, Any],
    paired_metrics: dict[str, Any],
) -> str:
    rows = [
        ("v2 expanded community", "All", "Sample", all_metrics["sample"]),
        (
            "v2 expanded community",
            "All",
            "Participant",
            all_metrics["participant_probability_averaged"],
        ),
        ("v2 expanded community", "Paired with v1", "Sample", paired_metrics["sample"]),
        (
            "v2 expanded community",
            "Paired with v1",
            "Participant",
            paired_metrics["participant_probability_averaged"],
        ),
    ]
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\caption{Exploratory transfer of the frozen IBDMDB Ricci model to the expanded-community external representation.}",
        r"\label{tab:external_ricci_expanded_v2}",
        r"\begin{tabular}{lllrrrrrrrr}",
        r"\toprule",
        r"Representation & Cohort & Unit & $n$ & Acc. & Bal. acc. & Macro F1 & ROC--AUC & AP$_{\mathrm{IBD}}$ & Rec. nonIBD & Rec. IBD \\",
        r"\midrule",
    ]
    for representation, cohort, unit, metric in rows:
        lines.append(
            f"{representation} & {cohort} & {unit} & {metric['n']} & "
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
            r"The model, scaler, coefficients, threshold and original IBDMDB edge order were frozen. Expanded-v2 graphs use external-cohort support normalization; these values therefore represent exploratory structural replication rather than a second strict external validation. The paired cohort contains exactly the 90 samples also analysed under frozen-support v1. AP denotes average precision for IBD.",
            r"\end{minipage}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(
    output_dir: Path,
    all_labelled: pd.DataFrame,
    all_participants: pd.DataFrame,
    all_metrics: dict[str, Any],
    all_sample_cm: np.ndarray,
    all_participant_cm: np.ndarray,
    paired_labelled: pd.DataFrame,
    paired_participants: pd.DataFrame,
    paired_metrics: dict[str, Any],
    paired_sample_cm: np.ndarray,
    paired_participant_cm: np.ndarray,
    comparison_samples: pd.DataFrame,
    comparison_participants: pd.DataFrame,
    comparison_summary: dict[str, Any],
    label_blind_predictions: pd.DataFrame,
    provenance: dict[str, Any],
    overwrite: bool,
) -> None:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if output_dir.exists():
        if not overwrite:
            die(f"Output directory exists; use --overwrite to move it to a backup: {output_dir}")
        backup = output_dir.with_name(
            output_dir.name + ".backup_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        if backup.exists():
            die(f"Backup path already exists: {backup}")
        output_dir.rename(backup)
        print(f"Existing output moved to recoverable backup: {backup}")

    temporary = Path(tempfile.mkdtemp(prefix=output_dir.name + ".tmp_", dir=output_dir.parent))
    try:
        label_blind_predictions.to_csv(temporary / "label_blind_probabilities_all_184.csv", index=False)
        all_dir = temporary / "all_184"
        paired_dir = temporary / "paired_90"
        all_dir.mkdir()
        paired_dir.mkdir()

        all_labelled.to_csv(all_dir / "sample_predictions.csv", index=False)
        all_participants.to_csv(
            all_dir / "participant_predictions_probability_averaged.csv", index=False
        )
        atomic_json(all_dir / "external_metrics.json", all_metrics)
        confusion_frame(all_sample_cm).to_csv(all_dir / "sample_confusion_matrix.csv")
        confusion_frame(all_participant_cm).to_csv(
            all_dir / "participant_confusion_matrix_probability_averaged.csv"
        )

        paired_labelled.to_csv(paired_dir / "sample_predictions.csv", index=False)
        paired_participants.to_csv(
            paired_dir / "participant_predictions_probability_averaged.csv", index=False
        )
        atomic_json(paired_dir / "external_metrics.json", paired_metrics)
        confusion_frame(paired_sample_cm).to_csv(paired_dir / "sample_confusion_matrix.csv")
        confusion_frame(paired_participant_cm).to_csv(
            paired_dir / "participant_confusion_matrix_probability_averaged.csv"
        )
        comparison_samples.to_csv(
            paired_dir / "v1_v2_sample_probability_comparison.csv", index=False
        )
        comparison_participants.to_csv(
            paired_dir / "v1_v2_participant_probability_comparison.csv", index=False
        )
        atomic_json(paired_dir / "v1_v2_representation_comparison.json", comparison_summary)

        (temporary / "external_ricci_expanded_v2_results_table.tex").write_text(
            latex_table(all_metrics, paired_metrics), encoding="utf-8"
        )
        atomic_json(temporary / "DEPLOYMENT_PROVENANCE.json", provenance)
        atomic_json(
            temporary / "RUN_COMPLETE.json",
            {
                "status": "complete",
                "completed_utc": utc_now(),
                "script_version": SCRIPT_VERSION,
                "model_sha256": EXPECTED_MODEL_SHA256,
                "all_v2_samples": EXPECTED_ALL_SAMPLES,
                "paired_v1_v2_samples": EXPECTED_PAIRED_SAMPLES,
                "methodological_role": "exploratory expanded-community structural replication",
            },
        )
        os.replace(temporary, output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def parse_args() -> argparse.Namespace:
    home = Path.home()
    ext = home / "Real_Data/external_validation/serrano_gomez_ibd/ge50_external_validation"
    expanded = ext / "expanded_community_structural_replication_v2"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--external-root", type=Path, default=ext)
    parser.add_argument(
        "--v1-deployment-source",
        type=Path,
        default=ext / "scripts/apply_frozen_ricci_ibd_model_v1.py",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=expanded / "08_exploratory_prediction/frozen_ricci_C002_v2_all184_and_paired90",
    )
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ext = args.external_root.expanduser().resolve()
    expanded = ext / "expanded_community_structural_replication_v2"
    feature_dir = expanded / "07_features/frozen_edge_projection"
    vectorization_marker = expanded / "07_features/VECTORIZATION_COMPLETE.json"
    metadata_path = expanded / "01_curated/matched_external_metadata_184.tsv.gz"
    paired_ids_path = expanded / "01_curated/paired_v1_sample_ids.csv"
    existing_prediction_path = expanded / "08_exploratory_prediction/label_blind_sample_probabilities.csv"
    v1_output = ext / "structural_frozen_ibdmdb/external_prediction_frozen_ricci_C002_v1"

    print("=" * 112)
    print("FROZEN RICCI IBD-vs-nonIBD C=0.02 — EXPANDED EXTERNAL V2 TRANSFER")
    print("=" * 112)
    print("Methodological role: exploratory structural replication; not strict external validation")
    print(f"v2 feature directory: {feature_dir}")
    print(f"output directory: {args.output_dir.expanduser().resolve()}")
    print("Evaluation sets: all 184 samples and locked 90-sample v1 intersection")
    print("Label firewall: all v2 probabilities are computed before metadata labels are read")

    v1_source_path = args.v1_deployment_source.expanduser().resolve()
    v1_module = import_locked_v1_deployment(v1_source_path)
    v1_required_paths = v1_module.verify_required_files()
    model = v1_module.load_and_verify_model(v1_required_paths)
    if model["model_sha256"] != EXPECTED_MODEL_SHA256:
        die("Audited v1 loader returned an unexpected model hash")
    scaler_proof, training_dtype = v1_module.prove_saved_scaler(v1_required_paths, model)
    model["training_input_dtype"] = training_dtype
    print("Saved scaler reproduces from the locked 1,317-sample source cohort: PASS")
    print(json.dumps(scaler_proof, indent=2))

    matrix, order, feature_audit = verify_v2_feature_set(
        feature_dir,
        vectorization_marker,
        model["source_axis"],
    )
    print(
        f"v2 frozen projection integrity: PASS; shape={matrix.shape}, "
        f"B_nnz={feature_audit['B_nnz']:,}, K0_nnz={feature_audit['K0_nnz']:,}"
    )

    label_blind = predict_label_blind(matrix, order, model)
    print(
        "Computed frozen probabilities without labels for 184 samples: "
        f"min/median/max={label_blind['proba_IBD'].min():.4f}/"
        f"{label_blind['proba_IBD'].median():.4f}/"
        f"{label_blind['proba_IBD'].max():.4f}"
    )
    crosscheck = optional_crosscheck_existing_prediction(existing_prediction_path, label_blind)
    if crosscheck["status"] == "pass":
        print("Independent cross-check against existing v2 pipeline probabilities: PASS")
    else:
        print("Existing v2 pipeline probabilities not present; independent cross-check skipped")

    # External metadata, labels, and participant IDs are first read here.
    all_labelled, metadata_audit = attach_metadata_after_prediction(label_blind, metadata_path)
    (
        all_labelled,
        all_sample_metrics,
        all_sample_cm,
        all_participants,
        all_participant_metrics,
        all_participant_cm,
    ) = evaluate_subset(all_labelled)
    all_metrics = {
        "interpretation": (
            "exploratory frozen-model transfer to the external-normalized expanded-community "
            "v2 frozen-edge projection; not strict external validation"
        ),
        "sample": all_sample_metrics,
        "participant_probability_averaged": all_participant_metrics,
    }

    paired_ids = load_paired_ids(paired_ids_path, set(all_labelled["sample_id"]))
    paired_labelled = (
        all_labelled.set_index("sample_id", verify_integrity=True)
        .loc[paired_ids]
        .reset_index()
    )
    (
        paired_labelled,
        paired_sample_metrics,
        paired_sample_cm,
        paired_participants,
        paired_participant_metrics,
        paired_participant_cm,
    ) = evaluate_subset(paired_labelled)
    paired_metrics = {
        "interpretation": (
            "exploratory expanded-v2 evaluation restricted to the exact 90 samples in strict v1"
        ),
        "sample": paired_sample_metrics,
        "participant_probability_averaged": paired_participant_metrics,
    }

    v1_predictions, v1_audit = load_and_verify_v1_predictions(
        v1_output / "sample_predictions.csv",
        v1_output / "external_metrics.json",
        v1_output / "RUN_COMPLETE.json",
        set(paired_ids),
    )
    comparison_samples, comparison_participants, comparison_summary = (
        paired_representation_comparison(paired_labelled, v1_predictions)
    )

    print("\nALL 184 V2 SAMPLE METRICS")
    print(json.dumps(all_sample_metrics, indent=2))
    print("All-184 sample confusion [rows=true; cols=pred; nonIBD, IBD]:")
    print(all_sample_cm)
    print("\nALL 184 V2 PARTICIPANT-AVERAGED METRICS")
    print(json.dumps(all_participant_metrics, indent=2))
    print("\nPAIRED 90 V2 SAMPLE METRICS")
    print(json.dumps(paired_sample_metrics, indent=2))
    print("Paired-90 sample confusion [rows=true; cols=pred; nonIBD, IBD]:")
    print(paired_sample_cm)
    print("\nPAIRED V1-V2 REPRESENTATION COMPARISON")
    print(json.dumps(comparison_summary, indent=2))

    script_path = Path(__file__).resolve()
    provenance = {
        "script_version": SCRIPT_VERSION,
        "script_path": str(script_path),
        "script_sha256": sha256_file(script_path),
        "created_utc": utc_now(),
        "method": {
            "model": "frozen IBDMDB Ricci IBD-vs-nonIBD logistic model at C=0.02",
            "model_refitting": False,
            "external_rescaling": False,
            "threshold_selection_on_external_data": False,
            "threshold": THRESHOLD,
            "feature_layout": "original frozen [all B coordinates, all K0 coordinates]",
            "evaluation_sets": {
                "all_v2": EXPECTED_ALL_SAMPLES,
                "paired_v1_v2": EXPECTED_PAIRED_SAMPLES,
            },
            "methodological_role": (
                "exploratory expanded-community structural replication; v2 is not strict "
                "external validation because graph normalization used the external cohort"
            ),
            "label_firewall": (
                "all 184 scores and probabilities were generated before the external metadata "
                "or labels were read"
            ),
        },
        "model": {
            "path": str(v1_required_paths["model"]),
            "sha256": model["model_sha256"],
            "selected_coefficients": int(np.count_nonzero(model["coef"])),
            "n_features": int(model["n_features"]),
            "training_input_dtype": training_dtype,
        },
        "source_scaler_proof": scaler_proof,
        "v1_deployment_source": {
            "path": str(v1_source_path),
            "sha256": EXPECTED_V1_DEPLOYMENT_SOURCE_SHA256,
        },
        "v2_features": feature_audit,
        "metadata": metadata_audit,
        "paired_ids": {
            "path": str(paired_ids_path),
            "sha256": sha256_file(paired_ids_path),
            "n": len(paired_ids),
        },
        "v1_strict_predictions": v1_audit,
        "existing_v2_pipeline_probability_crosscheck": crosscheck,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
        },
    }

    if args.verify_only:
        print("\nVERIFY-ONLY COMPLETE: no result files written.")
        return

    write_outputs(
        output_dir=args.output_dir.expanduser().resolve(),
        all_labelled=all_labelled,
        all_participants=all_participants,
        all_metrics=all_metrics,
        all_sample_cm=all_sample_cm,
        all_participant_cm=all_participant_cm,
        paired_labelled=paired_labelled,
        paired_participants=paired_participants,
        paired_metrics=paired_metrics,
        paired_sample_cm=paired_sample_cm,
        paired_participant_cm=paired_participant_cm,
        comparison_samples=comparison_samples,
        comparison_participants=comparison_participants,
        comparison_summary=comparison_summary,
        label_blind_predictions=label_blind,
        provenance=provenance,
        overwrite=args.overwrite,
    )
    print(f"\nRUN COMPLETE: {args.output_dir.expanduser().resolve()}")


if __name__ == "__main__":
    main()
