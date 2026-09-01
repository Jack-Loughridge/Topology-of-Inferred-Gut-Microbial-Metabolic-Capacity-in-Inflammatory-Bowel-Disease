#!/usr/bin/env python3
"""Verify and apply the frozen legacy Ricci IBD-vs-nonIBD C=0.02 model.

This deployment is fail-closed. It proves that the legacy artifact's saved
scaler matches the locked source IBDMDB cohort and faithful [B|K0] feature
matrix before producing any external probabilities. External labels are read
only after all sample probabilities have been computed.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from scipy.special import expit
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler


BASE = Path.home() / "Real_Data"
EXTERNAL_ROOT = (
    BASE
    / "external_validation"
    / "serrano_gomez_ibd"
    / "ge50_external_validation"
    / "structural_frozen_ibdmdb"
)
MODEL_PATH = (
    BASE
    / "Ricci_IBD_RepeatedCV_CPath"
    / "C_0p02"
    / "full_source_model"
    / "full_source_model.npz"
)
SOURCE_FEATURE_DIR = BASE / "Ricci_Classifier_Faithful_Eps0001_n250_v3"
SOURCE_MANIFEST = (
    BASE
    / "Ricci_IBD_RepeatedCV_CPath"
    / "splits"
    / "sample_split_manifest.csv"
)
EXTERNAL_FEATURE_DIR = EXTERNAL_ROOT / "ricci_features_frozen_training_order"
DEFAULT_OUTPUT_DIR = EXTERNAL_ROOT / "external_prediction_frozen_ricci_C002_v1"

REQUIRED_MODEL_KEYS = {
    "coef",
    "intercept",
    "scaler_mean",
    "scaler_scale",
    "schema_version",
}
CLASS_NAMES = ("nonIBD", "IBD")
PARTICIPANT_COLUMN = "host_subject_id"
MODEL_C = 0.02
EXPECTED_MODEL_SHA256 = "e3152d4006266962c63efdaf11f11cb2ca88ca6cdad1b728183b41e53304638d"
EXPECTED_EDGE_METADATA_SHA256 = "a3cb8e9b0f19677dbffdd5b8d6ca3119b14a06c6643d0781347872e6687936a3"
EXPECTED_RICCI_PARAMETERS = {
    "active_tol": 1e-12,
    "beta": 1.4,
    "c_single_out": 0.001,
    "epsilon_dist": 0.0001,
    "n_paths": 250,
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


def normalise_sample_id(value: object) -> str:
    text = Path(str(value).strip()).name
    text = re.sub(r"\.npy$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"_H0$", "", text, flags=re.IGNORECASE)
    return text


def find_column(frame: pd.DataFrame, candidates: tuple[str, ...], label: str) -> str:
    lower = {str(column).lower(): str(column) for column in frame.columns}
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    die(f"Could not find {label}; tried {candidates}; columns={list(frame.columns)}")


def binary_label(value: object) -> int:
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, (int, np.integer, float, np.floating)):
        numeric = float(value)
        if math.isfinite(numeric) and numeric in {0.0, 1.0}:
            return int(numeric)
        die(f"Unsupported IBD binary label: {value!r}")
    raw = str(value).strip().lower()
    if re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)", raw):
        numeric = float(raw)
        if math.isfinite(numeric) and numeric in {0.0, 1.0}:
            return int(numeric)
        die(f"Unsupported IBD binary label: {value!r}")
    compact = re.sub(r"[\s_-]+", "", raw)
    if compact in {"nonibd", "healthy", "healthycontrol", "control", "hc"}:
        return 0
    if compact in {"ibd", "uc", "ulcerativecolitis", "cd", "crohn", "crohns", "crohnsdisease"}:
        return 1
    die(f"Unsupported IBD binary label: {value!r}")


def json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    raise TypeError(type(value).__name__)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=json_value) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_csv(frame: pd.DataFrame, path: Path, index: bool = False) -> None:
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=index)
    os.replace(temporary, path)


def metric_bundle(y_true: np.ndarray, probability_ibd: np.ndarray) -> tuple[dict[str, Any], np.ndarray]:
    prediction = (probability_ibd >= 0.5).astype(int)
    cm = confusion_matrix(y_true, prediction, labels=[0, 1])
    metrics = {
        "n": int(len(y_true)),
        "class_counts": {
            "nonIBD": int(np.sum(y_true == 0)),
            "IBD": int(np.sum(y_true == 1)),
        },
        "threshold": 0.5,
        "accuracy": float(accuracy_score(y_true, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "macro_f1": float(f1_score(y_true, prediction, average="macro", zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probability_ibd)),
        "average_precision_ibd": float(average_precision_score(y_true, probability_ibd)),
        "recall_nonIBD": float(recall_score(y_true, prediction, pos_label=0, zero_division=0)),
        "recall_IBD": float(recall_score(y_true, prediction, pos_label=1, zero_division=0)),
    }
    return metrics, cm


def verify_required_files() -> dict[str, Path]:
    paths = {
        "model": MODEL_PATH,
        "source_matrix": SOURCE_FEATURE_DIR / "feature_matrix_B_K0.npz",
        "source_metadata": SOURCE_FEATURE_DIR / "matched_metadata.csv",
        "source_edges": SOURCE_FEATURE_DIR / "edge_metadata.csv",
        "source_manifest": SOURCE_MANIFEST,
        "external_matrix": EXTERNAL_FEATURE_DIR / "feature_matrix_B_K0.npz",
        "external_metadata": EXTERNAL_FEATURE_DIR / "matched_metadata.csv",
        "external_edges": EXTERNAL_FEATURE_DIR / "edge_metadata.csv",
        "external_feature_provenance": EXTERNAL_FEATURE_DIR / "FEATURE_MATRIX_PROVENANCE.json",
        "external_vectorization_audit": EXTERNAL_FEATURE_DIR / "vectorization_audit.csv",
    }
    missing = [f"{name}: {path}" for name, path in paths.items() if not path.is_file()]
    if missing:
        die("Required files are missing:\n" + "\n".join(missing))
    return paths


def load_and_verify_model(paths: dict[str, Path]) -> dict[str, Any]:
    model_sha256 = sha256_file(paths["model"])
    if model_sha256 != EXPECTED_MODEL_SHA256:
        die(
            "The model file is not the audited legacy C=0.02 IBD artifact: "
            f"expected SHA256={EXPECTED_MODEL_SHA256}, observed={model_sha256}"
        )

    artifact = np.load(paths["model"], allow_pickle=False)
    missing = sorted(REQUIRED_MODEL_KEYS - set(artifact.files))
    if missing:
        die(f"Legacy model artifact is missing keys: {missing}")

    coef = np.asarray(artifact["coef"], dtype=np.float64)
    intercept = np.asarray(artifact["intercept"], dtype=np.float64).reshape(-1)
    scaler_mean = np.asarray(artifact["scaler_mean"], dtype=np.float64).reshape(-1)
    scaler_scale = np.asarray(artifact["scaler_scale"], dtype=np.float64).reshape(-1)
    schema_version = np.asarray(artifact["schema_version"]).tolist()

    source_edges = pd.read_csv(paths["source_edges"], low_memory=False)
    external_edges = pd.read_csv(paths["external_edges"], low_memory=False)
    if "edge" not in source_edges or "edge" not in external_edges:
        die("Both edge metadata files must contain an 'edge' column")
    source_axis = source_edges["edge"].astype(str).tolist()
    external_axis = external_edges["edge"].astype(str).tolist()
    if len(source_axis) != len(set(source_axis)) or len(external_axis) != len(set(external_axis)):
        die("Frozen source or external edge axis contains duplicate edge identifiers")
    if source_axis != external_axis:
        die("External edge axis does not exactly equal the frozen source edge axis")
    source_edge_sha256 = sha256_file(paths["source_edges"])
    external_edge_sha256 = sha256_file(paths["external_edges"])
    if source_edge_sha256 != EXPECTED_EDGE_METADATA_SHA256:
        die(
            "Source edge metadata is not the audited frozen axis: "
            f"expected SHA256={EXPECTED_EDGE_METADATA_SHA256}, observed={source_edge_sha256}"
        )
    if external_edge_sha256 != EXPECTED_EDGE_METADATA_SHA256:
        die(
            "External edge metadata is not a byte-for-byte copy of the audited frozen axis: "
            f"expected SHA256={EXPECTED_EDGE_METADATA_SHA256}, observed={external_edge_sha256}"
        )

    n_features = 2 * len(source_axis)
    # The audited legacy artifact stores its binary coefficient vector flat,
    # while newer sklearn-style artifacts retain the leading class dimension.
    # Both encode the same single binary decision function.
    if coef.shape == (n_features,):
        coef = coef.reshape(1, n_features)
    elif coef.shape != (1, n_features):
        die(
            f"Expected binary coefficient shape ({n_features},) or (1,{n_features}), "
            f"found {coef.shape}"
        )
    if intercept.shape != (1,):
        die(f"Expected one binary intercept, found {intercept.shape}")
    if scaler_mean.shape != (n_features,) or scaler_scale.shape != (n_features,):
        die("Saved scaler dimensions do not match the frozen feature axis")
    if not all(np.isfinite(x).all() for x in (coef, intercept, scaler_mean, scaler_scale)):
        die("Model artifact contains non-finite parameters")
    if not np.all(scaler_scale > 0):
        die("Saved scaler contains a non-positive scale")

    return {
        "coef": coef,
        "intercept": intercept,
        "scaler_mean": scaler_mean,
        "scaler_scale": scaler_scale,
        "schema_version": schema_version,
        "model_sha256": model_sha256,
        "source_axis": source_axis,
        "n_features": n_features,
    }


def prove_saved_scaler(
    paths: dict[str, Path], model: dict[str, Any]
) -> tuple[dict[str, Any], str]:
    source_sparse = load_npz(paths["source_matrix"]).tocsr()
    source_meta = pd.read_csv(paths["source_metadata"], low_memory=False)
    manifest = pd.read_csv(paths["source_manifest"], low_memory=False)

    source_sample_col = find_column(
        source_meta,
        ("sample_id", "sample", "External ID", "external_id", "id"),
        "source metadata sample ID",
    )
    manifest_sample_col = find_column(
        manifest,
        ("sample_id", "sample", "External ID", "external_id", "id"),
        "manifest sample ID",
    )
    manifest_label_col = find_column(
        manifest,
        ("label", "class", "target", "condition", "cond"),
        "manifest task label",
    )

    if source_sparse.shape[0] != len(source_meta):
        die("Source feature rows do not match source metadata")
    if source_sparse.shape[1] != model["n_features"]:
        die("Source feature columns do not match the legacy model")
    if not np.isfinite(source_sparse.data).all():
        die("Source faithful feature matrix contains non-finite stored values")

    source_ids = source_meta[source_sample_col].map(normalise_sample_id)
    if source_ids.duplicated().any():
        die("Source metadata sample IDs are not unique")
    source_lookup = {sample_id: index for index, sample_id in enumerate(source_ids)}

    manifest_work = manifest[[manifest_sample_col, manifest_label_col]].copy()
    manifest_work["sample_id_normalised"] = manifest_work[manifest_sample_col].map(normalise_sample_id)
    manifest_work["y_binary"] = manifest_work[manifest_label_col].map(binary_label)
    master = manifest_work[["sample_id_normalised", "y_binary"]].drop_duplicates()
    if master["sample_id_normalised"].duplicated().any():
        die("Locked manifest contains conflicting task labels")
    if set(master["y_binary"]) != {0, 1}:
        die("Locked manifest does not contain both nonIBD and IBD")

    missing_source = sorted(set(master["sample_id_normalised"]) - set(source_lookup))
    if missing_source:
        die(f"Locked manifest samples missing from source matrix: {missing_source[:10]}")
    rows = np.asarray([source_lookup[s] for s in master["sample_id_normalised"]], dtype=int)

    print(f"Recomputing source scaler on {len(rows)} locked samples x {model['n_features']} features...")
    # Historical loaders exist in both float64 and float32 generations. Prove
    # which representation created this legacy artifact instead of assuming it.
    reproduction: dict[str, dict[str, Any]] = {}
    for dtype_name, dtype in (("float64", np.float64), ("float32", np.float32)):
        dense = source_sparse[rows].toarray().astype(dtype, copy=False)
        scaler = StandardScaler(with_mean=True, with_std=True)
        scaler.fit(dense)
        mean_diff = float(np.max(np.abs(scaler.mean_ - model["scaler_mean"])))
        scale_diff = float(np.max(np.abs(scaler.scale_ - model["scaler_scale"])))
        mean_match = bool(
            np.allclose(scaler.mean_, model["scaler_mean"], rtol=1e-7, atol=1e-9)
        )
        scale_match = bool(
            np.allclose(scaler.scale_, model["scaler_scale"], rtol=1e-7, atol=1e-9)
        )
        reproduction[dtype_name] = {
            "mean_matches": mean_match,
            "scale_matches": scale_match,
            "max_abs_mean_difference": mean_diff,
            "max_abs_scale_difference": scale_diff,
        }
        del dense, scaler
        gc.collect()
    del source_sparse
    gc.collect()

    compatible = [
        dtype_name
        for dtype_name, result in reproduction.items()
        if result["mean_matches"] and result["scale_matches"]
    ]
    if not compatible:
        die(
            "Legacy model scaler does not reproduce from the locked source cohort under "
            f"either historical input dtype: {json.dumps(reproduction, sort_keys=True)}"
        )
    training_input_dtype = min(
        compatible,
        key=lambda name: reproduction[name]["max_abs_mean_difference"]
        + reproduction[name]["max_abs_scale_difference"],
    )
    chosen = reproduction[training_input_dtype]

    proof = {
        "locked_source_samples": int(len(rows)),
        "source_class_counts": {
            "nonIBD": int(np.sum(master["y_binary"].to_numpy() == 0)),
            "IBD": int(np.sum(master["y_binary"].to_numpy() == 1)),
        },
        "training_input_dtype_selected": training_input_dtype,
        "saved_scaler_mean_matches_recomputed": chosen["mean_matches"],
        "saved_scaler_scale_matches_recomputed": chosen["scale_matches"],
        "max_abs_scaler_mean_difference": chosen["max_abs_mean_difference"],
        "max_abs_scaler_scale_difference": chosen["max_abs_scale_difference"],
        "dtype_reproduction_audit": reproduction,
    }
    return proof, training_input_dtype


def predict_without_labels(paths: dict[str, Path], model: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    external_sparse = load_npz(paths["external_matrix"]).tocsr()
    identity = pd.read_csv(
        paths["external_metadata"],
        usecols=["sample_id", PARTICIPANT_COLUMN],
        dtype=str,
    )
    if external_sparse.shape != (len(identity), model["n_features"]):
        die(
            f"External matrix/metadata/model dimension mismatch: matrix={external_sparse.shape}, "
            f"metadata={len(identity)}, model_features={model['n_features']}"
        )
    if not np.isfinite(external_sparse.data).all():
        die("External frozen-order feature matrix contains non-finite values")

    n_edges = model["n_features"] // 2
    B = external_sparse[:, :n_edges]
    K0 = external_sparse[:, n_edges:]
    if B.nnz and not np.isin(B.data, (0.0, 1.0)).all():
        die("External B block is not binary")
    orphan_k0 = K0 - K0.multiply(B)
    orphan_k0.eliminate_zeros()
    if orphan_k0.nnz:
        die("External K0 block has nonzero coordinates where B=0")

    feature_provenance = json.loads(paths["external_feature_provenance"].read_text())
    expected_shape = [int(external_sparse.shape[0]), int(external_sparse.shape[1])]
    if feature_provenance.get("shape") != expected_shape:
        die(
            "External feature provenance shape disagrees with the matrix: "
            f"provenance={feature_provenance.get('shape')}, matrix={expected_shape}"
        )
    if feature_provenance.get("n_frozen_edges") != n_edges:
        die("External feature provenance has the wrong frozen-edge count")
    if feature_provenance.get("training_edge_metadata_sha256") != EXPECTED_EDGE_METADATA_SHA256:
        die("External feature provenance does not name the audited frozen edge-axis hash")
    if feature_provenance.get("ricci_parameters") != EXPECTED_RICCI_PARAMETERS:
        die(
            "External feature provenance has unexpected Ricci parameters: "
            f"{feature_provenance.get('ricci_parameters')}"
        )
    if identity["sample_id"].duplicated().any():
        die("External sample IDs are not unique")

    participant = identity[PARTICIPANT_COLUMN].astype(str).str.strip()
    invalid = participant.str.lower().isin({"", "nan", "none", "null", "missing", "unknown"})
    if invalid.any():
        die(f"Invalid {PARTICIPANT_COLUMN} values for samples: {identity.loc[invalid, 'sample_id'].tolist()}")

    deployment_dtype = np.dtype(model["training_input_dtype"])
    dense = external_sparse.toarray().astype(deployment_dtype, copy=False)
    standardised = (dense - model["scaler_mean"]) / model["scaler_scale"]
    logits = standardised @ model["coef"].reshape(-1) + float(model["intercept"][0])
    probability_ibd = expit(logits)
    if not np.isfinite(probability_ibd).all():
        die("External probabilities are non-finite")

    predictions = identity.copy()
    predictions = predictions.rename(columns={PARTICIPANT_COLUMN: "participant_id"})
    predictions["logit_IBD"] = logits
    predictions["proba_nonIBD"] = 1.0 - probability_ibd
    predictions["proba_IBD"] = probability_ibd
    predictions["pred"] = (probability_ibd >= 0.5).astype(int)
    predictions["pred_label"] = np.where(predictions["pred"].eq(1), "IBD", "nonIBD")

    audit = pd.read_csv(paths["external_vectorization_audit"], low_memory=False)
    if "sample_id" not in audit or audit["sample_id"].astype(str).duplicated().any():
        die("Vectorization audit has missing or duplicated sample IDs")
    predictions = predictions.merge(audit, on="sample_id", how="left", validate="one_to_one")
    if predictions["ricci_active_edges"].isna().any():
        die("Some prediction samples are absent from the vectorization audit")

    # Return probabilities before the external label column has been read.
    return predictions, identity


def attach_labels_and_evaluate(
    paths: dict[str, Path],
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], np.ndarray, np.ndarray]:
    label_meta = pd.read_csv(
        paths["external_metadata"],
        usecols=["sample_id", "external_label_ibd_binary"],
        dtype=str,
    )
    label_meta["y_true"] = label_meta["external_label_ibd_binary"].map(binary_label)
    labelled = predictions.merge(
        label_meta[["sample_id", "external_label_ibd_binary", "y_true"]],
        on="sample_id",
        how="left",
        validate="one_to_one",
    )
    if labelled["y_true"].isna().any():
        die("Missing external binary labels after frozen probabilities were computed")
    labelled["y_true"] = labelled["y_true"].astype(int)
    labelled["true_label"] = labelled["y_true"].map({0: "nonIBD", 1: "IBD"})

    conflicts = labelled.groupby("participant_id")["y_true"].nunique().gt(1)
    if conflicts.any():
        die(f"Participants with conflicting external labels: {conflicts[conflicts].index.tolist()}")

    participant = labelled.groupby("participant_id", as_index=False).agg(
        sample_count=("sample_id", "size"),
        y_true=("y_true", "first"),
        true_label=("true_label", "first"),
        proba_nonIBD=("proba_nonIBD", "mean"),
        proba_IBD=("proba_IBD", "mean"),
    )
    participant["pred"] = (participant["proba_IBD"] >= 0.5).astype(int)
    participant["pred_label"] = participant["pred"].map({0: "nonIBD", 1: "IBD"})

    sample_metrics, sample_cm = metric_bundle(
        labelled["y_true"].to_numpy(dtype=int),
        labelled["proba_IBD"].to_numpy(dtype=float),
    )
    participant_metrics, participant_cm = metric_bundle(
        participant["y_true"].to_numpy(dtype=int),
        participant["proba_IBD"].to_numpy(dtype=float),
    )
    metrics = {
        "sample": sample_metrics,
        "participant_probability_averaged": participant_metrics,
    }
    return labelled, participant, metrics, sample_cm, participant_cm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    paths = verify_required_files()
    print("=" * 96)
    print("FROZEN RICCI IBD-vs-nonIBD C=0.02 EXTERNAL DEPLOYMENT")
    print("=" * 96)
    print("Model:", paths["model"])
    print("External feature matrix:", paths["external_matrix"])

    model = load_and_verify_model(paths)
    print(f"Legacy artifact schema: {model['schema_version']}")
    print(f"Frozen features: {model['n_features']}")
    print(f"Selected coefficients: {int(np.count_nonzero(model['coef']))}")

    scaler_proof, training_input_dtype = prove_saved_scaler(paths, model)
    model["training_input_dtype"] = training_input_dtype
    print("Saved scaler exactly reproduces from locked source cohort: PASS")
    print(json.dumps(scaler_proof, indent=2))

    predictions, _identity = predict_without_labels(paths, model)
    print(f"Frozen probabilities computed without external labels: {len(predictions)} samples")

    labelled, participants, metrics, sample_cm, participant_cm = attach_labels_and_evaluate(
        paths, predictions
    )
    print("\nSample metrics:")
    print(json.dumps(metrics["sample"], indent=2))
    print("Sample confusion [rows=true, cols=pred; nonIBD, IBD]:")
    print(sample_cm)
    print("\nParticipant probability-averaged metrics:")
    print(json.dumps(metrics["participant_probability_averaged"], indent=2))
    print("Participant confusion [rows=true, cols=pred; nonIBD, IBD]:")
    print(participant_cm)

    if args.verify_only:
        print("\nVERIFY-ONLY COMPLETE: no result files written.")
        return

    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        die(f"Output directory is non-empty; use --overwrite only after inspecting it: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    atomic_csv(labelled, output_dir / "sample_predictions.csv")
    atomic_csv(participants, output_dir / "participant_predictions_probability_averaged.csv")
    atomic_csv(
        pd.DataFrame(sample_cm, index=CLASS_NAMES, columns=CLASS_NAMES),
        output_dir / "sample_confusion_matrix.csv",
        index=True,
    )
    atomic_csv(
        pd.DataFrame(participant_cm, index=CLASS_NAMES, columns=CLASS_NAMES),
        output_dir / "participant_confusion_matrix_probability_averaged.csv",
        index=True,
    )
    atomic_json(output_dir / "external_metrics.json", metrics)

    feature_provenance = json.loads(paths["external_feature_provenance"].read_text())
    provenance = {
        "completed_utc": utc_now(),
        "analysis": "Serrano-Gomez GE50 structural-frozen-IBDMDB version 1 external validation",
        "model": {
            "path": str(paths["model"]),
            "sha256": model["model_sha256"],
            "C": MODEL_C,
            "class_order": list(CLASS_NAMES),
            "positive_class": "IBD",
            "threshold": 0.5,
            "artifact_schema_version": model["schema_version"],
            "selected_coefficients": int(np.count_nonzero(model["coef"])),
        },
        "source_validation": scaler_proof,
        "frozen_inputs": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in paths.items()
        },
        "external_feature_provenance": feature_provenance,
        "external_samples": int(len(labelled)),
        "external_participants": int(len(participants)),
        "participant_identifier": PARTICIPANT_COLUMN,
        "participant_prediction_rule": "arithmetic mean of frozen sample P(IBD), then fixed threshold 0.5",
        "leakage_control": (
            "External labels were not used for graph construction, vectorization, scaling, "
            "coefficient fitting, calibration, threshold selection, or probability generation. "
            "They were attached only after frozen probabilities had been computed."
        ),
    }
    atomic_json(output_dir / "DEPLOYMENT_PROVENANCE.json", provenance)
    atomic_json(
        output_dir / "RUN_COMPLETE.json",
        {
            "completed_utc": utc_now(),
            "status": "complete",
            "sample_predictions": int(len(labelled)),
            "participant_predictions": int(len(participants)),
            "model_sha256": provenance["model"]["sha256"],
        },
    )
    print("\nOUTPUTS WRITTEN:", output_dir)


if __name__ == "__main__":
    main()
