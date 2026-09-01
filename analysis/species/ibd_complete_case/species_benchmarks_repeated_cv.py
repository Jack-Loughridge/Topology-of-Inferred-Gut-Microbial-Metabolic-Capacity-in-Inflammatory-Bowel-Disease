#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Repeated participant-grouped species-abundance benchmarks for IBD vs non-IBD.

This program evaluates three fixed manuscript benchmark models on the exact
20 x 5 outer split manifest created by the repeated Ricci analysis:

  * L1 logistic regression
  * random forest
  * XGBoost

Methodological safeguards
-------------------------
* The split manifest is loaded directly; folds are never regenerated.
* Species, label, and participant metadata are restricted to manifest samples
  before consistency checks.
* Samples without any nonzero canonical species-level abundance are excluded
  explicitly as a complete-case rule while retaining their original fold assignments
  for every remaining sample.
* CLR transformation is per sample, while variance filtering and scaling are
  fitted on outer-training samples only.
* Class imbalance is handled exactly once through fold-specific balanced
  sample weights. XGBoost scale_pos_weight remains 1.0.
* Sample-level and participant-aggregated held-out metrics are both reported.
* The primary uncertainty distribution contains one pooled OOF estimate per
  repetition (20 values), not 100 fold values treated as independent.
* Every fold is resume-safe and stores predictions, preprocessing information,
  feature diagnostics, complexity diagnostics, and convergence diagnostics.

The default hyperparameters preserve the corrected manuscript baselines.
They are fixed rather than tuned on outer-test data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import sys
import tempfile
import time
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight


SCHEMA_VERSION = "species-benchmarks-repeated-cv-v1.1-complete-case"
CODE_VERSION = "1.1.0"
MODEL_ORDER = ["logistic_l1", "random_forest", "xgboost"]
MODEL_LABELS = {
    "logistic_l1": "Species L1 logistic",
    "random_forest": "Species random forest",
    "xgboost": "Species XGBoost",
}
CLASS_NAMES = {0: "non-IBD", 1: "IBD"}
EPS_METRIC = 1e-15


# =============================================================================
# Configuration
# =============================================================================

@dataclass(frozen=True)
class RunConfig:
    species_file: str
    species_sheet: str
    label_csv: str
    metadata_csv: str
    split_manifest: str
    output_dir: str
    models: tuple[str, ...]
    expected_repeats: int
    expected_folds: int
    clr_pseudocount: float
    variance_threshold: float
    logistic_c: float
    logistic_max_iter: int
    logistic_tol: float
    rf_trees: int
    rf_min_samples_leaf: int
    rf_max_features: str
    xgb_estimators: int
    xgb_max_depth: int
    xgb_learning_rate: float
    xgb_subsample: float
    xgb_colsample_bytree: float
    xgb_reg_lambda: float
    xgb_reg_alpha: float
    n_jobs: int
    base_model_seed: int
    weighting_mode: str
    save_fold_models: bool
    fit_full_source_models: bool
    make_plots: bool
    calibration_bins: int
    zero_profile_policy: str
    expected_zero_profiles: int


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    base = Path.home() / "Real_Data"
    parser = argparse.ArgumentParser(
        description="Run fixed species-abundance benchmarks on the exact repeated Ricci folds."
    )
    parser.add_argument(
        "--species-file",
        type=Path,
        default=base / "Real_Species_Abundances_canon.xlsx",
    )
    parser.add_argument("--species-sheet", default="Sheet1")
    parser.add_argument("--label-csv", type=Path, default=base / "sample_labels.csv")
    parser.add_argument("--metadata-csv", type=Path, default=base / "hmp2_metadata.csv")
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=base
        / "Ricci_IBD_RepeatedCV_CPath"
        / "splits"
        / "sample_split_manifest.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=base / "Species_Benchmarks_RepeatedCV_IBD_CompleteCase",
    )
    parser.add_argument(
        "--models",
        default=",".join(MODEL_ORDER),
        help="Comma-separated subset of: logistic_l1,random_forest,xgboost",
    )
    parser.add_argument("--expected-repeats", type=int, default=20)
    parser.add_argument("--expected-folds", type=int, default=5)
    parser.add_argument("--clr-pseudocount", type=float, default=1e-6)
    parser.add_argument("--variance-threshold", type=float, default=1e-10)

    parser.add_argument("--logistic-c", type=float, default=0.2)
    parser.add_argument("--logistic-max-iter", type=int, default=10000)
    parser.add_argument("--logistic-tol", type=float, default=1e-4)

    parser.add_argument("--rf-trees", type=int, default=1000)
    parser.add_argument("--rf-min-samples-leaf", type=int, default=3)
    parser.add_argument("--rf-max-features", default="sqrt")

    parser.add_argument("--xgb-estimators", type=int, default=700)
    parser.add_argument("--xgb-max-depth", type=int, default=3)
    parser.add_argument("--xgb-learning-rate", type=float, default=0.02)
    parser.add_argument("--xgb-subsample", type=float, default=0.85)
    parser.add_argument("--xgb-colsample-bytree", type=float, default=0.85)
    parser.add_argument("--xgb-reg-lambda", type=float, default=2.0)
    parser.add_argument("--xgb-reg-alpha", type=float, default=0.25)

    parser.add_argument(
        "--n-jobs",
        type=int,
        default=int(os.environ.get("BENCHMARK_N_JOBS", "2")),
        help="Threads used by RF/XGBoost. Keep small when Ricci/H0 run concurrently.",
    )
    parser.add_argument("--base-model-seed", type=int, default=25173)
    parser.add_argument(
        "--weighting-mode",
        choices=["class_balanced", "class_participant_balanced"],
        default="class_balanced",
        help=(
            "Default preserves manuscript benchmarks. class_participant_balanced is an "
            "optional sensitivity in which each participant has equal total weight within class."
        ),
    )
    parser.add_argument("--save-fold-models", action="store_true")
    parser.add_argument(
        "--no-full-source-models",
        action="store_true",
        help="Skip one final all-source model per benchmark.",
    )
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--calibration-bins", type=int, default=10)
    parser.add_argument(
        "--zero-profile-policy",
        choices=["exclude", "error"],
        default="exclude",
        help=(
            "How to handle manifest samples whose canonical species vector sums to zero. "
            "The IBDMDB complete-case analysis uses exclude."
        ),
    )
    parser.add_argument(
        "--expected-zero-profiles",
        type=int,
        default=10,
        help=(
            "Expected number of all-zero canonical species profiles. Use -1 to disable "
            "the exact-count assertion. The curated IBDMDB input is expected to have 10."
        ),
    )
    parser.add_argument(
        "--overwrite-incompatible-output",
        action="store_true",
        help="Delete an existing incompatible output directory instead of stopping.",
    )
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> RunConfig:
    models = tuple(x.strip() for x in args.models.split(",") if x.strip())
    invalid = sorted(set(models) - set(MODEL_ORDER))
    if invalid:
        raise ValueError(f"Unknown models: {invalid}. Allowed={MODEL_ORDER}")
    if not models:
        raise ValueError("At least one model must be requested.")
    if args.n_jobs < 1:
        raise ValueError("--n-jobs must be >= 1")
    if args.expected_repeats < 1 or args.expected_folds < 2:
        raise ValueError("Expected repeats/folds are invalid.")
    if args.clr_pseudocount <= 0:
        raise ValueError("CLR pseudocount must be positive.")
    if args.calibration_bins < 2:
        raise ValueError("Calibration bins must be >= 2.")
    if args.expected_zero_profiles < -1:
        raise ValueError("--expected-zero-profiles must be -1 or a nonnegative integer.")

    return RunConfig(
        species_file=str(Path(args.species_file).expanduser().resolve()),
        species_sheet=str(args.species_sheet),
        label_csv=str(Path(args.label_csv).expanduser().resolve()),
        metadata_csv=str(Path(args.metadata_csv).expanduser().resolve()),
        split_manifest=str(Path(args.split_manifest).expanduser().resolve()),
        output_dir=str(Path(args.output_dir).expanduser().resolve()),
        models=models,
        expected_repeats=int(args.expected_repeats),
        expected_folds=int(args.expected_folds),
        clr_pseudocount=float(args.clr_pseudocount),
        variance_threshold=float(args.variance_threshold),
        logistic_c=float(args.logistic_c),
        logistic_max_iter=int(args.logistic_max_iter),
        logistic_tol=float(args.logistic_tol),
        rf_trees=int(args.rf_trees),
        rf_min_samples_leaf=int(args.rf_min_samples_leaf),
        rf_max_features=str(args.rf_max_features),
        xgb_estimators=int(args.xgb_estimators),
        xgb_max_depth=int(args.xgb_max_depth),
        xgb_learning_rate=float(args.xgb_learning_rate),
        xgb_subsample=float(args.xgb_subsample),
        xgb_colsample_bytree=float(args.xgb_colsample_bytree),
        xgb_reg_lambda=float(args.xgb_reg_lambda),
        xgb_reg_alpha=float(args.xgb_reg_alpha),
        n_jobs=int(args.n_jobs),
        base_model_seed=int(args.base_model_seed),
        weighting_mode=str(args.weighting_mode),
        save_fold_models=bool(args.save_fold_models),
        fit_full_source_models=not bool(args.no_full_source_models),
        make_plots=not bool(args.no_plots),
        calibration_bins=int(args.calibration_bins),
        zero_profile_policy=str(args.zero_profile_policy),
        expected_zero_profiles=int(args.expected_zero_profiles),
    )


# =============================================================================
# General utilities
# =============================================================================

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalise_sample_id(value: Any) -> str:
    text = str(value).strip()
    text = os.path.basename(text)
    for suffix in (".npy", "_H0"):
        if text.endswith(suffix):
            text = text[: -len(suffix)]
    return text


def find_col(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    lookup = {str(c).strip().lower(): c for c in df.columns}
    for candidate in candidates:
        key = candidate.strip().lower()
        if key in lookup:
            return lookup[key]
    return None


def normalise_binary_label(value: Any) -> Optional[int]:
    text = str(value).strip().lower()
    if text in {"0", "nonibd", "non-ibd", "non ibd", "non_ibd", "control", "healthy"}:
        return 0
    if text in {"1", "ibd", "cd", "uc"}:
        return 1
    if "crohn" in text or "ulcerative" in text:
        return 1
    return None


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    return value


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8") as tmp:
        tmp.write(text)
        temp_name = tmp.name
    os.replace(temp_name, path)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_text(path, json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n")


def atomic_to_csv(df: pd.DataFrame, path: Path, index: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, suffix=".csv") as tmp:
        temp_name = tmp.name
    try:
        df.to_csv(temp_name, index=index)
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def atomic_joblib_dump(obj: Any, path: Path, compress: int = 3) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False, suffix=".joblib") as tmp:
        temp_name = tmp.name
    try:
        joblib.dump(obj, temp_name, compress=compress)
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def set_thread_environment(n_jobs: int) -> None:
    # Avoid nested oversubscription. Explicit launcher variables take precedence.
    for name in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        os.environ.setdefault(name, str(max(1, n_jobs)))


def package_versions() -> dict[str, Any]:
    result: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "joblib": joblib.__version__,
    }
    try:
        import xgboost

        result["xgboost"] = xgboost.__version__
    except Exception as exc:  # pragma: no cover - environment dependent
        result["xgboost"] = None
        result["xgboost_import_error"] = repr(exc)
    return result


def run_config_payload(config: RunConfig) -> dict[str, Any]:
    input_paths = {
        "species_file": Path(config.species_file),
        "label_csv": Path(config.label_csv),
        "metadata_csv": Path(config.metadata_csv),
        "split_manifest": Path(config.split_manifest),
    }
    fingerprints = {}
    for name, path in input_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing required input {name}: {path}")
        fingerprints[name] = {
            "path": str(path),
            "size_bytes": int(path.stat().st_size),
            "sha256": sha256_file(path),
        }
    return json_ready({
        "schema_version": SCHEMA_VERSION,
        "code_version": CODE_VERSION,
        "created_utc": now_iso(),
        "config": asdict(config),
        "input_fingerprints": fingerprints,
        "software": package_versions(),
        "method_notes": {
            "task": "IBD (UC+CD) versus non-IBD",
            "outer_folds": "Loaded exactly from Ricci sample_split_manifest.csv",
            "preprocessing": "Per-sample CLR; train-only variance filter; train-only StandardScaler",
            "weighting": config.weighting_mode,
            "xgboost_imbalance": "balanced sample_weight only; scale_pos_weight fixed to 1.0",
            "performance_distribution": "20 repetition-level pooled OOF metrics",
            "fold_metrics": "100 descriptive fold results per model; not independent replicates",
            "species_complete_case": (
                "Exclude canonical all-zero species vectors, preserve original repeat/fold/role "
                "assignments for every retained sample, and record exclusions explicitly"
            ),
            "zero_profile_policy": config.zero_profile_policy,
            "expected_zero_profiles": config.expected_zero_profiles,
        },
    })


def ensure_compatible_output(
    config: RunConfig,
    overwrite_incompatible: bool,
) -> dict[str, Any]:
    output_dir = Path(config.output_dir)
    payload = run_config_payload(config)
    config_path = output_dir / "run_config.json"

    if config_path.exists():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        comparable_existing = dict(existing)
        comparable_new = dict(payload)
        comparable_existing.pop("created_utc", None)
        comparable_new.pop("created_utc", None)
        # Software patch versions do not invalidate completed deterministic outputs;
        # input fingerprints and method parameters do.
        comparable_existing.pop("software", None)
        comparable_new.pop("software", None)
        if comparable_existing != comparable_new:
            if not overwrite_incompatible:
                raise RuntimeError(
                    f"Output directory {output_dir} contains an incompatible run. "
                    "Use a new output directory or --overwrite-incompatible-output."
                )
            shutil.rmtree(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            atomic_write_json(config_path, payload)
        else:
            print(f"[resume] Compatible run configuration found in {output_dir}", flush=True)
            payload = existing
    else:
        if output_dir.exists() and any(output_dir.iterdir()):
            if not overwrite_incompatible:
                raise RuntimeError(
                    f"Output directory {output_dir} is non-empty but has no run_config.json. "
                    "Use a new directory or --overwrite-incompatible-output."
                )
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(config_path, payload)

    return payload


# =============================================================================
# Manifest and input validation
# =============================================================================

def load_and_validate_manifest(
    path: Path,
    expected_repeats: int,
    expected_folds: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest = pd.read_csv(path, low_memory=False)
    required = {"repeat", "fold", "role", "sample_id", "participant_id", "label"}
    missing_cols = sorted(required - set(manifest.columns))
    if missing_cols:
        raise ValueError(f"Split manifest is missing columns: {missing_cols}")

    manifest = manifest.copy()
    manifest["repeat"] = pd.to_numeric(manifest["repeat"], errors="raise").astype(int)
    manifest["fold"] = pd.to_numeric(manifest["fold"], errors="raise").astype(int)
    manifest["role"] = manifest["role"].astype(str).str.strip().str.lower()
    manifest["sample_id"] = manifest["sample_id"].map(normalise_sample_id)
    manifest["participant_id"] = manifest["participant_id"].astype(str).str.strip()
    manifest["label"] = pd.to_numeric(manifest["label"], errors="raise").astype(int)

    if manifest[["sample_id", "participant_id"]].eq("").any().any():
        raise ValueError("Manifest contains blank sample_id or participant_id values.")
    if set(manifest["role"].unique()) != {"train", "test"}:
        raise ValueError(f"Manifest roles must be train/test; found {sorted(manifest['role'].unique())}")
    if not set(manifest["label"].unique()).issubset({0, 1}) or set(manifest["label"].unique()) != {0, 1}:
        raise ValueError(f"Manifest labels must contain exactly 0 and 1; found {sorted(manifest['label'].unique())}")

    repeats = sorted(manifest["repeat"].unique())
    folds = sorted(manifest["fold"].unique())
    if repeats != list(range(1, expected_repeats + 1)):
        raise ValueError(f"Expected repeats 1..{expected_repeats}, found {repeats}")
    if folds != list(range(1, expected_folds + 1)):
        raise ValueError(f"Expected folds 1..{expected_folds}, found {folds}")

    consistency = manifest.groupby("sample_id").agg(
        participant_n=("participant_id", "nunique"),
        label_n=("label", "nunique"),
    )
    bad = consistency[(consistency["participant_n"] != 1) | (consistency["label_n"] != 1)]
    if len(bad):
        raise ValueError(f"Manifest sample metadata conflicts: {bad.index.astype(str).tolist()[:10]}")

    sample_master = (
        manifest[["sample_id", "participant_id", "label"]]
        .drop_duplicates("sample_id")
        .sort_values("sample_id")
        .reset_index(drop=True)
    )
    participant_labels = sample_master.groupby("participant_id")["label"].nunique()
    mixed = participant_labels[participant_labels != 1]
    if len(mixed):
        raise ValueError(f"Participants have conflicting binary labels: {mixed.index.astype(str).tolist()[:10]}")

    all_samples = set(sample_master["sample_id"])
    n_samples = len(all_samples)
    for repeat_idx in repeats:
        repeat_rows = manifest[manifest["repeat"] == repeat_idx]
        test_count = repeat_rows[repeat_rows["role"] == "test"].groupby("sample_id").size()
        if set(test_count.index) != all_samples or not (test_count == 1).all():
            raise ValueError(f"Repeat {repeat_idx}: every sample must occur exactly once as test.")

        for fold_idx in folds:
            fold_rows = repeat_rows[repeat_rows["fold"] == fold_idx]
            duplicate_roles = fold_rows.groupby("sample_id")["role"].nunique()
            if (duplicate_roles != 1).any() or fold_rows["sample_id"].nunique() != n_samples:
                raise ValueError(
                    f"Repeat {repeat_idx}, fold {fold_idx}: train/test roles do not partition all samples."
                )
            if fold_rows.duplicated(["sample_id", "role"]).any():
                raise ValueError(f"Repeat {repeat_idx}, fold {fold_idx}: duplicate sample-role rows.")

            train = fold_rows[fold_rows["role"] == "train"]
            test = fold_rows[fold_rows["role"] == "test"]
            overlap = set(train["participant_id"]) & set(test["participant_id"])
            if overlap:
                raise ValueError(
                    f"Participant leakage in repeat {repeat_idx}, fold {fold_idx}: {sorted(overlap)[:10]}"
                )
            if set(train["label"]) != {0, 1} or set(test["label"]) != {0, 1}:
                raise ValueError(f"Repeat {repeat_idx}, fold {fold_idx} lacks a class.")

    return manifest, sample_master


def load_labels_for_verification(path: Path, required_ids: set[str]) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    sample_col = find_col(frame, ["External ID", "external_id", "sample_id", "sample", "id"])
    label_col = find_col(frame, ["label", "condition", "diagnosis", "disease", "status"])
    if sample_col is None or label_col is None:
        raise ValueError(f"Could not identify sample/label columns in {path}. Columns={list(frame.columns)}")

    out = frame[[sample_col, label_col]].copy()
    out.columns = ["sample_id", "source_label"]
    out["sample_id"] = out["sample_id"].map(normalise_sample_id)
    out = out[out["sample_id"].isin(required_ids)].copy()
    out["label"] = out["source_label"].map(normalise_binary_label)

    missing = sorted(required_ids - set(out["sample_id"]))
    if missing:
        raise ValueError(f"Label CSV is missing manifest samples: {missing[:10]}")
    if out["label"].isna().any():
        examples = out.loc[out["label"].isna(), ["sample_id", "source_label"]].head(10)
        raise ValueError(f"Unrecognized labels for required samples:\n{examples.to_string(index=False)}")
    conflicts = out.groupby("sample_id")["label"].nunique()
    bad = conflicts[conflicts != 1]
    if len(bad):
        raise ValueError(f"Conflicting required labels: {bad.index.astype(str).tolist()[:10]}")
    return out[["sample_id", "label"]].drop_duplicates("sample_id")


def load_metadata_for_verification(path: Path, required_ids: set[str]) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    sample_col = find_col(frame, ["External ID", "external_id", "sample_id", "sample", "id"])
    participant_col = find_col(
        frame,
        ["Participant ID", "participant_id", "participant", "subject_id", "host_subject_id"],
    )
    if sample_col is None or participant_col is None:
        raise ValueError(
            f"Could not identify sample/participant columns in {path}. Columns={list(frame.columns)}"
        )
    out = frame[[sample_col, participant_col]].copy()
    out.columns = ["sample_id", "participant_id"]
    out["sample_id"] = out["sample_id"].map(normalise_sample_id)
    out["participant_id"] = out["participant_id"].astype(str).str.strip()
    out = out[out["sample_id"].isin(required_ids)].copy()

    missing = sorted(required_ids - set(out["sample_id"]))
    if missing:
        raise ValueError(f"Metadata CSV is missing manifest samples: {missing[:10]}")
    if out["participant_id"].eq("").any() or out["participant_id"].str.lower().eq("nan").any():
        bad = out.loc[
            out["participant_id"].eq("") | out["participant_id"].str.lower().eq("nan"),
            "sample_id",
        ].tolist()
        raise ValueError(f"Missing participant IDs for manifest samples: {bad[:10]}")
    conflicts = out.groupby("sample_id")["participant_id"].nunique()
    bad = conflicts[conflicts != 1]
    if len(bad):
        raise ValueError(f"Conflicting participant IDs for manifest samples: {bad.index.astype(str).tolist()[:10]}")
    return out.drop_duplicates("sample_id")


def load_species_matrix(
    path: Path,
    sheet_name: str,
    sample_master: pd.DataFrame,
    zero_profile_policy: str,
    expected_zero_profiles: int,
) -> tuple[
    np.ndarray,
    list[str],
    pd.DataFrame,
    dict[str, Any],
    pd.DataFrame,
    pd.DataFrame,
]:
    """Load the canonical species matrix and apply the pre-specified complete-case rule.

    Samples whose full canonical species vector sums to zero contain no usable
    species-level composition. They are either rejected or excluded explicitly,
    depending on ``zero_profile_policy``. Exclusion is performed before any
    model fitting while preserving the original repeat/fold/role assignment for
    every retained sample.
    """
    frame = pd.read_excel(path, sheet_name=sheet_name)
    sample_col = find_col(frame, ["External ID", "external_id", "sample_id", "sample", "id"])
    if sample_col is None:
        raise ValueError(f"Could not find a sample ID column in {path}. Columns={list(frame.columns)}")

    frame = frame.copy()
    frame[sample_col] = frame[sample_col].map(normalise_sample_id)
    feature_cols = [c for c in frame.columns if c != sample_col]
    if not feature_cols:
        raise ValueError("Species table has no feature columns.")

    # Convert all feature columns in one operation. This avoids pandas frame
    # fragmentation when hundreds of columns are assigned one-by-one.
    original_features = frame[feature_cols]
    numeric = original_features.apply(pd.to_numeric, errors="coerce")
    nonblank = original_features.notna() & original_features.astype(str).apply(
        lambda column: column.str.strip().ne("")
    )
    invalid_mask = nonblank & numeric.isna()
    if invalid_mask.to_numpy().any():
        examples = []
        for column in feature_cols:
            bad = invalid_mask[column]
            if bad.any():
                examples.append(
                    {
                        "column": str(column),
                        "invalid_count": int(bad.sum()),
                        "examples": original_features.loc[bad, column].astype(str).head(3).tolist(),
                    }
                )
            if len(examples) >= 5:
                break
        raise ValueError(
            "Species feature columns contain nonnumeric values. Examples: "
            + json.dumps(examples)
        )

    raw_rows = len(frame)
    duplicate_rows = int(frame.duplicated(sample_col, keep=False).sum())
    missing_value_count = int(numeric.isna().to_numpy().sum())

    numeric_with_id = pd.concat(
        [
            frame[[sample_col]].rename(columns={sample_col: "sample_id"}),
            numeric.fillna(0.0),
        ],
        axis=1,
    )
    grouped = numeric_with_id.groupby("sample_id", as_index=False)[feature_cols].mean()

    required_ids = set(sample_master["sample_id"])
    available_ids = set(grouped["sample_id"])
    missing = sorted(required_ids - available_ids)
    if missing:
        raise ValueError(f"Species table is missing manifest samples: {missing[:10]}")
    extras = sorted(available_ids - required_ids)

    aligned = sample_master.merge(grouped, on="sample_id", how="left", validate="one_to_one")
    X_all = aligned[feature_cols].to_numpy(dtype=np.float64)
    if not np.isfinite(X_all).all():
        where = np.argwhere(~np.isfinite(X_all))[:10]
        raise ValueError(f"Species matrix contains nonfinite values after loading: {where.tolist()}")
    if (X_all < -1e-12).any():
        where = np.argwhere(X_all < -1e-12)[:10]
        raise ValueError(f"Species abundances contain negative values: {where.tolist()}")
    X_all = np.clip(X_all, 0.0, None)

    row_sums = X_all.sum(axis=1)
    nonzero_counts = (X_all > 0.0).sum(axis=1)
    usable_mask = row_sums > 0.0

    availability = aligned[["sample_id", "participant_id", "label"]].copy()
    availability["label_name"] = availability["label"].map(CLASS_NAMES)
    availability["species_row_sum"] = row_sums
    availability["nonzero_species_features"] = nonzero_counts.astype(int)
    availability["species_profile_available"] = usable_mask
    availability["analysis_status"] = np.where(usable_mask, "included", "excluded")
    availability["exclusion_reason"] = np.where(
        usable_mask,
        "",
        "No nonzero species-level abundance after canonical filtering",
    )

    excluded = availability.loc[~usable_mask].copy().reset_index(drop=True)
    n_zero = int(len(excluded))
    if expected_zero_profiles >= 0 and n_zero != expected_zero_profiles:
        raise ValueError(
            f"Expected {expected_zero_profiles} all-zero canonical species profiles, found {n_zero}. "
            "Inspect the species input or pass --expected-zero-profiles -1 only after documenting the difference."
        )
    if n_zero and zero_profile_policy == "error":
        raise ValueError(
            "Manifest samples have all-zero canonical species vectors: "
            + str(excluded["sample_id"].head(10).tolist())
        )

    included = availability.loc[usable_mask, ["sample_id", "participant_id", "label"]].copy()
    included = included.reset_index(drop=True)
    X = X_all[usable_mask]

    if len(included) == 0:
        raise ValueError("No usable species profiles remain after complete-case filtering.")
    if np.any(X.sum(axis=1) <= 0.0):
        raise RuntimeError("Internal complete-case filtering failure: an all-zero row remains.")

    diagnostics = {
        "raw_species_rows": int(raw_rows),
        "duplicate_input_rows": int(duplicate_rows),
        "unique_species_samples_after_aggregation": int(grouped["sample_id"].nunique()),
        "source_manifest_samples": int(len(sample_master)),
        "complete_case_samples": int(len(included)),
        "excluded_all_zero_profiles": int(n_zero),
        "excluded_sample_ids": excluded["sample_id"].astype(str).tolist(),
        "excluded_class_counts": excluded["label"].value_counts().sort_index().to_dict(),
        "excluded_participants": int(excluded["participant_id"].nunique()),
        "zero_profile_policy": zero_profile_policy,
        "expected_zero_profiles": int(expected_zero_profiles),
        "extra_species_samples_ignored": int(len(extras)),
        "extra_species_sample_examples": extras[:10],
        "n_features": int(len(feature_cols)),
        "missing_values_filled_with_zero": missing_value_count,
        "overall_zero_fraction_complete_case": float(np.mean(X == 0.0)),
        "row_sum_min_complete_case": float(X.sum(axis=1).min()),
        "row_sum_median_complete_case": float(np.median(X.sum(axis=1))),
        "row_sum_max_complete_case": float(X.sum(axis=1).max()),
    }
    return X, [str(c) for c in feature_cols], included, diagnostics, availability, excluded


def validate_complete_case_design(
    source_manifest: pd.DataFrame,
    source_master: pd.DataFrame,
    complete_master: pd.DataFrame,
    expected_repeats: int,
    expected_folds: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Filter the locked manifest and verify that its grouped design is preserved."""
    included_ids = set(complete_master["sample_id"].astype(str))
    manifest = source_manifest[source_manifest["sample_id"].isin(included_ids)].copy()
    if manifest.empty:
        raise ValueError("Complete-case split manifest is empty.")

    source_participants = set(source_master["participant_id"].astype(str))
    retained_participants = set(complete_master["participant_id"].astype(str))
    lost_participants = sorted(source_participants - retained_participants)
    if lost_participants:
        raise ValueError(
            "Complete-case filtering removed every species profile for participants: "
            + str(lost_participants[:10])
            + ". A new participant-level comparison design would be required."
        )

    fold_rows = []
    for repeat_idx in range(1, expected_repeats + 1):
        repeat_manifest = manifest[manifest["repeat"] == repeat_idx]
        test_counts = repeat_manifest.loc[repeat_manifest["role"] == "test", "sample_id"].value_counts()
        missing_test = sorted(included_ids - set(test_counts.index))
        if missing_test or not (test_counts == 1).all():
            raise ValueError(
                f"Complete-case samples must appear exactly once as test in repetition {repeat_idx}. "
                f"Missing examples={missing_test[:5]}"
            )

        for fold_idx in range(1, expected_folds + 1):
            rows = repeat_manifest[repeat_manifest["fold"] == fold_idx]
            train = rows[rows["role"] == "train"]
            test = rows[rows["role"] == "test"]
            train_participants = set(train["participant_id"])
            test_participants = set(test["participant_id"])
            overlap = sorted(train_participants & test_participants)
            if overlap:
                raise ValueError(
                    f"Participant leakage after complete-case filtering in repeat={repeat_idx}, "
                    f"fold={fold_idx}: {overlap[:10]}"
                )
            for role_name, role_frame in [("train", train), ("test", test)]:
                if set(role_frame["label"].unique()) != {0, 1}:
                    raise ValueError(
                        f"Complete-case {role_name} set lacks a class in repeat={repeat_idx}, fold={fold_idx}."
                    )
            fold_rows.append(
                {
                    "repeat": repeat_idx,
                    "fold": fold_idx,
                    "train_samples": int(len(train)),
                    "test_samples": int(len(test)),
                    "train_participants": int(train["participant_id"].nunique()),
                    "test_participants": int(test["participant_id"].nunique()),
                    "train_nonIBD_samples": int((train["label"] == 0).sum()),
                    "train_IBD_samples": int((train["label"] == 1).sum()),
                    "test_nonIBD_samples": int((test["label"] == 0).sum()),
                    "test_IBD_samples": int((test["label"] == 1).sum()),
                }
            )

    participant_counts = (
        complete_master.groupby(["participant_id", "label"], as_index=False)
        .size()
        .rename(columns={"size": "complete_case_sample_count"})
    )
    source_counts = (
        source_master.groupby(["participant_id", "label"], as_index=False)
        .size()
        .rename(columns={"size": "source_manifest_sample_count"})
    )
    participant_counts = source_counts.merge(
        participant_counts,
        on=["participant_id", "label"],
        how="left",
    )
    participant_counts["complete_case_sample_count"] = (
        participant_counts["complete_case_sample_count"].fillna(0).astype(int)
    )
    participant_counts["excluded_sample_count"] = (
        participant_counts["source_manifest_sample_count"]
        - participant_counts["complete_case_sample_count"]
    )
    participant_counts["label_name"] = participant_counts["label"].map(CLASS_NAMES)
    return manifest.reset_index(drop=True), pd.DataFrame(fold_rows), participant_counts


def load_and_verify_inputs(config: RunConfig) -> dict[str, Any]:
    source_manifest, source_master = load_and_validate_manifest(
        Path(config.split_manifest),
        expected_repeats=config.expected_repeats,
        expected_folds=config.expected_folds,
    )
    required_ids = set(source_master["sample_id"])
    labels = load_labels_for_verification(Path(config.label_csv), required_ids)
    metadata = load_metadata_for_verification(Path(config.metadata_csv), required_ids)

    check = source_master.merge(labels, on="sample_id", suffixes=("_manifest", "_labels"))
    label_mismatch = check[check["label_manifest"] != check["label_labels"]]
    if len(label_mismatch):
        raise ValueError(
            "Manifest/label CSV disagreement:\n"
            + label_mismatch.head(10).to_string(index=False)
        )

    check = source_master.merge(metadata, on="sample_id", suffixes=("_manifest", "_metadata"))
    participant_mismatch = check[
        check["participant_id_manifest"] != check["participant_id_metadata"]
    ]
    if len(participant_mismatch):
        raise ValueError(
            "Manifest/metadata participant disagreement:\n"
            + participant_mismatch.head(10).to_string(index=False)
        )

    X, feature_names, complete_master, species_diagnostics, availability, excluded = load_species_matrix(
        Path(config.species_file),
        config.species_sheet,
        source_master,
        zero_profile_policy=config.zero_profile_policy,
        expected_zero_profiles=config.expected_zero_profiles,
    )

    manifest, fold_counts, participant_counts = validate_complete_case_design(
        source_manifest,
        source_master,
        complete_master,
        expected_repeats=config.expected_repeats,
        expected_folds=config.expected_folds,
    )

    return {
        "manifest": manifest,
        "sample_master": complete_master,
        "source_manifest": source_manifest,
        "source_sample_master": source_master,
        "X": X,
        "feature_names": feature_names,
        "species_diagnostics": species_diagnostics,
        "species_availability": availability,
        "excluded_samples": excluded,
        "complete_case_fold_counts": fold_counts,
        "participant_complete_case_counts": participant_counts,
    }


# =============================================================================
# Preprocessing and weighting
# =============================================================================

def clr_transform(X: np.ndarray, pseudocount: float) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    if (X < 0).any() or not np.isfinite(X).all():
        raise ValueError("CLR input must be finite and nonnegative.")
    log_x = np.log(X + pseudocount)
    return log_x - log_x.mean(axis=1, keepdims=True)


@dataclass
class FittedPreprocessor:
    support_mask: np.ndarray
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    scaler_var: np.ndarray
    input_feature_names: list[str]
    selected_feature_names: list[str]
    pseudocount: float
    variance_threshold: float

    def transform(self, X_raw: np.ndarray) -> np.ndarray:
        clr = clr_transform(X_raw, self.pseudocount)
        selected = clr[:, self.support_mask]
        return (selected - self.scaler_mean) / self.scaler_scale


def fit_preprocessor(
    X_train_raw: np.ndarray,
    feature_names: list[str],
    pseudocount: float,
    variance_threshold: float,
) -> tuple[FittedPreprocessor, np.ndarray]:
    X_clr = clr_transform(X_train_raw, pseudocount)
    selector = VarianceThreshold(threshold=variance_threshold)
    X_selected = selector.fit_transform(X_clr)
    support = selector.get_support()
    if X_selected.shape[1] == 0:
        raise RuntimeError("VarianceThreshold removed every feature.")
    scaler = StandardScaler(with_mean=True, with_std=True)
    X_processed = scaler.fit_transform(X_selected)
    if not np.isfinite(X_processed).all():
        raise RuntimeError("Nonfinite values after training preprocessing.")
    selected_names = [feature_names[i] for i in np.flatnonzero(support)]
    fitted = FittedPreprocessor(
        support_mask=support.astype(bool),
        scaler_mean=np.asarray(scaler.mean_, dtype=np.float64),
        scaler_scale=np.asarray(scaler.scale_, dtype=np.float64),
        scaler_var=np.asarray(scaler.var_, dtype=np.float64),
        input_feature_names=list(feature_names),
        selected_feature_names=selected_names,
        pseudocount=float(pseudocount),
        variance_threshold=float(variance_threshold),
    )
    return fitted, X_processed




def preprocessor_to_payload(preprocessor: FittedPreprocessor) -> dict[str, Any]:
    """Return a joblib-safe, module-independent preprocessing payload."""
    return {
        "support_mask": np.asarray(preprocessor.support_mask, dtype=bool),
        "scaler_mean": np.asarray(preprocessor.scaler_mean, dtype=np.float64),
        "scaler_scale": np.asarray(preprocessor.scaler_scale, dtype=np.float64),
        "scaler_var": np.asarray(preprocessor.scaler_var, dtype=np.float64),
        "input_feature_names": list(preprocessor.input_feature_names),
        "selected_feature_names": list(preprocessor.selected_feature_names),
        "pseudocount": float(preprocessor.pseudocount),
        "variance_threshold": float(preprocessor.variance_threshold),
    }


def transform_with_preprocessor_payload(
    payload: dict[str, Any],
    X_raw: np.ndarray,
) -> np.ndarray:
    """Apply a saved preprocessing payload without fitting anything new."""
    clr = clr_transform(X_raw, float(payload["pseudocount"]))
    support = np.asarray(payload["support_mask"], dtype=bool)
    mean = np.asarray(payload["scaler_mean"], dtype=np.float64)
    scale = np.asarray(payload["scaler_scale"], dtype=np.float64)
    selected = clr[:, support]
    if selected.shape[1] != len(mean) or len(mean) != len(scale):
        raise ValueError("Saved preprocessing payload has incompatible dimensions.")
    transformed = (selected - mean) / scale
    if not np.isfinite(transformed).all():
        raise ValueError("Saved preprocessing produced nonfinite values.")
    return transformed


def training_weights(
    y: np.ndarray,
    participant_ids: np.ndarray,
    mode: str,
) -> np.ndarray:
    y = np.asarray(y, dtype=int)
    participant_ids = np.asarray(participant_ids, dtype=str)
    weights = compute_sample_weight(class_weight="balanced", y=y).astype(np.float64)
    if mode == "class_balanced":
        return weights
    if mode != "class_participant_balanced":
        raise ValueError(f"Unknown weighting mode: {mode}")

    counts = pd.Series(participant_ids).value_counts()
    participant_factor = np.array([1.0 / counts[x] for x in participant_ids], dtype=np.float64)
    weights = weights * participant_factor
    # Preserve mean weight 1 for stable optimizer/tree semantics.
    weights = weights / weights.mean()
    return weights


# =============================================================================
# Models
# =============================================================================

def model_seed(base_seed: int, model_name: str, repeat_idx: int, fold_idx: int) -> int:
    model_offset = {"logistic_l1": 11003, "random_forest": 23003, "xgboost": 37003}[model_name]
    return int(base_seed + model_offset + repeat_idx * 1009 + fold_idx * 97)


def build_model(model_name: str, config: RunConfig, seed: int) -> BaseEstimator:
    if model_name == "logistic_l1":
        return LogisticRegression(
            penalty="l1",
            solver="saga",
            C=config.logistic_c,
            class_weight=None,
            max_iter=config.logistic_max_iter,
            tol=config.logistic_tol,
            random_state=seed,
            n_jobs=config.n_jobs,
        )
    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=config.rf_trees,
            max_depth=None,
            min_samples_leaf=config.rf_min_samples_leaf,
            max_features=config.rf_max_features,
            class_weight=None,
            random_state=seed,
            n_jobs=config.n_jobs,
        )
    if model_name == "xgboost":
        try:
            from xgboost import XGBClassifier
        except Exception as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "XGBoost was requested but cannot be imported. Install requirements.txt."
            ) from exc
        return XGBClassifier(
            n_estimators=config.xgb_estimators,
            max_depth=config.xgb_max_depth,
            learning_rate=config.xgb_learning_rate,
            subsample=config.xgb_subsample,
            colsample_bytree=config.xgb_colsample_bytree,
            reg_lambda=config.xgb_reg_lambda,
            reg_alpha=config.xgb_reg_alpha,
            objective="binary:logistic",
            eval_metric="auc",
            random_state=seed,
            n_jobs=config.n_jobs,
            scale_pos_weight=1.0,
            importance_type="gain",
        )
    raise ValueError(f"Unknown model {model_name}")


def fit_model(
    model_name: str,
    model: BaseEstimator,
    X_train: np.ndarray,
    y_train: np.ndarray,
    sample_weight: np.ndarray,
) -> tuple[BaseEstimator, list[str], float]:
    start = time.perf_counter()
    caught_messages: list[str] = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if model_name == "xgboost":
            model.fit(X_train, y_train, sample_weight=sample_weight, verbose=False)
        else:
            model.fit(X_train, y_train, sample_weight=sample_weight)
        for warning in caught:
            caught_messages.append(f"{warning.category.__name__}: {warning.message}")
    runtime = time.perf_counter() - start
    return model, caught_messages, runtime


def predict_probability(model: BaseEstimator, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = np.asarray(model.predict_proba(X), dtype=np.float64)
        classes = np.asarray(model.classes_, dtype=int)
        where = np.flatnonzero(classes == 1)
        if len(where) != 1:
            raise RuntimeError(f"Model classes do not contain binary positive class exactly once: {classes}")
        result = proba[:, int(where[0])]
    else:  # Defensive fallback; all default models support predict_proba.
        score = np.asarray(model.decision_function(X), dtype=np.float64)
        result = 1.0 / (1.0 + np.exp(-score))
    if not np.isfinite(result).all():
        raise RuntimeError("Model produced nonfinite probabilities.")
    return np.clip(result, 0.0, 1.0)


def model_complexity(model_name: str, model: BaseEstimator) -> dict[str, Any]:
    if model_name == "logistic_l1":
        coef = np.asarray(model.coef_).reshape(-1)
        n_iter = int(np.max(np.asarray(model.n_iter_)))
        return {
            "n_nonzero_model_features": int(np.count_nonzero(np.abs(coef) > 1e-12)),
            "coefficient_l1_norm": float(np.abs(coef).sum()),
            "coefficient_l2_norm": float(np.linalg.norm(coef)),
            "n_iter": n_iter,
        }
    if model_name == "random_forest":
        depths = np.array([tree.tree_.max_depth for tree in model.estimators_], dtype=float)
        leaves = np.array([tree.tree_.n_leaves for tree in model.estimators_], dtype=float)
        return {
            "n_trees": int(len(model.estimators_)),
            "mean_tree_depth": float(depths.mean()),
            "max_tree_depth": int(depths.max()),
            "mean_tree_leaves": float(leaves.mean()),
            "max_tree_leaves": int(leaves.max()),
        }
    if model_name == "xgboost":
        booster = model.get_booster()
        importance = booster.get_score(importance_type="gain")
        return {
            "n_boosting_rounds": int(boosted_round_count(booster)),
            "n_features_with_positive_gain": int(len(importance)),
            "scale_pos_weight": float(model.get_params().get("scale_pos_weight", np.nan)),
        }
    return {}


def boosted_round_count(booster: Any) -> int:
    try:
        return int(booster.num_boosted_rounds())
    except Exception:
        try:
            return int(len(booster.get_dump()))
        except Exception:
            return -1


def full_feature_diagnostics(
    model_name: str,
    model: BaseEstimator,
    preprocessor: FittedPreprocessor,
) -> pd.DataFrame:
    n_features = len(preprocessor.input_feature_names)
    selected_idx = np.flatnonzero(preprocessor.support_mask)
    standardized_value = np.zeros(n_features, dtype=np.float64)
    clr_space_value = np.zeros(n_features, dtype=np.float64)

    if model_name == "logistic_l1":
        selected_values = np.asarray(model.coef_, dtype=np.float64).reshape(-1)
        standardized_value[selected_idx] = selected_values
        clr_space_value[selected_idx] = selected_values / preprocessor.scaler_scale
        value_type = "signed_logit_coefficient"
    elif model_name in {"random_forest", "xgboost"}:
        selected_values = np.asarray(model.feature_importances_, dtype=np.float64).reshape(-1)
        if len(selected_values) != len(selected_idx):
            raise RuntimeError(
                f"{model_name} feature_importances length {len(selected_values)} != selected features {len(selected_idx)}"
            )
        standardized_value[selected_idx] = selected_values
        clr_space_value[selected_idx] = selected_values
        value_type = "gain_importance" if model_name == "xgboost" else "impurity_importance"
    else:
        raise ValueError(model_name)

    return pd.DataFrame(
        {
            "feature_index": np.arange(n_features, dtype=int),
            "feature": preprocessor.input_feature_names,
            "variance_selected": preprocessor.support_mask.astype(int),
            "model_value": standardized_value,
            "clr_space_logit_coefficient": clr_space_value if model_name == "logistic_l1" else np.nan,
            "model_used": (np.abs(standardized_value) > 1e-12).astype(int),
            "value_type": value_type,
        }
    )


# =============================================================================
# Metrics and aggregation
# =============================================================================

def safe_binary_metrics(y_true: np.ndarray, proba_ibd: np.ndarray) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=int)
    proba_ibd = np.asarray(proba_ibd, dtype=np.float64)
    if len(y_true) != len(proba_ibd) or len(y_true) == 0:
        raise ValueError("Invalid metric inputs.")
    pred = (proba_ibd >= 0.5).astype(int)
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    tn, fp, fn, tp = [int(x) for x in cm.ravel()]
    metrics: dict[str, Any] = {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "positive_f1": float(f1_score(y_true, pred, pos_label=1, zero_division=0)),
        "nonibd_recall": float(tn / (tn + fp)) if (tn + fp) else np.nan,
        "ibd_recall": float(tp / (tp + fn)) if (tp + fn) else np.nan,
        "brier": float(brier_score_loss(y_true, proba_ibd)),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, proba_ibd))
    except Exception:
        metrics["roc_auc"] = np.nan
    try:
        probability_matrix = np.column_stack([1.0 - proba_ibd, proba_ibd])
        metrics["log_loss"] = float(
            log_loss(y_true, np.clip(probability_matrix, EPS_METRIC, 1.0 - EPS_METRIC), labels=[0, 1])
        )
    except Exception:
        metrics["log_loss"] = np.nan
    return metrics


def participant_predictions(sample_predictions: pd.DataFrame) -> pd.DataFrame:
    label_counts = sample_predictions.groupby("participant_id")["y_true"].nunique()
    if (label_counts != 1).any():
        raise RuntimeError("Participant has conflicting labels in prediction frame.")
    result = (
        sample_predictions.groupby("participant_id", as_index=False)
        .agg(
            y_true=("y_true", "first"),
            proba_ibd=("proba_ibd", "mean"),
            n_samples=("sample_id", "nunique"),
        )
    )
    result["pred"] = (result["proba_ibd"] >= 0.5).astype(int)
    result["true_label"] = result["y_true"].map(CLASS_NAMES)
    result["pred_label"] = result["pred"].map(CLASS_NAMES)
    return result


def calibration_table(
    y_true: np.ndarray,
    proba: np.ndarray,
    n_bins: int,
) -> pd.DataFrame:
    y_true = np.asarray(y_true, dtype=int)
    proba = np.asarray(proba, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_index = np.clip(np.digitize(proba, edges[1:-1], right=False), 0, n_bins - 1)
    rows = []
    for idx in range(n_bins):
        mask = bin_index == idx
        rows.append(
            {
                "bin": idx + 1,
                "lower": float(edges[idx]),
                "upper": float(edges[idx + 1]),
                "n": int(mask.sum()),
                "mean_predicted_probability": float(proba[mask].mean()) if mask.any() else np.nan,
                "observed_ibd_fraction": float(y_true[mask].mean()) if mask.any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def fold_indices(
    manifest: pd.DataFrame,
    sample_to_index: dict[str, int],
    repeat_idx: int,
    fold_idx: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    rows = manifest[(manifest["repeat"] == repeat_idx) & (manifest["fold"] == fold_idx)]
    train_ids = rows.loc[rows["role"] == "train", "sample_id"].astype(str).tolist()
    test_ids = rows.loc[rows["role"] == "test", "sample_id"].astype(str).tolist()
    train_idx = np.array([sample_to_index[x] for x in train_ids], dtype=int)
    test_idx = np.array([sample_to_index[x] for x in test_ids], dtype=int)
    split_seed = int(rows["split_seed"].iloc[0]) if "split_seed" in rows.columns else -1
    return train_idx, test_idx, split_seed


def fold_dir(output_dir: Path, model_name: str, repeat_idx: int, fold_idx: int) -> Path:
    return output_dir / "folds" / model_name / f"repeat_{repeat_idx:02d}" / f"fold_{fold_idx}"


def fold_is_complete(path: Path) -> bool:
    required = [
        path / "FOLD_COMPLETE.json",
        path / "metrics.json",
        path / "test_predictions.csv",
        path / "participant_test_predictions.csv",
        path / "feature_diagnostics.csv.gz",
        path / "preprocessing.joblib",
    ]
    return all(x.exists() for x in required)


def load_completed_fold(path: Path) -> dict[str, Any]:
    if not fold_is_complete(path):
        raise RuntimeError(f"Fold is not complete: {path}")
    return {
        "metrics": json.loads((path / "metrics.json").read_text(encoding="utf-8")),
        "sample_predictions": pd.read_csv(path / "test_predictions.csv"),
        "participant_predictions": pd.read_csv(path / "participant_test_predictions.csv"),
        "feature_diagnostics": pd.read_csv(path / "feature_diagnostics.csv.gz"),
    }


# =============================================================================
# Fold fitting
# =============================================================================

def run_one_fold(
    config: RunConfig,
    model_name: str,
    repeat_idx: int,
    fold_idx: int,
    split_seed: int,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    X: np.ndarray,
    sample_master: pd.DataFrame,
    feature_names: list[str],
    output_dir: Path,
) -> dict[str, Any]:
    out = fold_dir(output_dir, model_name, repeat_idx, fold_idx)
    if fold_is_complete(out):
        print(f"[resume] {model_name} repeat={repeat_idx} fold={fold_idx}", flush=True)
        return load_completed_fold(out)

    out.mkdir(parents=True, exist_ok=True)
    start_total = time.perf_counter()
    train_meta = sample_master.iloc[train_idx].reset_index(drop=True)
    test_meta = sample_master.iloc[test_idx].reset_index(drop=True)
    y_train = train_meta["label"].to_numpy(dtype=int)
    y_test = test_meta["label"].to_numpy(dtype=int)

    train_participants = set(train_meta["participant_id"])
    test_participants_set = set(test_meta["participant_id"])
    overlap = train_participants & test_participants_set
    if overlap:
        raise RuntimeError(f"Participant leakage at runtime: {sorted(overlap)[:10]}")

    preprocessor, X_train = fit_preprocessor(
        X[train_idx],
        feature_names,
        config.clr_pseudocount,
        config.variance_threshold,
    )
    X_test = preprocessor.transform(X[test_idx])
    weights = training_weights(
        y_train,
        train_meta["participant_id"].astype(str).to_numpy(),
        config.weighting_mode,
    )

    seed = model_seed(config.base_model_seed, model_name, repeat_idx, fold_idx)
    model = build_model(model_name, config, seed)
    model, warning_messages, fit_runtime = fit_model(
        model_name, model, X_train, y_train, weights
    )

    proba_train = predict_probability(model, X_train)
    proba_test = predict_probability(model, X_test)
    train_metrics = safe_binary_metrics(y_train, proba_train)
    test_metrics = safe_binary_metrics(y_test, proba_test)

    sample_pred = test_meta.copy()
    sample_pred.insert(0, "model", model_name)
    sample_pred.insert(1, "repeat", repeat_idx)
    sample_pred.insert(2, "fold", fold_idx)
    sample_pred["proba_ibd"] = proba_test
    sample_pred["pred"] = (proba_test >= 0.5).astype(int)
    sample_pred["true_label"] = sample_pred["label"].map(CLASS_NAMES)
    sample_pred["pred_label"] = sample_pred["pred"].map(CLASS_NAMES)
    sample_pred = sample_pred.rename(columns={"label": "y_true"})

    participant_pred = participant_predictions(sample_pred)
    participant_pred.insert(0, "model", model_name)
    participant_pred.insert(1, "repeat", repeat_idx)
    participant_pred.insert(2, "fold", fold_idx)
    participant_metrics_test = safe_binary_metrics(
        participant_pred["y_true"].to_numpy(), participant_pred["proba_ibd"].to_numpy()
    )

    train_sample_frame = train_meta.rename(columns={"label": "y_true"}).copy()
    train_sample_frame["proba_ibd"] = proba_train
    train_participant_frame = participant_predictions(train_sample_frame)
    participant_metrics_train = safe_binary_metrics(
        train_participant_frame["y_true"].to_numpy(),
        train_participant_frame["proba_ibd"].to_numpy(),
    )

    feature_diag = full_feature_diagnostics(model_name, model, preprocessor)
    feature_diag.insert(0, "model", model_name)
    feature_diag.insert(1, "repeat", repeat_idx)
    feature_diag.insert(2, "fold", fold_idx)

    complexity = model_complexity(model_name, model)
    convergence_warning = any("ConvergenceWarning" in msg for msg in warning_messages)
    if model_name == "logistic_l1":
        convergence_at_limit = int(complexity.get("n_iter", -1)) >= config.logistic_max_iter
    else:
        convergence_at_limit = False

    metrics_payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "model": model_name,
        "model_label": MODEL_LABELS[model_name],
        "repeat": repeat_idx,
        "fold": fold_idx,
        "split_seed": split_seed,
        "model_seed": seed,
        "n_train_samples": int(len(train_idx)),
        "n_test_samples": int(len(test_idx)),
        "n_train_participants": int(train_meta["participant_id"].nunique()),
        "n_test_participants": int(test_meta["participant_id"].nunique()),
        "n_input_features": int(len(feature_names)),
        "n_variance_selected_features": int(preprocessor.support_mask.sum()),
        "weighting_mode": config.weighting_mode,
        "sample_weight_min": float(weights.min()),
        "sample_weight_mean": float(weights.mean()),
        "sample_weight_max": float(weights.max()),
        "fit_runtime_seconds": float(fit_runtime),
        "total_runtime_seconds": float(time.perf_counter() - start_total),
        "warning_messages": warning_messages,
        "convergence_warning": bool(convergence_warning),
        "convergence_at_iteration_limit": bool(convergence_at_limit),
        "train_sample_metrics": train_metrics,
        "test_sample_metrics": test_metrics,
        "train_participant_metrics": participant_metrics_train,
        "test_participant_metrics": participant_metrics_test,
        "complexity": complexity,
        "xgboost_weighting_check": (
            {
                "balanced_sample_weight_used": True,
                "scale_pos_weight": float(model.get_params().get("scale_pos_weight", np.nan)),
                "double_weighting_present": False,
            }
            if model_name == "xgboost"
            else None
        ),
    }

    atomic_to_csv(sample_pred, out / "test_predictions.csv", index=False)
    atomic_to_csv(participant_pred, out / "participant_test_predictions.csv", index=False)
    feature_diag.to_csv(out / "feature_diagnostics.csv.gz", index=False, compression="gzip")
    atomic_joblib_dump(preprocessor_to_payload(preprocessor), out / "preprocessing.joblib", compress=3)
    atomic_write_json(out / "metrics.json", metrics_payload)

    if config.save_fold_models:
        atomic_joblib_dump(
            {
                "schema_version": SCHEMA_VERSION,
                "model_name": model_name,
                "model": model,
                "preprocessor": preprocessor_to_payload(preprocessor),
                "class_names": CLASS_NAMES,
                "feature_names": feature_names,
                "repeat": repeat_idx,
                "fold": fold_idx,
            },
            out / "fitted_model.joblib",
            compress=3,
        )

    atomic_write_json(
        out / "FOLD_COMPLETE.json",
        {
            "completed_utc": now_iso(),
            "model": model_name,
            "repeat": repeat_idx,
            "fold": fold_idx,
            "test_prediction_rows": int(len(sample_pred)),
            "participant_prediction_rows": int(len(participant_pred)),
        },
    )
    print(
        f"[{model_name} r{repeat_idx:02d} f{fold_idx}] "
        f"bal={test_metrics['balanced_accuracy']:.4f} "
        f"auc={test_metrics['roc_auc']:.4f} "
        f"participant_bal={participant_metrics_test['balanced_accuracy']:.4f} "
        f"features={int(preprocessor.support_mask.sum())} "
        f"runtime={metrics_payload['total_runtime_seconds']:.1f}s",
        flush=True,
    )
    return {
        "metrics": metrics_payload,
        "sample_predictions": sample_pred,
        "participant_predictions": participant_pred,
        "feature_diagnostics": feature_diag,
    }


# =============================================================================
# Incremental and final summaries
# =============================================================================

def flatten_fold_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    row = {
        k: payload[k]
        for k in [
            "model",
            "model_label",
            "repeat",
            "fold",
            "split_seed",
            "model_seed",
            "n_train_samples",
            "n_test_samples",
            "n_train_participants",
            "n_test_participants",
            "n_input_features",
            "n_variance_selected_features",
            "weighting_mode",
            "sample_weight_min",
            "sample_weight_mean",
            "sample_weight_max",
            "fit_runtime_seconds",
            "total_runtime_seconds",
            "convergence_warning",
            "convergence_at_iteration_limit",
        ]
    }
    for level in ["train_sample", "test_sample", "train_participant", "test_participant"]:
        source = payload[f"{level}_metrics"]
        for key, value in source.items():
            row[f"{level}_{key}"] = value
    for key, value in payload.get("complexity", {}).items():
        row[f"complexity_{key}"] = value
    return row


def repetition_metrics_from_predictions(
    predictions: pd.DataFrame,
    model_name: str,
    expected_repeats: int,
    level: str,
) -> pd.DataFrame:
    rows = []
    id_col = "sample_id" if level == "sample" else "participant_id"
    for repeat_idx in range(1, expected_repeats + 1):
        subset = predictions[
            (predictions["model"] == model_name) & (predictions["repeat"] == repeat_idx)
        ].copy()
        if subset.empty:
            continue
        duplicate = subset.duplicated(id_col)
        if duplicate.any():
            examples = subset.loc[duplicate, id_col].astype(str).head(10).tolist()
            raise RuntimeError(
                f"{model_name} repeat {repeat_idx} has duplicate {level} predictions: {examples}"
            )
        metrics = safe_binary_metrics(subset["y_true"].to_numpy(), subset["proba_ibd"].to_numpy())
        rows.append(
            {
                "model": model_name,
                "model_label": MODEL_LABELS[model_name],
                "repeat": repeat_idx,
                "evaluation_level": level,
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def summarize_repetition_metrics(repetition_metrics: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "positive_f1",
        "roc_auc",
        "brier",
        "log_loss",
        "nonibd_recall",
        "ibd_recall",
    ]
    rows = []
    for (model, level), group in repetition_metrics.groupby(["model", "evaluation_level"], sort=False):
        row: dict[str, Any] = {
            "model": model,
            "model_label": MODEL_LABELS[model],
            "evaluation_level": level,
            "n_repetitions": int(group["repeat"].nunique()),
        }
        for metric in metric_cols:
            values = pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy(dtype=float)
            if len(values) == 0:
                for suffix in ["mean", "sd", "median", "q25", "q75", "q025", "q975"]:
                    row[f"{metric}_{suffix}"] = np.nan
                continue
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_sd"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            row[f"{metric}_median"] = float(np.median(values))
            row[f"{metric}_q25"] = float(np.quantile(values, 0.25))
            row[f"{metric}_q75"] = float(np.quantile(values, 0.75))
            row[f"{metric}_q025"] = float(np.quantile(values, 0.025))
            row[f"{metric}_q975"] = float(np.quantile(values, 0.975))
        rows.append(row)
    return pd.DataFrame(rows)


def paired_model_comparisons(repetition_metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = ["accuracy", "balanced_accuracy", "macro_f1", "roc_auc", "brier", "log_loss"]
    detail_rows = []
    summary_rows = []
    for level in sorted(repetition_metrics["evaluation_level"].unique()):
        subset = repetition_metrics[repetition_metrics["evaluation_level"] == level]
        for i, model_a in enumerate(MODEL_ORDER):
            if model_a not in set(subset["model"]):
                continue
            for model_b in MODEL_ORDER[i + 1 :]:
                if model_b not in set(subset["model"]):
                    continue
                merged = subset[subset["model"] == model_a].merge(
                    subset[subset["model"] == model_b],
                    on=["repeat", "evaluation_level"],
                    suffixes=("_a", "_b"),
                    validate="one_to_one",
                )
                for metric in metrics:
                    # For error metrics, positive means A is better by reversing the sign.
                    if metric in {"brier", "log_loss"}:
                        delta = merged[f"{metric}_b"] - merged[f"{metric}_a"]
                    else:
                        delta = merged[f"{metric}_a"] - merged[f"{metric}_b"]
                    for repeat_idx, value in zip(merged["repeat"], delta):
                        detail_rows.append(
                            {
                                "evaluation_level": level,
                                "model_a": model_a,
                                "model_b": model_b,
                                "metric": metric,
                                "repeat": int(repeat_idx),
                                "delta_a_better_positive": float(value),
                            }
                        )
                    values = delta.to_numpy(dtype=float)
                    summary_rows.append(
                        {
                            "evaluation_level": level,
                            "model_a": model_a,
                            "model_b": model_b,
                            "metric": metric,
                            "n_repetitions": int(len(values)),
                            "mean_delta_a_better_positive": float(values.mean()),
                            "sd_delta": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                            "median_delta": float(np.median(values)),
                            "q025_delta": float(np.quantile(values, 0.025)),
                            "q975_delta": float(np.quantile(values, 0.975)),
                            "fraction_a_better": float(np.mean(values > 0)),
                            "fraction_equal": float(np.mean(np.isclose(values, 0.0))),
                        }
                    )
    return pd.DataFrame(detail_rows), pd.DataFrame(summary_rows)


def consensus_predictions(
    sample_predictions: pd.DataFrame,
    participant_preds: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sample_consensus = (
        sample_predictions.groupby(["model", "sample_id", "participant_id"], as_index=False)
        .agg(
            y_true=("y_true", "first"),
            proba_ibd_mean=("proba_ibd", "mean"),
            proba_ibd_sd=("proba_ibd", "std"),
            proba_ibd_min=("proba_ibd", "min"),
            proba_ibd_max=("proba_ibd", "max"),
            n_repetitions=("repeat", "nunique"),
        )
    )
    sample_consensus["pred"] = (sample_consensus["proba_ibd_mean"] >= 0.5).astype(int)
    sample_consensus["true_label"] = sample_consensus["y_true"].map(CLASS_NAMES)
    sample_consensus["pred_label"] = sample_consensus["pred"].map(CLASS_NAMES)

    participant_consensus = (
        participant_preds.groupby(["model", "participant_id"], as_index=False)
        .agg(
            y_true=("y_true", "first"),
            proba_ibd_mean=("proba_ibd", "mean"),
            proba_ibd_sd=("proba_ibd", "std"),
            proba_ibd_min=("proba_ibd", "min"),
            proba_ibd_max=("proba_ibd", "max"),
            n_repetitions=("repeat", "nunique"),
            mean_samples_per_repeat=("n_samples", "mean"),
        )
    )
    participant_consensus["pred"] = (participant_consensus["proba_ibd_mean"] >= 0.5).astype(int)
    participant_consensus["true_label"] = participant_consensus["y_true"].map(CLASS_NAMES)
    participant_consensus["pred_label"] = participant_consensus["pred"].map(CLASS_NAMES)

    metric_rows = []
    for model_name in sorted(sample_consensus["model"].unique(), key=MODEL_ORDER.index):
        for level, frame in [
            ("sample", sample_consensus[sample_consensus["model"] == model_name]),
            ("participant", participant_consensus[participant_consensus["model"] == model_name]),
        ]:
            metrics = safe_binary_metrics(frame["y_true"], frame["proba_ibd_mean"])
            metric_rows.append(
                {
                    "model": model_name,
                    "model_label": MODEL_LABELS[model_name],
                    "evaluation_level": level,
                    **metrics,
                }
            )
    return sample_consensus, participant_consensus, pd.DataFrame(metric_rows)


def aggregate_feature_stability(feature_diagnostics: pd.DataFrame) -> dict[str, pd.DataFrame]:
    outputs = {}
    for model_name, group in feature_diagnostics.groupby("model", sort=False):
        rows = []
        for feature, values in group.groupby("feature", sort=False):
            model_values = values["model_value"].to_numpy(dtype=float)
            nonzero = np.abs(model_values) > 1e-12
            selected_values = model_values[nonzero]
            positive = selected_values > 0
            negative = selected_values < 0
            row = {
                "model": model_name,
                "model_label": MODEL_LABELS[model_name],
                "feature": feature,
                "n_fits": int(len(values)),
                "variance_selected_frequency": float(values["variance_selected"].mean()),
                "model_used_frequency": float(nonzero.mean()),
                "model_value_mean_including_zero": float(model_values.mean()),
                "model_value_sd_including_zero": float(model_values.std(ddof=1)) if len(model_values) > 1 else 0.0,
                "model_value_median_including_zero": float(np.median(model_values)),
                "model_value_q025_including_zero": float(np.quantile(model_values, 0.025)),
                "model_value_q975_including_zero": float(np.quantile(model_values, 0.975)),
                "model_value_mean_when_used": float(selected_values.mean()) if len(selected_values) else np.nan,
                "model_value_median_when_used": float(np.median(selected_values)) if len(selected_values) else np.nan,
                "positive_when_used_fraction": float(positive.mean()) if len(selected_values) else np.nan,
                "negative_when_used_fraction": float(negative.mean()) if len(selected_values) else np.nan,
                "absolute_value_mean_including_zero": float(np.abs(model_values).mean()),
            }
            rows.append(row)
        result = pd.DataFrame(rows)
        if model_name == "logistic_l1":
            result = result.sort_values(
                ["model_used_frequency", "absolute_value_mean_including_zero"],
                ascending=[False, False],
            )
        else:
            result = result.sort_values(
                ["absolute_value_mean_including_zero", "model_used_frequency"],
                ascending=[False, False],
            )
        outputs[model_name] = result.reset_index(drop=True)
    return outputs


def update_partial_outputs(
    output_dir: Path,
    config: RunConfig,
    completed: list[dict[str, Any]],
) -> None:
    partial = output_dir / "partial_progress"
    partial.mkdir(parents=True, exist_ok=True)
    fold_metrics = pd.DataFrame([flatten_fold_metrics(x["metrics"]) for x in completed])
    if not fold_metrics.empty:
        atomic_to_csv(fold_metrics.sort_values(["model", "repeat", "fold"]), partial / "completed_fold_metrics.csv")
    sample_predictions = pd.concat([x["sample_predictions"] for x in completed], ignore_index=True)
    participant_preds = pd.concat([x["participant_predictions"] for x in completed], ignore_index=True)

    repetition_frames = []
    for model_name in config.models:
        subset = fold_metrics[fold_metrics["model"] == model_name] if not fold_metrics.empty else pd.DataFrame()
        completed_repeats = []
        if not subset.empty:
            counts = subset.groupby("repeat")["fold"].nunique()
            completed_repeats = counts[counts == config.expected_folds].index.astype(int).tolist()
        if completed_repeats:
            sample_complete = sample_predictions[
                (sample_predictions["model"] == model_name)
                & (sample_predictions["repeat"].isin(completed_repeats))
            ]
            participant_complete = participant_preds[
                (participant_preds["model"] == model_name)
                & (participant_preds["repeat"].isin(completed_repeats))
            ]
            repetition_frames.append(
                repetition_metrics_from_predictions(
                    sample_complete, model_name, config.expected_repeats, "sample"
                )
            )
            repetition_frames.append(
                repetition_metrics_from_predictions(
                    participant_complete, model_name, config.expected_repeats, "participant"
                )
            )
    if repetition_frames:
        repetition = pd.concat(repetition_frames, ignore_index=True)
        atomic_to_csv(repetition, partial / "completed_repetition_metrics.csv")
        atomic_to_csv(
            summarize_repetition_metrics(repetition),
            partial / "completed_repetition_summary.csv",
        )

    expected_per_model = config.expected_repeats * config.expected_folds
    for model_name in config.models:
        model_completed = [x for x in completed if x["metrics"]["model"] == model_name]
        if len(model_completed) == expected_per_model:
            diagnostics = pd.concat(
                [x["feature_diagnostics"] for x in model_completed],
                ignore_index=True,
            )
            stability = aggregate_feature_stability(diagnostics)[model_name]
            atomic_to_csv(
                stability,
                partial / f"completed_feature_stability_{model_name}.csv",
            )

    atomic_write_json(
        partial / "progress.json",
        {
            "updated_utc": now_iso(),
            "completed_folds": int(len(completed)),
            "target_folds": int(len(config.models) * config.expected_repeats * config.expected_folds),
            "completed_by_model": (
                fold_metrics.groupby("model").size().astype(int).to_dict()
                if not fold_metrics.empty
                else {}
            ),
        },
    )


# =============================================================================
# Full-source models
# =============================================================================

def fit_full_source_model(
    config: RunConfig,
    model_name: str,
    X: np.ndarray,
    sample_master: pd.DataFrame,
    feature_names: list[str],
    output_dir: Path,
) -> None:
    out = output_dir / "full_source_models" / model_name
    marker = out / "FULL_MODEL_COMPLETE.json"
    model_path = out / "full_source_model.joblib"
    if marker.exists() and model_path.exists():
        print(f"[resume] full-source model {model_name}", flush=True)
        return
    out.mkdir(parents=True, exist_ok=True)
    preprocessor, X_processed = fit_preprocessor(
        X, feature_names, config.clr_pseudocount, config.variance_threshold
    )
    y = sample_master["label"].to_numpy(dtype=int)
    groups = sample_master["participant_id"].astype(str).to_numpy()
    weights = training_weights(y, groups, config.weighting_mode)
    seed = model_seed(config.base_model_seed, model_name, 0, 0)
    model = build_model(model_name, config, seed)
    model, warning_messages, runtime = fit_model(
        model_name, model, X_processed, y, weights
    )
    proba = predict_probability(model, X_processed)
    training_metrics = safe_binary_metrics(y, proba)
    feature_diag = full_feature_diagnostics(model_name, model, preprocessor)
    atomic_to_csv(feature_diag, out / "full_source_feature_diagnostics.csv")
    atomic_joblib_dump(
        {
            "schema_version": SCHEMA_VERSION,
            "code_version": CODE_VERSION,
            "model_name": model_name,
            "model_label": MODEL_LABELS[model_name],
            "model": model,
            "preprocessor": preprocessor_to_payload(preprocessor),
            "feature_names": feature_names,
            "class_names": CLASS_NAMES,
            "task": "IBD (UC+CD) versus non-IBD",
            "sample_ids": sample_master["sample_id"].tolist(),
            "participant_ids": sample_master["participant_id"].tolist(),
            "weighting_mode": config.weighting_mode,
            "training_metrics_apparent_only": training_metrics,
        },
        model_path,
        compress=3,
    )
    metadata = {
        "model": model_name,
        "model_seed": seed,
        "fit_runtime_seconds": runtime,
        "warning_messages": warning_messages,
        "n_samples": int(len(sample_master)),
        "n_participants": int(sample_master["participant_id"].nunique()),
        "n_input_features": int(len(feature_names)),
        "n_variance_selected_features": int(preprocessor.support_mask.sum()),
        "training_metrics_apparent_only": training_metrics,
        "complexity": model_complexity(model_name, model),
        "external_use_warning": (
            "Apply this saved training preprocessor and model without refitting or external rescaling. "
            "Apparent training metrics are not generalization estimates."
        ),
    }
    atomic_write_json(out / "full_source_model_metadata.json", metadata)
    atomic_write_json(marker, {"completed_utc": now_iso(), **metadata})


# =============================================================================
# Plots
# =============================================================================

def plot_performance_distributions(
    repetition_metrics: pd.DataFrame,
    output_dir: Path,
) -> None:
    plot_dir = output_dir / "figures"
    plot_dir.mkdir(parents=True, exist_ok=True)
    metrics = ["balanced_accuracy", "roc_auc", "macro_f1", "brier"]
    for level in ["sample", "participant"]:
        subset = repetition_metrics[repetition_metrics["evaluation_level"] == level]
        if subset.empty:
            continue
        for metric in metrics:
            groups = []
            labels = []
            for model_name in MODEL_ORDER:
                values = subset.loc[subset["model"] == model_name, metric].dropna().to_numpy()
                if len(values):
                    groups.append(values)
                    labels.append(MODEL_LABELS[model_name])
            if not groups:
                continue
            plt.figure(figsize=(8, 5))
            plt.boxplot(groups, labels=labels, showmeans=True)
            plt.ylabel(metric.replace("_", " ").title())
            plt.title(f"Repeated pooled OOF {metric.replace('_', ' ')} ({level} level)")
            plt.xticks(rotation=20, ha="right")
            plt.tight_layout()
            plt.savefig(plot_dir / f"{level}_{metric}_distribution.png", dpi=220)
            plt.close()


def plot_consensus_calibration(
    sample_consensus: pd.DataFrame,
    participant_consensus: pd.DataFrame,
    output_dir: Path,
    n_bins: int,
) -> None:
    plot_dir = output_dir / "figures"
    plot_dir.mkdir(parents=True, exist_ok=True)
    for level, frame, probability_col in [
        ("sample", sample_consensus, "proba_ibd_mean"),
        ("participant", participant_consensus, "proba_ibd_mean"),
    ]:
        plt.figure(figsize=(6, 6))
        plotted = False
        for model_name in MODEL_ORDER:
            subset = frame[frame["model"] == model_name]
            if subset.empty:
                continue
            cal = calibration_table(subset["y_true"], subset[probability_col], n_bins)
            cal = cal[cal["n"] > 0]
            plt.plot(
                cal["mean_predicted_probability"],
                cal["observed_ibd_fraction"],
                marker="o",
                label=MODEL_LABELS[model_name],
            )
            plotted = True
        if plotted:
            plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
            plt.xlabel("Mean predicted IBD probability")
            plt.ylabel("Observed IBD fraction")
            plt.title(f"Consensus calibration ({level} level)")
            plt.legend()
            plt.tight_layout()
            plt.savefig(plot_dir / f"consensus_calibration_{level}.png", dpi=220)
        plt.close()


def plot_top_features(feature_stability: dict[str, pd.DataFrame], output_dir: Path) -> None:
    plot_dir = output_dir / "figures"
    plot_dir.mkdir(parents=True, exist_ok=True)
    for model_name, frame in feature_stability.items():
        top = frame.head(20).iloc[::-1]
        if top.empty:
            continue
        value_col = (
            "model_value_mean_including_zero"
            if model_name == "logistic_l1"
            else "absolute_value_mean_including_zero"
        )
        plt.figure(figsize=(10, 8))
        plt.barh(top["feature"], top[value_col])
        plt.xlabel(
            "Mean signed standardized coefficient"
            if model_name == "logistic_l1"
            else "Mean feature importance including zeros"
        )
        plt.title(f"Top repeated-CV features: {MODEL_LABELS[model_name]}")
        plt.tight_layout()
        plt.savefig(plot_dir / f"top_features_{model_name}.png", dpi=220)
        plt.close()


def plot_train_test_gaps(fold_metrics: pd.DataFrame, output_dir: Path) -> None:
    plot_dir = output_dir / "figures"
    plot_dir.mkdir(parents=True, exist_ok=True)
    for metric in ["balanced_accuracy", "roc_auc"]:
        groups = []
        labels = []
        for model_name in MODEL_ORDER:
            subset = fold_metrics[fold_metrics["model"] == model_name]
            if subset.empty:
                continue
            gap = (
                subset[f"train_sample_{metric}"].to_numpy(dtype=float)
                - subset[f"test_sample_{metric}"].to_numpy(dtype=float)
            )
            groups.append(gap[np.isfinite(gap)])
            labels.append(MODEL_LABELS[model_name])
        if groups:
            plt.figure(figsize=(8, 5))
            plt.boxplot(groups, labels=labels, showmeans=True)
            plt.axhline(0.0, linestyle="--")
            plt.ylabel(f"Train minus test {metric.replace('_', ' ')}")
            plt.title(f"Outer-fold train-test gap: {metric.replace('_', ' ')}")
            plt.xticks(rotation=20, ha="right")
            plt.tight_layout()
            plt.savefig(plot_dir / f"train_test_gap_{metric}.png", dpi=220)
            plt.close()


# =============================================================================
# Main aggregation
# =============================================================================

def finalize_outputs(
    config: RunConfig,
    bundle: dict[str, Any],
    completed: list[dict[str, Any]],
) -> None:
    output_dir = Path(config.output_dir)
    fold_metrics = pd.DataFrame([flatten_fold_metrics(x["metrics"]) for x in completed])
    fold_metrics = fold_metrics.sort_values(["model", "repeat", "fold"]).reset_index(drop=True)
    sample_predictions = pd.concat([x["sample_predictions"] for x in completed], ignore_index=True)
    participant_preds = pd.concat([x["participant_predictions"] for x in completed], ignore_index=True)
    feature_diagnostics = pd.concat([x["feature_diagnostics"] for x in completed], ignore_index=True)

    expected_total = len(config.models) * config.expected_repeats * config.expected_folds
    if len(fold_metrics) != expected_total:
        raise RuntimeError(f"Expected {expected_total} completed folds, found {len(fold_metrics)}")

    atomic_to_csv(fold_metrics, output_dir / "all_outer_fold_metrics.csv")
    sample_predictions.to_csv(
        output_dir / "all_outer_test_sample_predictions.csv.gz",
        index=False,
        compression="gzip",
    )
    participant_preds.to_csv(
        output_dir / "all_outer_test_participant_predictions.csv.gz",
        index=False,
        compression="gzip",
    )

    repetition_frames = []
    for model_name in config.models:
        repetition_frames.append(
            repetition_metrics_from_predictions(
                sample_predictions, model_name, config.expected_repeats, "sample"
            )
        )
        repetition_frames.append(
            repetition_metrics_from_predictions(
                participant_preds, model_name, config.expected_repeats, "participant"
            )
        )
    repetition_metrics = pd.concat(repetition_frames, ignore_index=True)
    repetition_metrics = repetition_metrics.sort_values(
        ["evaluation_level", "model", "repeat"]
    ).reset_index(drop=True)
    atomic_to_csv(repetition_metrics, output_dir / "repetition_pooled_oof_metrics.csv")

    summary = summarize_repetition_metrics(repetition_metrics)
    atomic_to_csv(summary, output_dir / "repetition_performance_summary.csv")

    paired_detail, paired_summary = paired_model_comparisons(repetition_metrics)
    atomic_to_csv(paired_detail, output_dir / "paired_model_differences_by_repetition.csv")
    atomic_to_csv(paired_summary, output_dir / "paired_model_difference_summary.csv")

    sample_consensus, participant_consensus, consensus_metrics = consensus_predictions(
        sample_predictions, participant_preds
    )
    atomic_to_csv(sample_consensus, output_dir / "consensus_predictions_by_sample.csv")
    atomic_to_csv(participant_consensus, output_dir / "consensus_predictions_by_participant.csv")
    atomic_to_csv(consensus_metrics, output_dir / "consensus_prediction_metrics.csv")

    calibration_frames = []
    for model_name in config.models:
        for level, frame in [
            ("sample", sample_consensus[sample_consensus["model"] == model_name]),
            (
                "participant",
                participant_consensus[participant_consensus["model"] == model_name],
            ),
        ]:
            cal = calibration_table(
                frame["y_true"], frame["proba_ibd_mean"], config.calibration_bins
            )
            cal.insert(0, "evaluation_level", level)
            cal.insert(0, "model", model_name)
            calibration_frames.append(cal)
    calibration = pd.concat(calibration_frames, ignore_index=True)
    atomic_to_csv(calibration, output_dir / "consensus_calibration_tables.csv")

    feature_stability = aggregate_feature_stability(feature_diagnostics)
    for model_name, frame in feature_stability.items():
        atomic_to_csv(frame, output_dir / f"feature_stability_{model_name}.csv")
        atomic_to_csv(frame.head(100), output_dir / f"top_100_features_{model_name}.csv")

    complexity_cols = [c for c in fold_metrics.columns if c.startswith("complexity_")]
    diagnostic_cols = [
        "model",
        "repeat",
        "fold",
        "n_variance_selected_features",
        "fit_runtime_seconds",
        "total_runtime_seconds",
        "convergence_warning",
        "convergence_at_iteration_limit",
        "train_sample_balanced_accuracy",
        "test_sample_balanced_accuracy",
        "train_sample_roc_auc",
        "test_sample_roc_auc",
    ] + complexity_cols
    atomic_to_csv(fold_metrics[diagnostic_cols], output_dir / "model_fit_diagnostics.csv")

    input_summary = {
        "created_utc": now_iso(),
        "species_diagnostics": bundle["species_diagnostics"],
        "source_manifest_samples": int(len(bundle["source_sample_master"])),
        "complete_case_samples": int(len(bundle["sample_master"])),
        "excluded_species_profiles": int(len(bundle["excluded_samples"])),
        "n_participants": int(bundle["sample_master"]["participant_id"].nunique()),
        "class_sample_counts": bundle["sample_master"]["label"].value_counts().sort_index().to_dict(),
        "class_participant_counts": (
            bundle["sample_master"]
            .drop_duplicates("participant_id")["label"]
            .value_counts()
            .sort_index()
            .to_dict()
        ),
        "n_features": int(len(bundle["feature_names"])),
        "models": list(config.models),
        "expected_repeats": config.expected_repeats,
        "expected_folds": config.expected_folds,
    }
    atomic_write_json(output_dir / "input_and_design_summary.json", input_summary)

    if config.make_plots:
        plot_performance_distributions(repetition_metrics, output_dir)
        plot_consensus_calibration(
            sample_consensus, participant_consensus, output_dir, config.calibration_bins
        )
        plot_top_features(feature_stability, output_dir)
        plot_train_test_gaps(fold_metrics, output_dir)

    atomic_write_json(
        output_dir / "RUN_COMPLETE.json",
        {
            "completed_utc": now_iso(),
            "schema_version": SCHEMA_VERSION,
            "code_version": CODE_VERSION,
            "models": list(config.models),
            "completed_folds": int(len(fold_metrics)),
            "repetition_metric_rows": int(len(repetition_metrics)),
            "primary_results": [
                "repetition_pooled_oof_metrics.csv",
                "repetition_performance_summary.csv",
                "paired_model_difference_summary.csv",
            ],
        },
    )


def run_analysis(args: argparse.Namespace) -> Path:
    config = config_from_args(args)
    set_thread_environment(config.n_jobs)
    ensure_compatible_output(config, bool(args.overwrite_incompatible_output))
    output_dir = Path(config.output_dir)

    print("=" * 110, flush=True)
    print("Repeated species-abundance benchmarks: IBD vs non-IBD", flush=True)
    print("=" * 110, flush=True)
    print(f"Species file: {config.species_file}", flush=True)
    print(f"Split manifest: {config.split_manifest}", flush=True)
    print(f"Output: {config.output_dir}", flush=True)
    print(f"Models: {config.models}", flush=True)
    print(f"Expected design: {config.expected_repeats} x {config.expected_folds}", flush=True)
    print(f"Threads per model: {config.n_jobs}", flush=True)
    print(f"Weighting: {config.weighting_mode}", flush=True)
    print(
        f"Zero-profile rule: {config.zero_profile_policy}; expected count={config.expected_zero_profiles}",
        flush=True,
    )
    print("XGBoost imbalance rule: balanced sample_weight only; scale_pos_weight=1.0", flush=True)
    print("=" * 110, flush=True)

    bundle = load_and_verify_inputs(config)

    # Persist the complete-case cohort definition before fitting any model.
    atomic_to_csv(bundle["source_sample_master"], output_dir / "source_manifest_sample_master.csv")
    atomic_to_csv(bundle["sample_master"], output_dir / "complete_case_sample_master.csv")
    atomic_to_csv(bundle["species_availability"], output_dir / "sample_species_availability.csv")
    atomic_to_csv(bundle["excluded_samples"], output_dir / "excluded_species_profiles.csv")
    atomic_to_csv(bundle["manifest"], output_dir / "complete_case_split_manifest.csv")
    atomic_to_csv(bundle["complete_case_fold_counts"], output_dir / "complete_case_fold_counts.csv")
    atomic_to_csv(
        bundle["participant_complete_case_counts"],
        output_dir / "participant_complete_case_counts.csv",
    )

    atomic_write_json(
        output_dir / "preflight_summary.json",
        {
            "passed_utc": now_iso(),
            "source_manifest_samples": int(len(bundle["source_sample_master"])),
            "complete_case_samples": int(len(bundle["sample_master"])),
            "excluded_samples": int(len(bundle["excluded_samples"])),
            "n_participants": int(bundle["sample_master"]["participant_id"].nunique()),
            "n_features": int(len(bundle["feature_names"])),
            "species_diagnostics": bundle["species_diagnostics"],
            "cohort_definition_files": [
                "source_manifest_sample_master.csv",
                "complete_case_sample_master.csv",
                "sample_species_availability.csv",
                "excluded_species_profiles.csv",
                "complete_case_split_manifest.csv",
                "complete_case_fold_counts.csv",
                "participant_complete_case_counts.csv",
            ],
        },
    )
    print(
        f"[inputs] {len(bundle['sample_master'])} complete-case samples "
        f"from {len(bundle['source_sample_master'])} manifest samples; "
        f"excluded={len(bundle['excluded_samples'])}; "
        f"participants={bundle['sample_master']['participant_id'].nunique()}; "
        f"features={len(bundle['feature_names'])}",
        flush=True,
    )
    if len(bundle["excluded_samples"]):
        print(
            "[complete-case exclusions] "
            + ", ".join(bundle["excluded_samples"]["sample_id"].astype(str).tolist()),
            flush=True,
        )

    sample_to_index = {
        sid: idx for idx, sid in enumerate(bundle["sample_master"]["sample_id"].astype(str))
    }
    completed: list[dict[str, Any]] = []
    for model_name in config.models:
        print("\n" + "#" * 110, flush=True)
        print(f"MODEL: {MODEL_LABELS[model_name]}", flush=True)
        print("#" * 110, flush=True)
        for repeat_idx in range(1, config.expected_repeats + 1):
            for fold_idx in range(1, config.expected_folds + 1):
                train_idx, test_idx, split_seed = fold_indices(
                    bundle["manifest"], sample_to_index, repeat_idx, fold_idx
                )
                result = run_one_fold(
                    config=config,
                    model_name=model_name,
                    repeat_idx=repeat_idx,
                    fold_idx=fold_idx,
                    split_seed=split_seed,
                    train_idx=train_idx,
                    test_idx=test_idx,
                    X=bundle["X"],
                    sample_master=bundle["sample_master"],
                    feature_names=bundle["feature_names"],
                    output_dir=output_dir,
                )
                completed.append(result)
                update_partial_outputs(output_dir, config, completed)

        if config.fit_full_source_models:
            fit_full_source_model(
                config,
                model_name,
                bundle["X"],
                bundle["sample_master"],
                bundle["feature_names"],
                output_dir,
            )

    finalize_outputs(config, bundle, completed)
    print("\n" + "=" * 110, flush=True)
    print("RUN COMPLETE", flush=True)
    print(f"Primary summary: {output_dir / 'repetition_performance_summary.csv'}", flush=True)
    print(f"Paired comparisons: {output_dir / 'paired_model_difference_summary.csv'}", flush=True)
    print("=" * 110, flush=True)
    return output_dir


def main() -> None:
    args = parse_args()
    run_analysis(args)


if __name__ == "__main__":
    main()
