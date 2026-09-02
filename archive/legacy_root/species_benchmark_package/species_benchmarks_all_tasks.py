#!/usr/bin/env python3
from __future__ import annotations

"""Repeated participant-grouped species-abundance benchmarks for all five IBD tasks.

This runner consumes the exact task-specific 20 x 5 split manifests written by
``h0_ricci_joint_repeated_cv_all_tasks_v2``.  It deliberately does not generate
folds.  Every retained sample keeps the same task, repetition, fold, and role as
in the joint H0+Ricci analysis.

The five tasks are run in this fixed order:
  1. IBD vs non-IBD
  2. non-IBD vs UC vs CD
  3. non-IBD vs UC
  4. non-IBD vs CD
  5. UC vs CD (manifest folder: CD_vs_UC).
"""

import argparse
import hashlib
import json
import math
import os
import platform
import re
import sys
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
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

SCHEMA_VERSION = 1
CODE_VERSION = "2.0.0"
EPS = 1e-15

MODEL_ORDER = ("logistic_l1", "random_forest", "xgboost")
MODEL_LABELS = {
    "logistic_l1": "Species L1 logistic",
    "random_forest": "Species random forest",
    "xgboost": "Species XGBoost",
}


@dataclass(frozen=True)
class TaskSpec:
    folder: str
    report_name: str
    class_order: tuple[str, ...]


ALL_TASKS: tuple[TaskSpec, ...] = (
    TaskSpec("IBD_vs_nonIBD", "IBD vs non-IBD", ("nonIBD", "IBD")),
    TaskSpec("three_way_nonIBD_UC_CD", "non-IBD vs UC vs CD", ("nonIBD", "UC", "CD")),
    TaskSpec("nonIBD_vs_UC", "non-IBD vs UC", ("nonIBD", "UC")),
    TaskSpec("nonIBD_vs_CD", "non-IBD vs CD", ("nonIBD", "CD")),
    TaskSpec("CD_vs_UC", "UC vs CD", ("CD", "UC")),
)
TASK_BY_FOLDER = {task.folder: task for task in ALL_TASKS}
TASK_ORDER = tuple(task.folder for task in ALL_TASKS)


@dataclass(frozen=True)
class RunConfig:
    species_file: str
    species_sheet: str
    split_dir: str
    output_dir: str
    tasks: tuple[str, ...]
    models: tuple[str, ...]
    expected_repeats: int = 20
    expected_folds: int = 5
    expected_global_zero_profiles: int = 10
    pseudocount: float = 1e-6
    variance_threshold: float = 0.0
    weighting: str = "class_balanced"
    n_jobs: int = 1
    base_seed: int = 20260720
    logistic_c: float = 0.2
    logistic_max_iter: int = 10000
    logistic_tol: float = 1e-4
    rf_trees: int = 1000
    rf_min_samples_leaf: int = 3
    rf_max_features: str = "sqrt"
    xgb_estimators: int = 700
    xgb_max_depth: int = 3
    xgb_learning_rate: float = 0.02
    xgb_subsample: float = 0.85
    xgb_colsample_bytree: float = 0.85
    xgb_reg_lambda: float = 2.0
    xgb_reg_alpha: float = 0.25
    calibration_bins: int = 10
    make_plots: bool = True
    save_full_source_models: bool = True


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

    def transform(self, x_raw: np.ndarray) -> np.ndarray:
        clr = clr_transform(x_raw, self.pseudocount)
        selected = clr[:, self.support_mask]
        transformed = (selected - self.scaler_mean) / self.scaler_scale
        if not np.isfinite(transformed).all():
            raise RuntimeError("Nonfinite values after applying fitted preprocessing.")
        return transformed


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_text(path, json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n")


def atomic_csv(frame: pd.DataFrame, path: Path, index: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    frame.to_csv(tmp, index=index)
    os.replace(tmp, path)


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(payload: dict[str, Any]) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=json_default)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalise_sample_id(value: object) -> str:
    text = str(value).strip()
    text = Path(text).name
    text = re.sub(r"\.npy$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"_H0$", "", text, flags=re.IGNORECASE)
    return text


def normalise_label(value: object) -> str:
    text = str(value).strip()
    compact = text.lower().replace("_", "-").replace(" ", "")
    aliases = {
        "nonibd": "nonIBD",
        "non-ibd": "nonIBD",
        "nonibdcontrol": "nonIBD",
        "ibd": "IBD",
        "uc": "UC",
        "ulcerativecolitis": "UC",
        "cd": "CD",
        "crohn'sdisease": "CD",
        "crohnsdisease": "CD",
    }
    return aliases.get(compact, text)


def find_col(frame: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    lookup = {str(column).strip().lower(): str(column) for column in frame.columns}
    for candidate in candidates:
        if candidate.strip().lower() in lookup:
            return lookup[candidate.strip().lower()]
    return None


def set_thread_environment(n_jobs: int) -> None:
    value = str(max(1, int(n_jobs)))
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(key, value)


def parse_csv_tuple(value: str) -> tuple[str, ...]:
    items = tuple(item.strip() for item in value.split(",") if item.strip())
    if not items:
        raise argparse.ArgumentTypeError("Expected a non-empty comma-separated list.")
    return items


def build_parser() -> argparse.ArgumentParser:
    base = Path.home() / "Real_Data"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--species-file", type=Path, default=base / "Real_Species_Abundances_canon.xlsx")
    parser.add_argument("--species-sheet", default="Sheet1")
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=base / "H0_Ricci_JointSparse_RepeatedCV_AllTasks" / "splits",
        help="Directory containing the five exact task manifests written by the joint H0+Ricci preflight.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=base / "Species_Benchmarks_RepeatedCV_RemainingTasks",
    )
    parser.add_argument("--tasks", type=parse_csv_tuple, default=TASK_ORDER)
    parser.add_argument("--models", type=parse_csv_tuple, default=MODEL_ORDER)
    parser.add_argument("--expected-repeats", type=int, default=20)
    parser.add_argument("--expected-folds", type=int, default=5)
    parser.add_argument("--expected-global-zero-profiles", type=int, default=10)
    parser.add_argument("--pseudocount", type=float, default=1e-6)
    parser.add_argument("--variance-threshold", type=float, default=0.0)
    parser.add_argument(
        "--weighting",
        choices=("class_balanced", "class_participant_balanced"),
        default="class_balanced",
    )
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--base-seed", type=int, default=20260720)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--no-full-source-models", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--overwrite-incompatible-output", action="store_true")
    return parser


def config_from_args(args: argparse.Namespace) -> RunConfig:
    tasks = tuple(args.tasks)
    models = tuple(args.models)
    invalid_tasks = sorted(set(tasks) - set(TASK_ORDER))
    if invalid_tasks:
        raise ValueError(f"Unknown task(s): {invalid_tasks}")
    if len(set(tasks)) != len(tasks):
        raise ValueError("Task list contains duplicates.")
    ordered = tuple(task for task in TASK_ORDER if task in set(tasks))
    if ordered != tasks:
        raise ValueError(f"Tasks must preserve the locked order: {TASK_ORDER}")
    invalid_models = sorted(set(models) - set(MODEL_ORDER))
    if invalid_models:
        raise ValueError(f"Unknown models: {invalid_models}")
    if len(set(models)) != len(models):
        raise ValueError("Model list contains duplicates.")
    if args.expected_repeats < 1 or args.expected_folds < 2:
        raise ValueError("Invalid repeated-CV dimensions.")
    return RunConfig(
        species_file=str(args.species_file.expanduser().resolve()),
        species_sheet=str(args.species_sheet),
        split_dir=str(args.split_dir.expanduser().resolve()),
        output_dir=str(args.output_dir.expanduser().resolve()),
        tasks=tasks,
        models=models,
        expected_repeats=int(args.expected_repeats),
        expected_folds=int(args.expected_folds),
        expected_global_zero_profiles=int(args.expected_global_zero_profiles),
        pseudocount=float(args.pseudocount),
        variance_threshold=float(args.variance_threshold),
        weighting=str(args.weighting),
        n_jobs=max(1, int(args.n_jobs)),
        base_seed=int(args.base_seed),
        make_plots=not bool(args.no_plots),
        save_full_source_models=not bool(args.no_full_source_models),
    )


def scientific_payload(config: RunConfig) -> dict[str, Any]:
    payload = asdict(config)
    for runtime_only in ("output_dir", "n_jobs", "make_plots", "save_full_source_models"):
        payload.pop(runtime_only, None)
    payload["schema_version"] = SCHEMA_VERSION
    payload["code_version"] = CODE_VERSION
    payload["task_order"] = list(TASK_ORDER)
    payload["model_order"] = list(MODEL_ORDER)
    payload["xgboost_weighting_rule"] = "balanced sample_weight only; scale_pos_weight=1.0 for binary"
    return payload


def input_paths(config: RunConfig) -> dict[str, Path]:
    paths = {"species_file": Path(config.species_file)}
    for task in ALL_TASKS:
        paths[f"manifest_{task.folder}"] = Path(config.split_dir) / f"{task.folder}_split_manifest.csv"
    return paths


def ensure_inputs_exist(config: RunConfig) -> None:
    missing = [str(path) for path in input_paths(config).values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Required inputs are missing:\n" + "\n".join(missing) +
            "\nRun the H0+Ricci all-task validate_inputs.py first so it writes the shared manifests."
        )


def current_run_lock(config: RunConfig) -> dict[str, Any]:
    paths = input_paths(config)
    return {
        "created_utc": now_iso(),
        "scientific_config": scientific_payload(config),
        "scientific_config_sha256": canonical_json_sha256(scientific_payload(config)),
        "input_sha256": {name: sha256_file(path) for name, path in paths.items()},
        "code_sha256": sha256_file(Path(__file__).resolve()),
    }


def ensure_compatible_output(config: RunConfig, overwrite: bool) -> dict[str, Any]:
    output = Path(config.output_dir)
    lock_path = output / "run_config.json"
    current = current_run_lock(config)
    if not output.exists():
        output.mkdir(parents=True)
        atomic_json(lock_path, current)
        return current
    contents = [path for path in output.iterdir()]
    if not contents:
        atomic_json(lock_path, current)
        return current
    if not lock_path.exists():
        if overwrite:
            import shutil
            shutil.rmtree(output)
            output.mkdir(parents=True)
            atomic_json(lock_path, current)
            return current
        raise RuntimeError(
            f"Output directory {output} is non-empty but has no run_config.json. "
            "Use a new directory or --overwrite-incompatible-output."
        )
    existing = json.loads(lock_path.read_text())
    keys = ("scientific_config_sha256", "input_sha256", "code_sha256")
    compatible = all(existing.get(key) == current.get(key) for key in keys)
    if not compatible:
        if overwrite:
            import shutil
            shutil.rmtree(output)
            output.mkdir(parents=True)
            atomic_json(lock_path, current)
            return current
        raise RuntimeError(
            "Existing output was created with different code, data, manifests, or scientific settings. "
            "Use a new output directory or --overwrite-incompatible-output."
        )
    print(f"[resume] Compatible run configuration found in {output}", flush=True)
    return existing


def load_manifest(path: Path, task: TaskSpec, config: RunConfig) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    required = {"task_folder", "repeat", "fold", "split_seed", "role", "sample_id", "participant_id", "label"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    frame = frame.copy()
    frame["task_folder"] = frame["task_folder"].astype(str).str.strip()
    frame = frame[frame["task_folder"].eq(task.folder)].copy()
    if frame.empty:
        raise ValueError(f"{path} contains no rows for {task.folder}")
    frame["repeat"] = pd.to_numeric(frame["repeat"], errors="raise").astype(int)
    frame["fold"] = pd.to_numeric(frame["fold"], errors="raise").astype(int)
    frame["split_seed"] = pd.to_numeric(frame["split_seed"], errors="raise").astype(int)
    frame["role"] = frame["role"].astype(str).str.strip().str.lower()
    frame["sample_id"] = frame["sample_id"].map(normalise_sample_id)
    frame["participant_id"] = frame["participant_id"].astype(str).str.strip()
    frame["label"] = frame["label"].map(normalise_label)
    if frame[["sample_id", "participant_id", "label"]].eq("").any().any():
        raise ValueError(f"{task.folder} manifest contains blank metadata.")
    if set(frame["role"]) != {"train", "test"}:
        raise ValueError(f"{task.folder} roles must be exactly train/test.")
    if sorted(frame["repeat"].unique()) != list(range(1, config.expected_repeats + 1)):
        raise ValueError(f"{task.folder} does not have repeats 1..{config.expected_repeats}.")
    if sorted(frame["fold"].unique()) != list(range(1, config.expected_folds + 1)):
        raise ValueError(f"{task.folder} does not have folds 1..{config.expected_folds}.")
    master = frame[["sample_id", "participant_id", "label"]].drop_duplicates()
    if master["sample_id"].duplicated().any():
        bad = master.loc[master["sample_id"].duplicated(False), "sample_id"].tolist()[:10]
        raise ValueError(f"{task.folder} has conflicting sample metadata: {bad}")
    if master.groupby("participant_id")["label"].nunique().gt(1).any():
        raise ValueError(f"{task.folder} has participants with conflicting labels.")
    expected_classes = set(task.class_order)
    if set(master["label"]) != expected_classes:
        raise ValueError(f"{task.folder} classes {sorted(set(master['label']))} != expected {task.class_order}")
    expected_samples = set(master["sample_id"])
    for repeat in range(1, config.expected_repeats + 1):
        repeat_rows = frame[frame["repeat"].eq(repeat)]
        if repeat_rows["split_seed"].nunique() != 1:
            raise ValueError(f"{task.folder} repeat {repeat} has multiple split seeds.")
        tested: set[str] = set()
        for fold in range(1, config.expected_folds + 1):
            rows = repeat_rows[repeat_rows["fold"].eq(fold)]
            train = rows[rows["role"].eq("train")]
            test = rows[rows["role"].eq("test")]
            if train.empty or test.empty:
                raise ValueError(f"{task.folder} repeat {repeat} fold {fold} lacks train/test.")
            if set(train["participant_id"]) & set(test["participant_id"]):
                raise ValueError(f"Participant leakage in {task.folder} repeat {repeat} fold {fold}.")
            if set(train["label"]) != expected_classes or set(test["label"]) != expected_classes:
                raise ValueError(f"A class is absent in {task.folder} repeat {repeat} fold {fold}.")
            test_ids = set(test["sample_id"])
            if tested & test_ids:
                raise ValueError(f"Repeated test samples in {task.folder} repeat {repeat}.")
            tested |= test_ids
        if tested != expected_samples:
            raise ValueError(f"Test folds do not cover every sample in {task.folder} repeat {repeat}.")
    return frame.reset_index(drop=True)


def load_all_manifests(config: RunConfig) -> dict[str, pd.DataFrame]:
    manifests = {}
    for task in ALL_TASKS:
        path = Path(config.split_dir) / f"{task.folder}_split_manifest.csv"
        manifests[task.folder] = load_manifest(path, task, config)
    return manifests


def load_species_matrix(
    path: Path,
    sheet: str,
    required_ids: set[str],
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    raw = pd.read_excel(path, sheet_name=sheet)
    sample_col = find_col(raw, ("External ID", "external_id", "sample_id", "sample", "id"))
    if sample_col is None:
        raise ValueError(f"Could not identify sample column in {path}; columns={list(raw.columns)[:20]}")
    raw = raw.copy()
    raw["_sample_id"] = raw[sample_col].map(normalise_sample_id)
    feature_cols = [str(column) for column in raw.columns if column not in {sample_col, "_sample_id"}]
    if not feature_cols:
        raise ValueError("No species feature columns found.")
    numeric = raw[feature_cols].apply(pd.to_numeric, errors="coerce")
    nonnumeric_populated = raw[feature_cols].notna() & numeric.isna()
    if nonnumeric_populated.to_numpy().any():
        examples = np.argwhere(nonnumeric_populated.to_numpy())[:10]
        raise ValueError(f"Species matrix contains nonnumeric populated cells at {examples.tolist()}")
    missing_count = int(numeric.isna().to_numpy().sum())
    numeric = numeric.fillna(0.0)
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("Species matrix contains infinite values.")
    if (numeric.to_numpy(dtype=float) < 0).any():
        raise ValueError("Species matrix contains negative values.")
    combined = pd.concat(
        [raw[["_sample_id"]].rename(columns={"_sample_id": "sample_id"}), numeric],
        axis=1,
    )
    grouped = combined.groupby("sample_id", as_index=False)[feature_cols].mean()
    missing = sorted(required_ids - set(grouped["sample_id"]))
    if missing:
        raise ValueError(f"Species table is missing required samples: {missing[:10]}")
    extra = sorted(set(grouped["sample_id"]) - required_ids)
    grouped = grouped[grouped["sample_id"].isin(required_ids)].copy()
    grouped = grouped.sort_values("sample_id", kind="mergesort").reset_index(drop=True)
    values = grouped[feature_cols].to_numpy(dtype=np.float64)
    row_sums = values.sum(axis=1)
    diagnostics = {
        "raw_rows": int(len(raw)),
        "unique_samples": int(len(grouped)),
        "duplicate_input_rows": int(len(raw) - raw["_sample_id"].nunique()),
        "n_features": int(len(feature_cols)),
        "missing_values_filled_with_zero": missing_count,
        "extra_species_samples_ignored": int(len(extra)),
        "extra_species_sample_examples": extra[:10],
        "zero_fraction": float(np.mean(values == 0.0)),
        "row_sum_min": float(row_sums.min()),
        "row_sum_median": float(np.median(row_sums)),
        "row_sum_max": float(row_sums.max()),
    }
    return grouped, feature_cols, diagnostics


def validate_complete_case_manifest(
    manifest: pd.DataFrame,
    task: TaskSpec,
    usable_ids: set[str],
    config: RunConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    filtered = manifest[manifest["sample_id"].isin(usable_ids)].copy()
    source_master = manifest[["sample_id", "participant_id", "label"]].drop_duplicates()
    complete_master = filtered[["sample_id", "participant_id", "label"]].drop_duplicates()
    excluded = source_master[~source_master["sample_id"].isin(usable_ids)].copy()
    excluded["exclusion_reason"] = "No nonzero species-level abundance after canonical filtering"
    if complete_master.empty:
        raise ValueError(f"Complete-case cohort is empty for {task.folder}")
    lost_participants = sorted(set(source_master["participant_id"]) - set(complete_master["participant_id"]))
    if lost_participants:
        raise ValueError(f"Complete-case filtering removes participants in {task.folder}: {lost_participants[:10]}")
    expected_classes = set(task.class_order)
    rows_out = []
    expected_samples = set(complete_master["sample_id"])
    for repeat in range(1, config.expected_repeats + 1):
        tested: set[str] = set()
        for fold in range(1, config.expected_folds + 1):
            rows = filtered[filtered["repeat"].eq(repeat) & filtered["fold"].eq(fold)]
            train = rows[rows["role"].eq("train")]
            test = rows[rows["role"].eq("test")]
            if set(train["participant_id"]) & set(test["participant_id"]):
                raise ValueError(f"Participant leakage after filtering in {task.folder} r{repeat} f{fold}")
            if set(train["label"]) != expected_classes or set(test["label"]) != expected_classes:
                raise ValueError(f"A class vanished after filtering in {task.folder} r{repeat} f{fold}")
            test_ids = set(test["sample_id"])
            if tested & test_ids:
                raise ValueError(f"Duplicate complete-case test samples in {task.folder} repeat {repeat}")
            tested |= test_ids
            row = {
                "repeat": repeat,
                "fold": fold,
                "train_samples": int(train["sample_id"].nunique()),
                "test_samples": int(test["sample_id"].nunique()),
                "train_participants": int(train["participant_id"].nunique()),
                "test_participants": int(test["participant_id"].nunique()),
            }
            for label in task.class_order:
                row[f"train_{label}_samples"] = int(train["label"].eq(label).sum())
                row[f"test_{label}_samples"] = int(test["label"].eq(label).sum())
            rows_out.append(row)
        if tested != expected_samples:
            raise ValueError(f"Complete-case test coverage mismatch in {task.folder} repeat {repeat}")
    return filtered.reset_index(drop=True), complete_master.reset_index(drop=True), pd.DataFrame(rows_out)


def load_bundle(config: RunConfig) -> dict[str, Any]:
    ensure_inputs_exist(config)
    manifests = load_all_manifests(config)
    all_master = pd.concat(
        [frame[["sample_id", "participant_id", "label"]] for frame in manifests.values()],
        ignore_index=True,
    )[["sample_id"]].drop_duplicates()
    required_ids = set(all_master["sample_id"])
    species, feature_names, species_diag = load_species_matrix(
        Path(config.species_file), config.species_sheet, required_ids
    )
    value_matrix = species[feature_names].to_numpy(dtype=np.float64)
    zero_mask = np.isclose(value_matrix.sum(axis=1), 0.0, rtol=0.0, atol=0.0)
    zero_ids = sorted(species.loc[zero_mask, "sample_id"].tolist())
    if config.expected_global_zero_profiles >= 0 and len(zero_ids) != config.expected_global_zero_profiles:
        raise ValueError(
            f"Expected {config.expected_global_zero_profiles} global all-zero profiles; found {len(zero_ids)}: {zero_ids[:20]}"
        )
    usable_ids = set(species.loc[~zero_mask, "sample_id"])
    species_indexed = species.set_index("sample_id", verify_integrity=True)
    task_bundles = {}
    audit_rows = []
    for task_folder in config.tasks:
        task = TASK_BY_FOLDER[task_folder]
        manifest = manifests[task_folder]
        filtered, master, fold_counts = validate_complete_case_manifest(
            manifest, task, usable_ids, config
        )
        x = species_indexed.loc[master["sample_id"], feature_names].to_numpy(dtype=np.float64)
        sample_to_index = {sid: idx for idx, sid in enumerate(master["sample_id"])}
        excluded = (
            manifest[["sample_id", "participant_id", "label"]]
            .drop_duplicates()
            .loc[lambda d: ~d["sample_id"].isin(usable_ids)]
            .copy()
        )
        excluded["exclusion_reason"] = "No nonzero species-level abundance after canonical filtering"
        audit_rows.append({
            "task_folder": task.folder,
            "task": task.report_name,
            "source_samples": int(manifest["sample_id"].nunique()),
            "complete_case_samples": int(len(master)),
            "excluded_zero_profiles": int(len(excluded)),
            "participants": int(master["participant_id"].nunique()),
            "classes": "|".join(task.class_order),
            "class_sample_counts": json.dumps(master["label"].value_counts().to_dict(), sort_keys=True),
            "class_participant_counts": json.dumps(master.groupby("label")["participant_id"].nunique().to_dict(), sort_keys=True),
        })
        task_bundles[task_folder] = {
            "task": task,
            "manifest": filtered,
            "source_manifest": manifest,
            "sample_master": master,
            "X": x,
            "sample_to_index": sample_to_index,
            "excluded": excluded,
            "fold_counts": fold_counts,
        }
    return {
        "feature_names": feature_names,
        "species_diagnostics": species_diag,
        "global_zero_sample_ids": zero_ids,
        "task_bundles": task_bundles,
        "task_audit": pd.DataFrame(audit_rows),
    }


def clr_transform(x: np.ndarray, pseudocount: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if not np.isfinite(x).all() or (x < 0).any():
        raise ValueError("CLR input must be finite and nonnegative.")
    log_x = np.log(x + pseudocount)
    return log_x - log_x.mean(axis=1, keepdims=True)


def fit_preprocessor(
    x_train_raw: np.ndarray,
    feature_names: list[str],
    pseudocount: float,
    variance_threshold: float,
) -> tuple[FittedPreprocessor, np.ndarray]:
    x_clr = clr_transform(x_train_raw, pseudocount)
    selector = VarianceThreshold(threshold=variance_threshold)
    x_selected = selector.fit_transform(x_clr)
    support = selector.get_support()
    if x_selected.shape[1] == 0:
        raise RuntimeError("VarianceThreshold removed every feature.")
    scaler = StandardScaler(with_mean=True, with_std=True)
    x_processed = scaler.fit_transform(x_selected)
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
    return fitted, x_processed


def preprocessor_payload(pre: FittedPreprocessor) -> dict[str, Any]:
    return {
        "support_mask": np.asarray(pre.support_mask, dtype=bool),
        "scaler_mean": np.asarray(pre.scaler_mean, dtype=np.float64),
        "scaler_scale": np.asarray(pre.scaler_scale, dtype=np.float64),
        "scaler_var": np.asarray(pre.scaler_var, dtype=np.float64),
        "input_feature_names": list(pre.input_feature_names),
        "selected_feature_names": list(pre.selected_feature_names),
        "pseudocount": float(pre.pseudocount),
        "variance_threshold": float(pre.variance_threshold),
    }


def training_weights(y: np.ndarray, participants: np.ndarray, mode: str) -> np.ndarray:
    weights = compute_sample_weight(class_weight="balanced", y=y).astype(np.float64)
    if mode == "class_balanced":
        return weights
    counts = pd.Series(participants.astype(str)).value_counts()
    participant_factor = np.array([1.0 / counts[str(pid)] for pid in participants], dtype=np.float64)
    weights *= participant_factor
    return weights / weights.mean()


def model_seed(config: RunConfig, task_index: int, model_name: str, repeat: int, fold: int) -> int:
    model_offset = {"logistic_l1": 11003, "random_forest": 23003, "xgboost": 37003}[model_name]
    return int(config.base_seed + task_index * 100_003 + model_offset + repeat * 1009 + fold * 97)


def build_model(model_name: str, config: RunConfig, seed: int, n_classes: int) -> BaseEstimator:
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
        except Exception as exc:
            raise RuntimeError("XGBoost is required. Install requirements.txt.") from exc
        common = dict(
            n_estimators=config.xgb_estimators,
            max_depth=config.xgb_max_depth,
            learning_rate=config.xgb_learning_rate,
            subsample=config.xgb_subsample,
            colsample_bytree=config.xgb_colsample_bytree,
            reg_lambda=config.xgb_reg_lambda,
            reg_alpha=config.xgb_reg_alpha,
            random_state=seed,
            n_jobs=config.n_jobs,
            importance_type="gain",
        )
        if n_classes == 2:
            return XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                scale_pos_weight=1.0,
                **common,
            )
        return XGBClassifier(
            objective="multi:softprob",
            num_class=n_classes,
            eval_metric="mlogloss",
            **common,
        )
    raise ValueError(model_name)


def fit_model(
    model_name: str,
    model: BaseEstimator,
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
) -> tuple[BaseEstimator, list[str], float]:
    start = time.perf_counter()
    messages: list[str] = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if model_name == "xgboost":
            model.fit(x, y, sample_weight=weights, verbose=False)
        else:
            model.fit(x, y, sample_weight=weights)
        for warning in caught:
            messages.append(f"{warning.category.__name__}: {warning.message}")
    return model, messages, time.perf_counter() - start


def predict_proba_aligned(model: BaseEstimator, x: np.ndarray, n_classes: int) -> np.ndarray:
    raw = np.asarray(model.predict_proba(x), dtype=np.float64)
    classes = np.asarray(model.classes_, dtype=int)
    output = np.zeros((len(x), n_classes), dtype=np.float64)
    for source_col, class_index in enumerate(classes):
        if class_index < 0 or class_index >= n_classes:
            raise RuntimeError(f"Unexpected model class index {class_index}")
        output[:, int(class_index)] = raw[:, source_col]
    row_sums = output.sum(axis=1)
    if not np.isfinite(output).all() or np.any(row_sums <= 0):
        raise RuntimeError("Model produced invalid probabilities.")
    output /= row_sums[:, None]
    return np.clip(output, 0.0, 1.0)


def metric_dict(y_true: np.ndarray, proba: np.ndarray, class_names: tuple[str, ...]) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=int)
    proba = np.asarray(proba, dtype=np.float64)
    n_classes = len(class_names)
    if proba.shape != (len(y_true), n_classes):
        raise ValueError("Probability matrix shape mismatch.")
    pred = np.argmax(proba, axis=1)
    onehot = np.eye(n_classes, dtype=np.float64)[y_true]
    result: dict[str, Any] = {
        "n_observations": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "brier": float(
            np.mean((proba[:, 1] - y_true) ** 2)
            if n_classes == 2
            else np.mean(np.sum((proba - onehot) ** 2, axis=1))
        ),
        "log_loss": float(log_loss(y_true, np.clip(proba, EPS, 1.0 - EPS), labels=list(range(n_classes)))),
    }
    try:
        if n_classes == 2:
            result["roc_auc"] = float(roc_auc_score(y_true, proba[:, 1]))
        else:
            result["roc_auc"] = float(
                roc_auc_score(y_true, proba, multi_class="ovr", average="macro", labels=list(range(n_classes)))
            )
    except ValueError:
        result["roc_auc"] = float("nan")
    cm = confusion_matrix(y_true, pred, labels=list(range(n_classes)))
    result["confusion_matrix"] = cm.tolist()
    recalls = recall_score(y_true, pred, labels=list(range(n_classes)), average=None, zero_division=0)
    f1s = f1_score(y_true, pred, labels=list(range(n_classes)), average=None, zero_division=0)
    for idx, label in enumerate(class_names):
        result[f"recall_{label}"] = float(recalls[idx])
        result[f"f1_{label}"] = float(f1s[idx])
        binary_truth = (y_true == idx).astype(int)
        try:
            result[f"ovr_auc_{label}"] = float(roc_auc_score(binary_truth, proba[:, idx]))
        except ValueError:
            result[f"ovr_auc_{label}"] = float("nan")
    return result


def participant_predictions(sample_predictions: pd.DataFrame, class_names: tuple[str, ...]) -> pd.DataFrame:
    prob_cols = [f"proba_{label}" for label in class_names]
    label_counts = sample_predictions.groupby("participant_id")["y_true"].nunique()
    if label_counts.gt(1).any():
        raise RuntimeError("Participant has conflicting labels in sample predictions.")
    grouped = sample_predictions.groupby("participant_id", as_index=False).agg(
        y_true=("y_true", "first"),
        n_samples=("sample_id", "nunique"),
        **{column: (column, "mean") for column in prob_cols},
    )
    matrix = grouped[prob_cols].to_numpy(dtype=np.float64)
    grouped["pred"] = np.argmax(matrix, axis=1)
    grouped["true_label"] = grouped["y_true"].map(dict(enumerate(class_names)))
    grouped["pred_label"] = grouped["pred"].map(dict(enumerate(class_names)))
    return grouped


def model_complexity(model_name: str, model: BaseEstimator) -> dict[str, Any]:
    if model_name == "logistic_l1":
        coef = np.asarray(model.coef_, dtype=np.float64)
        return {
            "n_nonzero_coefficients": int(np.count_nonzero(np.abs(coef) > 1e-12)),
            "coefficient_l1_norm": float(np.abs(coef).sum()),
            "coefficient_l2_norm": float(np.linalg.norm(coef)),
            "n_iter": int(np.max(np.asarray(model.n_iter_))),
        }
    if model_name == "random_forest":
        depths = np.asarray([tree.tree_.max_depth for tree in model.estimators_], dtype=float)
        leaves = np.asarray([tree.tree_.n_leaves for tree in model.estimators_], dtype=float)
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
        try:
            rounds = int(booster.num_boosted_rounds())
        except Exception:
            rounds = int(len(booster.get_dump()))
        scale_pos_weight = model.get_params().get("scale_pos_weight", 1.0)
        return {
            "n_boosting_rounds": rounds,
            "n_features_with_positive_gain": int(len(importance)),
            "scale_pos_weight": float(scale_pos_weight) if scale_pos_weight is not None else np.nan,
        }
    return {}


def feature_diagnostics(
    model_name: str,
    model: BaseEstimator,
    pre: FittedPreprocessor,
    class_names: tuple[str, ...],
) -> pd.DataFrame:
    n_features = len(pre.input_feature_names)
    selected = np.flatnonzero(pre.support_mask)
    if model_name == "logistic_l1":
        coef = np.asarray(model.coef_, dtype=np.float64)
        if len(class_names) == 2 and coef.shape[0] == 1:
            labels = (class_names[1],)
        elif coef.shape[0] == len(class_names):
            labels = class_names
        else:
            raise RuntimeError(f"Unexpected logistic coefficient shape {coef.shape}")
        frames = []
        for row_idx, target_class in enumerate(labels):
            values = np.zeros(n_features, dtype=np.float64)
            clr_values = np.zeros(n_features, dtype=np.float64)
            values[selected] = coef[row_idx]
            clr_values[selected] = coef[row_idx] / pre.scaler_scale
            frames.append(pd.DataFrame({
                "target_class": target_class,
                "feature_index": np.arange(n_features, dtype=int),
                "feature": pre.input_feature_names,
                "variance_selected": pre.support_mask.astype(int),
                "model_value": values,
                "clr_space_logit_coefficient": clr_values,
                "model_used": (np.abs(values) > 1e-12).astype(int),
                "value_type": "signed_logit_coefficient",
            }))
        return pd.concat(frames, ignore_index=True)
    values = np.zeros(n_features, dtype=np.float64)
    importances = np.asarray(model.feature_importances_, dtype=np.float64)
    if len(importances) != len(selected):
        raise RuntimeError(f"Feature importance length mismatch for {model_name}")
    values[selected] = importances
    return pd.DataFrame({
        "target_class": "__global__",
        "feature_index": np.arange(n_features, dtype=int),
        "feature": pre.input_feature_names,
        "variance_selected": pre.support_mask.astype(int),
        "model_value": values,
        "clr_space_logit_coefficient": np.nan,
        "model_used": (values > 0).astype(int),
        "value_type": "gain_importance" if model_name == "xgboost" else "impurity_importance",
    })


def fold_rows(
    manifest: pd.DataFrame,
    master: pd.DataFrame,
    sample_to_index: dict[str, int],
    repeat: int,
    fold: int,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    rows = manifest[manifest["repeat"].eq(repeat) & manifest["fold"].eq(fold)]
    train = rows[rows["role"].eq("train")][["sample_id", "participant_id", "label"]].drop_duplicates()
    test = rows[rows["role"].eq("test")][["sample_id", "participant_id", "label"]].drop_duplicates()
    train_order = master[master["sample_id"].isin(train["sample_id"])][["sample_id"]]
    train = train_order.merge(
        rows[rows["role"].eq("train")][["sample_id", "participant_id", "label"]].drop_duplicates(),
        on="sample_id", validate="one_to_one"
    )
    test_order = master[master["sample_id"].isin(test["sample_id"])][["sample_id"]]
    test = test_order.merge(
        rows[rows["role"].eq("test")][["sample_id", "participant_id", "label"]].drop_duplicates(),
        on="sample_id", validate="one_to_one"
    )
    train_idx = np.array([sample_to_index[sid] for sid in train["sample_id"]], dtype=int)
    test_idx = np.array([sample_to_index[sid] for sid in test["sample_id"]], dtype=int)
    return train, test, train_idx, test_idx


def fold_dir(root: Path, task_folder: str, model: str, repeat: int, fold: int) -> Path:
    return root / task_folder / "folds" / model / f"repeat_{repeat:02d}" / f"fold_{fold}"


def fold_marker_payload(
    config_fingerprint: str,
    task: TaskSpec,
    model: str,
    repeat: int,
    fold: int,
    class_names: tuple[str, ...],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "code_version": CODE_VERSION,
        "config_fingerprint": config_fingerprint,
        "task_folder": task.folder,
        "model": model,
        "repeat": repeat,
        "fold": fold,
        "class_names": list(class_names),
    }


def load_completed_fold(
    directory: Path,
    expected: dict[str, Any],
) -> Optional[dict[str, Any]]:
    marker = directory / "FOLD_COMPLETE.json"
    required = (
        directory / "metrics.json",
        directory / "test_predictions.csv",
        directory / "participant_test_predictions.csv",
        directory / "feature_diagnostics.csv.gz",
    )
    if not marker.exists():
        return None
    if any(not path.exists() for path in required):
        raise RuntimeError(f"Corrupt completed fold at {directory}")
    payload = json.loads(marker.read_text())
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError(f"Completed fold incompatibility at {directory}: {key}")
    return {
        "metrics": json.loads((directory / "metrics.json").read_text()),
        "sample_predictions": pd.read_csv(directory / "test_predictions.csv"),
        "participant_predictions": pd.read_csv(directory / "participant_test_predictions.csv"),
        "feature_diagnostics": pd.read_csv(directory / "feature_diagnostics.csv.gz"),
    }


def run_fold(
    config: RunConfig,
    config_fingerprint: str,
    root: Path,
    task_index: int,
    bundle: dict[str, Any],
    feature_names: list[str],
    model_name: str,
    repeat: int,
    fold: int,
) -> dict[str, Any]:
    task: TaskSpec = bundle["task"]
    class_names = task.class_order
    class_to_index = {label: idx for idx, label in enumerate(class_names)}
    out = fold_dir(root, task.folder, model_name, repeat, fold)
    expected = fold_marker_payload(config_fingerprint, task, model_name, repeat, fold, class_names)
    completed = load_completed_fold(out, expected)
    if completed is not None:
        print(f"[resume] {task.folder} {model_name} r{repeat:02d} f{fold}", flush=True)
        return completed
    out.mkdir(parents=True, exist_ok=True)
    train_meta, test_meta, train_idx, test_idx = fold_rows(
        bundle["manifest"], bundle["sample_master"], bundle["sample_to_index"], repeat, fold
    )
    x_raw = bundle["X"]
    y_train = train_meta["label"].map(class_to_index).to_numpy(dtype=int)
    y_test = test_meta["label"].map(class_to_index).to_numpy(dtype=int)
    start = time.perf_counter()
    pre, x_train = fit_preprocessor(
        x_raw[train_idx], feature_names, config.pseudocount, config.variance_threshold
    )
    x_test = pre.transform(x_raw[test_idx])
    weights = training_weights(
        y_train, train_meta["participant_id"].to_numpy(dtype=str), config.weighting
    )
    seed = model_seed(config, task_index, model_name, repeat, fold)
    model = build_model(model_name, config, seed, len(class_names))
    model, warning_messages, fit_runtime = fit_model(model_name, model, x_train, y_train, weights)
    train_proba = predict_proba_aligned(model, x_train, len(class_names))
    test_proba = predict_proba_aligned(model, x_test, len(class_names))
    train_metrics = metric_dict(y_train, train_proba, class_names)
    test_metrics = metric_dict(y_test, test_proba, class_names)
    pred = np.argmax(test_proba, axis=1)
    sample_pred = test_meta.copy()
    sample_pred["y_true"] = y_test
    sample_pred["pred"] = pred
    sample_pred["true_label"] = sample_pred["label"]
    sample_pred["pred_label"] = [class_names[idx] for idx in pred]
    sample_pred = sample_pred.drop(columns="label")
    for idx, label in enumerate(class_names):
        sample_pred[f"proba_{label}"] = test_proba[:, idx]
    sample_pred.insert(0, "fold", fold)
    sample_pred.insert(0, "repeat", repeat)
    sample_pred.insert(0, "model", model_name)
    sample_pred.insert(0, "task_folder", task.folder)
    participant_pred = participant_predictions(sample_pred, class_names)
    participant_matrix = participant_pred[[f"proba_{label}" for label in class_names]].to_numpy(dtype=float)
    participant_metrics = metric_dict(participant_pred["y_true"].to_numpy(), participant_matrix, class_names)
    participant_pred.insert(0, "fold", fold)
    participant_pred.insert(0, "repeat", repeat)
    participant_pred.insert(0, "model", model_name)
    participant_pred.insert(0, "task_folder", task.folder)
    diag = feature_diagnostics(model_name, model, pre, class_names)
    diag.insert(0, "fold", fold)
    diag.insert(0, "repeat", repeat)
    diag.insert(0, "model", model_name)
    diag.insert(0, "task_folder", task.folder)
    convergence_messages = [message for message in warning_messages if "ConvergenceWarning" in message]
    complexity = model_complexity(model_name, model)
    max_iter_reached = (
        model_name == "logistic_l1" and int(complexity.get("n_iter", -1)) >= config.logistic_max_iter
    )
    metrics = {
        **expected,
        "task": task.report_name,
        "model_label": MODEL_LABELS[model_name],
        "seed": seed,
        "n_train_samples": int(len(train_meta)),
        "n_test_samples": int(len(test_meta)),
        "n_train_participants": int(train_meta["participant_id"].nunique()),
        "n_test_participants": int(test_meta["participant_id"].nunique()),
        "n_input_features": int(len(feature_names)),
        "n_variance_selected_features": int(pre.support_mask.sum()),
        "weighting": config.weighting,
        "fit_runtime_seconds": fit_runtime,
        "total_runtime_seconds": time.perf_counter() - start,
        "warning_messages": warning_messages,
        "convergence_warning": bool(convergence_messages),
        "convergence_at_iteration_limit": bool(max_iter_reached),
        "train_sample": train_metrics,
        "test_sample": test_metrics,
        "test_participant": participant_metrics,
        "complexity": complexity,
    }
    atomic_json(out / "metrics.json", metrics)
    atomic_csv(sample_pred, out / "test_predictions.csv")
    atomic_csv(participant_pred, out / "participant_test_predictions.csv")
    tmp_diag = out / "feature_diagnostics.csv.gz.tmp"
    diag.to_csv(tmp_diag, index=False, compression="gzip")
    os.replace(tmp_diag, out / "feature_diagnostics.csv.gz")
    atomic_json(out / "FOLD_COMPLETE.json", {**expected, "completed_utc": now_iso()})
    print(
        f"[{task.folder} {model_name} r{repeat:02d} f{fold}] "
        f"bal={test_metrics['balanced_accuracy']:.4f} auc={test_metrics['roc_auc']:.4f} "
        f"participant_bal={participant_metrics['balanced_accuracy']:.4f} "
        f"features={int(pre.support_mask.sum())} runtime={metrics['total_runtime_seconds']:.1f}s",
        flush=True,
    )
    return {
        "metrics": metrics,
        "sample_predictions": sample_pred,
        "participant_predictions": participant_pred,
        "feature_diagnostics": diag,
    }


def flatten_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    row = {key: value for key, value in metrics.items() if key not in {"train_sample", "test_sample", "test_participant", "complexity", "warning_messages"}}
    row["warning_messages"] = " | ".join(metrics.get("warning_messages", []))
    for prefix in ("train_sample", "test_sample", "test_participant"):
        for key, value in metrics[prefix].items():
            if key == "confusion_matrix":
                row[f"{prefix}_{key}"] = json.dumps(value)
            else:
                row[f"{prefix}_{key}"] = value
    for key, value in metrics.get("complexity", {}).items():
        row[f"complexity_{key}"] = value
    return row


def repetition_metrics(
    predictions: pd.DataFrame,
    model_name: str,
    class_names: tuple[str, ...],
    expected_repeats: int,
    level: str,
) -> pd.DataFrame:
    rows = []
    id_col = "sample_id" if level == "sample" else "participant_id"
    prob_cols = [f"proba_{label}" for label in class_names]
    for repeat in range(1, expected_repeats + 1):
        subset = predictions[predictions["model"].eq(model_name) & predictions["repeat"].eq(repeat)]
        if subset.empty:
            continue
        if subset[id_col].duplicated().any():
            raise RuntimeError(f"Duplicate {level} OOF predictions for {model_name} repeat {repeat}")
        metrics = metric_dict(subset["y_true"].to_numpy(), subset[prob_cols].to_numpy(), class_names)
        rows.append({
            "model": model_name,
            "model_label": MODEL_LABELS[model_name],
            "repeat": repeat,
            "evaluation_level": level,
            **{key: value for key, value in metrics.items() if key != "confusion_matrix"},
            "confusion_matrix": json.dumps(metrics["confusion_matrix"]),
        })
    return pd.DataFrame(rows)


def summarize_repetitions(frame: pd.DataFrame) -> pd.DataFrame:
    id_cols = {"model", "model_label", "repeat", "evaluation_level", "confusion_matrix", "n_observations"}
    metric_cols = [column for column in frame.columns if column not in id_cols]
    rows = []
    for (model, label, level), group in frame.groupby(["model", "model_label", "evaluation_level"], sort=False):
        row = {
            "model": model,
            "model_label": label,
            "evaluation_level": level,
            "n_repetitions": int(group["repeat"].nunique()),
        }
        for metric in metric_cols:
            values = pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy(dtype=float)
            if not len(values):
                continue
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_sd"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            row[f"{metric}_median"] = float(np.median(values))
            row[f"{metric}_q025"] = float(np.quantile(values, 0.025))
            row[f"{metric}_q975"] = float(np.quantile(values, 0.975))
        rows.append(row)
    return pd.DataFrame(rows)


def paired_comparisons(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = ("accuracy", "balanced_accuracy", "macro_f1", "roc_auc", "brier", "log_loss")
    detail = []
    summary = []
    for level in sorted(frame["evaluation_level"].unique()):
        subset = frame[frame["evaluation_level"].eq(level)]
        available = set(subset["model"])
        for i, model_a in enumerate(MODEL_ORDER):
            if model_a not in available:
                continue
            for model_b in MODEL_ORDER[i + 1:]:
                if model_b not in available:
                    continue
                merged = subset[subset["model"].eq(model_a)].merge(
                    subset[subset["model"].eq(model_b)], on=["repeat", "evaluation_level"], suffixes=("_a", "_b"), validate="one_to_one"
                )
                for metric in metrics:
                    if f"{metric}_a" not in merged or f"{metric}_b" not in merged:
                        continue
                    if metric in {"brier", "log_loss"}:
                        delta = merged[f"{metric}_b"] - merged[f"{metric}_a"]
                    else:
                        delta = merged[f"{metric}_a"] - merged[f"{metric}_b"]
                    values = delta.to_numpy(dtype=float)
                    for rep, value in zip(merged["repeat"], values):
                        detail.append({
                            "evaluation_level": level,
                            "model_a": model_a,
                            "model_b": model_b,
                            "metric": metric,
                            "repeat": int(rep),
                            "delta_a_better_positive": float(value),
                        })
                    summary.append({
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
                    })
    return pd.DataFrame(detail), pd.DataFrame(summary)


def feature_stability(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, target_class, feature), group in frame.groupby(["model", "target_class", "feature"], sort=False):
        values = group["model_value"].to_numpy(dtype=float)
        used = np.abs(values) > 1e-12
        used_values = values[used]
        rows.append({
            "model": model,
            "model_label": MODEL_LABELS[model],
            "target_class": target_class,
            "feature": feature,
            "n_fits": int(len(values)),
            "variance_selected_frequency": float(group["variance_selected"].mean()),
            "model_used_frequency": float(used.mean()),
            "model_value_mean_including_zero": float(values.mean()),
            "model_value_sd_including_zero": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "model_value_median_including_zero": float(np.median(values)),
            "model_value_q025_including_zero": float(np.quantile(values, 0.025)),
            "model_value_q975_including_zero": float(np.quantile(values, 0.975)),
            "model_value_mean_when_used": float(used_values.mean()) if len(used_values) else np.nan,
            "positive_when_used_fraction": float(np.mean(used_values > 0)) if len(used_values) else np.nan,
            "negative_when_used_fraction": float(np.mean(used_values < 0)) if len(used_values) else np.nan,
            "absolute_value_mean_including_zero": float(np.abs(values).mean()),
        })
    output = pd.DataFrame(rows)
    return output.sort_values(
        ["model", "target_class", "model_used_frequency", "absolute_value_mean_including_zero"],
        ascending=[True, True, False, False], kind="mergesort"
    ).reset_index(drop=True)


def consensus_predictions(
    sample_predictions: pd.DataFrame,
    participant_preds: pd.DataFrame,
    class_names: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prob_cols = [f"proba_{label}" for label in class_names]
    sample_agg = {
        "y_true": ("y_true", "first"),
        "n_repetitions": ("repeat", "nunique"),
    }
    for column in prob_cols:
        sample_agg[f"{column}_mean"] = (column, "mean")
        sample_agg[f"{column}_sd"] = (column, "std")
    sample = sample_predictions.groupby(["model", "sample_id", "participant_id"], as_index=False).agg(**sample_agg)
    sample_matrix = sample[[f"{column}_mean" for column in prob_cols]].to_numpy(dtype=float)
    sample["pred"] = np.argmax(sample_matrix, axis=1)
    sample["true_label"] = sample["y_true"].map(dict(enumerate(class_names)))
    sample["pred_label"] = sample["pred"].map(dict(enumerate(class_names)))

    part_agg = {
        "y_true": ("y_true", "first"),
        "n_repetitions": ("repeat", "nunique"),
        "mean_samples_per_repeat": ("n_samples", "mean"),
    }
    for column in prob_cols:
        part_agg[f"{column}_mean"] = (column, "mean")
        part_agg[f"{column}_sd"] = (column, "std")
    participant = participant_preds.groupby(["model", "participant_id"], as_index=False).agg(**part_agg)
    participant_matrix = participant[[f"{column}_mean" for column in prob_cols]].to_numpy(dtype=float)
    participant["pred"] = np.argmax(participant_matrix, axis=1)
    participant["true_label"] = participant["y_true"].map(dict(enumerate(class_names)))
    participant["pred_label"] = participant["pred"].map(dict(enumerate(class_names)))

    metric_rows = []
    for model in MODEL_ORDER:
        if model not in set(sample["model"]):
            continue
        for level, frame, matrix in (
            ("sample", sample[sample["model"].eq(model)], None),
            ("participant", participant[participant["model"].eq(model)], None),
        ):
            cols = [f"proba_{label}_mean" for label in class_names]
            metrics = metric_dict(frame["y_true"].to_numpy(), frame[cols].to_numpy(), class_names)
            metric_rows.append({
                "model": model,
                "model_label": MODEL_LABELS[model],
                "evaluation_level": level,
                **{key: value for key, value in metrics.items() if key != "confusion_matrix"},
                "confusion_matrix": json.dumps(metrics["confusion_matrix"]),
            })
    return sample, participant, pd.DataFrame(metric_rows)


def make_plots(repetition: pd.DataFrame, out: Path, task: TaskSpec) -> None:
    plot_dir = out / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    for level in ("sample", "participant"):
        subset = repetition[repetition["evaluation_level"].eq(level)]
        for metric in ("balanced_accuracy", "macro_f1", "roc_auc"):
            groups, labels = [], []
            for model in MODEL_ORDER:
                values = subset.loc[subset["model"].eq(model), metric].dropna().to_numpy(dtype=float)
                if len(values):
                    groups.append(values)
                    labels.append(MODEL_LABELS[model])
            if groups:
                plt.figure(figsize=(8, 5))
                plt.boxplot(groups, labels=labels, showmeans=True)
                plt.ylabel(metric.replace("_", " ").title())
                plt.title(f"{task.report_name}: pooled OOF {metric.replace('_', ' ')} ({level})")
                plt.xticks(rotation=20, ha="right")
                plt.tight_layout()
                plt.savefig(plot_dir / f"{level}_{metric}.png", dpi=220)
                plt.close()


def train_full_source_model(
    config: RunConfig,
    config_fingerprint: str,
    task_index: int,
    bundle: dict[str, Any],
    feature_names: list[str],
    model_name: str,
    task_root: Path,
) -> None:
    if not config.save_full_source_models:
        return
    task: TaskSpec = bundle["task"]
    class_names = task.class_order
    out = task_root / "full_source_models" / model_name
    marker = out / "MODEL_COMPLETE.json"
    expected = {
        "config_fingerprint": config_fingerprint,
        "task_folder": task.folder,
        "model": model_name,
        "class_names": list(class_names),
    }
    if marker.exists():
        payload = json.loads(marker.read_text())
        if all(payload.get(key) == value for key, value in expected.items()) and (out / "full_source_model.joblib").exists():
            print(f"[resume] full-source model {task.folder} {model_name}", flush=True)
            return
        raise RuntimeError(f"Incompatible full-source model marker at {out}")
    out.mkdir(parents=True, exist_ok=True)
    master = bundle["sample_master"]
    class_to_index = {label: idx for idx, label in enumerate(class_names)}
    y = master["label"].map(class_to_index).to_numpy(dtype=int)
    pre, x = fit_preprocessor(bundle["X"], feature_names, config.pseudocount, config.variance_threshold)
    weights = training_weights(y, master["participant_id"].to_numpy(dtype=str), config.weighting)
    seed = model_seed(config, task_index, model_name, 0, 0)
    model = build_model(model_name, config, seed, len(class_names))
    model, warning_messages, runtime = fit_model(model_name, model, x, y, weights)
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "code_version": CODE_VERSION,
        "task_folder": task.folder,
        "task": task.report_name,
        "model_name": model_name,
        "class_names": class_names,
        "model": model,
        "preprocessor": preprocessor_payload(pre),
        "training_sample_ids": master["sample_id"].tolist(),
        "training_participant_ids": master["participant_id"].tolist(),
        "training_labels": master["label"].tolist(),
        "seed": seed,
        "fit_runtime_seconds": runtime,
        "warning_messages": warning_messages,
    }
    joblib.dump(artifact, out / "full_source_model.joblib", compress=3)
    diag = feature_diagnostics(model_name, model, pre, class_names)
    atomic_csv(diag, out / "full_source_feature_diagnostics.csv")
    atomic_json(marker, {**expected, "completed_utc": now_iso()})


def finalize_task(
    config: RunConfig,
    config_fingerprint: str,
    task_index: int,
    bundle: dict[str, Any],
    feature_names: list[str],
    completed: list[dict[str, Any]],
    root: Path,
) -> None:
    task: TaskSpec = bundle["task"]
    task_root = root / task.folder
    fold_metrics = pd.DataFrame([flatten_metrics(item["metrics"]) for item in completed])
    expected = len(config.models) * config.expected_repeats * config.expected_folds
    if len(fold_metrics) != expected:
        raise RuntimeError(f"{task.folder}: expected {expected} folds, found {len(fold_metrics)}")
    fold_metrics = fold_metrics.sort_values(["model", "repeat", "fold"]).reset_index(drop=True)
    samples = pd.concat([item["sample_predictions"] for item in completed], ignore_index=True)
    participants = pd.concat([item["participant_predictions"] for item in completed], ignore_index=True)
    diagnostics = pd.concat([item["feature_diagnostics"] for item in completed], ignore_index=True)
    atomic_csv(fold_metrics, task_root / "all_outer_fold_metrics.csv")
    samples.to_csv(task_root / "all_outer_test_sample_predictions.csv.gz", index=False, compression="gzip")
    participants.to_csv(task_root / "all_outer_test_participant_predictions.csv.gz", index=False, compression="gzip")
    rep_frames = []
    for model in config.models:
        rep_frames.append(repetition_metrics(samples, model, task.class_order, config.expected_repeats, "sample"))
        rep_frames.append(repetition_metrics(participants, model, task.class_order, config.expected_repeats, "participant"))
    repetition = pd.concat(rep_frames, ignore_index=True).sort_values(["evaluation_level", "model", "repeat"])
    atomic_csv(repetition, task_root / "repetition_pooled_oof_metrics.csv")
    summary = summarize_repetitions(repetition)
    atomic_csv(summary, task_root / "repetition_performance_summary.csv")
    paired_detail, paired_summary = paired_comparisons(repetition)
    atomic_csv(paired_detail, task_root / "paired_model_differences_by_repetition.csv")
    atomic_csv(paired_summary, task_root / "paired_model_difference_summary.csv")
    sample_consensus, participant_consensus, consensus_metrics = consensus_predictions(samples, participants, task.class_order)
    atomic_csv(sample_consensus, task_root / "consensus_predictions_by_sample.csv")
    atomic_csv(participant_consensus, task_root / "consensus_predictions_by_participant.csv")
    atomic_csv(consensus_metrics, task_root / "consensus_prediction_metrics.csv")
    stability = feature_stability(diagnostics)
    atomic_csv(stability, task_root / "feature_stability_all_models.csv")
    for model in config.models:
        subset = stability[stability["model"].eq(model)]
        atomic_csv(subset, task_root / f"feature_stability_{model}.csv")
        atomic_csv(subset.groupby("target_class", group_keys=False).head(100), task_root / f"top_100_features_{model}.csv")
    atomic_csv(bundle["sample_master"], task_root / "complete_case_sample_master.csv")
    atomic_csv(bundle["excluded"], task_root / "excluded_species_profiles.csv")
    atomic_csv(bundle["manifest"], task_root / "complete_case_split_manifest.csv")
    atomic_csv(bundle["fold_counts"], task_root / "complete_case_fold_counts.csv")
    for model in config.models:
        train_full_source_model(config, config_fingerprint, task_index, bundle, feature_names, model, task_root)
    if config.make_plots:
        make_plots(repetition, task_root, task)
    atomic_json(task_root / "RUN_COMPLETE.json", {
        "completed_utc": now_iso(),
        "schema_version": SCHEMA_VERSION,
        "code_version": CODE_VERSION,
        "task_folder": task.folder,
        "task": task.report_name,
        "class_names": list(task.class_order),
        "completed_folds": int(len(fold_metrics)),
        "models": list(config.models),
        "repetitions": config.expected_repeats,
        "folds": config.expected_folds,
    })


def aggregate_root(config: RunConfig, bundle: dict[str, Any], root: Path) -> None:
    aggregate = root / "aggregate"
    aggregate.mkdir(parents=True, exist_ok=True)
    summary_frames, repetition_frames, paired_frames, fold_frames = [], [], [], []
    completed_tasks = []
    for task_folder in config.tasks:
        task_root = root / task_folder
        if not (task_root / "RUN_COMPLETE.json").exists():
            continue
        task = TASK_BY_FOLDER[task_folder]
        completed_tasks.append(task_folder)
        for filename, target in (
            ("repetition_performance_summary.csv", summary_frames),
            ("repetition_pooled_oof_metrics.csv", repetition_frames),
            ("paired_model_difference_summary.csv", paired_frames),
            ("all_outer_fold_metrics.csv", fold_frames),
        ):
            frame = pd.read_csv(task_root / filename)
            for column in ("task_order", "task_folder", "task"):
                if column in frame.columns:
                    frame = frame.drop(columns=column)
            frame.insert(0, "task", task.report_name)
            frame.insert(0, "task_folder", task.folder)
            frame.insert(0, "task_order", list(TASK_ORDER).index(task.folder) + 1)
            target.append(frame)
    if summary_frames:
        atomic_csv(pd.concat(summary_frames, ignore_index=True), aggregate / "repetition_performance_summary_all_tasks.csv")
        atomic_csv(pd.concat(repetition_frames, ignore_index=True), aggregate / "repetition_pooled_oof_metrics_all_tasks.csv")
        atomic_csv(pd.concat(paired_frames, ignore_index=True), aggregate / "paired_model_difference_summary_all_tasks.csv")
        atomic_csv(pd.concat(fold_frames, ignore_index=True), aggregate / "all_outer_fold_metrics_all_tasks.csv")
    atomic_csv(bundle["task_audit"], aggregate / "task_complete_case_audit.csv")
    atomic_json(root / "progress.json", {
        "updated_utc": now_iso(),
        "requested_tasks": list(config.tasks),
        "completed_tasks": completed_tasks,
        "completed_task_count": len(completed_tasks),
        "requested_task_count": len(config.tasks),
        "completed_fold_markers": len(list(root.glob("*/folds/*/repeat_*/fold_*/FOLD_COMPLETE.json"))),
        "expected_fold_markers": len(config.tasks) * len(config.models) * config.expected_repeats * config.expected_folds,
    })


def print_preflight(config: RunConfig, bundle: dict[str, Any]) -> None:
    print("=" * 118)
    print("REPEATED SPECIES BENCHMARKS — ALL FIVE TASKS PREFLIGHT")
    print("=" * 118)
    print(f"Species file: {config.species_file}")
    print(f"Shared split directory: {config.split_dir}")
    print(f"Global all-zero canonical profiles: {len(bundle['global_zero_sample_ids'])}")
    print(f"Global zero sample IDs: {', '.join(bundle['global_zero_sample_ids'])}")
    print(f"Species features: {len(bundle['feature_names'])}")
    print("Task order:")
    for index, row in bundle["task_audit"].iterrows():
        print(
            f"  {index + 1}. {row['task']}: {int(row['complete_case_samples'])} / {int(row['source_samples'])} samples, "
            f"{int(row['participants'])} participants, excluded={int(row['excluded_zero_profiles'])}"
        )
    print(f"Models: {', '.join(config.models)}")
    print(f"Design per task/model: {config.expected_repeats} x {config.expected_folds}")
    print(f"Total outer fits requested: {len(config.tasks) * len(config.models) * config.expected_repeats * config.expected_folds}")
    print("Weighting: balanced sample weights; XGBoost binary scale_pos_weight=1.0")
    print("REPEATED SPECIES BENCHMARK ALL-TASK PREFLIGHT: PASSED")
    print("=" * 118)


def run(config: RunConfig, args: argparse.Namespace) -> Path:
    set_thread_environment(config.n_jobs)
    lock = ensure_compatible_output(config, bool(args.overwrite_incompatible_output))
    config_fingerprint = str(lock["scientific_config_sha256"])
    bundle = load_bundle(config)
    root = Path(config.output_dir)
    atomic_csv(bundle["task_audit"], root / "task_complete_case_audit.csv")
    atomic_json(root / "input_and_design_summary.json", {
        "created_utc": now_iso(),
        "schema_version": SCHEMA_VERSION,
        "code_version": CODE_VERSION,
        "species_diagnostics": bundle["species_diagnostics"],
        "global_zero_sample_ids": bundle["global_zero_sample_ids"],
        "tasks": list(config.tasks),
        "models": list(config.models),
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    })
    print_preflight(config, bundle)
    if args.validate_only:
        return root
    if args.aggregate_only:
        aggregate_root(config, bundle, root)
        return root
    for task_folder in config.tasks:
        task_index = next(index for index, task in enumerate(ALL_TASKS, start=1) if task.folder == task_folder)
        task_bundle = bundle["task_bundles"][task_folder]
        task: TaskSpec = task_bundle["task"]
        print("#" * 118)
        print(f"TASK {task_index}/5: {task.report_name} ({task.folder})")
        print("#" * 118, flush=True)
        completed = []
        for model_name in config.models:
            print("-" * 118)
            print(f"MODEL: {MODEL_LABELS[model_name]}")
            print("-" * 118, flush=True)
            for repeat in range(1, config.expected_repeats + 1):
                for fold in range(1, config.expected_folds + 1):
                    completed.append(run_fold(
                        config,
                        config_fingerprint,
                        root,
                        task_index,
                        task_bundle,
                        bundle["feature_names"],
                        model_name,
                        repeat,
                        fold,
                    ))
            aggregate_root(config, bundle, root)
        finalize_task(
            config, config_fingerprint, task_index, task_bundle, bundle["feature_names"], completed, root
        )
        aggregate_root(config, bundle, root)
        print(f"[complete] {task.report_name}: {root / task.folder / 'repetition_performance_summary.csv'}", flush=True)
    aggregate_root(config, bundle, root)
    atomic_json(root / "RUN_COMPLETE.json", {
        "completed_utc": now_iso(),
        "schema_version": SCHEMA_VERSION,
        "code_version": CODE_VERSION,
        "tasks": list(config.tasks),
        "models": list(config.models),
        "completed_folds": len(config.tasks) * len(config.models) * config.expected_repeats * config.expected_folds,
    })
    print("=" * 118)
    print("ALL FIVE SPECIES BENCHMARK TASKS COMPLETE")
    print(f"Output: {root}")
    print(f"Combined summary: {root / 'aggregate' / 'repetition_performance_summary_all_tasks.csv'}")
    print("=" * 118)
    return root


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = config_from_args(args)
    run(config, args)


if __name__ == "__main__":
    main()
