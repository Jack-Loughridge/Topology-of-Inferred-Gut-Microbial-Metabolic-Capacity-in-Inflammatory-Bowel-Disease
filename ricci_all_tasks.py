#!/usr/bin/env python3
from __future__ import annotations

"""Repeated participant-grouped Ricci [B|K0] classifiers for all five IBD tasks.

The runner consumes the exact locked task manifests written by the all-task
H0+Ricci preflight. It never regenerates outer folds. By default it runs the
manuscript model C=0.02 for 20 repetitions x 5 folds on each task. Additional
C values can be supplied for a sensitivity path.
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn import __version__ as sklearn_version
from sklearn.exceptions import ConvergenceWarning
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

SCHEMA_VERSION = 1
CODE_VERSION = "1.0.0"
EPS = 1e-15


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
TASK_ORDER = tuple(task.folder for task in ALL_TASKS)
TASK_BY_FOLDER = {task.folder: task for task in ALL_TASKS}


@dataclass(frozen=True)
class RunConfig:
    feature_dir: str
    split_dir: str
    output_dir: str
    tasks: tuple[str, ...]
    c_values: tuple[float, ...]
    expected_repeats: int = 20
    expected_folds: int = 5
    max_iter: int = 10000
    tol: float = 1e-4
    n_jobs: int = 1
    base_model_seed: int = 20260721
    coef_eps: float = 1e-12
    make_plots: bool = True
    fit_full_source_models: bool = True
    top_n: int = 100


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def atomic_json(path: Path, payload: Any) -> None:
    atomic_text(path, json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n")


def atomic_csv(frame: pd.DataFrame, path: Path, index: bool = False, compression: Optional[str] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    frame.to_csv(tmp, index=index, compression=compression)
    os.replace(tmp, path)


def atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
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


def canonical_sha256(payload: Any) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=json_default)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalise_sample_id(value: object) -> str:
    text = str(value).strip()
    text = Path(text).name
    text = re.sub(r"\.npy$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"_H0$", "", text, flags=re.IGNORECASE)
    return text


def normalise_condition(value: object) -> str:
    text = str(value).strip()
    compact = re.sub(r"[\s_-]+", "", text.lower())
    aliases = {
        "nonibd": "nonIBD",
        "nonibdcontrol": "nonIBD",
        "healthy": "nonIBD",
        "control": "nonIBD",
        "uc": "UC",
        "ulcerativecolitis": "UC",
        "cd": "CD",
        "crohn'sdisease": "CD",
        "crohnsdisease": "CD",
        "ibd": "IBD",
    }
    return aliases.get(compact, text)


def task_label_from_condition(condition: str, task: TaskSpec) -> Optional[str]:
    cond = normalise_condition(condition)
    if task.folder == "IBD_vs_nonIBD":
        if cond == "nonIBD":
            return "nonIBD"
        if cond in {"UC", "CD", "IBD"}:
            return "IBD"
        return None
    if cond in task.class_order:
        return cond
    return None


def c_tag(value: float) -> str:
    return "C_" + f"{float(value):.12g}".replace("-", "m").replace(".", "p")


def parse_csv_tuple(value: str) -> tuple[str, ...]:
    items = tuple(item.strip() for item in value.split(",") if item.strip())
    if not items:
        raise argparse.ArgumentTypeError("Expected a non-empty comma-separated list.")
    return items


def parse_float_tuple(value: str) -> tuple[float, ...]:
    try:
        items = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if not items or any(item <= 0 for item in items) or len(set(items)) != len(items):
        raise argparse.ArgumentTypeError("C values must be unique positive numbers.")
    return items


def build_parser() -> argparse.ArgumentParser:
    base = Path.home() / "Real_Data"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", type=Path, default=base / "Ricci_Classifier_Faithful_Eps0001_n250_v3")
    parser.add_argument("--split-dir", type=Path, default=base / "H0_Ricci_JointSparse_RepeatedCV_AllTasks" / "splits")
    parser.add_argument("--output-dir", type=Path, default=base / "Ricci_RepeatedCV_AllTasks")
    parser.add_argument("--tasks", type=parse_csv_tuple, default=TASK_ORDER)
    parser.add_argument("--c-values", type=parse_float_tuple, default=(0.02,))
    parser.add_argument("--expected-repeats", type=int, default=20)
    parser.add_argument("--expected-folds", type=int, default=5)
    parser.add_argument("--max-iter", type=int, default=10000)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--base-model-seed", type=int, default=20260721)
    parser.add_argument("--coef-eps", type=float, default=1e-12)
    parser.add_argument("--top-n", type=int, default=100)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--no-full-source-models", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--overwrite-incompatible-output", action="store_true")
    return parser


def config_from_args(args: argparse.Namespace) -> RunConfig:
    tasks = tuple(args.tasks)
    invalid = sorted(set(tasks) - set(TASK_ORDER))
    if invalid:
        raise ValueError(f"Unknown tasks: {invalid}")
    ordered = tuple(task for task in TASK_ORDER if task in set(tasks))
    if tasks != ordered or len(set(tasks)) != len(tasks):
        raise ValueError(f"Tasks must be unique and preserve this order: {TASK_ORDER}")
    if args.expected_repeats < 1 or args.expected_folds < 2:
        raise ValueError("Invalid repeated-CV dimensions.")
    return RunConfig(
        feature_dir=str(args.feature_dir.expanduser().resolve()),
        split_dir=str(args.split_dir.expanduser().resolve()),
        output_dir=str(args.output_dir.expanduser().resolve()),
        tasks=tasks,
        c_values=tuple(float(v) for v in args.c_values),
        expected_repeats=int(args.expected_repeats),
        expected_folds=int(args.expected_folds),
        max_iter=int(args.max_iter),
        tol=float(args.tol),
        n_jobs=max(1, int(args.n_jobs)),
        base_model_seed=int(args.base_model_seed),
        coef_eps=float(args.coef_eps),
        make_plots=not bool(args.no_plots),
        fit_full_source_models=not bool(args.no_full_source_models),
        top_n=int(args.top_n),
    )


def input_paths(config: RunConfig) -> dict[str, Path]:
    feature_dir = Path(config.feature_dir)
    paths = {
        "feature_matrix": feature_dir / "feature_matrix_B_K0.npz",
        "matched_metadata": feature_dir / "matched_metadata.csv",
        "edge_metadata": feature_dir / "edge_metadata.csv",
    }
    for task in ALL_TASKS:
        paths[f"manifest_{task.folder}"] = Path(config.split_dir) / f"{task.folder}_split_manifest.csv"
    return paths


def scientific_payload(config: RunConfig) -> dict[str, Any]:
    payload = asdict(config)
    for key in ("output_dir", "n_jobs", "make_plots", "fit_full_source_models"):
        payload.pop(key, None)
    payload.update({
        "schema_version": SCHEMA_VERSION,
        "code_version": CODE_VERSION,
        "task_order": list(TASK_ORDER),
        "feature_definition": "[B|K0] with active-edge presence B and active-edge Ricci curvature K0; absent K0=0",
        "preprocessing": "StandardScaler fit on each outer-training fold only",
        "classifier": "L1 LogisticRegression(solver=saga,class_weight=balanced)",
    })
    return payload


def ensure_inputs(config: RunConfig) -> None:
    missing = [str(path) for path in input_paths(config).values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required inputs:\n" + "\n".join(missing))


def current_lock(config: RunConfig) -> dict[str, Any]:
    paths = input_paths(config)
    scientific = scientific_payload(config)
    return {
        "created_utc": now_iso(),
        "scientific_config": scientific,
        "scientific_config_sha256": canonical_sha256(scientific),
        "input_sha256": {name: sha256_file(path) for name, path in paths.items()},
        "code_sha256": sha256_file(Path(__file__).resolve()),
    }


def ensure_compatible_output(config: RunConfig, overwrite: bool) -> dict[str, Any]:
    root = Path(config.output_dir)
    lock_path = root / "run_config.json"
    current = current_lock(config)
    if not root.exists() or not any(root.iterdir()):
        root.mkdir(parents=True, exist_ok=True)
        atomic_json(lock_path, current)
        return current
    if not lock_path.exists():
        if overwrite:
            import shutil
            shutil.rmtree(root); root.mkdir(parents=True); atomic_json(lock_path, current); return current
        raise RuntimeError(f"Output directory {root} is non-empty but has no run_config.json.")
    existing = json.loads(lock_path.read_text())
    keys = ("scientific_config_sha256", "input_sha256", "code_sha256")
    if not all(existing.get(key) == current.get(key) for key in keys):
        if overwrite:
            import shutil
            shutil.rmtree(root); root.mkdir(parents=True); atomic_json(lock_path, current); return current
        raise RuntimeError("Existing output is incompatible with the current code, inputs or scientific settings.")
    print(f"[resume] Compatible run configuration found in {root}", flush=True)
    return existing


def load_manifest(path: Path, task: TaskSpec, config: RunConfig) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    required = {"task_folder", "repeat", "fold", "split_seed", "role", "sample_id", "participant_id", "label"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    frame = frame.copy()
    frame["task_folder"] = frame["task_folder"].astype(str).str.strip()
    frame = frame[frame["task_folder"].eq(task.folder)].copy()
    if frame.empty:
        raise ValueError(f"No rows for {task.folder} in {path}")
    for column in ("repeat", "fold", "split_seed"):
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(int)
    frame["role"] = frame["role"].astype(str).str.lower().str.strip()
    frame["sample_id"] = frame["sample_id"].map(normalise_sample_id)
    frame["participant_id"] = frame["participant_id"].astype(str).str.strip()
    frame["label"] = frame["label"].map(normalise_condition)
    if set(frame["role"]) != {"train", "test"}:
        raise ValueError(f"{task.folder}: roles must be train/test")
    if sorted(frame["repeat"].unique()) != list(range(1, config.expected_repeats + 1)):
        raise ValueError(f"{task.folder}: repeats do not equal 1..{config.expected_repeats}")
    if sorted(frame["fold"].unique()) != list(range(1, config.expected_folds + 1)):
        raise ValueError(f"{task.folder}: folds do not equal 1..{config.expected_folds}")
    master = frame[["sample_id", "participant_id", "label"]].drop_duplicates()
    if master["sample_id"].duplicated().any():
        raise ValueError(f"{task.folder}: conflicting sample metadata")
    if master.groupby("participant_id")["label"].nunique().gt(1).any():
        raise ValueError(f"{task.folder}: participant has conflicting labels")
    expected_classes = set(task.class_order)
    if set(master["label"]) != expected_classes:
        raise ValueError(f"{task.folder}: classes {sorted(set(master['label']))} != {task.class_order}")
    expected_ids = set(master["sample_id"])
    for repeat in range(1, config.expected_repeats + 1):
        rep = frame[frame["repeat"].eq(repeat)]
        if rep["split_seed"].nunique() != 1:
            raise ValueError(f"{task.folder} repeat {repeat}: multiple split seeds")
        tested: set[str] = set()
        for fold in range(1, config.expected_folds + 1):
            rows = rep[rep["fold"].eq(fold)]
            train = rows[rows["role"].eq("train")]
            test = rows[rows["role"].eq("test")]
            if train.empty or test.empty:
                raise ValueError(f"{task.folder} repeat {repeat} fold {fold}: missing train/test")
            if set(train["participant_id"]) & set(test["participant_id"]):
                raise ValueError(f"{task.folder} repeat {repeat} fold {fold}: participant leakage")
            if set(train["label"]) != expected_classes or set(test["label"]) != expected_classes:
                raise ValueError(f"{task.folder} repeat {repeat} fold {fold}: class absent")
            ids = set(test["sample_id"])
            if tested & ids:
                raise ValueError(f"{task.folder} repeat {repeat}: repeated test sample")
            tested |= ids
        if tested != expected_ids:
            raise ValueError(f"{task.folder} repeat {repeat}: test folds do not cover all samples")
    return frame.reset_index(drop=True)


def load_bundle(config: RunConfig) -> dict[str, Any]:
    ensure_inputs(config)
    feature_dir = Path(config.feature_dir)
    sparse = load_npz(feature_dir / "feature_matrix_B_K0.npz").tocsr()
    metadata = pd.read_csv(feature_dir / "matched_metadata.csv", low_memory=False)
    edge = pd.read_csv(feature_dir / "edge_metadata.csv", low_memory=False)
    for column in ("sample_id", "participant_id", "cond"):
        if column not in metadata:
            raise ValueError(f"matched_metadata.csv missing {column}")
    if "edge" not in edge:
        raise ValueError("edge_metadata.csv missing edge")
    if "process" not in edge:
        edge["process"] = "Unannotated"
    metadata = metadata.copy()
    metadata["sample_id"] = metadata["sample_id"].map(normalise_sample_id)
    metadata["participant_id"] = metadata["participant_id"].astype(str).str.strip()
    metadata["condition"] = metadata["cond"].map(normalise_condition)
    if metadata["sample_id"].duplicated().any():
        raise ValueError("matched_metadata.csv sample_id must be unique")
    if sparse.shape[0] != len(metadata):
        raise ValueError("Feature rows do not match metadata rows")
    n_edges = len(edge)
    if sparse.shape[1] != 2 * n_edges:
        raise ValueError("Feature columns must equal 2 * number of edges")
    edge = edge.copy()
    edge["edge"] = edge["edge"].astype(str)
    edge["process"] = edge["process"].fillna("Unannotated").astype(str)
    feature_meta = pd.DataFrame({
        "feature_index": np.arange(2 * n_edges, dtype=int),
        "feature": [f"B__{x}" for x in edge["edge"]] + [f"K0__{x}" for x in edge["edge"]],
        "feature_type": ["B"] * n_edges + ["K0"] * n_edges,
        "edge_index": list(range(n_edges)) + list(range(n_edges)),
        "edge": edge["edge"].tolist() + edge["edge"].tolist(),
        "process": edge["process"].tolist() + edge["process"].tolist(),
    })
    print("[load] Converting faithful Ricci matrix to dense float32...", flush=True)
    dense = sparse.toarray().astype(np.float32, copy=False)
    if not np.isfinite(dense).all():
        raise ValueError("Ricci feature matrix contains nonfinite values")
    metadata_index = {sid: idx for idx, sid in enumerate(metadata["sample_id"])}
    task_bundles: dict[str, Any] = {}
    audit_rows = []
    for folder in config.tasks:
        task = TASK_BY_FOLDER[folder]
        manifest = load_manifest(Path(config.split_dir) / f"{folder}_split_manifest.csv", task, config)
        master = manifest[["sample_id", "participant_id", "label"]].drop_duplicates().sort_values("sample_id", kind="mergesort").reset_index(drop=True)
        missing = sorted(set(master["sample_id"]) - set(metadata_index))
        if missing:
            raise ValueError(f"{folder}: feature metadata missing samples {missing[:10]}")
        rows = np.array([metadata_index[sid] for sid in master["sample_id"]], dtype=int)
        derived = [task_label_from_condition(metadata.iloc[idx]["condition"], task) for idx in rows]
        if any(label is None for label in derived):
            raise ValueError(f"{folder}: unsupported condition in matched metadata")
        mismatches = master.loc[np.asarray(derived, dtype=object) != master["label"].to_numpy(dtype=object), "sample_id"].tolist()
        if mismatches:
            raise ValueError(f"{folder}: manifest labels disagree with matched metadata: {mismatches[:10]}")
        task_bundles[folder] = {
            "task": task,
            "manifest": manifest,
            "master": master,
            "X": dense[rows],
            "sample_to_index": {sid: idx for idx, sid in enumerate(master["sample_id"])},
        }
        audit_rows.append({
            "task_order": TASK_ORDER.index(folder) + 1,
            "task_folder": folder,
            "task": task.report_name,
            "samples": len(master),
            "participants": master["participant_id"].nunique(),
            "classes": "|".join(task.class_order),
            "sample_class_counts": json.dumps(master["label"].value_counts().to_dict(), sort_keys=True),
            "participant_class_counts": json.dumps(master.groupby("label")["participant_id"].nunique().to_dict(), sort_keys=True),
        })
    return {"feature_meta": feature_meta, "edge_meta": edge, "task_bundles": task_bundles, "audit": pd.DataFrame(audit_rows), "dense_shape": dense.shape}


def indices_for_fold(bundle: dict[str, Any], repeat: int, fold: int) -> tuple[np.ndarray, np.ndarray, int]:
    rows = bundle["manifest"]
    rows = rows[rows["repeat"].eq(repeat) & rows["fold"].eq(fold)]
    train_ids = rows[rows["role"].eq("train")]["sample_id"].drop_duplicates().tolist()
    test_ids = rows[rows["role"].eq("test")]["sample_id"].drop_duplicates().tolist()
    mapping = bundle["sample_to_index"]
    return (
        np.array([mapping[sid] for sid in train_ids], dtype=int),
        np.array([mapping[sid] for sid in test_ids], dtype=int),
        int(rows["split_seed"].iloc[0]),
    )


def safe_auc(y_true: np.ndarray, proba: np.ndarray, n_classes: int) -> float:
    try:
        if n_classes == 2:
            return float(roc_auc_score(y_true, proba[:, 1]))
        return float(roc_auc_score(y_true, proba, labels=list(range(n_classes)), multi_class="ovr", average="macro"))
    except Exception:
        return float("nan")


def metric_dict(y_true: np.ndarray, pred: np.ndarray, proba: np.ndarray, class_names: tuple[str, ...]) -> dict[str, Any]:
    n_classes = len(class_names)
    proba = np.asarray(proba, dtype=float)
    proba = np.clip(proba, EPS, 1.0)
    proba = proba / proba.sum(axis=1, keepdims=True)
    out: dict[str, Any] = {
        "n_observations": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "roc_auc": safe_auc(y_true, proba, n_classes),
        "log_loss": float(log_loss(y_true, np.clip(proba, EPS, 1.0 - EPS), labels=list(range(n_classes)))),
        "confusion_matrix": json.dumps(confusion_matrix(y_true, pred, labels=list(range(n_classes))).tolist()),
    }
    recalls = recall_score(y_true, pred, labels=list(range(n_classes)), average=None, zero_division=0)
    for idx, name in enumerate(class_names):
        out[f"recall_{name}"] = float(recalls[idx])
        truth = (y_true == idx).astype(int)
        out[f"ovr_auc_{name}"] = float(roc_auc_score(truth, proba[:, idx])) if len(np.unique(truth)) == 2 else float("nan")
    return out


def participant_predictions(sample: pd.DataFrame, class_names: tuple[str, ...]) -> pd.DataFrame:
    prob_cols = [f"proba_{name}" for name in class_names]
    if sample.groupby("participant_id")["y_true"].nunique().gt(1).any():
        raise RuntimeError("Participant has conflicting labels in OOF predictions")
    grouped = sample.groupby("participant_id", as_index=False).agg(
        sample_count=("sample_id", "size"), y_true=("y_true", "first"), **{col: (col, "mean") for col in prob_cols}
    )
    probs = grouped[prob_cols].to_numpy(dtype=float)
    grouped["pred"] = probs.argmax(axis=1)
    grouped["true_label"] = grouped["y_true"].map(dict(enumerate(class_names)))
    grouped["pred_label"] = grouped["pred"].map(dict(enumerate(class_names)))
    return grouped


def model_seed(config: RunConfig, task_index: int, c_index: int, repeat: int, fold: int) -> int:
    return int(config.base_model_seed + task_index * 1_000_003 + c_index * 100_003 + repeat * 1_009 + fold * 97)


def fold_dir(root: Path, task: TaskSpec, C: float, repeat: int, fold: int) -> Path:
    return root / task.folder / c_tag(C) / "folds" / f"repeat_{repeat:02d}" / f"fold_{fold}"


def interpretation_coef(coef: np.ndarray, class_names: tuple[str, ...]) -> tuple[np.ndarray, tuple[str, ...]]:
    if coef.shape[0] == 1:
        return coef.astype(np.float64), (class_names[1],)
    return coef.astype(np.float64), class_names


def fit_or_load_fold(config: RunConfig, root: Path, bundle: dict[str, Any], feature_meta: pd.DataFrame, task_index: int, c_index: int, C: float, repeat: int, fold: int) -> dict[str, Any]:
    task: TaskSpec = bundle["task"]
    directory = fold_dir(root, task, C, repeat, fold)
    marker = directory / "FOLD_COMPLETE.json"
    metrics_path = directory / "fold_metrics.json"
    sample_path = directory / "sample_predictions.csv"
    participant_path = directory / "participant_predictions.csv"
    artifact_path = directory / "model_artifact.npz"
    required = [marker, metrics_path, sample_path, participant_path, artifact_path]
    if all(path.exists() for path in required):
        print(f"[resume] {task.folder} C={C:g} r{repeat:02d} f{fold}", flush=True)
        return json.loads(metrics_path.read_text())
    if directory.exists():
        import shutil
        shutil.rmtree(directory)
    directory.mkdir(parents=True)
    train_idx, test_idx, split_seed = indices_for_fold(bundle, repeat, fold)
    master = bundle["master"]
    X = bundle["X"]
    class_names = task.class_order
    class_to_index = {name: idx for idx, name in enumerate(class_names)}
    y = master["label"].map(class_to_index).to_numpy(dtype=int)
    scaler = StandardScaler(with_mean=True, with_std=True)
    X_train = scaler.fit_transform(X[train_idx])
    X_test = scaler.transform(X[test_idx])
    seed = model_seed(config, task_index, c_index, repeat, fold)
    model = LogisticRegression(
        penalty="l1", solver="saga", class_weight="balanced", C=float(C), max_iter=config.max_iter,
        tol=config.tol, random_state=seed, n_jobs=config.n_jobs,
    )
    started = time.time()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(X_train, y[train_idx])
    convergence = [str(w.message) for w in caught if issubclass(w.category, ConvergenceWarning)]
    proba = model.predict_proba(X_test)
    pred = proba.argmax(axis=1)
    test_meta = master.iloc[test_idx].reset_index(drop=True)
    sample = test_meta[["sample_id", "participant_id", "label"]].copy()
    sample["repeat"] = repeat; sample["fold"] = fold; sample["split_seed"] = split_seed
    sample["y_true"] = y[test_idx]; sample["pred"] = pred
    sample["true_label"] = sample["label"]; sample["pred_label"] = [class_names[i] for i in pred]
    sample = sample.drop(columns="label")
    for idx, name in enumerate(class_names):
        sample[f"proba_{name}"] = proba[:, idx]
    participant = participant_predictions(sample, class_names)
    participant["repeat"] = repeat
    participant["fold"] = fold
    participant["split_seed"] = split_seed
    sample_metrics = metric_dict(y[test_idx], pred, proba, class_names)
    part_proba = participant[[f"proba_{name}" for name in class_names]].to_numpy(dtype=float)
    participant_metrics = metric_dict(participant["y_true"].to_numpy(dtype=int), participant["pred"].to_numpy(dtype=int), part_proba, class_names)
    coef, interpretation_classes = interpretation_coef(model.coef_, class_names)
    contributions = X_test.astype(np.float64)[:, None, :] * coef[None, :, :]
    participant_ids = test_meta["participant_id"].astype(str).to_numpy()
    unique_participants = sorted(set(participant_ids))
    p_abs, p_signed = [], []
    for pid in unique_participants:
        values = contributions[participant_ids == pid]
        p_abs.append(np.mean(np.abs(values), axis=0))
        p_signed.append(np.mean(values, axis=0))
    heldout_abs = np.mean(np.stack(p_abs), axis=0)
    heldout_signed = np.mean(np.stack(p_signed), axis=0)
    metrics = {
        "task_order": task_index, "task_folder": task.folder, "task": task.report_name,
        "C": float(C), "repeat": repeat, "fold": fold, "split_seed": split_seed, "model_seed": seed,
        "train_samples": len(train_idx), "test_samples": len(test_idx),
        "train_participants": master.iloc[train_idx]["participant_id"].nunique(),
        "test_participants": master.iloc[test_idx]["participant_id"].nunique(),
        "selected_coefficients": int(np.sum(np.abs(coef) > config.coef_eps)),
        "convergence_warning": bool(convergence), "warning_messages": convergence,
        "n_iter_max": int(np.max(model.n_iter_)), "at_iteration_limit": bool(np.max(model.n_iter_) >= config.max_iter),
        "elapsed_seconds": float(time.time() - started),
    }
    metrics.update({f"sample_{k}": v for k, v in sample_metrics.items()})
    metrics.update({f"participant_{k}": v for k, v in participant_metrics.items()})
    atomic_csv(sample, sample_path)
    atomic_csv(participant, participant_path)
    atomic_npz(
        artifact_path,
        scaler_mean=np.asarray(scaler.mean_, dtype=np.float64), scaler_scale=np.asarray(scaler.scale_, dtype=np.float64),
        scaler_var=np.asarray(scaler.var_, dtype=np.float64), coef=coef, intercept=np.asarray(model.intercept_, dtype=np.float64),
        model_classes=np.asarray(model.classes_, dtype=int), interpretation_classes=np.asarray(interpretation_classes, dtype=str),
        heldout_abs_participant_mean=heldout_abs, heldout_signed_participant_mean=heldout_signed,
        feature_names=feature_meta["feature"].to_numpy(dtype=str), class_names=np.asarray(class_names, dtype=str),
        train_sample_ids=master.iloc[train_idx]["sample_id"].to_numpy(dtype=str), test_sample_ids=master.iloc[test_idx]["sample_id"].to_numpy(dtype=str),
    )
    atomic_json(metrics_path, metrics)
    atomic_json(marker, {"completed_utc": now_iso(), "metrics_sha256": sha256_file(metrics_path), "artifact_sha256": sha256_file(artifact_path)})
    print(f"[{task.folder} C={C:g} r{repeat:02d} f{fold}] bal={sample_metrics['balanced_accuracy']:.4f} auc={sample_metrics['roc_auc']:.4f} features={metrics['selected_coefficients']} runtime={metrics['elapsed_seconds']:.1f}s", flush=True)
    return metrics


def repetition_metrics(predictions: pd.DataFrame, class_names: tuple[str, ...], repeats: int, level: str) -> pd.DataFrame:
    rows = []
    prob_cols = [f"proba_{name}" for name in class_names]
    for repeat in range(1, repeats + 1):
        frame = predictions[predictions["repeat"].eq(repeat)]
        if frame.empty:
            raise RuntimeError(f"Missing {level} predictions for repeat {repeat}")
        y = frame["y_true"].to_numpy(dtype=int); pred = frame["pred"].to_numpy(dtype=int); proba = frame[prob_cols].to_numpy(dtype=float)
        result = metric_dict(y, pred, proba, class_names)
        result.update({"repeat": repeat, "evaluation_level": level})
        rows.append(result)
    return pd.DataFrame(rows)


def summary_from_repetitions(frame: pd.DataFrame) -> pd.DataFrame:
    id_cols = {"repeat", "evaluation_level", "confusion_matrix", "n_observations"}
    metrics = [c for c in frame.columns if c not in id_cols and pd.api.types.is_numeric_dtype(frame[c])]
    rows = []
    for level, group in frame.groupby("evaluation_level", sort=False):
        row: dict[str, Any] = {"evaluation_level": level, "n_repetitions": len(group)}
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy(dtype=float)
            row[f"{metric}_mean"] = float(np.mean(values)) if len(values) else float("nan")
            row[f"{metric}_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            row[f"{metric}_q025"] = float(np.quantile(values, 0.025)) if len(values) else float("nan")
            row[f"{metric}_q975"] = float(np.quantile(values, 0.975)) if len(values) else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def stats(values: np.ndarray, prefix: str) -> dict[str, Any]:
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if not len(finite):
        return {f"{prefix}_mean": np.nan, f"{prefix}_sd": np.nan, f"{prefix}_median": np.nan, f"{prefix}_q025": np.nan, f"{prefix}_q975": np.nan}
    return {
        f"{prefix}_mean": float(np.mean(finite)), f"{prefix}_sd": float(np.std(finite, ddof=1)) if len(finite)>1 else 0.0,
        f"{prefix}_median": float(np.median(finite)), f"{prefix}_q025": float(np.quantile(finite,0.025)), f"{prefix}_q975": float(np.quantile(finite,0.975)),
    }


def coefficient_stability(root: Path, task: TaskSpec, C: float, config: RunConfig, feature_meta: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    artifacts = []
    for repeat in range(1, config.expected_repeats + 1):
        for fold in range(1, config.expected_folds + 1):
            path = fold_dir(root, task, C, repeat, fold) / "model_artifact.npz"
            with np.load(path, allow_pickle=False) as data:
                artifacts.append({
                    "coef": data["coef"].astype(float),
                    "classes": tuple(data["interpretation_classes"].astype(str).tolist()),
                    "abs": data["heldout_abs_participant_mean"].astype(float),
                    "signed": data["heldout_signed_participant_mean"].astype(float),
                    "repeat": repeat, "fold": fold,
                })
    classes = artifacts[0]["classes"]
    if any(item["classes"] != classes for item in artifacts):
        raise RuntimeError("Interpretation class order changed across folds")
    feature_rows: list[pd.DataFrame] = []
    process_rows: list[dict[str, Any]] = []
    for class_idx, class_name in enumerate(classes):
        coef = np.stack([item["coef"][class_idx] for item in artifacts])
        abs_contrib = np.stack([item["abs"][class_idx] for item in artifacts])
        signed_contrib = np.stack([item["signed"][class_idx] for item in artifacts])
        selected = np.abs(coef) > config.coef_eps
        sign_nonzero = np.sign(coef)
        positive = np.sum(sign_nonzero > 0, axis=0); negative = np.sum(sign_nonzero < 0, axis=0)
        selected_count = selected.sum(axis=0)
        sign_consistency = np.divide(np.maximum(positive, negative), selected_count, out=np.zeros_like(selected_count, dtype=float), where=selected_count>0)
        frame = feature_meta.copy()
        frame.insert(0, "class", class_name)
        frame["n_fits"] = coef.shape[0]
        frame["selected_count"] = selected_count
        frame["selection_frequency"] = selected.mean(axis=0)
        frame["positive_count"] = positive; frame["negative_count"] = negative
        frame["sign_consistency_given_selected"] = sign_consistency
        frame["coef_mean"] = coef.mean(axis=0); frame["coef_sd"] = coef.std(axis=0, ddof=1)
        frame["coef_median"] = np.median(coef, axis=0); frame["coef_q025"] = np.quantile(coef,0.025,axis=0); frame["coef_q975"] = np.quantile(coef,0.975,axis=0)
        conditional_mean = np.zeros(coef.shape[1]); conditional_sd = np.zeros(coef.shape[1])
        for j in np.flatnonzero(selected_count):
            vals = coef[selected[:,j],j]
            conditional_mean[j] = vals.mean(); conditional_sd[j] = vals.std(ddof=1) if len(vals)>1 else 0.0
        frame["coef_selected_mean"] = conditional_mean; frame["coef_selected_sd"] = conditional_sd
        frame["heldout_abs_contribution_mean"] = abs_contrib.mean(axis=0); frame["heldout_abs_contribution_sd"] = abs_contrib.std(axis=0,ddof=1)
        frame["heldout_signed_contribution_mean"] = signed_contrib.mean(axis=0); frame["heldout_signed_contribution_sd"] = signed_contrib.std(axis=0,ddof=1)
        feature_rows.append(frame)
        for artifact_index, item in enumerate(artifacts):
            fold_frame = feature_meta[["process","feature_type"]].copy()
            fold_frame["abs_coef"] = np.abs(item["coef"][class_idx])
            fold_frame["signed_coef"] = item["coef"][class_idx]
            fold_frame["heldout_abs"] = item["abs"][class_idx]
            fold_frame["heldout_signed"] = item["signed"][class_idx]
            grouped = fold_frame.groupby(["process","feature_type"], as_index=False).agg(
                abs_coefficient_sum=("abs_coef","sum"), signed_coefficient_sum=("signed_coef","sum"),
                heldout_abs_contribution_sum=("heldout_abs","sum"), heldout_signed_contribution_sum=("heldout_signed","sum"),
                feature_count=("abs_coef","size"), selected_feature_count=("abs_coef",lambda s: int(np.sum(np.asarray(s)>config.coef_eps))),
            )
            for row in grouped.itertuples(index=False):
                process_rows.append({"class":class_name,"repeat":item["repeat"],"fold":item["fold"],**row._asdict()})
    features = pd.concat(feature_rows, ignore_index=True)
    processes = pd.DataFrame(process_rows)
    return features, processes


def aggregate_process(processes: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    metrics=["abs_coefficient_sum","signed_coefficient_sum","heldout_abs_contribution_sum","heldout_signed_contribution_sum","selected_feature_count"]
    for keys, group in processes.groupby(["class","process","feature_type"], sort=False):
        row={"class":keys[0],"process":keys[1],"feature_type":keys[2],"n_fits":len(group),"feature_count":int(group["feature_count"].iloc[0])}
        for metric in metrics: row.update(stats(group[metric].to_numpy(dtype=float),metric))
        rows.append(row)
    return pd.DataFrame(rows)


def consensus_predictions(sample: pd.DataFrame, class_names: tuple[str,...]) -> tuple[pd.DataFrame,pd.DataFrame,dict[str,Any],dict[str,Any]]:
    prob_cols=[f"proba_{name}" for name in class_names]
    base=sample.groupby("sample_id",as_index=False).agg(participant_id=("participant_id","first"),y_true=("y_true","first"),**{f"{c}_mean":(c,"mean") for c in prob_cols})
    matrix=base[[f"{c}_mean" for c in prob_cols]].to_numpy(dtype=float)
    base["pred"]=matrix.argmax(axis=1); base["true_label"]=base["y_true"].map(dict(enumerate(class_names))); base["pred_label"]=base["pred"].map(dict(enumerate(class_names)))
    sample_metrics=metric_dict(base["y_true"].to_numpy(int),base["pred"].to_numpy(int),matrix,class_names)
    tmp=base.rename(columns={f"{c}_mean":c for c in prob_cols})
    part=participant_predictions(tmp,class_names)
    part_matrix=part[prob_cols].to_numpy(dtype=float)
    part_metrics=metric_dict(part["y_true"].to_numpy(int),part["pred"].to_numpy(int),part_matrix,class_names)
    return base,part,sample_metrics,part_metrics


def plot_confusion(cm: np.ndarray, labels: tuple[str,...], path: Path, title: str) -> None:
    fig,ax=plt.subplots(figsize=(5.2,4.4)); im=ax.imshow(cm)
    ax.set_xticks(range(len(labels)),labels=labels,rotation=30,ha="right"); ax.set_yticks(range(len(labels)),labels=labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    for i in range(len(labels)):
        for j in range(len(labels)): ax.text(j,i,str(int(cm[i,j])),ha="center",va="center")
    fig.colorbar(im,ax=ax,fraction=0.046,pad=0.04); fig.tight_layout(); fig.savefig(path,dpi=180); plt.close(fig)


def fit_full_source(config: RunConfig, root: Path, bundle: dict[str,Any], feature_meta: pd.DataFrame, task_index:int,c_index:int,C:float)->None:
    task:TaskSpec=bundle["task"]; directory=root/task.folder/c_tag(C)/"full_source_model"; marker=directory/"FULL_SOURCE_COMPLETE.json"
    if marker.exists(): return
    directory.mkdir(parents=True,exist_ok=True)
    class_to_index={name:i for i,name in enumerate(task.class_order)}; y=bundle["master"]["label"].map(class_to_index).to_numpy(int)
    scaler=StandardScaler(); Xz=scaler.fit_transform(bundle["X"])
    seed=model_seed(config,task_index,c_index,999,1)
    model=LogisticRegression(penalty="l1",solver="saga",class_weight="balanced",C=C,max_iter=config.max_iter,tol=config.tol,random_state=seed,n_jobs=config.n_jobs)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always",ConvergenceWarning); model.fit(Xz,y)
    coef,interpretation_classes=interpretation_coef(model.coef_,task.class_order)
    atomic_npz(directory/"full_source_model.npz",scaler_mean=scaler.mean_,scaler_scale=scaler.scale_,scaler_var=scaler.var_,coef=coef,intercept=model.intercept_,model_classes=model.classes_,interpretation_classes=np.asarray(interpretation_classes,dtype=str),class_names=np.asarray(task.class_order,dtype=str),feature_names=feature_meta["feature"].to_numpy(dtype=str),sample_ids=bundle["master"]["sample_id"].to_numpy(dtype=str))
    atomic_json(marker,{"completed_utc":now_iso(),"C":C,"model_seed":seed,"n_iter_max":int(np.max(model.n_iter_)),"convergence_warning":any(issubclass(w.category,ConvergenceWarning) for w in caught)})


def finalize_task_c(config:RunConfig,root:Path,bundle:dict[str,Any],feature_meta:pd.DataFrame,C:float)->None:
    task:TaskSpec=bundle["task"]; croot=root/task.folder/c_tag(C)
    metric_rows=[]; samples=[]; participants=[]
    for repeat in range(1,config.expected_repeats+1):
        for fold in range(1,config.expected_folds+1):
            directory=fold_dir(root,task,C,repeat,fold)
            metric_rows.append(json.loads((directory/"fold_metrics.json").read_text()))
            samples.append(pd.read_csv(directory/"sample_predictions.csv")); participants.append(pd.read_csv(directory/"participant_predictions.csv"))
    fold_metrics=pd.DataFrame(metric_rows); sample=pd.concat(samples,ignore_index=True); participant=pd.concat(participants,ignore_index=True)
    atomic_csv(fold_metrics,croot/"all_outer_fold_metrics.csv"); atomic_csv(sample,croot/"all_oof_sample_predictions.csv"); atomic_csv(participant,croot/"all_oof_participant_predictions.csv")
    sample_rep=repetition_metrics(sample,task.class_order,config.expected_repeats,"sample"); part_rep=repetition_metrics(participant,task.class_order,config.expected_repeats,"participant")
    rep=pd.concat([sample_rep,part_rep],ignore_index=True); summary=summary_from_repetitions(rep)
    atomic_csv(rep,croot/"repetition_pooled_oof_metrics.csv"); atomic_csv(summary,croot/"repetition_performance_summary.csv")
    features,process_folds=coefficient_stability(root,task,C,config,feature_meta); process_summary=aggregate_process(process_folds)
    atomic_csv(features,croot/"feature_coefficient_stability.csv.gz",compression="gzip")
    top=features.sort_values(["selection_frequency","heldout_abs_contribution_mean","coef_mean"],ascending=[False,False,False]).groupby("class",group_keys=False).head(config.top_n)
    atomic_csv(top,croot/"top_stable_features.csv")
    atomic_csv(process_folds,croot/"process_fold_distributions.csv"); atomic_csv(process_summary,croot/"process_summary.csv")
    consensus_sample,consensus_part,sample_metrics,part_metrics=consensus_predictions(sample,task.class_order)
    atomic_csv(consensus_sample,croot/"consensus_sample_predictions.csv"); atomic_csv(consensus_part,croot/"consensus_participant_predictions.csv")
    atomic_json(croot/"consensus_metrics.json",{"sample":sample_metrics,"participant":part_metrics})
    sample_cm=confusion_matrix(consensus_sample["y_true"],consensus_sample["pred"],labels=list(range(len(task.class_order))))
    part_cm=confusion_matrix(consensus_part["y_true"],consensus_part["pred"],labels=list(range(len(task.class_order))))
    atomic_csv(pd.DataFrame(sample_cm,index=task.class_order,columns=task.class_order),croot/"consensus_confusion_sample.csv",index=True)
    atomic_csv(pd.DataFrame(part_cm,index=task.class_order,columns=task.class_order),croot/"consensus_confusion_participant.csv",index=True)
    if config.make_plots:
        plot_confusion(sample_cm,task.class_order,croot/"consensus_confusion_sample.png",f"{task.report_name}: sample consensus")
        plot_confusion(part_cm,task.class_order,croot/"consensus_confusion_participant.png",f"{task.report_name}: participant consensus")
        for metric in ("balanced_accuracy","macro_f1","roc_auc"):
            values=[sample_rep[metric].dropna().to_numpy(),part_rep[metric].dropna().to_numpy()]
            fig,ax=plt.subplots(figsize=(5.5,4.2)); ax.boxplot(values,labels=["Sample","Participant"],showmeans=True); ax.set_ylabel(metric.replace("_"," ").title()); ax.set_title(f"{task.report_name}, C={C:g}"); fig.tight_layout(); fig.savefig(croot/f"repetition_{metric}.png",dpi=180); plt.close(fig)
    atomic_json(croot/"RUN_COMPLETE.json",{"completed_utc":now_iso(),"task":task.report_name,"C":C,"fold_models":config.expected_repeats*config.expected_folds})


def aggregate_root(config:RunConfig,bundle:dict[str,Any],root:Path)->None:
    aggregate=root/"aggregate"; aggregate.mkdir(parents=True,exist_ok=True)
    summaries=[]; repetitions=[]; folds=[]; completed=[]
    for folder in config.tasks:
        task=TASK_BY_FOLDER[folder]
        for C in config.c_values:
            croot=root/folder/c_tag(C)
            if not (croot/"RUN_COMPLETE.json").exists(): continue
            completed.append({"task_folder":folder,"C":C})
            for filename,target in (("repetition_performance_summary.csv",summaries),("repetition_pooled_oof_metrics.csv",repetitions),("all_outer_fold_metrics.csv",folds)):
                frame=pd.read_csv(croot/filename)
                for column in ("task_order", "task_folder", "task", "C"):
                    if column in frame.columns:
                        frame = frame.drop(columns=column)
                frame.insert(0,"C",C); frame.insert(0,"task",task.report_name); frame.insert(0,"task_folder",folder); frame.insert(0,"task_order",TASK_ORDER.index(folder)+1); target.append(frame)
    if summaries: atomic_csv(pd.concat(summaries,ignore_index=True),aggregate/"repetition_performance_summary_all_tasks.csv")
    if repetitions: atomic_csv(pd.concat(repetitions,ignore_index=True),aggregate/"repetition_pooled_oof_metrics_all_tasks.csv")
    if folds: atomic_csv(pd.concat(folds,ignore_index=True),aggregate/"all_outer_fold_metrics_all_tasks.csv")
    atomic_csv(bundle["audit"],aggregate/"task_audit.csv")
    marker_count=len(list(root.glob("*/C_*/folds/repeat_*/fold_*/FOLD_COMPLETE.json")))
    atomic_json(root/"progress.json",{"updated_utc":now_iso(),"completed_task_C":completed,"completed_fold_markers":marker_count,"expected_fold_markers":len(config.tasks)*len(config.c_values)*config.expected_repeats*config.expected_folds})


def print_preflight(config:RunConfig,bundle:dict[str,Any])->None:
    print("="*118); print("REPEATED RICCI [B|K0] — ALL FIVE TASKS PREFLIGHT"); print("="*118)
    print(f"Feature directory: {config.feature_dir}"); print(f"Shared split directory: {config.split_dir}"); print(f"Full matrix shape: {bundle['dense_shape']}")
    for row in bundle["audit"].itertuples(index=False): print(f"  {row.task_order}. {row.task}: {row.samples} samples, {row.participants} participants, classes={row.classes}")
    print(f"Design per task/C: {config.expected_repeats} x {config.expected_folds}"); print(f"C values: {config.c_values}")
    print(f"Total outer fits requested: {len(config.tasks)*len(config.c_values)*config.expected_repeats*config.expected_folds}")
    print("REPEATED RICCI ALL-TASK PREFLIGHT: PASSED"); print("="*118)


def run(config:RunConfig,args:argparse.Namespace)->Path:
    for key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ.setdefault(key,str(config.n_jobs))
    lock=ensure_compatible_output(config,bool(args.overwrite_incompatible_output)); root=Path(config.output_dir); bundle=load_bundle(config)
    atomic_csv(bundle["audit"],root/"task_audit.csv"); atomic_csv(bundle["feature_meta"],root/"feature_metadata_used.csv"); atomic_csv(bundle["edge_meta"],root/"edge_metadata_used.csv")
    atomic_json(root/"input_and_design_summary.json",{"created_utc":now_iso(),"schema_version":SCHEMA_VERSION,"code_version":CODE_VERSION,"scientific_config_sha256":lock["scientific_config_sha256"],"software":{"python":sys.version,"platform":platform.platform(),"numpy":np.__version__,"pandas":pd.__version__,"scikit_learn":sklearn_version}})
    print_preflight(config,bundle)
    if args.validate_only: return root
    if args.aggregate_only: aggregate_root(config,bundle,root); return root
    for task_index,folder in enumerate(config.tasks,start=1):
        task_bundle=bundle["task_bundles"][folder]; task=task_bundle["task"]
        print("#"*118); print(f"TASK {TASK_ORDER.index(folder)+1}/5: {task.report_name} ({folder})"); print("#"*118)
        for c_index,C in enumerate(config.c_values,start=1):
            print("-"*118); print(f"C={C:g}"); print("-"*118)
            for repeat in range(1,config.expected_repeats+1):
                for fold in range(1,config.expected_folds+1): fit_or_load_fold(config,root,task_bundle,bundle["feature_meta"],TASK_ORDER.index(folder)+1,c_index,C,repeat,fold)
            if config.fit_full_source_models: fit_full_source(config,root,task_bundle,bundle["feature_meta"],TASK_ORDER.index(folder)+1,c_index,C)
            finalize_task_c(config,root,task_bundle,bundle["feature_meta"],C); aggregate_root(config,bundle,root)
            print(f"[complete] {task.report_name}, C={C:g}",flush=True)
    aggregate_root(config,bundle,root)
    atomic_json(root/"RUN_COMPLETE.json",{"completed_utc":now_iso(),"tasks":list(config.tasks),"c_values":list(config.c_values),"completed_fold_models":len(config.tasks)*len(config.c_values)*config.expected_repeats*config.expected_folds})
    print("="*118); print("ALL FIVE RICCI TASKS COMPLETE"); print(f"Combined summary: {root/'aggregate'/'repetition_performance_summary_all_tasks.csv'}"); print("="*118)
    return root


def main(argv:Optional[list[str]]=None)->None:
    parser=build_parser(); args=parser.parse_args(argv); run(config_from_args(args),args)


if __name__=="__main__": main()
