#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, hstack, save_npz, load_npz
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold


SOURCE_CANDIDATES = ["source", "src", "u", "a", "from", "tail", "substrate", "input", "from_node"]
TARGET_CANDIDATES = ["target", "dst", "v", "b", "to", "head", "product", "output", "to_node"]
EDGE_CANDIDATES = ["edge", "edge_id", "edge_key", "reaction_edge", "directed_edge"]
KAPPA_CANDIDATES = ["kappa", "curvature", "ricci", "ricci_curvature", "k0", "K0", "k_ab", "ollivier_ricci"]
SAMPLE_CANDIDATES = ["sample_id", "sample", "External ID", "external_id", "id"]
PROCESS_CANDIDATES = [
    "process",
    "process_class",
    "process_category",
    "biological_process",
    "category",
    "edge_process",
    "reaction_process",
]

TASKS = {
    "three_way": {
        "display": "non-IBD vs UC vs CD",
        "folder": "three_way_nonIBD_UC_CD",
        "kind": "multiclass",
        "class_names": ["non-IBD", "UC", "CD"],
        "label_map": {"nonibd": 0, "uc": 1, "cd": 2},
        "positive_name": None,
    },
    "ibd_vs_nonibd": {
        "display": "IBD vs non-IBD",
        "folder": "IBD_vs_nonIBD",
        "kind": "binary",
        "class_names": ["non-IBD", "IBD"],
        "label_map": {"nonibd": 0, "uc": 1, "cd": 1},
        "positive_name": "IBD",
    },
    "nonibd_vs_uc": {
        "display": "non-IBD vs UC",
        "folder": "nonIBD_vs_UC",
        "kind": "binary",
        "class_names": ["non-IBD", "UC"],
        "label_map": {"nonibd": 0, "uc": 1},
        "positive_name": "UC",
    },
    "nonibd_vs_cd": {
        "display": "non-IBD vs CD",
        "folder": "nonIBD_vs_CD",
        "kind": "binary",
        "class_names": ["non-IBD", "CD"],
        "label_map": {"nonibd": 0, "cd": 1},
        "positive_name": "CD",
    },
    "cd_vs_uc": {
        "display": "CD vs UC",
        "folder": "CD_vs_UC",
        "kind": "binary",
        "class_names": ["CD", "UC"],
        "label_map": {"cd": 0, "uc": 1},
        "positive_name": "UC",
    },
}


def norm_cond(x: Any) -> str:
    s = str(x).strip().lower()
    if "crohn" in s or s == "cd":
        return "cd"
    if "ulcerative" in s or s == "uc":
        return "uc"
    if "non" in s:
        return "nonibd"
    return s


def latex_escape(x: Any) -> str:
    s = str(x)
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for a, b in repl.items():
        s = s.replace(a, b)
    return s


def fmt(x, digits=3):
    try:
        x = float(x)
    except Exception:
        return "--"
    if not math.isfinite(x):
        return "--"
    return f"{x:.{digits}f}"


def fmt_mean_sd(mean, sd, digits=3):
    if pd.isna(mean):
        return "--"
    if pd.isna(sd):
        return fmt(mean, digits)
    return f"{fmt(mean, digits)} $\\pm$ {fmt(sd, digits)}"


def pick_col(columns, candidates, required=True, label="column"):
    cols = list(columns)
    lower = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand in cols:
            return cand
        if cand.lower() in lower:
            return lower[cand.lower()]
    if required:
        raise ValueError(f"Could not find {label}. Tried {candidates}. Columns: {cols}")
    return None


def infer_sample_id_from_filename(path: Path) -> str:
    name = path.name
    for suffix in [".csv.gz", ".tsv.gz", ".csv", ".tsv", ".gz"]:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break

    suffixes = [
        "_ricci_active",
        "_ricci",
        "_faithful_pairwise_active",
        "_active_ricci",
        ".ricci_active",
        ".ricci",
    ]
    for s in suffixes:
        if name.endswith(s):
            name = name[: -len(s)]
    return name


def read_csv_chunks(path: Path, chunksize: int):
    sep = "\t" if path.name.endswith(".tsv") or path.name.endswith(".tsv.gz") else ","
    return pd.read_csv(path, sep=sep, low_memory=False, chunksize=chunksize)


def detect_columns(chunk: pd.DataFrame):
    edge_col = pick_col(chunk.columns, EDGE_CANDIDATES, required=False, label="edge column")
    source_col = None
    target_col = None

    if edge_col is None:
        source_col = pick_col(chunk.columns, SOURCE_CANDIDATES, required=True, label="source column")
        target_col = pick_col(chunk.columns, TARGET_CANDIDATES, required=True, label="target column")

    kappa_col = pick_col(chunk.columns, KAPPA_CANDIDATES, required=True, label="curvature column")
    sample_col = pick_col(chunk.columns, SAMPLE_CANDIDATES, required=False, label="sample id column")
    process_col = pick_col(chunk.columns, PROCESS_CANDIDATES, required=False, label="process column")

    return {
        "edge_col": edge_col,
        "source_col": source_col,
        "target_col": target_col,
        "kappa_col": kappa_col,
        "sample_col": sample_col,
        "process_col": process_col,
    }


def build_edge_series(chunk: pd.DataFrame, detected: dict[str, str | None]) -> pd.Series:
    if detected["edge_col"] is not None:
        return chunk[detected["edge_col"]].astype(str)
    return chunk[detected["source_col"]].astype(str) + " -> " + chunk[detected["target_col"]].astype(str)


def find_ricci_files(ricci_dir: Path, explicit_glob: str | None):
    if explicit_glob:
        files = sorted(ricci_dir.glob(explicit_glob))
    else:
        files = sorted(ricci_dir.rglob("*ricci*.csv*"))
        files = [p for p in files if "diagnostic" not in p.name.lower()]
        if not files:
            files = sorted(ricci_dir.rglob("*.csv.gz")) + sorted(ricci_dir.rglob("*.csv"))

    files = [p for p in files if p.is_file()]
    if not files:
        raise FileNotFoundError(f"No Ricci CSV/CSV.GZ files found in {ricci_dir}")
    return files


def load_metadata(metadata_path: Path) -> pd.DataFrame:
    meta = pd.read_csv(metadata_path, low_memory=False)
    sample_col = pick_col(meta.columns, ["External ID", "sample_id", "sample", "external_id"], label="sample id in metadata")
    participant_col = pick_col(meta.columns, ["Participant ID", "participant_id", "participant"], label="participant id in metadata")
    diagnosis_col = pick_col(meta.columns, ["diagnosis", "Diagnosis", "condition", "cond"], label="diagnosis in metadata")

    out = meta[[sample_col, participant_col, diagnosis_col]].copy()
    out.columns = ["sample_id", "participant_id", "diagnosis"]
    out["sample_id"] = out["sample_id"].astype(str)
    out["participant_id"] = out["participant_id"].astype(str)
    out["cond"] = out["diagnosis"].apply(norm_cond)
    out = out[out["cond"].isin(["cd", "uc", "nonibd"])].drop_duplicates("sample_id").reset_index(drop=True)
    return out


def pass1_discover_edges(files, chunksize):
    edge_to_idx: dict[str, int] = {}
    edge_process: dict[str, str] = {}
    sample_ids = set()
    detected_by_file = {}

    print("[pass1] Discovering samples, edges, and process annotations...", flush=True)
    t0 = time.time()

    for fi, path in enumerate(files, start=1):
        file_sample = infer_sample_id_from_filename(path)
        first = True
        detected = None
        n_rows = 0

        for chunk in read_csv_chunks(path, chunksize):
            if first:
                detected = detect_columns(chunk)
                detected_by_file[str(path)] = detected
                first = False

            edges = build_edge_series(chunk, detected)
            kappa = pd.to_numeric(chunk[detected["kappa_col"]], errors="coerce")
            ok = kappa.notna() & np.isfinite(kappa.to_numpy(dtype=float))

            if detected["sample_col"] is not None:
                samples = chunk.loc[ok, detected["sample_col"]].astype(str)
                sample_ids.update(samples.unique().tolist())
            else:
                sample_ids.add(file_sample)

            if detected["process_col"] is not None:
                procs = chunk.loc[ok, detected["process_col"]].fillna("Unannotated").astype(str)
            else:
                procs = pd.Series(["Unannotated"] * int(ok.sum()))

            for e, proc in zip(edges.loc[ok].astype(str), procs):
                if e not in edge_to_idx:
                    edge_to_idx[e] = len(edge_to_idx)
                    edge_process[e] = proc if proc else "Unannotated"
                elif edge_process.get(e, "Unannotated") == "Unannotated" and proc:
                    edge_process[e] = proc

            n_rows += len(chunk)

        print(f"  [{fi}/{len(files)}] {path.name}: rows={n_rows:,}, edges_so_far={len(edge_to_idx):,}, samples_so_far={len(sample_ids):,}", flush=True)

    print(f"[pass1 done] edges={len(edge_to_idx):,}, samples={len(sample_ids):,}, time={(time.time()-t0)/60:.1f} min", flush=True)
    return edge_to_idx, edge_process, sample_ids, detected_by_file


def build_sparse_features(files, metadata, edge_to_idx, chunksize, out_dir):
    n_edges = len(edge_to_idx)
    samples = metadata["sample_id"].tolist()
    sample_to_row = {s: i for i, s in enumerate(samples)}

    rows_b, cols_b, data_b = [], [], []
    rows_k, cols_k, data_k = [], [], []

    print("[pass2] Building sparse [B | K0] feature matrix...", flush=True)
    t0 = time.time()

    for fi, path in enumerate(files, start=1):
        file_sample = infer_sample_id_from_filename(path)
        first = True
        detected = None
        used_rows = 0

        for chunk in read_csv_chunks(path, chunksize):
            if first:
                detected = detect_columns(chunk)
                first = False

            edges = build_edge_series(chunk, detected)
            kappa = pd.to_numeric(chunk[detected["kappa_col"]], errors="coerce")
            ok = kappa.notna() & np.isfinite(kappa.to_numpy(dtype=float))

            if detected["sample_col"] is not None:
                sid_series = chunk[detected["sample_col"]].astype(str)
            else:
                sid_series = pd.Series([file_sample] * len(chunk), index=chunk.index)

            small = pd.DataFrame({
                "sample_id": sid_series.loc[ok].astype(str),
                "edge": edges.loc[ok].astype(str),
                "kappa": kappa.loc[ok].astype(float),
            })

            small = small[small["sample_id"].isin(sample_to_row)]
            if small.empty:
                continue

            # Duplicate safety: average duplicate curvature rows for the same sample-edge.
            small = small.groupby(["sample_id", "edge"], as_index=False)["kappa"].mean()

            for sid, e, kap in small.itertuples(index=False):
                ei = edge_to_idx.get(e)
                if ei is None:
                    continue
                ri = sample_to_row[sid]
                rows_b.append(ri)
                cols_b.append(ei)
                data_b.append(1.0)

                rows_k.append(ri)
                cols_k.append(ei)
                data_k.append(float(kap))

            used_rows += len(small)

        print(f"  [{fi}/{len(files)}] {path.name}: used sample-edge rows={used_rows:,}", flush=True)

    n_samples = len(samples)

    B = coo_matrix((data_b, (rows_b, cols_b)), shape=(n_samples, n_edges), dtype=np.float32).tocsr()
    B.data[:] = 1.0

    K = coo_matrix((data_k, (rows_k, cols_k)), shape=(n_samples, n_edges), dtype=np.float32).tocsr()
    X = hstack([B, K], format="csr")

    save_npz(out_dir / "feature_matrix_B_K0.npz", X)

    feature_meta = pd.DataFrame({
        "edge_index": np.arange(n_edges),
        "edge": list(edge_to_idx.keys()),
        "process": [None] * n_edges,
    })
    inv = {idx: edge for edge, idx in edge_to_idx.items()}
    feature_meta["edge"] = [inv[i] for i in range(n_edges)]
    return X, feature_meta



def balanced_limit_metadata(metadata: pd.DataFrame, max_samples: int | None, seed: int) -> pd.DataFrame:
    """
    Balanced sample-level limit for smoke tests.

    The previous smoke-test behaviour used .head(max_samples), which can produce
    a subset with too few non-IBD participants or even a single effective class
    in grouped CV. This function samples approximately equally from CD, UC, and
    non-IBD while preserving participant IDs for grouped splitting.
    """
    if max_samples is None or max_samples >= len(metadata):
        return metadata.copy()

    rng = np.random.default_rng(seed)
    conds = [c for c in ["cd", "uc", "nonibd"] if c in set(metadata["cond"])]
    if not conds:
        return metadata.sample(n=max_samples, random_state=seed).copy()

    base = max_samples // len(conds)
    rem = max_samples % len(conds)

    parts = []
    for i, cond in enumerate(conds):
        sub = metadata[metadata["cond"] == cond].copy()
        if sub.empty:
            continue

        n_take = base + (1 if i < rem else 0)
        n_take = min(n_take, len(sub))

        if n_take > 0:
            take_idx = rng.choice(sub.index.to_numpy(), size=n_take, replace=False)
            parts.append(sub.loc[take_idx])

    limited = pd.concat(parts, axis=0)

    # If one class had too few samples, top up from the remaining rows.
    if len(limited) < max_samples:
        remaining = metadata.drop(index=limited.index, errors="ignore")
        n_extra = min(max_samples - len(limited), len(remaining))
        if n_extra > 0:
            extra_idx = rng.choice(remaining.index.to_numpy(), size=n_extra, replace=False)
            limited = pd.concat([limited, remaining.loc[extra_idx]], axis=0)

    limited = limited.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    print(f"[max_samples] Balanced smoke subset: {len(limited)} samples", flush=True)
    print(limited["cond"].value_counts().to_string(), flush=True)
    print("participants:", limited["participant_id"].nunique(), flush=True)

    return limited


def build_or_load_features(args, metadata):
    out = Path(args.output_dir)
    matrix_path = out / "feature_matrix_B_K0.npz"
    edge_meta_path = out / "edge_metadata.csv"

    if args.reuse_features and matrix_path.exists() and edge_meta_path.exists():
        print("[reuse] Loading existing feature matrix and edge metadata.", flush=True)
        X = load_npz(matrix_path)
        edge_meta = pd.read_csv(edge_meta_path)
        return X, edge_meta

    files = find_ricci_files(Path(args.ricci_dir), args.ricci_glob)
    print("Ricci files found:", len(files), flush=True)
    for p in files[:10]:
        print(" ", p, flush=True)
    if len(files) > 10:
        print(" ...", flush=True)

    edge_to_idx, edge_process_dict, sample_ids, detected = pass1_discover_edges(files, args.chunksize)

    matched = metadata[metadata["sample_id"].isin(sample_ids)].copy()
    if args.max_samples is not None:
        matched = balanced_limit_metadata(matched, args.max_samples, args.seed)

    if matched.empty:
        raise ValueError("No metadata sample IDs matched Ricci files.")

    print("Matched samples:", matched["sample_id"].nunique(), flush=True)
    print("Matched participants:", matched["participant_id"].nunique(), flush=True)
    print("Condition counts:", flush=True)
    print(matched["cond"].value_counts().to_string(), flush=True)

    X, edge_meta = build_sparse_features(files, matched, edge_to_idx, args.chunksize, out)

    inv_edges = [None] * len(edge_to_idx)
    for e, idx in edge_to_idx.items():
        inv_edges[idx] = e

    edge_meta = pd.DataFrame({
        "edge_index": np.arange(len(inv_edges)),
        "edge": inv_edges,
        "process": [edge_process_dict.get(e, "Unannotated") for e in inv_edges],
    })

    matched.to_csv(out / "matched_metadata.csv", index=False)
    edge_meta.to_csv(edge_meta_path, index=False)
    (out / "detected_columns_by_file.json").write_text(json.dumps(detected, indent=2, default=str))

    print("[features done] X shape:", X.shape, "nnz:", X.nnz, flush=True)
    return X, edge_meta


def mean_abs_sparse(X):
    Y = X.copy()
    Y.data = np.abs(Y.data)
    return np.asarray(Y.mean(axis=0)).ravel()


def make_model(args, n_classes):
    cw = None if args.class_weight == "none" else args.class_weight

    return LogisticRegression(
        penalty="l1",
        solver="saga",
        C=args.C,
        max_iter=args.max_iter,
        tol=args.tol,
        class_weight=cw,
        random_state=args.seed,
        n_jobs=args.n_jobs,
        multi_class="multinomial" if n_classes > 2 else "auto",
    )


def task_subset(metadata, task_key):
    spec = TASKS[task_key]
    rows = []
    y = []
    for i, cond in enumerate(metadata["cond"].tolist()):
        if cond in spec["label_map"]:
            rows.append(i)
            y.append(spec["label_map"][cond])
    return np.asarray(rows, dtype=int), np.asarray(y, dtype=int)


def safe_auc(y_true, proba, n_classes):
    try:
        if n_classes == 2:
            return float(roc_auc_score(y_true, proba[:, 1]))
        return float(roc_auc_score(y_true, proba, labels=list(range(n_classes)), multi_class="ovr", average="macro"))
    except Exception:
        return np.nan



def effective_grouped_n_splits(y: np.ndarray, groups: np.ndarray, requested: int, task_display: str) -> int:
    """
    Determine a safe number of grouped CV splits.

    Each class must be represented by at least n_splits distinct participant
    groups. This prevents folds where the training data contains only one class.
    """
    y = np.asarray(y)
    groups = np.asarray(groups)

    class_group_counts = []
    for cls in sorted(np.unique(y)):
        cls_groups = set(groups[y == cls])
        class_group_counts.append(len(cls_groups))

    if len(class_group_counts) < 2:
        raise ValueError(
            f"{task_display}: only one class is present after filtering. "
            "Use a larger or more balanced smoke subset."
        )

    max_safe = min(class_group_counts)
    n_splits = min(int(requested), int(max_safe))

    if n_splits < 2:
        raise ValueError(
            f"{task_display}: not enough participant groups per class for grouped CV. "
            f"Participant groups per class: {class_group_counts}. "
            "Use a larger smoke subset or run the full dataset."
        )

    if n_splits < requested:
        print(
            f"[CV] Reducing n_splits from {requested} to {n_splits} for {task_display} "
            f"because participant groups per class are {class_group_counts}.",
            flush=True,
        )

    return n_splits


def cv_for_task(X, metadata, task_key, args, out_dir):
    spec = TASKS[task_key]
    rows, y = task_subset(metadata, task_key)
    X_task = X[rows]
    meta_task = metadata.iloc[rows].reset_index(drop=True)
    groups = meta_task["participant_id"].to_numpy()

    n_splits_eff = effective_grouped_n_splits(y, groups, args.n_splits, spec["display"])
    sgkf = StratifiedGroupKFold(n_splits=n_splits_eff, shuffle=True, random_state=args.seed)

    fold_rows = []
    y_true_all = []
    y_pred_all = []
    proba_all = []
    fold_models = []

    for fold, (tr, te) in enumerate(sgkf.split(X_task, y, groups), start=1):
        model = make_model(args, len(spec["class_names"]))

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(X_task[tr], y[tr])
            conv_warn = any(isinstance(w.message, ConvergenceWarning) for w in caught)

        pred = model.predict(X_task[te])
        proba = model.predict_proba(X_task[te])

        acc = accuracy_score(y[te], pred)
        bacc = balanced_accuracy_score(y[te], pred)
        macro_f1 = f1_score(y[te], pred, average="macro")
        auc = safe_auc(y[te], proba, len(spec["class_names"]))

        fold_rows.append({
            "fold": fold,
            "train_samples": int(len(tr)),
            "test_samples": int(len(te)),
            "train_participants": int(len(set(groups[tr]))),
            "test_participants": int(len(set(groups[te]))),
            "accuracy": float(acc),
            "balanced_accuracy": float(bacc),
            "macro_f1": float(macro_f1),
            "roc_auc": float(auc) if not pd.isna(auc) else np.nan,
            "convergence_warning": bool(conv_warn),
        })

        y_true_all.extend(y[te].tolist())
        y_pred_all.extend(pred.tolist())
        proba_all.append(proba)
        fold_models.append(model)

        print(
            f"  {spec['display']} fold {fold}: acc={acc:.3f}, bacc={bacc:.3f}, f1={macro_f1:.3f}, auc={auc:.3f}",
            flush=True,
        )

    fold_df = pd.DataFrame(fold_rows)
    cm = confusion_matrix(y_true_all, y_pred_all, labels=list(range(len(spec["class_names"]))))

    fold_df.to_csv(out_dir / "fold_results.csv", index=False)
    pd.DataFrame(cm, index=spec["class_names"], columns=spec["class_names"]).to_csv(out_dir / "confusion_matrix.csv")

    write_fold_table(fold_df, spec, out_dir / "latex_fold_results.tex")
    write_confusion_table(cm, spec["class_names"], spec["display"], out_dir / "latex_confusion_matrix.tex")

    # Refit interpretation model on all samples in the task.
    interp_model = make_model(args, len(spec["class_names"]))
    interp_model.fit(X_task, y)

    return {
        "task_key": task_key,
        "spec": spec,
        "rows": rows,
        "y": y,
        "X_task": X_task,
        "metadata_task": meta_task,
        "fold_df": fold_df,
        "cm": cm,
        "model": interp_model,
    }


def write_fold_table(fold_df, spec, out_path):
    metrics = ["accuracy", "balanced_accuracy", "macro_f1", "roc_auc"]

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        rf"\caption{{Participant-grouped five-fold performance for {latex_escape(spec['display'])}.}}",
        r"\begin{tabular}{rrrrrrrr}",
        r"\toprule",
        r"Fold & Train & Test & Train participants & Test participants & Accuracy & Balanced acc. & Macro F1 / ROC-AUC \\",
        r"\midrule",
    ]

    for _, r in fold_df.iterrows():
        lines.append(
            f"{int(r['fold'])} & {int(r['train_samples'])} & {int(r['test_samples'])} & "
            f"{int(r['train_participants'])} & {int(r['test_participants'])} & "
            f"{fmt(r['accuracy'])} & {fmt(r['balanced_accuracy'])} & {fmt(r['macro_f1'])} / {fmt(r['roc_auc'])} \\\\"
        )

    mean = fold_df[metrics + ["train_samples", "test_samples", "train_participants", "test_participants"]].mean(numeric_only=True)
    sd = fold_df[metrics + ["train_samples", "test_samples", "train_participants", "test_participants"]].std(ddof=0, numeric_only=True)

    lines += [
        r"\midrule",
        f"Mean & {fmt(mean['train_samples'],1)} & {fmt(mean['test_samples'],1)} & {fmt(mean['train_participants'],1)} & {fmt(mean['test_participants'],1)} & "
        f"{fmt(mean['accuracy'])} & {fmt(mean['balanced_accuracy'])} & {fmt(mean['macro_f1'])} / {fmt(mean['roc_auc'])} \\\\",
        f"SD & {fmt(sd['train_samples'],1)} & {fmt(sd['test_samples'],1)} & {fmt(sd['train_participants'],1)} & {fmt(sd['test_participants'],1)} & "
        f"{fmt(sd['accuracy'])} & {fmt(sd['balanced_accuracy'])} & {fmt(sd['macro_f1'])} / {fmt(sd['roc_auc'])} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    out_path.write_text("\n".join(lines) + "\n")


def write_confusion_table(cm, class_names, task_display, out_path):
    header = " & ".join([f"Predicted {latex_escape(c)}" for c in class_names])
    align = "l" + "r" * len(class_names)

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        rf"\caption{{Aggregated confusion matrix for {latex_escape(task_display)}. Rows denote true labels and columns denote predicted labels.}}",
        rf"\begin{{tabular}}{{{align}}}",
        r"\toprule",
        " & " + header + r" \\",
        r"\midrule",
    ]

    for i, true_name in enumerate(class_names):
        vals = " & ".join(str(int(x)) for x in cm[i])
        lines.append(f"True {latex_escape(true_name)} & {vals} \\\\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    out_path.write_text("\n".join(lines) + "\n")


def write_pairwise_summary(results, out_path):
    task_keys = ["nonibd_vs_uc", "nonibd_vs_cd", "cd_vs_uc"]
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Summary of participant-grouped pairwise Ricci binary classification performance. Values are mean $\pm$ SD across five folds.}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Task & Accuracy & Balanced acc. & Macro F1 & ROC-AUC \\",
        r"\midrule",
    ]

    for key in task_keys:
        if key not in results:
            continue
        spec = TASKS[key]
        df = results[key]["fold_df"]
        vals = []
        for m in ["accuracy", "balanced_accuracy", "macro_f1", "roc_auc"]:
            vals.append(fmt_mean_sd(df[m].mean(), df[m].std(ddof=0)))
        lines.append(f"{latex_escape(spec['display'])} & " + " & ".join(vals) + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    out_path.write_text("\n".join(lines) + "\n")




def make_ricci_density_plots(K, metadata: pd.DataFrame, out_dir: Path, n_bins: int = 260) -> None:
    """
    Make participant-average Ricci curvature density curves.

    For each sample, a histogram density is computed across active-edge Ricci
    values. Sample-level density curves are averaged within participant. The
    plotted condition curves are then mean ± SD across participants.

    Outputs:
      plots/ricci_curvature_density_by_condition.png/pdf
      plots/ricci_curvature_log_density_by_condition.png/pdf
      ricci_curvature_density_participant_average.csv
      ricci_curvature_density_condition_summary.csv
    """
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    all_vals = np.asarray(K.data, dtype=float)
    all_vals = all_vals[np.isfinite(all_vals)]

    if all_vals.size == 0:
        print("[density] No finite Ricci values found; skipping density plots.", flush=True)
        return

    # Robust plotting window: wide enough for the negative tail but not dominated
    # by a tiny number of extreme outliers.
    x_min = float(np.percentile(all_vals, 0.2))
    x_max = float(np.percentile(all_vals, 99.8))

    x_min = max(-40.0, min(x_min, -5.0))
    x_max = min(1.0, max(x_max, 0.5))

    if x_min >= x_max:
        x_min, x_max = -30.0, 1.0

    bins = np.linspace(x_min, x_max, n_bins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])

    sample_rows = []

    meta = metadata.reset_index(drop=True)

    for i, row_meta in meta.iterrows():
        vals = np.asarray(K.getrow(i).data, dtype=float)
        vals = vals[np.isfinite(vals)]
        vals = vals[(vals >= x_min) & (vals <= x_max)]

        if vals.size:
            density, _ = np.histogram(vals, bins=bins, density=True)
        else:
            density = np.zeros(len(centers), dtype=float)

        for c, d in zip(centers, density):
            sample_rows.append({
                "sample_id": row_meta["sample_id"],
                "participant_id": row_meta["participant_id"],
                "cond": row_meta["cond"],
                "kappa_bin_center": float(c),
                "density": float(d),
            })

    sample_density = pd.DataFrame(sample_rows)
    sample_density.to_csv(out_dir / "ricci_curvature_density_samplewise.csv", index=False)

    participant_density = (
        sample_density
        .groupby(["participant_id", "cond", "kappa_bin_center"], as_index=False)["density"]
        .mean()
    )
    participant_density.to_csv(out_dir / "ricci_curvature_density_participant_average.csv", index=False)

    summary_rows = []

    for cond in ["overall", "cd", "uc", "nonibd"]:
        if cond == "overall":
            sub = participant_density.copy()
        else:
            sub = participant_density[participant_density["cond"] == cond].copy()

        for center, g in sub.groupby("kappa_bin_center"):
            vals = pd.to_numeric(g["density"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            summary_rows.append({
                "cond": cond,
                "kappa_bin_center": float(center),
                "n_participants": int(g["participant_id"].nunique()),
                "density_mean": float(vals.mean()) if len(vals) else np.nan,
                "density_sd": float(vals.std(ddof=0)) if len(vals) else np.nan,
            })

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "ricci_curvature_density_condition_summary.csv", index=False)

    def display_condition(c):
        return {"overall": "overall", "cd": "CD", "uc": "UC", "nonibd": "non-IBD"}.get(str(c), str(c))

    def draw(log_scale: bool) -> None:
        fig, ax = plt.subplots(figsize=(7.2, 4.8))

        for cond in ["overall", "cd", "uc", "nonibd"]:
            sub = summary[summary["cond"] == cond].sort_values("kappa_bin_center")
            if sub.empty:
                continue

            x = sub["kappa_bin_center"].to_numpy(dtype=float)
            y = sub["density_mean"].to_numpy(dtype=float)
            sd = sub["density_sd"].to_numpy(dtype=float)
            sd = np.where(np.isfinite(sd), sd, 0.0)

            if log_scale:
                eps = 1e-8
                y_plot = np.log10(y + eps)
                lo = np.log10(np.maximum(y - sd, 0.0) + eps)
                hi = np.log10(y + sd + eps)
                y_label = r"$\log_{10}$ density"
                title = "Participant-average Ricci curvature log-density"
                out_base = plot_dir / "ricci_curvature_log_density_by_condition"
            else:
                y_plot = y
                lo = np.maximum(y - sd, 0.0)
                hi = y + sd
                y_label = "Density"
                title = "Participant-average Ricci curvature density"
                out_base = plot_dir / "ricci_curvature_density_by_condition"

            line, = ax.plot(x, y_plot, linewidth=2.0, label=display_condition(cond))
            ax.fill_between(x, lo, hi, alpha=0.15, color=line.get_color(), linewidth=0)

        ax.set_xlabel(r"Ricci curvature $\kappa$")
        ax.set_ylabel(y_label)
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()

        fig.savefig(str(out_base) + ".png", dpi=300)
        fig.savefig(str(out_base) + ".pdf")
        plt.close(fig)

    draw(log_scale=False)
    draw(log_scale=True)

    print("[density] Wrote Ricci density plots:", flush=True)
    print(" ", plot_dir / "ricci_curvature_density_by_condition.png", flush=True)
    print(" ", plot_dir / "ricci_curvature_log_density_by_condition.png", flush=True)


def curvature_percentile_table(X, metadata, edge_meta, out_dir):
    """
    Write Ricci curvature percentile diagnostic tables.

    Two tables are produced:

    1. Pooled active-edge percentile table:
       pools all active Ricci curvature values across samples within each
       condition. This matches the older "distribution of Ricci feature values"
       style table.

    2. Participant-average percentile table:
       computes percentiles within each sample graph, averages sample-level
       percentiles within participant, then reports mean ± SD across
       participants by condition. This matches the main graph-diagnostic
       protocol used elsewhere in the manuscript.
    """
    n_edges = len(edge_meta)
    K = X[:, n_edges:].tocsr()

    pct_specs = [
        (0.1, "p0p1", "0.1"),
        (1, "p1", "1"),
        (5, "p5", "5"),
        (10, "p10", "10"),
        (20, "p20", "20"),
        (30, "p30", "30"),
        (40, "p40", "40"),
        (50, "p50", "50"),
        (60, "p60", "60"),
        (70, "p70", "70"),
        (80, "p80", "80"),
        (90, "p90", "90"),
        (95, "p95", "95"),
        (99, "p99", "99"),
    ]

    pct_values = [x[0] for x in pct_specs]

    def condition_label(c):
        return {"overall": "overall", "cd": "CD", "uc": "UC", "nonibd": "non-IBD"}.get(str(c), str(c))

    def write_pooled_latex(df, path):
        lines = [
            r"\begin{table}[ht]",
            r"\centering",
            r"\caption{Pooled percentiles of Ricci curvature feature values across active graph edges, shown overall and by diagnostic condition.}",
            r"\resizebox{\textwidth}{!}{%",
            r"\begin{tabular}{lr" + "r" * len(pct_specs) + "}",
            r"\toprule",
            "Group & $n$ & " + " & ".join([rf"$p_{{{lab}}}$" for _, _, lab in pct_specs]) + r" \\",
            r"\midrule",
        ]

        for _, row in df.iterrows():
            vals = [fmt(row[col]) for _, col, _ in pct_specs]
            lines.append(f"{latex_escape(row['group'])} & {int(row['n'])} & " + " & ".join(vals) + r" \\")

        lines += [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\end{table}",
        ]
        path.write_text("\n".join(lines) + "\n")

    def write_participant_latex(df, path):
        lines = [
            r"\begin{table}[ht]",
            r"\centering",
            r"\caption{Participant-average percentile bands of Ricci curvature values. Percentiles are first computed within each sample graph across active directed edges, then averaged within participant; condition values are mean $\pm$ SD across participants.}",
            r"\resizebox{\textwidth}{!}{%",
            r"\begin{tabular}{lr" + "r" * len(pct_specs) + "}",
            r"\toprule",
            "Condition & $n$ & " + " & ".join([rf"$p_{{{lab}}}$" for _, _, lab in pct_specs]) + r" \\",
            r"\midrule",
        ]

        for cond in ["overall", "cd", "uc", "nonibd"]:
            sub = df[df["cond"] == cond]
            if sub.empty:
                continue

            row = sub.iloc[0]
            vals = [
                fmt_mean_sd(row.get(f"{col}_mean"), row.get(f"{col}_sd"))
                for _, col, _ in pct_specs
            ]
            lines.append(f"{condition_label(cond)} & {int(row['n'])} & " + " & ".join(vals) + r" \\")

        lines += [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\end{table}",
        ]
        path.write_text("\n".join(lines) + "\n")

    # --------------------------------------------------------------
    # A. Pooled percentiles
    # --------------------------------------------------------------
    pooled_rows = []

    for cond in ["overall", "cd", "uc", "nonibd"]:
        if cond == "overall":
            vals = K.data
            label = "Overall"
        else:
            idx = np.where(metadata["cond"].to_numpy() == cond)[0]
            vals = K[idx].data
            label = condition_label(cond)

        vals = np.asarray(vals, dtype=float)
        vals = vals[np.isfinite(vals)]

        row = {"group": label, "cond": cond, "n": int(len(vals))}
        if len(vals):
            qs = np.percentile(vals, pct_values)
            for (_, col, _), q in zip(pct_specs, qs):
                row[col] = float(q)
        else:
            for _, col, _ in pct_specs:
                row[col] = np.nan

        pooled_rows.append(row)

    pooled_df = pd.DataFrame(pooled_rows)
    pooled_df.to_csv(out_dir / "ricci_curvature_percentiles_pooled_by_condition.csv", index=False)

    # Keep the old filename for compatibility with the combined LaTeX collector.
    write_pooled_latex(
        pooled_df,
        out_dir / "latex_ricci_curvature_percentiles_by_condition.tex",
    )

    # --------------------------------------------------------------
    # B. Samplewise -> participant-average percentiles
    # --------------------------------------------------------------
    sample_rows = []
    meta = metadata.reset_index(drop=True)

    for i, row_meta in meta.iterrows():
        vals = np.asarray(K.getrow(i).data, dtype=float)
        vals = vals[np.isfinite(vals)]

        row = {
            "sample_id": row_meta["sample_id"],
            "participant_id": row_meta["participant_id"],
            "cond": row_meta["cond"],
            "n_active_ricci_edges": int(len(vals)),
        }

        if len(vals):
            qs = np.percentile(vals, pct_values)
            for (_, col, _), q in zip(pct_specs, qs):
                row[col] = float(q)
        else:
            for _, col, _ in pct_specs:
                row[col] = np.nan

        sample_rows.append(row)

    sample_df = pd.DataFrame(sample_rows)
    sample_df.to_csv(out_dir / "ricci_curvature_percentiles_samplewise.csv", index=False)

    participant_rows = []
    numeric_cols = ["n_active_ricci_edges"] + [col for _, col, _ in pct_specs]

    for pid, sub in sample_df.groupby("participant_id", dropna=False):
        row = {
            "participant_id": pid,
            "cond": sub["cond"].astype(str).value_counts().idxmax(),
            "n_samples_for_participant": int(sub["sample_id"].nunique()),
        }

        for col in numeric_cols:
            vals = pd.to_numeric(sub[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            row[col] = float(vals.mean()) if len(vals) else np.nan

        participant_rows.append(row)

    participant_df = pd.DataFrame(participant_rows)
    participant_df.to_csv(out_dir / "ricci_curvature_percentiles_participant_average.csv", index=False)

    summary_rows = []

    for cond in ["overall", "cd", "uc", "nonibd"]:
        sub = participant_df if cond == "overall" else participant_df[participant_df["cond"] == cond]

        row = {
            "cond": cond,
            "n": int(len(sub)),
        }

        for col in numeric_cols:
            vals = pd.to_numeric(sub[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            row[f"{col}_mean"] = float(vals.mean()) if len(vals) else np.nan
            row[f"{col}_sd"] = float(vals.std(ddof=0)) if len(vals) else np.nan

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "ricci_curvature_percentiles_participant_condition_summary.csv", index=False)

    write_participant_latex(
        summary_df,
        out_dir / "latex_ricci_curvature_percentiles_participant_average.tex",
    )

    make_ricci_density_plots(K, metadata, out_dir)

def coef_tables_for_task(result, edge_meta, out_dir, max_process_rows, top_n):
    spec = result["spec"]
    model = result["model"]
    X_task = result["X_task"]
    n_edges = len(edge_meta)

    mean_abs_x = mean_abs_sparse(X_task)
    edge_names = edge_meta["edge"].astype(str).tolist()
    processes = edge_meta["process"].fillna("Unannotated").astype(str).tolist()

    task_dir = out_dir / TASKS[result["task_key"]]["folder"]
    interp_dir = task_dir / "interpretation_full_refit"
    interp_dir.mkdir(parents=True, exist_ok=True)

    if spec["kind"] == "binary":
        beta = model.coef_[0].astype(float)
        make_binary_interpretation_tables(
            beta=beta,
            mean_abs_x=mean_abs_x,
            edge_names=edge_names,
            processes=processes,
            n_edges=n_edges,
            positive_name=spec["positive_name"],
            task_display=spec["display"],
            out_dir=interp_dir,
            max_process_rows=max_process_rows,
            top_n=top_n,
        )
    else:
        beta = model.coef_.astype(float)
        make_multiclass_interpretation_tables(
            beta=beta,
            mean_abs_x=mean_abs_x,
            edge_names=edge_names,
            processes=processes,
            n_edges=n_edges,
            class_names=spec["class_names"],
            task_display=spec["display"],
            out_dir=interp_dir,
            max_process_rows=max_process_rows,
            top_n=top_n,
        )


def make_feature_info(j, n_edges, edge_names, processes):
    if j < n_edges:
        return "B", j, edge_names[j], processes[j]
    idx = j - n_edges
    return "K0", idx, edge_names[idx], processes[idx]


def make_binary_interpretation_tables(beta, mean_abs_x, edge_names, processes, n_edges, positive_name, task_display, out_dir, max_process_rows, top_n):
    tol = 1e-12
    nz = np.where(np.abs(beta) > tol)[0]

    rows = []
    for j in nz:
        ftype, ei, edge, proc = make_feature_info(j, n_edges, edge_names, processes)
        rows.append({
            "feature_index": int(j),
            "edge_index": int(ei),
            "edge": edge,
            "process": proc,
            "feature_type": ftype,
            "beta": float(beta[j]),
            "average_importance": float(abs(beta[j]) * mean_abs_x[j]),
            "positive_class": positive_name,
        })

    coef_df = pd.DataFrame(rows)
    coef_df.to_csv(out_dir / "feature_coefficients.csv", index=False)

    if coef_df.empty:
        return

    proc_rows = []
    for proc, sub in coef_df.groupby("process"):
        ksub = sub[sub["feature_type"] == "K0"]
        bsub = sub[sub["feature_type"] == "B"]
        proc_rows.append({
            "Process": proc,
            "Total Coeff.": sub["beta"].abs().sum(),
            "Features": len(sub),
            "Edges": sub["edge"].nunique(),
            "Mean |coeff|": sub["beta"].abs().mean(),
            "Mean B coeff": bsub["beta"].mean() if len(bsub) else np.nan,
            "Mean K0 coeff": ksub["beta"].mean() if len(ksub) else np.nan,
            "% positive K0": 100.0 * (ksub["beta"] > 0).mean() if len(ksub) else np.nan,
            "K0 features": len(ksub),
        })

    proc_df = pd.DataFrame(proc_rows).sort_values("Total Coeff.", ascending=False)
    proc_df.to_csv(out_dir / "process_coefficient_table.csv", index=False)
    write_df_latex(proc_df.head(max_process_rows), out_dir / "latex_process_coefficient_table.tex",
                   f"Biological process coefficient table for {task_display}. Positive coefficients correspond to {positive_name}.")

    # Realised importance
    imp_rows = []
    for proc, sub in coef_df.groupby("process"):
        b_imp = sub.loc[sub["feature_type"] == "B", "average_importance"].sum()
        k_imp = sub.loc[sub["feature_type"] == "K0", "average_importance"].sum()
        imp_rows.append({
            "Process category": proc,
            "Total imp.": b_imp + k_imp,
            "B imp.": b_imp,
            "K0 imp.": k_imp,
        })
    imp_df = pd.DataFrame(imp_rows).sort_values("Total imp.", ascending=False)
    imp_df.to_csv(out_dir / "process_realised_importance_table.csv", index=False)
    write_df_latex(imp_df.head(max_process_rows), out_dir / "latex_process_realised_importance_table.tex",
                   f"Total realised push by process category for {task_display}. Push is the mean across samples of summed absolute realised contributions.")

    top = coef_df.sort_values("beta", key=lambda s: s.abs(), ascending=False).head(top_n).copy()
    top = top[["positive_class", "edge", "process", "feature_type", "beta", "average_importance"]]
    top.columns = ["Class", "Edge", "Process", "Dominant", "Coefficient $\\beta$", "Average imp."]
    top.to_csv(out_dir / "top_absolute_features.csv", index=False)
    write_df_latex(top, out_dir / "latex_top_absolute_features.tex",
                   f"Top absolute-value Ricci features for {task_display}.")


def make_multiclass_interpretation_tables(beta, mean_abs_x, edge_names, processes, n_edges, class_names, task_display, out_dir, max_process_rows, top_n):
    tol = 1e-12
    rows = []

    for ci, cname in enumerate(class_names):
        nz = np.where(np.abs(beta[ci]) > tol)[0]
        for j in nz:
            ftype, ei, edge, proc = make_feature_info(j, n_edges, edge_names, processes)
            rows.append({
                "class": cname,
                "class_index": ci,
                "feature_index": int(j),
                "edge_index": int(ei),
                "edge": edge,
                "process": proc,
                "feature_type": ftype,
                "beta": float(beta[ci, j]),
                "average_importance": float(abs(beta[ci, j]) * mean_abs_x[j]),
            })

    coef_df = pd.DataFrame(rows)
    coef_df.to_csv(out_dir / "feature_coefficients_multiclass_long.csv", index=False)

    if coef_df.empty:
        return

    proc_rows = []
    for proc, sub in coef_df.groupby("process"):
        row = {
            "Process": proc,
            "Total Coeff.": sub["beta"].abs().sum(),
            "Features": len(sub),
            "Edges": sub["edge"].nunique(),
            "Mean |β|": sub["beta"].abs().mean(),
        }
        for cname in class_names:
            for ftype in ["B", "K0"]:
                ss = sub[(sub["class"] == cname) & (sub["feature_type"] == ftype)]
                row[f"{ftype}_{cname}"] = ss["beta"].mean() if len(ss) else np.nan
        proc_rows.append(row)

    proc_df = pd.DataFrame(proc_rows).sort_values("Total Coeff.", ascending=False)
    proc_df.to_csv(out_dir / "process_coefficient_table.csv", index=False)
    write_df_latex(proc_df.head(max_process_rows), out_dir / "latex_process_coefficient_table.tex",
                   f"Biological process coefficient table for the {task_display} Ricci classifier.")

    imp_rows = []
    for proc, sub in coef_df.groupby("process"):
        row = {"Process": proc, "Total imp.": sub["average_importance"].sum()}
        for cname in class_names:
            for ftype in ["K0", "B"]:
                ss = sub[(sub["class"] == cname) & (sub["feature_type"] == ftype)]
                row[f"{cname} {ftype}"] = ss["average_importance"].sum()
        imp_rows.append(row)

    imp_df = pd.DataFrame(imp_rows).sort_values("Total imp.", ascending=False)
    imp_df.to_csv(out_dir / "process_realised_importance_table.csv", index=False)
    write_df_latex(imp_df.head(max_process_rows), out_dir / "latex_process_realised_importance_table.tex",
                   f"Total realised push by process category for the {task_display} Ricci classifier.")

    top = coef_df.sort_values("beta", key=lambda s: s.abs(), ascending=False).head(top_n).copy()
    top = top[["class", "edge", "process", "feature_type", "beta", "average_importance"]]
    top.columns = ["Class", "Edge", "Process", "Dom.", "$\\beta$", "Avg. imp."]
    top.to_csv(out_dir / "top_absolute_features.csv", index=False)
    write_df_latex(top, out_dir / "latex_top_absolute_features.tex",
                   f"Top absolute-value features from the {task_display} Ricci classifier.")



def write_df_latex(df, out_path, caption):
    """
    Minimal dependency-free LaTeX table writer.

    This avoids pandas.to_latex(), which depends on a sufficiently recent
    jinja2 installation in newer pandas versions.
    """
    if df is None or len(df) == 0:
        out_path.write_text(
            "\\begin{table}[ht]\n"
            "\\centering\n"
            f"\\caption{{{latex_escape(caption)}}}\n"
            "No rows.\n"
            "\\end{table}\n"
        )
        return

    pretty = df.copy()

    # Format floats but leave integers/strings readable.
    for c in pretty.columns:
        if pd.api.types.is_float_dtype(pretty[c]):
            pretty[c] = pretty[c].map(lambda x: "--" if pd.isna(x) else f"{float(x):.3f}")
        elif pd.api.types.is_integer_dtype(pretty[c]):
            pretty[c] = pretty[c].map(lambda x: "--" if pd.isna(x) else str(int(x)))
        else:
            pretty[c] = pretty[c].map(lambda x: "--" if pd.isna(x) else str(x))

    n_cols = len(pretty.columns)
    align = "l" * n_cols

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        rf"\caption{{{latex_escape(caption)}}}",
        r"\resizebox{\textwidth}{!}{%",
        rf"\begin{{tabular}}{{{align}}}",
        r"\toprule",
        " & ".join(latex_escape(c) for c in pretty.columns) + r" \\",
        r"\midrule",
    ]

    for _, row in pretty.iterrows():
        vals = [latex_escape(row[c]) for c in pretty.columns]
        lines.append(" & ".join(vals) + r" \\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\end{table}",
    ]

    out_path.write_text("\n".join(lines) + "\n")

def collect_all_latex(out_dir, results):
    parts = []

    for name in ["latex_ricci_curvature_percentiles_by_condition.tex", "latex_ricci_curvature_percentiles_participant_average.tex", "latex_pairwise_summary.tex"]:
        p = out_dir / name
        if p.exists():
            parts.append(p.read_text())

    for key, res in results.items():
        folder = TASKS[key]["folder"]
        task_dir = out_dir / folder
        for name in [
            "latex_fold_results.tex",
            "latex_confusion_matrix.tex",
            "interpretation_full_refit/latex_process_coefficient_table.tex",
            "interpretation_full_refit/latex_process_realised_importance_table.tex",
            "interpretation_full_refit/latex_top_absolute_features.tex",
        ]:
            p = task_dir / name
            if p.exists():
                parts.append(p.read_text())

    (out_dir / "latex_all_ricci_classifier_tables.tex").write_text("\n\n".join(parts) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ricci-dir", required=True)
    ap.add_argument("--metadata", default=str(Path.home() / "Real_Data" / "hmp2_metadata.csv"))
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--ricci-glob", default=None, help='Optional glob such as "*_ricci_active.csv.gz".')
    ap.add_argument("--tasks", nargs="+", default=list(TASKS.keys()), choices=list(TASKS.keys()))
    ap.add_argument("--chunksize", type=int, default=250000)
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20260709)
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--class-weight", choices=["balanced", "none"], default="balanced")
    ap.add_argument("--max-iter", type=int, default=5000)
    ap.add_argument("--tol", type=float, default=1e-4)
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--reuse-features", action="store_true")
    ap.add_argument("--max-process-rows", type=int, default=40)
    ap.add_argument("--top-n", type=int, default=25)
    args = ap.parse_args()

    out_dir = Path(args.output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    config = vars(args).copy()
    config["feature_definition"] = "[B | K0], where B(e)=1 if active edge e is present and K0(e)=Ricci curvature if present, else 0."
    config["cv_definition"] = "StratifiedGroupKFold over participant IDs; no participant appears in both train and test."
    config["interpretation_tables"] = "Fit one final L1 logistic model on all samples in each task after CV; use this model for coefficient tables."
    (out_dir / "run_config.json").write_text(json.dumps(config, indent=2, default=str))

    metadata_all = load_metadata(Path(args.metadata))
    X, edge_meta = build_or_load_features(args, metadata_all)

    matched_meta = pd.read_csv(out_dir / "matched_metadata.csv")
    matched_meta["sample_id"] = matched_meta["sample_id"].astype(str)
    matched_meta["participant_id"] = matched_meta["participant_id"].astype(str)
    matched_meta["cond"] = matched_meta["cond"].astype(str)

    print("\nFinal feature matrix:", X.shape, "nnz=", X.nnz, flush=True)
    print("Matched metadata rows:", len(matched_meta), flush=True)

    curvature_percentile_table(X, matched_meta, edge_meta, out_dir)

    results = {}
    for task_key in args.tasks:
        print("\n" + "=" * 100, flush=True)
        print("Running task:", TASKS[task_key]["display"], flush=True)
        print("=" * 100, flush=True)

        task_dir = out_dir / TASKS[task_key]["folder"]
        task_dir.mkdir(parents=True, exist_ok=True)

        res = cv_for_task(X, matched_meta, task_key, args, task_dir)
        results[task_key] = res
        coef_tables_for_task(res, edge_meta, out_dir, args.max_process_rows, args.top_n)

    write_pairwise_summary(results, out_dir / "latex_pairwise_summary.tex")
    collect_all_latex(out_dir, results)

    print("\n[READY] Ricci classifier finished.", flush=True)
    print("Main output:", out_dir, flush=True)
    print("Combined LaTeX:", out_dir / "latex_all_ricci_classifier_tables.tex", flush=True)


if __name__ == "__main__":
    main()
