#!/usr/bin/env python3
from __future__ import annotations

"""Build manuscript-ready IBD-vs-non-IBD Ricci C-path outputs.

Purpose
-------
This script is a *post-processing only* utility.  It never refits a classifier.
It reads the completed repeated participant-grouped OOF predictions from the
Ricci C-path run, aligns them with the already-completed species-abundance
benchmark evaluation set, and writes publication-ready tables/figures.

The key comparison rule is deliberately conservative:

* Ricci-only performance is reported on the full Ricci OOF cohort.
* Ricci-vs-species comparisons are recomputed on the exact species-benchmark
  complete-case sample set.  No Ricci model is refit; only held-out OOF rows
  are filtered before metrics are recomputed.
* Repetition is the unit of uncertainty: each of the 20 repeated estimates
  pools all five participant-grouped outer folds.
* C=0.02 is treated as the prespecified manuscript/reference model.  Other C
  values are sensitivity analyses, not post-hoc model selection.

Default expected inputs
-----------------------
Ricci:
    ~/Real_Data/Ricci_IBD_RepeatedCV_CPath
Species benchmarks:
    ~/Real_Data/Species_Benchmarks_RepeatedCV_IBD_CompleteCase
Canonical species table (fallback only for recovering the exact benchmark
complete-case sample IDs):
    ~/Real_Data/Real_Species_Abundances_canon.xlsx

The loader is intentionally tolerant of the two output layouts used by the
project: per-C directories such as C_0p02 and combined CSVs containing a C
column.  It discovers files by their columns and validates the result before
writing anything.
"""

import argparse
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score

VERSION = "1.0.0"
DEFAULT_C_VALUES = (0.005, 0.01, 0.02, 0.05, 0.1)
DEFAULT_REFERENCE_C = 0.02
DEFAULT_REPEATS = 20
METRICS = ("accuracy", "balanced_accuracy", "macro_f1", "roc_auc")
FIGURE_METRICS = ("balanced_accuracy", "macro_f1", "roc_auc")
METRIC_LABELS = {
    "accuracy": "Accuracy",
    "balanced_accuracy": "Balanced accuracy",
    "macro_f1": "Macro F1",
    "roc_auc": "ROC-AUC",
}

MODEL_ORDER = (
    "Species random forest",
    "Species XGBoost",
    "Species L1 logistic",
)


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------

def now_iso() -> str:
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


def parse_float_list(text: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in text.split(",") if item.strip())
    if not values or any(v <= 0 for v in values) or len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("C values must be unique positive numbers")
    return values


def c_key(value: float) -> float:
    return round(float(value), 12)


def c_display(value: float) -> str:
    return f"{float(value):g}"


def c_tag(value: float) -> str:
    return "C_" + f"{float(value):.12g}".replace("-", "m").replace(".", "p")


def latex_escape(value: object) -> str:
    text = str(value)
    replacements = {
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
    return "".join(replacements.get(ch, ch) for ch in text)


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def atomic_json(path: Path, payload: Any) -> None:
    atomic_text(path, json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n")


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def read_csv_header(path: Path) -> list[str]:
    try:
        return list(pd.read_csv(path, nrows=0).columns)
    except Exception:
        return []


def first_existing(columns: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    lookup = {str(c).strip().lower(): c for c in columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    return None


def normalise_sample_id(value: object) -> str:
    text = str(value).strip()
    text = Path(text).name
    text = re.sub(r"\.npy$", "", text, flags=re.I)
    text = re.sub(r"_H0$", "", text, flags=re.I)
    return text


def normalise_binary_label(value: object) -> int:
    if pd.isna(value):
        raise ValueError("Missing binary label")
    if isinstance(value, (int, np.integer, bool, np.bool_)):
        iv = int(value)
        if iv in (0, 1):
            return iv
    if isinstance(value, (float, np.floating)) and math.isfinite(float(value)):
        fv = float(value)
        if fv in (0.0, 1.0):
            return int(fv)
    text = str(value).strip().lower()
    compact = re.sub(r"[\s_\-]+", "", text)
    if compact in {"0", "nonibd", "nonibdcontrol", "healthy", "control", "false"}:
        return 0
    if compact in {"1", "ibd", "uc", "cd", "crohns", "crohnsdisease", "ulcerativecolitis", "true"}:
        return 1
    raise ValueError(f"Could not interpret IBD-vs-non-IBD label: {value!r}")


def normalise_model_name(value: object) -> Optional[str]:
    text = str(value).strip().lower()
    compact = re.sub(r"[^a-z0-9]+", "", text)
    if "randomforest" in compact or compact in {"rf", "speciesrf"}:
        return "Species random forest"
    if "xgboost" in compact or compact in {"xgb", "speciesxgb"}:
        return "Species XGBoost"
    if ("logistic" in compact and ("l1" in compact or "lasso" in compact)) or compact in {
        "logisticl1", "l1logistic", "speciesl1", "logisticregressionl1"
    }:
        return "Species L1 logistic"
    return None


def infer_model_from_path(path: Path) -> Optional[str]:
    return normalise_model_name(str(path))


def infer_c_from_text(text: str) -> Optional[float]:
    # Explicit C_0p02 / C-0.02 / C=0.02 / C0.02 style.
    patterns = [
        r"(?:^|[/\\_\-])C[_=\-]?([0-9]+(?:[p\.][0-9]+)?)",
        r"(?:^|[/\\])C([0-9]+(?:[p\.][0-9]+)?)",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.I)
        if matches:
            raw = matches[-1].replace("p", ".").replace("P", ".")
            try:
                return float(raw)
            except ValueError:
                pass
    return None


def float_match(a: float, b: float, atol: float = 1e-12) -> bool:
    return abs(float(a) - float(b)) <= atol


def require_finite_unit_interval(frame: pd.DataFrame, columns: Iterable[str], context: str) -> None:
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
        if not np.isfinite(values).all():
            raise RuntimeError(f"{context}: {column} contains nonfinite values")
        if ((values < -1e-12) | (values > 1.0 + 1e-12)).any():
            raise RuntimeError(f"{context}: {column} has values outside [0,1]")


# -----------------------------------------------------------------------------
# Prediction standardisation and metrics
# -----------------------------------------------------------------------------

@dataclass
class PredictionBundle:
    frame: pd.DataFrame
    source_files: list[Path]


def identify_prediction_columns(columns: Iterable[str]) -> dict[str, Optional[str]]:
    cols = list(columns)
    return {
        "sample_id": first_existing(cols, ["sample_id", "External ID", "external_id", "sample", "sampleid"]),
        "repeat": first_existing(cols, ["repeat", "repetition", "rep"]),
        "fold": first_existing(cols, ["fold", "outer_fold", "test_fold"]),
        "true": first_existing(cols, ["y_true", "true_label", "label", "label_name", "truth", "actual"]),
        "pred": first_existing(cols, ["pred", "y_pred", "predicted_label", "pred_label", "prediction"]),
        "model": first_existing(cols, ["model", "model_name", "classifier"]),
        "C": first_existing(cols, ["C", "c", "c_value", "regularization_C"]),
    }


def find_ibd_probability_column(columns: Iterable[str]) -> Optional[str]:
    cols = list(columns)
    exact = first_existing(
        cols,
        [
            "proba_IBD", "probability_IBD", "probability_ibd", "proba_ibd",
            "p_IBD", "p_ibd", "prob_IBD", "prob_ibd", "y_probability",
            "positive_probability", "probability_1", "proba_1",
        ],
    )
    if exact is not None:
        return exact
    # Conservative fuzzy fallback: column mentions probability/proba and IBD,
    # but not non-IBD.
    for col in cols:
        low = str(col).lower()
        if ("prob" in low or "proba" in low) and "ibd" in low and "non" not in low:
            return col
    return None


def standardise_prediction_frame(
    frame: pd.DataFrame,
    source: Path,
    *,
    forced_model: Optional[str] = None,
    forced_c: Optional[float] = None,
) -> pd.DataFrame:
    mapping = identify_prediction_columns(frame.columns)
    probability_col = find_ibd_probability_column(frame.columns)
    required = ("sample_id", "repeat", "true")
    if any(mapping[key] is None for key in required) or probability_col is None:
        raise ValueError("not a usable OOF prediction table")

    out = pd.DataFrame()
    out["sample_id"] = frame[mapping["sample_id"]].map(normalise_sample_id)
    out["repeat"] = pd.to_numeric(frame[mapping["repeat"]], errors="raise").astype(int)
    if mapping["fold"] is not None:
        out["fold"] = pd.to_numeric(frame[mapping["fold"]], errors="coerce").astype("Int64")
    else:
        out["fold"] = pd.Series([pd.NA] * len(frame), dtype="Int64")
    out["y_true"] = frame[mapping["true"]].map(normalise_binary_label).astype(int)
    out["probability_IBD"] = pd.to_numeric(frame[probability_col], errors="raise").astype(float)

    if mapping["pred"] is not None:
        try:
            out["pred"] = frame[mapping["pred"]].map(normalise_binary_label).astype(int)
        except Exception:
            # Numeric argmax-style predictions are common.
            pred = pd.to_numeric(frame[mapping["pred"]], errors="coerce")
            if pred.isna().any() or not pred.isin([0, 1]).all():
                raise
            out["pred"] = pred.astype(int)
    else:
        out["pred"] = (out["probability_IBD"] >= 0.5).astype(int)

    if forced_model is not None:
        out["model"] = forced_model
    elif mapping["model"] is not None:
        out["model"] = frame[mapping["model"]].map(normalise_model_name)
    else:
        out["model"] = infer_model_from_path(source)

    if forced_c is not None:
        out["C"] = float(forced_c)
    elif mapping["C"] is not None:
        out["C"] = pd.to_numeric(frame[mapping["C"]], errors="coerce")
    else:
        out["C"] = infer_c_from_text(str(source))

    out["source_file"] = str(source)
    return out


def binary_metrics(frame: pd.DataFrame) -> dict[str, float]:
    y = frame["y_true"].to_numpy(dtype=int)
    pred = frame["pred"].to_numpy(dtype=int)
    proba = frame["probability_IBD"].to_numpy(dtype=float)
    if len(np.unique(y)) != 2:
        raise RuntimeError("Metric subset does not contain both IBD and non-IBD")
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "roc_auc": float(roc_auc_score(y, proba)),
    }


def repetition_metrics_from_predictions(
    frame: pd.DataFrame,
    expected_repeats: int,
    *,
    sample_ids: Optional[set[str]] = None,
    model_name: Optional[str] = None,
    c_value: Optional[float] = None,
) -> pd.DataFrame:
    work = frame.copy()
    if sample_ids is not None:
        work = work[work["sample_id"].isin(sample_ids)].copy()
    rows: list[dict[str, Any]] = []
    expected = list(range(1, expected_repeats + 1))
    observed = sorted(work["repeat"].unique().tolist())
    if observed != expected:
        raise RuntimeError(f"Expected repeats 1..{expected_repeats}, observed {observed}")

    # The pooled OOF representation must contain exactly one held-out row per
    # sample per repetition.  This catches accidental concatenation of duplicate
    # prediction files before any metric is reported.
    duplicate = work.duplicated(["repeat", "sample_id"], keep=False)
    if duplicate.any():
        example = work.loc[duplicate, ["repeat", "sample_id", "source_file"]].head(8)
        raise RuntimeError("Duplicate OOF sample rows detected:\n" + example.to_string(index=False))

    sample_sets = []
    for repeat in expected:
        rep = work[work["repeat"].eq(repeat)].copy()
        if rep.empty:
            raise RuntimeError(f"No rows for repeat {repeat}")
        sample_sets.append(set(rep["sample_id"]))
        result = binary_metrics(rep)
        row: dict[str, Any] = {
            "repeat": repeat,
            "n_samples": len(rep),
            **result,
        }
        if model_name is not None:
            row["model"] = model_name
        if c_value is not None:
            row["C"] = float(c_value)
        rows.append(row)
    if any(s != sample_sets[0] for s in sample_sets[1:]):
        raise RuntimeError("OOF sample set changes across repetitions")
    return pd.DataFrame(rows)


def summary_metrics(frame: pd.DataFrame, id_values: dict[str, Any]) -> dict[str, Any]:
    row = dict(id_values)
    row["n_repetitions"] = int(len(frame))
    if "n_samples" in frame:
        unique_n = sorted(pd.to_numeric(frame["n_samples"]).astype(int).unique().tolist())
        row["n_samples"] = unique_n[0] if len(unique_n) == 1 else "/".join(map(str, unique_n))
    for metric in METRICS:
        values = pd.to_numeric(frame[metric], errors="raise").to_numpy(float)
        row[f"{metric}_mean"] = float(np.mean(values))
        row[f"{metric}_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return row


# -----------------------------------------------------------------------------
# Ricci OOF discovery
# -----------------------------------------------------------------------------

def iter_csv_files(root: Path) -> list[Path]:
    files = list(root.rglob("*.csv")) + list(root.rglob("*.csv.gz"))
    return sorted(set(files))


def likely_prediction_files(root: Path) -> list[Path]:
    candidates = []
    for path in iter_csv_files(root):
        name = path.name.lower()
        if any(token in name for token in ("prediction", "oof", "heldout")):
            candidates.append(path)
    return sorted(candidates)


def load_ricci_predictions(root: Path, c_values: tuple[float, ...], expected_repeats: int) -> PredictionBundle:
    requested = {c_key(v): v for v in c_values}
    usable: list[pd.DataFrame] = []
    sources: list[Path] = []
    diagnostics: list[str] = []

    for path in likely_prediction_files(root):
        cols = read_csv_header(path)
        mapping = identify_prediction_columns(cols)
        if mapping["sample_id"] is None or mapping["repeat"] is None or mapping["true"] is None:
            continue
        if find_ibd_probability_column(cols) is None:
            continue
        # Reject obvious participant-level prediction tables for the primary
        # sample-level manuscript analysis.
        if "participant" in path.name.lower() and "sample" not in path.name.lower():
            continue
        try:
            frame = pd.read_csv(path, low_memory=False)
            standard = standardise_prediction_frame(frame, path)
        except Exception as exc:
            diagnostics.append(f"{path}: {exc}")
            continue

        if standard["model"].notna().any():
            # If a model column exists and contains species models, this is not Ricci.
            known = set(standard["model"].dropna().unique())
            if known & set(MODEL_ORDER):
                continue
        if standard["C"].isna().all():
            continue
        standard["C"] = pd.to_numeric(standard["C"], errors="coerce")
        standard = standard[standard["C"].map(lambda x: c_key(x) if pd.notna(x) else None).isin(requested)].copy()
        if standard.empty:
            continue
        usable.append(standard)
        sources.append(path)

    if not usable:
        details = "\n".join(diagnostics[:10])
        raise RuntimeError(
            "Could not discover Ricci sample-level OOF prediction CSVs under "
            f"{root}. Expected columns like sample_id, repeat, y_true/true_label and proba_IBD/probability_IBD, "
            "with C either in the file or directory name.\n" + details
        )

    combined = pd.concat(usable, ignore_index=True)
    combined["C_key"] = combined["C"].map(c_key)

    # Multiple aggregate copies can exist.  For each C, choose the source file
    # that by itself forms the cleanest complete 20-repeat OOF table.
    selected_parts: list[pd.DataFrame] = []
    selected_sources: list[Path] = []
    for key, original_c in requested.items():
        cframe = combined[combined["C_key"].eq(key)].copy()
        if cframe.empty:
            raise RuntimeError(f"No Ricci OOF predictions discovered for C={original_c:g}")
        best: Optional[tuple[tuple[int, int, int], str, pd.DataFrame]] = None
        for source_name, group in cframe.groupby("source_file", sort=False):
            repeats = sorted(group["repeat"].unique().tolist())
            expected_ok = repeats == list(range(1, expected_repeats + 1))
            duplicate_count = int(group.duplicated(["repeat", "sample_id"]).sum())
            n_unique = int(group[["repeat", "sample_id"]].drop_duplicates().shape[0])
            score = (1 if expected_ok else 0, -duplicate_count, n_unique)
            if best is None or score > best[0]:
                best = (score, source_name, group)
        assert best is not None
        chosen = best[2].drop(columns="C_key").copy()
        # Full validation also checks duplicate/sample-set consistency.
        repetition_metrics_from_predictions(chosen, expected_repeats, c_value=original_c)
        selected_parts.append(chosen)
        selected_sources.append(Path(best[1]))

    final = pd.concat(selected_parts, ignore_index=True)
    require_finite_unit_interval(final, ["probability_IBD"], "Ricci predictions")
    return PredictionBundle(final, selected_sources)


# -----------------------------------------------------------------------------
# Species benchmark recovery
# -----------------------------------------------------------------------------

def load_species_complete_case_ids(species_file: Path) -> set[str]:
    if not species_file.exists():
        raise FileNotFoundError(species_file)
    frame = pd.read_excel(species_file)
    sample_col = first_existing(frame.columns, ["External ID", "sample_id", "external_id"])
    if sample_col is None:
        raise RuntimeError(f"Could not find sample-ID column in {species_file}")
    feature_cols = [c for c in frame.columns if c != sample_col]
    numeric = frame[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    work = pd.concat([frame[[sample_col]].copy(), numeric], axis=1)
    work[sample_col] = work[sample_col].map(normalise_sample_id)
    # The production benchmark aggregates duplicate sample rows before applying
    # the all-zero rule.  Whether mean or sum is used does not change all-zero
    # status, so max-absolute is used here purely to recover inclusion IDs.
    grouped = work.groupby(sample_col, sort=False)[feature_cols].max()
    keep = grouped.abs().sum(axis=1).gt(0)
    return set(grouped.index[keep].astype(str))


def discover_benchmark_prediction_bundle(root: Path, expected_repeats: int) -> Optional[PredictionBundle]:
    frames: list[pd.DataFrame] = []
    sources: list[Path] = []
    for path in likely_prediction_files(root):
        cols = read_csv_header(path)
        mapping = identify_prediction_columns(cols)
        if mapping["sample_id"] is None or mapping["repeat"] is None or mapping["true"] is None:
            continue
        if find_ibd_probability_column(cols) is None:
            continue
        if "participant" in path.name.lower() and "sample" not in path.name.lower():
            continue
        try:
            raw = pd.read_csv(path, low_memory=False)
            forced = infer_model_from_path(path) if mapping["model"] is None else None
            standard = standardise_prediction_frame(raw, path, forced_model=forced)
        except Exception:
            continue
        standard = standard[standard["model"].isin(MODEL_ORDER)].copy()
        if standard.empty:
            continue
        frames.append(standard)
        sources.append(path)
    if not frames:
        return None
    combined = pd.concat(frames, ignore_index=True)

    # Select one clean prediction source per model, avoiding duplicate aggregate
    # copies.  A combined file with a model column is allowed and can serve all
    # three models.
    parts = []
    chosen_sources: list[Path] = []
    for model in MODEL_ORDER:
        m = combined[combined["model"].eq(model)].copy()
        if m.empty:
            return None
        best = None
        for source_name, group in m.groupby("source_file", sort=False):
            repeats = sorted(group["repeat"].unique().tolist())
            expected_ok = repeats == list(range(1, expected_repeats + 1))
            duplicates = int(group.duplicated(["repeat", "sample_id"]).sum())
            score = (1 if expected_ok else 0, -duplicates, group[["repeat", "sample_id"]].drop_duplicates().shape[0])
            if best is None or score > best[0]:
                best = (score, source_name, group)
        assert best is not None
        chosen = best[2].copy()
        repetition_metrics_from_predictions(chosen, expected_repeats, model_name=model)
        parts.append(chosen)
        chosen_sources.append(Path(best[1]))
    return PredictionBundle(pd.concat(parts, ignore_index=True), chosen_sources)


def identify_metric_columns(columns: Iterable[str]) -> dict[str, Optional[str]]:
    cols = list(columns)
    return {
        "repeat": first_existing(cols, ["repeat", "repetition", "rep"]),
        "level": first_existing(cols, ["evaluation_level", "level", "aggregation_level"]),
        "model": first_existing(cols, ["model", "model_name", "classifier"]),
        "accuracy": first_existing(cols, ["accuracy", "sample_accuracy"]),
        "balanced_accuracy": first_existing(cols, ["balanced_accuracy", "balanced accuracy", "sample_balanced_accuracy"]),
        "macro_f1": first_existing(cols, ["macro_f1", "macro f1", "sample_macro_f1"]),
        "roc_auc": first_existing(cols, ["roc_auc", "roc-auc", "auc", "sample_roc_auc"]),
    }


def discover_benchmark_repetition_metrics(root: Path, expected_repeats: int) -> Optional[tuple[pd.DataFrame, list[Path]]]:
    candidates: list[tuple[int, Path, pd.DataFrame]] = []
    for path in iter_csv_files(root):
        name = path.name.lower()
        if not any(token in name for token in ("repetition", "performance", "metric", "summary")):
            continue
        cols = read_csv_header(path)
        mapping = identify_metric_columns(cols)
        if any(mapping[k] is None for k in ("repeat", "model", *METRICS)):
            continue
        try:
            frame = pd.read_csv(path, low_memory=False)
        except Exception:
            continue
        work = pd.DataFrame()
        work["repeat"] = pd.to_numeric(frame[mapping["repeat"]], errors="coerce")
        work["model"] = frame[mapping["model"]].map(normalise_model_name)
        for metric in METRICS:
            work[metric] = pd.to_numeric(frame[mapping[metric]], errors="coerce")
        if mapping["level"] is not None:
            level = frame[mapping["level"]].astype(str).str.lower()
            sample_mask = level.str.contains("sample") & ~level.str.contains("participant")
            if sample_mask.any():
                work = work[sample_mask].copy()
        work = work.dropna(subset=["repeat", "model", *METRICS]).copy()
        work["repeat"] = work["repeat"].astype(int)
        work = work[work["model"].isin(MODEL_ORDER)]
        if work.empty:
            continue
        complete_models = sum(
            sorted(work.loc[work["model"].eq(model), "repeat"].unique().tolist()) == list(range(1, expected_repeats + 1))
            for model in MODEL_ORDER
        )
        score = complete_models * 100000 + len(work)
        candidates.append((score, path, work))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    best = candidates[0]
    work = best[2]
    if all(
        sorted(work.loc[work["model"].eq(model), "repeat"].unique().tolist()) == list(range(1, expected_repeats + 1))
        for model in MODEL_ORDER
    ):
        return work.drop_duplicates(["model", "repeat"], keep="last").reset_index(drop=True), [best[1]]
    return None


def benchmark_metrics(
    benchmark_root: Path,
    species_file: Path,
    expected_repeats: int,
) -> tuple[pd.DataFrame, set[str], list[Path], str]:
    prediction_bundle = discover_benchmark_prediction_bundle(benchmark_root, expected_repeats)
    if prediction_bundle is not None:
        ids_by_model = []
        metric_parts = []
        for model in MODEL_ORDER:
            frame = prediction_bundle.frame[prediction_bundle.frame["model"].eq(model)].copy()
            ids = set(frame["sample_id"])
            ids_by_model.append(ids)
            metric_parts.append(repetition_metrics_from_predictions(frame, expected_repeats, model_name=model))
        if any(ids != ids_by_model[0] for ids in ids_by_model[1:]):
            raise RuntimeError("Species benchmark OOF sample set differs between benchmark models")
        metrics = pd.concat(metric_parts, ignore_index=True)
        return metrics, ids_by_model[0], prediction_bundle.source_files, "benchmark_oof_predictions"

    recovered = discover_benchmark_repetition_metrics(benchmark_root, expected_repeats)
    if recovered is None:
        raise RuntimeError(
            "Could not recover species-benchmark repetition metrics or OOF predictions. "
            "Expected a CSV with repeat/model/accuracy/balanced_accuracy/macro_f1/roc_auc columns."
        )
    metrics, sources = recovered
    complete_ids = load_species_complete_case_ids(species_file)
    return metrics, complete_ids, sources + [species_file], "benchmark_metrics_plus_species_complete_case_recovery"


# -----------------------------------------------------------------------------
# Sparsity discovery
# -----------------------------------------------------------------------------

def standardise_sparsity_columns(frame: pd.DataFrame, path: Path) -> pd.DataFrame:
    out = pd.DataFrame()
    c_col = first_existing(frame.columns, ["C", "c", "c_value"])
    if c_col is not None:
        out["C"] = pd.to_numeric(frame[c_col], errors="coerce")
    else:
        inferred = infer_c_from_text(str(path))
        out["C"] = inferred
    repeat_col = first_existing(frame.columns, ["repeat", "repetition", "rep"])
    fold_col = first_existing(frame.columns, ["fold", "outer_fold"])
    out["repeat"] = pd.to_numeric(frame[repeat_col], errors="coerce") if repeat_col else np.nan
    out["fold"] = pd.to_numeric(frame[fold_col], errors="coerce") if fold_col else np.nan

    aliases = {
        "selected_B": ["selected_B", "selected_b", "n_selected_B", "selected_presence"],
        "selected_K0": ["selected_K0", "selected_k0", "n_selected_K0", "selected_curvature"],
        "selected_total": ["selected_total", "selected_coefficients", "n_selected", "selected_features"],
        "abs_coef_mass_B": ["abs_coef_mass_B", "absolute_coefficient_mass_B", "coef_mass_B"],
        "abs_coef_mass_K0": ["abs_coef_mass_K0", "absolute_coefficient_mass_K0", "coef_mass_K0"],
    }
    found = 0
    for target, candidates in aliases.items():
        source = first_existing(frame.columns, candidates)
        if source is not None:
            out[target] = pd.to_numeric(frame[source], errors="coerce")
            found += 1
        else:
            out[target] = np.nan
    if found == 0:
        raise ValueError("no sparsity columns")
    if out["selected_total"].isna().all() and not out["selected_B"].isna().all() and not out["selected_K0"].isna().all():
        out["selected_total"] = out["selected_B"] + out["selected_K0"]
    out["source_file"] = str(path)
    return out


def discover_sparsity(root: Path, c_values: tuple[float, ...]) -> tuple[pd.DataFrame, list[Path]]:
    requested = {c_key(v): v for v in c_values}
    candidates: list[pd.DataFrame] = []
    sources: list[Path] = []
    for path in iter_csv_files(root):
        name = path.name.lower()
        if not any(token in name for token in ("metric", "spars", "fold", "diagnostic")):
            continue
        cols = read_csv_header(path)
        if not any(str(c).lower() in {"selected_b", "selected_k0", "selected_total", "selected_coefficients", "abs_coef_mass_b", "abs_coef_mass_k0"} for c in cols):
            continue
        try:
            raw = pd.read_csv(path, low_memory=False)
            std = standardise_sparsity_columns(raw, path)
        except Exception:
            continue
        std["C_key"] = std["C"].map(lambda x: c_key(x) if pd.notna(x) else None)
        std = std[std["C_key"].isin(requested)].copy()
        if std.empty:
            continue
        candidates.append(std)
        sources.append(path)

    # JSON fold metrics fallback.
    if not candidates:
        rows = []
        json_sources = []
        for path in root.rglob("*.json"):
            if not any(token in path.name.lower() for token in ("metric", "spars", "diagnostic")):
                continue
            try:
                payload = json.loads(path.read_text())
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue
            keys = {str(k).lower() for k in payload}
            if not keys & {"selected_b", "selected_k0", "selected_total", "selected_coefficients"}:
                continue
            frame = pd.DataFrame([payload])
            try:
                std = standardise_sparsity_columns(frame, path)
            except Exception:
                continue
            rows.append(std)
            json_sources.append(path)
        if rows:
            candidates = rows
            sources = json_sources

    if not candidates:
        return pd.DataFrame(), []
    combined = pd.concat(candidates, ignore_index=True)
    combined["C_key"] = combined["C"].map(lambda x: c_key(x) if pd.notna(x) else None)

    selected_parts = []
    selected_sources = []
    for key, cval in requested.items():
        cframe = combined[combined["C_key"].eq(key)].copy()
        if cframe.empty:
            continue
        # Prefer a source with B+K0 detail, then most repeat/fold rows.
        best = None
        for source_name, group in cframe.groupby("source_file", sort=False):
            detailed = int(group["selected_B"].notna().any()) + int(group["selected_K0"].notna().any())
            unique_folds = group[["repeat", "fold"]].dropna().drop_duplicates().shape[0]
            nonnull_total = int(group["selected_total"].notna().sum())
            score = (detailed, unique_folds, nonnull_total)
            if best is None or score > best[0]:
                best = (score, source_name, group)
        assert best is not None
        chosen = best[2].drop(columns="C_key").copy()
        selected_parts.append(chosen)
        selected_sources.append(Path(best[1]))
    return (pd.concat(selected_parts, ignore_index=True) if selected_parts else pd.DataFrame()), selected_sources


# -----------------------------------------------------------------------------
# Output tables
# -----------------------------------------------------------------------------

def fmt_mean_sd(mean: float, sd: float, digits: int = 3) -> str:
    if not (math.isfinite(float(mean)) and math.isfinite(float(sd))):
        return "--"
    return f"{float(mean):.{digits}f} ± {float(sd):.{digits}f}"


def fmt_mean_sd_latex(mean: float, sd: float, digits: int = 3) -> str:
    if not (math.isfinite(float(mean)) and math.isfinite(float(sd))):
        return "--"
    return f"${float(mean):.{digits}f} \\pm {float(sd):.{digits}f}$"


def write_performance_latex(frame: pd.DataFrame, path: Path, caption: str, label: str) -> None:
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Model & Accuracy & Balanced accuracy & Macro F1 & ROC-AUC \\",
        r"\midrule",
    ]
    for _, row in frame.iterrows():
        model = latex_escape(row["model"])
        values = " & ".join(
            fmt_mean_sd_latex(row[f"{metric}_mean"], row[f"{metric}_sd"])
            for metric in METRICS
        )
        lines.append(f"{model} & {values} \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    atomic_text(path, "\n".join(lines))


def build_performance_table(summary_rows: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for item in summary_rows:
        row = {"model": item["model"], "n_samples": item.get("n_samples"), "n_repetitions": item.get("n_repetitions")}
        for metric in METRICS:
            row[f"{metric}_mean"] = item[f"{metric}_mean"]
            row[f"{metric}_sd"] = item[f"{metric}_sd"]
            row[METRIC_LABELS[metric]] = fmt_mean_sd(item[f"{metric}_mean"], item[f"{metric}_sd"])
        rows.append(row)
    display_cols = ["model", "n_samples", "n_repetitions", *METRICS]
    # Retain both numeric and publication-formatted values in the CSV.
    ordered = ["model", "n_samples", "n_repetitions"]
    for metric in METRICS:
        ordered.extend([f"{metric}_mean", f"{metric}_sd", METRIC_LABELS[metric]])
    return pd.DataFrame(rows)[ordered]


def paired_difference_table(reference_metrics: pd.DataFrame, benchmark_metrics_frame: pd.DataFrame, reference_c: float) -> pd.DataFrame:
    rows = []
    ref = reference_metrics.set_index("repeat")
    for model in MODEL_ORDER:
        bench = benchmark_metrics_frame[benchmark_metrics_frame["model"].eq(model)].set_index("repeat")
        common_repeats = sorted(set(ref.index) & set(bench.index))
        if len(common_repeats) != len(ref):
            raise RuntimeError(f"Paired comparison with {model} does not have all repeats")
        for metric in ("balanced_accuracy", "macro_f1", "roc_auc"):
            delta = ref.loc[common_repeats, metric].to_numpy(float) - bench.loc[common_repeats, metric].to_numpy(float)
            rows.append({
                "comparison": f"Ricci C={reference_c:g} minus {model}",
                "metric": metric,
                "mean_delta_ricci_better_positive": float(np.mean(delta)),
                "sd_delta": float(np.std(delta, ddof=1)),
                "q025_delta": float(np.quantile(delta, 0.025)),
                "q975_delta": float(np.quantile(delta, 0.975)),
                "fraction_ricci_better": float(np.mean(delta > 0)),
                "n_paired_repetitions": len(delta),
            })
    return pd.DataFrame(rows)


def write_paired_latex(frame: pd.DataFrame, path: Path, reference_c: float) -> None:
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        f"\\caption{{Paired repetition-level performance differences between the reference Ricci model ($C={reference_c:g}$) and species-abundance benchmarks on the common complete-case sample subset. Positive differences favour Ricci. The 2.5th and 97.5th percentiles describe the empirical distribution of the 20 paired repetition differences and are not a confidence interval fitted from independent samples.}}",
        r"\label{tab:ricci_ibd_paired_benchmarks}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"Comparison & Metric & Mean $\Delta$ & SD $\Delta$ & 2.5\% & 97.5\% & Fraction Ricci better \\",
        r"\midrule",
    ]
    for _, row in frame.iterrows():
        metric = METRIC_LABELS.get(row["metric"], row["metric"])
        lines.append(
            f"{latex_escape(row['comparison'])} & {latex_escape(metric)} & "
            f"{row['mean_delta_ricci_better_positive']:.4f} & {row['sd_delta']:.4f} & "
            f"{row['q025_delta']:.4f} & {row['q975_delta']:.4f} & {row['fraction_ricci_better']:.2f} \\")
    lines += [r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}", ""]
    atomic_text(path, "\n".join(lines))


def summarize_sparsity(sparsity: pd.DataFrame, common_ricci_metrics: pd.DataFrame, c_values: tuple[float, ...]) -> pd.DataFrame:
    rows = []
    for cval in c_values:
        key = c_key(cval)
        group = sparsity[sparsity["C"].map(lambda x: c_key(x) if pd.notna(x) else None).eq(key)].copy() if not sparsity.empty else pd.DataFrame()
        metric_group = common_ricci_metrics[common_ricci_metrics["C"].map(c_key).eq(key)]
        row: dict[str, Any] = {"C": cval}
        for column in ("selected_B", "selected_K0", "selected_total", "abs_coef_mass_B", "abs_coef_mass_K0"):
            values = pd.to_numeric(group[column], errors="coerce").dropna().to_numpy(float) if column in group else np.asarray([])
            row[f"{column}_mean"] = float(np.mean(values)) if len(values) else np.nan
            row[f"{column}_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else (0.0 if len(values) == 1 else np.nan)
        for metric in ("balanced_accuracy", "macro_f1", "roc_auc"):
            values = metric_group[metric].to_numpy(float)
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_sd"] = float(np.std(values, ddof=1))
        if math.isfinite(row["selected_K0_mean"]) and math.isfinite(row["selected_total_mean"]) and row["selected_total_mean"] > 0:
            row["selected_K0_fraction_mean_approx"] = row["selected_K0_mean"] / row["selected_total_mean"]
        else:
            row["selected_K0_fraction_mean_approx"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def write_sparsity_latex(frame: pd.DataFrame, path: Path, reference_c: float) -> None:
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Ricci $C$-path sensitivity and sparsity for IBD vs non-IBD. Predictive metrics are mean $\pm$ SD across 20 pooled OOF repetitions on the common species-complete-case subset. Selected-feature counts are mean $\pm$ SD across outer fitted models when the corresponding diagnostics are available. $C=0.02$ is the prespecified reference model.}",
        r"\label{tab:ricci_ibd_cpath_sparsity}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"$C$ & Selected B & Selected $K_0$ & Selected total & Balanced accuracy & Macro F1 & ROC-AUC \\",
        r"\midrule",
    ]
    for _, row in frame.iterrows():
        def count_cell(prefix: str) -> str:
            mean, sd = row[f"{prefix}_mean"], row[f"{prefix}_sd"]
            if not (pd.notna(mean) and pd.notna(sd)):
                return "--"
            return f"${mean:.1f} \\pm {sd:.1f}$"
        marker = r"\textbf{" + c_display(row["C"]) + "}" if float_match(row["C"], reference_c) else c_display(row["C"])
        lines.append(
            f"{marker} & {count_cell('selected_B')} & {count_cell('selected_K0')} & {count_cell('selected_total')} & "
            f"{fmt_mean_sd_latex(row['balanced_accuracy_mean'], row['balanced_accuracy_sd'])} & "
            f"{fmt_mean_sd_latex(row['macro_f1_mean'], row['macro_f1_sd'])} & "
            f"{fmt_mean_sd_latex(row['roc_auc_mean'], row['roc_auc_sd'])} \\")
    lines += [r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}", ""]
    atomic_text(path, "\n".join(lines))


# -----------------------------------------------------------------------------
# Figures
# -----------------------------------------------------------------------------

def plot_metric_bars(comparison_table: pd.DataFrame, metric: str, reference_c: float, out_dir: Path) -> None:
    labels = comparison_table["model"].tolist()
    means = comparison_table[f"{metric}_mean"].to_numpy(float)
    sds = comparison_table[f"{metric}_sd"].to_numpy(float)
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(11.0, 6.2))
    # Two calls intentionally use Matplotlib's default colour cycle, avoiding
    # hand-picked colours while keeping benchmark and Ricci families distinct.
    benchmark_bars = ax.bar(x[:3], means[:3], yerr=sds[:3], capsize=4, label="Species abundance benchmark")
    ricci_bars = ax.bar(x[3:], means[3:], yerr=sds[3:], capsize=4, label="Ricci [B | K0]")

    ref_label = f"Ricci C={reference_c:g}"
    if ref_label in labels:
        index = labels.index(ref_label) - 3
        if 0 <= index < len(ricci_bars):
            ricci_bars[index].set_hatch("///")
            ricci_bars[index].set_linewidth(2.0)
            ricci_bars[index].set_label(f"Reference Ricci C={reference_c:g}")

    ax.set_ylabel(METRIC_LABELS[metric])
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=32, ha="right")
    ax.set_title(f"IBD vs non-IBD: {METRIC_LABELS[metric]} across species benchmarks and Ricci C path")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    stem = f"figure_ibd_benchmark_ricci_{metric}"
    fig.savefig(out_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_consensus_confusion(matrix: np.ndarray, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(5.4, 4.7))
    image = ax.imshow(matrix)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    labels = ["non-IBD", "IBD"]
    ax.set_xticks([0, 1]); ax.set_xticklabels(labels)
    ax.set_yticks([0, 1]); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(title)
    threshold = matrix.max() / 2.0 if matrix.size else 0
    for i in range(2):
        for j in range(2):
            # Use default text colour selection; no fixed palette is specified.
            ax.text(j, i, f"{int(matrix[i, j])}", ha="center", va="center")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Consensus confusion
# -----------------------------------------------------------------------------

def consensus_predictions(frame: pd.DataFrame, sample_ids: Optional[set[str]] = None) -> pd.DataFrame:
    work = frame.copy()
    if sample_ids is not None:
        work = work[work["sample_id"].isin(sample_ids)].copy()
    if work.groupby("sample_id")["y_true"].nunique().gt(1).any():
        raise RuntimeError("True labels change across repeats for the same sample")
    grouped = work.groupby("sample_id", as_index=False).agg(
        y_true=("y_true", "first"),
        mean_probability_IBD=("probability_IBD", "mean"),
        n_repetitions=("repeat", "nunique"),
    )
    grouped["pred"] = (grouped["mean_probability_IBD"] >= 0.5).astype(int)
    return grouped


def confusion_matrix_binary(consensus: pd.DataFrame) -> np.ndarray:
    matrix = np.zeros((2, 2), dtype=int)
    for truth, pred in consensus[["y_true", "pred"]].itertuples(index=False):
        matrix[int(truth), int(pred)] += 1
    return matrix


def write_confusion_outputs(consensus: pd.DataFrame, csv_path: Path, tex_path: Path, caption: str, label: str) -> np.ndarray:
    matrix = confusion_matrix_binary(consensus)
    table = pd.DataFrame(
        {
            "true_class": ["non-IBD", "IBD"],
            "predicted_nonIBD": matrix[:, 0],
            "predicted_IBD": matrix[:, 1],
        }
    )
    atomic_csv(table, csv_path)
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r" & Predicted non-IBD & Predicted IBD \\",
        r"\midrule",
        f"True non-IBD & {matrix[0,0]} & {matrix[0,1]} \\",
        f"True IBD & {matrix[1,0]} & {matrix[1,1]} \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    atomic_text(tex_path, "\n".join(lines))
    return matrix


# -----------------------------------------------------------------------------
# Main build
# -----------------------------------------------------------------------------

def build_outputs(args: argparse.Namespace) -> Path:
    ricci_root = args.ricci_dir.expanduser().resolve()
    benchmark_root = args.benchmark_dir.expanduser().resolve()
    species_file = args.species_file.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    c_values = tuple(float(x) for x in args.c_values)
    reference_c = float(args.reference_c)
    if not any(float_match(reference_c, c) for c in c_values):
        raise ValueError("reference C must be included in --c-values")

    print("=" * 100)
    print("RICCI IBD C-PATH MANUSCRIPT OUTPUTS")
    print("=" * 100)
    print(f"Ricci root:     {ricci_root}")
    print(f"Benchmark root: {benchmark_root}")
    print(f"Output:         {out_dir}")
    print(f"C values:       {', '.join(c_display(c) for c in c_values)}")
    print(f"Reference C:    {reference_c:g}")

    ricci_bundle = load_ricci_predictions(ricci_root, c_values, args.expected_repeats)
    ricci = ricci_bundle.frame.copy()
    print("[OK] Ricci OOF predictions discovered:")
    for source in ricci_bundle.source_files:
        print(f"     {source}")

    benchmark_rep, benchmark_ids, benchmark_sources, benchmark_mode = benchmark_metrics(
        benchmark_root, species_file, args.expected_repeats
    )
    print(f"[OK] Species benchmark repetitions recovered via {benchmark_mode}")
    print(f"[OK] Species benchmark complete-case sample IDs: {len(benchmark_ids)}")

    # Full Ricci metrics and exact full sample counts.
    full_ricci_parts = []
    full_summary_rows = []
    full_sets: dict[float, set[str]] = {}
    for cval in c_values:
        cframe = ricci[ricci["C"].map(c_key).eq(c_key(cval))].copy()
        full_sets[cval] = set(cframe["sample_id"])
        rep = repetition_metrics_from_predictions(cframe, args.expected_repeats, model_name=f"Ricci C={cval:g}", c_value=cval)
        full_ricci_parts.append(rep)
        full_summary_rows.append(summary_metrics(rep, {"model": f"Ricci C={cval:g}", "C": cval}))
    full_ricci_rep = pd.concat(full_ricci_parts, ignore_index=True)
    atomic_csv(full_ricci_rep, out_dir / "ricci_full_cohort_repetition_metrics.csv")
    full_table = build_performance_table(full_summary_rows)
    atomic_csv(full_table, out_dir / "table_ricci_full_cohort_cpath.csv")
    write_performance_latex(
        full_table,
        out_dir / "table_ricci_full_cohort_cpath.tex",
        "IBD vs non-IBD Ricci $C$-path performance on the full Ricci cohort. Values are mean $\\pm$ SD across 20 pooled out-of-fold repetitions; each repetition pools five participant-grouped outer folds. $C=0.02$ is the prespecified reference model.",
        "tab:ricci_ibd_cpath_full",
    )

    # Common comparison set: exact benchmark complete-case IDs intersected with
    # every requested Ricci C so every bar uses the same held-out samples.
    common_ids = set(benchmark_ids)
    for ids in full_sets.values():
        common_ids &= ids
    if not common_ids:
        raise RuntimeError("No common samples between Ricci C path and species benchmark complete-case set")
    missing_benchmark_in_ricci = len(benchmark_ids - common_ids)
    if args.require_exact_benchmark_sample_set and missing_benchmark_in_ricci:
        examples = sorted(benchmark_ids - common_ids)[:15]
        raise RuntimeError(
            f"Ricci is missing {missing_benchmark_in_ricci} species-benchmark complete-case samples; examples={examples}. "
            "Refusing to create a nominally paired comparison."
        )
    print(f"[OK] Common comparison sample set: {len(common_ids)}")
    if missing_benchmark_in_ricci:
        print(f"[WARN] {missing_benchmark_in_ricci} benchmark complete-case samples are absent from at least one Ricci C; using intersection")

    # Validate benchmark metrics themselves.
    require_finite_unit_interval(benchmark_rep, METRICS, "Species benchmark repetition metrics")
    for model in MODEL_ORDER:
        observed = sorted(benchmark_rep.loc[benchmark_rep["model"].eq(model), "repeat"].unique().tolist())
        if observed != list(range(1, args.expected_repeats + 1)):
            raise RuntimeError(f"Species benchmark {model} repeats are incomplete: {observed}")

    common_ricci_parts = []
    common_ricci_summary = []
    for cval in c_values:
        cframe = ricci[ricci["C"].map(c_key).eq(c_key(cval))].copy()
        rep = repetition_metrics_from_predictions(
            cframe,
            args.expected_repeats,
            sample_ids=common_ids,
            model_name=f"Ricci C={cval:g}",
            c_value=cval,
        )
        common_ricci_parts.append(rep)
        common_ricci_summary.append(summary_metrics(rep, {"model": f"Ricci C={cval:g}", "C": cval}))
    common_ricci_rep = pd.concat(common_ricci_parts, ignore_index=True)
    atomic_csv(common_ricci_rep, out_dir / "ricci_common_subset_repetition_metrics.csv")

    benchmark_summary = []
    for model in MODEL_ORDER:
        rep = benchmark_rep[benchmark_rep["model"].eq(model)].sort_values("repeat")
        row = summary_metrics(rep, {"model": model})
        # The benchmark metrics are already calculated on the complete-case set.
        # n_samples is filled explicitly from the recovered benchmark ID set.
        row["n_samples"] = len(benchmark_ids)
        benchmark_summary.append(row)

    comparison_rows = benchmark_summary + common_ricci_summary
    comparison_table = build_performance_table(comparison_rows)
    atomic_csv(comparison_table, out_dir / "table_benchmark_comparison_common_subset.csv")
    write_performance_latex(
        comparison_table,
        out_dir / "table_benchmark_comparison_common_subset.tex",
        f"IBD vs non-IBD comparison of species-abundance benchmarks and the Ricci $C$ path on the common complete-case evaluation subset ($n={len(common_ids)}$). Values are mean $\\pm$ SD across 20 pooled OOF repetitions. $C=0.02$ is the prespecified reference Ricci model; the remaining $C$ values are sensitivity analyses.",
        "tab:ricci_ibd_benchmark_comparison",
    )

    # Three standalone bar figures, one per requested headline metric.  This is
    # deliberately not a multi-panel plotting object, which keeps each panel
    # independently reusable in the manuscript/Overleaf layout.
    for metric in FIGURE_METRICS:
        plot_metric_bars(comparison_table, metric, reference_c, out_dir)

    # Overleaf wrapper that assembles the three standalone plots into the
    # agreed three-panel main-text figure without baking them into one raster.
    threepanel = r"""\begin{figure}[ht]
\centering
\begin{minipage}[t]{0.32\textwidth}
\centering
\includegraphics[width=\linewidth]{figure_ibd_benchmark_ricci_balanced_accuracy.pdf}\\
\small (a) Balanced accuracy
\end{minipage}
\hfill
\begin{minipage}[t]{0.32\textwidth}
\centering
\includegraphics[width=\linewidth]{figure_ibd_benchmark_ricci_macro_f1.pdf}\\
\small (b) Macro F1
\end{minipage}
\hfill
\begin{minipage}[t]{0.32\textwidth}
\centering
\includegraphics[width=\linewidth]{figure_ibd_benchmark_ricci_roc_auc.pdf}\\
\small (c) ROC-AUC
\end{minipage}
\caption{IBD vs non-IBD performance of species-abundance benchmarks and the Ricci $C$ path on the common complete-case evaluation subset. Bars show the mean across 20 pooled five-fold OOF repetitions and error bars show one standard deviation. $C=0.02$ is the prespecified reference Ricci model; the other $C$ values are sensitivity analyses.}
\label{fig:ricci_ibd_benchmark_cpath}
\end{figure}
"""
    atomic_text(out_dir / "figure_ibd_benchmark_ricci_threepanel.tex", threepanel)

    # Paired repetition differences at the reference C.
    ref_common = common_ricci_rep[common_ricci_rep["C"].map(c_key).eq(c_key(reference_c))].copy()
    paired = paired_difference_table(ref_common, benchmark_rep, reference_c)
    atomic_csv(paired, out_dir / "table_reference_c_paired_differences.csv")
    write_paired_latex(paired, out_dir / "table_reference_c_paired_differences.tex", reference_c)

    # Reference-C repetition-level appendix table.
    ref_common.sort_values("repeat").to_csv(out_dir / "reference_c_repetition_metrics_common_subset.csv", index=False)
    ref_full = full_ricci_rep[full_ricci_rep["C"].map(c_key).eq(c_key(reference_c))].sort_values("repeat")
    ref_full.to_csv(out_dir / "reference_c_repetition_metrics_full_cohort.csv", index=False)

    # Sparsity/sensitivity table.
    sparsity, sparsity_sources = discover_sparsity(ricci_root, c_values)
    sparsity_table = summarize_sparsity(sparsity, common_ricci_rep, c_values)
    atomic_csv(sparsity_table, out_dir / "table_c_sparsity.csv")
    write_sparsity_latex(sparsity_table, out_dir / "table_c_sparsity.tex", reference_c)
    if sparsity.empty:
        print("[WARN] No detailed selected-B/selected-K0 fold diagnostics were discoverable; sparsity count cells are left blank.")
    else:
        print(f"[OK] Sparsity diagnostics recovered from {len(sparsity_sources)} source file(s)")

    # Consensus confusion matrices: primary full Ricci cohort and common subset.
    ref_predictions = ricci[ricci["C"].map(c_key).eq(c_key(reference_c))].copy()
    consensus_full = consensus_predictions(ref_predictions)
    atomic_csv(consensus_full, out_dir / "reference_c_consensus_predictions_full_cohort.csv")
    full_matrix = write_confusion_outputs(
        consensus_full,
        out_dir / "reference_c_consensus_confusion_full_cohort.csv",
        out_dir / "reference_c_consensus_confusion_full_cohort.tex",
        "Consensus held-out sample confusion matrix for the reference Ricci IBD vs non-IBD classifier ($C=0.02$). Each sample's OOF IBD probability is averaged across 20 repetitions, so every sample contributes once. Rows denote true labels and columns denote predicted labels.",
        "tab:ricci_ibd_consensus_confusion",
    )
    plot_consensus_confusion(
        full_matrix,
        out_dir / "reference_c_consensus_confusion_full_cohort.png",
        "Ricci C=0.02 consensus confusion matrix",
    )

    consensus_common = consensus_predictions(ref_predictions, common_ids)
    atomic_csv(consensus_common, out_dir / "reference_c_consensus_predictions_common_subset.csv")
    common_matrix = write_confusion_outputs(
        consensus_common,
        out_dir / "reference_c_consensus_confusion_common_subset.csv",
        out_dir / "reference_c_consensus_confusion_common_subset.tex",
        "Consensus held-out sample confusion matrix for the reference Ricci classifier ($C=0.02$) restricted to the common species-complete-case comparison subset. Each sample contributes once after averaging its OOF probability across 20 repetitions.",
        "tab:ricci_ibd_consensus_confusion_common",
    )

    # A tidy long-form comparison file is useful for any later re-plotting.
    tidy_rows = []
    for _, row in comparison_table.iterrows():
        family = "Species abundance" if row["model"].startswith("Species") else "Ricci"
        for metric in METRICS:
            tidy_rows.append({
                "model": row["model"],
                "family": family,
                "metric": metric,
                "metric_label": METRIC_LABELS[metric],
                "mean": row[f"{metric}_mean"],
                "sd": row[f"{metric}_sd"],
                "n_repetitions": row["n_repetitions"],
                "n_samples": row["n_samples"],
                "reference_model": row["model"] == f"Ricci C={reference_c:g}",
            })
    atomic_csv(pd.DataFrame(tidy_rows), out_dir / "benchmark_ricci_comparison_tidy.csv")
    atomic_csv(pd.DataFrame({"sample_id": sorted(common_ids)}), out_dir / "common_comparison_sample_ids.csv")

    # Provenance with hashes of the exact files actually used.
    used_files = sorted(set(ricci_bundle.source_files + benchmark_sources + sparsity_sources))
    provenance = {
        "created_utc": now_iso(),
        "script_version": VERSION,
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "ricci_root": str(ricci_root),
        "benchmark_root": str(benchmark_root),
        "species_file": str(species_file),
        "output_dir": str(out_dir),
        "c_values": list(c_values),
        "reference_c": reference_c,
        "expected_repeats": args.expected_repeats,
        "uncertainty_unit": "SD across pooled five-fold OOF repetition-level estimates",
        "comparison_rule": "Ricci OOF predictions filtered to exact benchmark complete-case sample IDs; no refitting",
        "benchmark_recovery_mode": benchmark_mode,
        "benchmark_complete_case_n": len(benchmark_ids),
        "common_comparison_n": len(common_ids),
        "ricci_full_n_by_c": {c_display(c): len(full_sets[c]) for c in c_values},
        "reference_consensus_full_n": len(consensus_full),
        "reference_consensus_common_n": len(consensus_common),
        "input_files": [
            {"path": str(path), "sha256": sha256_file(path) if path.exists() and path.is_file() else None}
            for path in used_files
        ],
    }
    atomic_json(out_dir / "provenance.json", provenance)

    manifest_rows = []
    for path in sorted(out_dir.iterdir()):
        if path.is_file():
            manifest_rows.append({"file": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    atomic_csv(pd.DataFrame(manifest_rows), out_dir / "MANIFEST.sha256.csv")

    readme = f"""IBD vs non-IBD Ricci C-path manuscript outputs
================================================

Reference model
---------------
C = {reference_c:g} is the prespecified manuscript/reference Ricci model.  The
other C values are sensitivity analyses and must not be described as if the
best-performing C had been selected after observing the test results.

Evaluation unit
---------------
Means and SDs are calculated across {args.expected_repeats} repetition-level
estimates.  Each repetition pools the five participant-grouped outer held-out
folds before a metric is calculated.

Species benchmark comparison
----------------------------
The species benchmark complete-case set contains {len(benchmark_ids)} recovered
samples.  The exact comparison intersection used by every bar is n={len(common_ids)}.
Ricci predictions are not refit for this comparison: existing held-out OOF rows
are simply restricted to the common IDs and metrics are recomputed.

Recommended main-text files
---------------------------
1. table_benchmark_comparison_common_subset.tex
2. figure_ibd_benchmark_ricci_balanced_accuracy.pdf
3. figure_ibd_benchmark_ricci_macro_f1.pdf
4. figure_ibd_benchmark_ricci_roc_auc.pdf
5. figure_ibd_benchmark_ricci_threepanel.tex (Overleaf wrapper for 2--4)
6. table_reference_c_paired_differences.tex
7. table_c_sparsity.tex
8. reference_c_consensus_confusion_full_cohort.tex

The full-cohort Ricci-only C-path table is in table_ricci_full_cohort_cpath.tex.
Repetition-level CSVs are retained for the appendix/supplement and provenance.json
records exact input hashes.
"""
    atomic_text(out_dir / "README.txt", readme)

    print("=" * 100)
    print("MANUSCRIPT OUTPUT BUILD COMPLETE")
    print(f"Common comparison n: {len(common_ids)}")
    print(f"Reference full-cohort consensus n: {len(consensus_full)}")
    print(f"Outputs: {out_dir}")
    print("=" * 100)
    return out_dir


# -----------------------------------------------------------------------------
# Synthetic self-test
# -----------------------------------------------------------------------------

def make_selftest_fixture(root: Path, repeats: int = 20) -> tuple[Path, Path, Path, Path]:
    rng = np.random.default_rng(20260813)
    ricci_root = root / "ricci"
    benchmark_root = root / "benchmark"
    out = root / "out"
    ricci_root.mkdir(); benchmark_root.mkdir()

    n_full = 60
    sample_ids = np.asarray([f"S{i:03d}" for i in range(n_full)])
    y = np.asarray([0] * 20 + [1] * 40, dtype=int)
    benchmark_ids = set(sample_ids[:-3])

    for c in DEFAULT_C_VALUES:
        croot = ricci_root / c_tag(c)
        croot.mkdir()
        rows = []
        sparsity_rows = []
        signal = 0.65 + 0.03 * math.exp(-abs(math.log(c / 0.02)))
        for repeat in range(1, repeats + 1):
            noise = rng.normal(0, 0.12, size=n_full)
            p = np.clip((1 - signal) + (2 * signal - 1) * y + noise, 0.01, 0.99)
            pred = (p >= 0.5).astype(int)
            for i, sid in enumerate(sample_ids):
                rows.append({
                    "sample_id": sid,
                    "repeat": repeat,
                    "fold": (i % 5) + 1,
                    "y_true": y[i],
                    "pred": pred[i],
                    "proba_IBD": p[i],
                    "C": c,
                })
            for fold in range(1, 6):
                base = int(20 + c * 1000)
                sparsity_rows.append({
                    "C": c,
                    "repeat": repeat,
                    "fold": fold,
                    "selected_B": base + fold,
                    "selected_K0": base * 2 + fold,
                    "selected_total": base * 3 + 2 * fold,
                    "abs_coef_mass_B": 1.0 + c,
                    "abs_coef_mass_K0": 2.0 + c,
                })
        pd.DataFrame(rows).to_csv(croot / "all_oof_sample_predictions.csv", index=False)
        pd.DataFrame(sparsity_rows).to_csv(croot / "all_outer_fold_metrics.csv", index=False)

    # Benchmark prediction file: exact common sample set, three models.
    bench_rows = []
    model_offsets = {
        "random_forest": 0.64,
        "xgboost": 0.62,
        "logistic_l1": 0.60,
    }
    keep_mask = np.asarray([sid in benchmark_ids for sid in sample_ids])
    for repeat in range(1, repeats + 1):
        for model, signal in model_offsets.items():
            noise = rng.normal(0, 0.13, size=n_full)
            p = np.clip((1 - signal) + (2 * signal - 1) * y + noise, 0.01, 0.99)
            pred = (p >= 0.5).astype(int)
            for i, sid in enumerate(sample_ids):
                if not keep_mask[i]:
                    continue
                bench_rows.append({
                    "sample_id": sid,
                    "repeat": repeat,
                    "fold": (i % 5) + 1,
                    "y_true": y[i],
                    "pred": pred[i],
                    "probability_IBD": p[i],
                    "model": model,
                })
    pd.DataFrame(bench_rows).to_csv(benchmark_root / "all_oof_sample_predictions.csv", index=False)

    # Minimal species workbook fallback, though self-test should recover IDs from OOF predictions.
    species = pd.DataFrame({
        "External ID": sample_ids,
        "Species_A": [1.0 if sid in benchmark_ids else 0.0 for sid in sample_ids],
        "Species_B": [0.5 if sid in benchmark_ids else 0.0 for sid in sample_ids],
    })
    species_file = root / "species.xlsx"
    species.to_excel(species_file, index=False)
    return ricci_root, benchmark_root, species_file, out


def run_self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="ricci_ibd_manuscript_selftest_") as td:
        root = Path(td)
        ricci, benchmark, species, out = make_selftest_fixture(root)
        args = argparse.Namespace(
            ricci_dir=ricci,
            benchmark_dir=benchmark,
            species_file=species,
            out_dir=out,
            c_values=DEFAULT_C_VALUES,
            reference_c=DEFAULT_REFERENCE_C,
            expected_repeats=DEFAULT_REPEATS,
            require_exact_benchmark_sample_set=True,
        )
        build_outputs(args)
        required = [
            "table_benchmark_comparison_common_subset.csv",
            "table_benchmark_comparison_common_subset.tex",
            "figure_ibd_benchmark_ricci_balanced_accuracy.png",
            "figure_ibd_benchmark_ricci_macro_f1.pdf",
            "figure_ibd_benchmark_ricci_roc_auc.pdf",
            "figure_ibd_benchmark_ricci_threepanel.tex",
            "table_reference_c_paired_differences.csv",
            "table_c_sparsity.csv",
            "reference_c_consensus_confusion_full_cohort.csv",
            "reference_c_repetition_metrics_common_subset.csv",
            "provenance.json",
            "README.txt",
        ]
        missing = [name for name in required if not (out / name).exists()]
        if missing:
            raise AssertionError(f"Self-test missing outputs: {missing}")
        comparison = pd.read_csv(out / "table_benchmark_comparison_common_subset.csv")
        if len(comparison) != 8:
            raise AssertionError(f"Expected 8 comparison rows, found {len(comparison)}")
        common = pd.read_csv(out / "common_comparison_sample_ids.csv")
        if len(common) != 57:
            raise AssertionError(f"Expected 57 common samples, found {len(common)}")
        paired = pd.read_csv(out / "table_reference_c_paired_differences.csv")
        if len(paired) != 9:
            raise AssertionError(f"Expected 9 paired rows, found {len(paired)}")
        print("SELF-TEST PASSED: C discovery, 20-repeat OOF validation, common-subset filtering, benchmark alignment, paired deltas, sparsity summaries, consensus confusion, LaTeX/CSV tables, and standalone bar figures all succeeded.")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    base = Path.home() / "Real_Data"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ricci-dir", type=Path, default=base / "Ricci_IBD_RepeatedCV_CPath")
    parser.add_argument("--benchmark-dir", type=Path, default=base / "Species_Benchmarks_RepeatedCV_IBD_CompleteCase")
    parser.add_argument("--species-file", type=Path, default=base / "Real_Species_Abundances_canon.xlsx")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=base / "Ricci_IBD_RepeatedCV_CPath" / "manuscript_outputs_cpath",
    )
    parser.add_argument("--c-values", type=parse_float_list, default=DEFAULT_C_VALUES)
    parser.add_argument("--reference-c", type=float, default=DEFAULT_REFERENCE_C)
    parser.add_argument("--expected-repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument(
        "--allow-smaller-intersection",
        action="store_true",
        help="Allow the common comparison set to be smaller than the full species-benchmark complete-case set.",
    )
    parser.add_argument("--self-test", action="store_true")
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.self_test:
        run_self_test()
        return
    args.require_exact_benchmark_sample_set = not args.allow_smaller_intersection
    if args.expected_repeats < 2:
        raise ValueError("expected repeats must be at least 2")
    build_outputs(args)


if __name__ == "__main__":
    main()
