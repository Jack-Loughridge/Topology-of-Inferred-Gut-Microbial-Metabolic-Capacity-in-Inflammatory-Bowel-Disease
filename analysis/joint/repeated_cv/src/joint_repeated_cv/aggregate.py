from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import RunConfig
from .core import EXPECTED_CORE_OUTPUTS, core_repeat_complete, repeat_dir
from .metrics import (
    classification_metrics,
    participant_average_predictions,
    standardise_prediction_frame,
    summary_across_repetitions,
)
from .tasks import TASK_BY_FOLDER, TASK_SPECS, normalise_label, ordered_classes
from .util import atomic_csv, atomic_json, normalise_sample_id, now_iso


def _prediction_id_column(frame: pd.DataFrame) -> str:
    column = next((name for name in ("sample_id", "sample", "External ID") if name in frame.columns), None)
    if column is None:
        raise ValueError("Core prediction file lacks a sample-ID column.")
    return column


def _numeric_sequence_equal(observed: Any, expected: tuple[float, ...] | tuple[int, ...]) -> bool:
    try:
        observed_values = np.asarray(list(observed), dtype=float)
        expected_values = np.asarray(expected, dtype=float)
    except (TypeError, ValueError):
        return False
    return observed_values.shape == expected_values.shape and np.allclose(
        observed_values, expected_values, rtol=0, atol=1e-12
    )


def verify_core_run_config(
    config: RunConfig, task_folder: str, repeat: int, split_seed: int
) -> dict[str, Any]:
    path = repeat_dir(config, task_folder, repeat) / "RUN_CONFIG.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    exact_expected = {
        "tasks": task_folder,
        "n_outer_splits": config.expected_folds,
        "n_inner_splits": config.n_inner_splits,
        "seed": split_seed,
        "selection_metric": config.selection_metric,
        "logistic_max_iter": config.logistic_max_iter,
        "n_alternations": config.alternations,
        "quad_points": config.quad_points,
        "active_set_initial": config.active_set_initial,
        "active_set_batch": config.active_set_batch,
        "active_set_max_rounds": config.active_set_max_rounds,
    }
    mismatches: dict[str, Any] = {}
    for key, expected in exact_expected.items():
        observed = payload.get(key)
        if observed != expected:
            mismatches[key] = {"observed": observed, "expected": expected}
    floating_expected = {
        "c_value": config.c_value,
        "logistic_tolerance": config.logistic_tolerance,
        "alpha_smoothness_gamma": config.alpha_smoothness_gamma,
    }
    for key, expected in floating_expected.items():
        try:
            equal = np.isclose(float(payload.get(key)), float(expected), rtol=0, atol=1e-12)
        except (TypeError, ValueError):
            equal = False
        if not equal:
            mismatches[key] = {"observed": payload.get(key), "expected": expected}
    sequence_expected = {
        "interval_grid": config.interval_grid,
        "lambda_h0_grid": config.lambda_h0_grid,
        "lambda_ricci_grid": config.lambda_ricci_grid,
    }
    for key, expected in sequence_expected.items():
        if not _numeric_sequence_equal(payload.get(key), expected):
            mismatches[key] = {"observed": payload.get(key), "expected": list(expected)}
    if mismatches:
        raise RuntimeError(
            f"Core RUN_CONFIG.json is incompatible for {task_folder} repeat {repeat}: {mismatches}"
        )
    return payload


def verify_repeat_output(
    config: RunConfig,
    manifest: pd.DataFrame,
    task_folder: str,
    repeat: int,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    task = TASK_BY_FOLDER[task_folder]
    root = repeat_dir(config, task_folder, repeat)
    if not core_repeat_complete(config, task_folder, repeat):
        raise FileNotFoundError(
            f"Core output is not structurally complete for {task_folder} repeat {repeat}: {root}"
        )
    missing = [name for name in EXPECTED_CORE_OUTPUTS if not (root / name).exists()]
    if missing:
        raise FileNotFoundError(f"Incomplete core output for {task_folder} repeat {repeat}: {missing}")
    split_seed = int(
        manifest.loc[
            manifest["task_folder"].eq(task_folder) & manifest["repeat"].eq(repeat),
            "split_seed",
        ].iloc[0]
    )
    verify_core_run_config(config, task_folder, repeat, split_seed)
    raw = pd.read_csv(root / "ALL_TEST_PREDICTIONS.csv", low_memory=False)
    if "task_folder" in raw.columns:
        subset = raw[raw["task_folder"].astype(str).eq(task_folder)].copy()
        if not subset.empty:
            raw = subset
    id_column = _prediction_id_column(raw)
    raw["sample_id"] = raw[id_column].map(normalise_sample_id)
    if "participant_id" not in raw.columns:
        raise ValueError(f"Predictions for {task_folder} repeat {repeat} lack participant_id.")
    raw["participant_id"] = raw["participant_id"].astype(str).str.strip()
    if "fold" not in raw.columns:
        raise ValueError(f"Predictions for {task_folder} repeat {repeat} lack fold.")
    raw["fold"] = pd.to_numeric(raw["fold"], errors="raise").astype(int)
    if raw["sample_id"].duplicated().any():
        bad = raw.loc[raw["sample_id"].duplicated(False), "sample_id"].tolist()[:10]
        raise ValueError(f"Duplicate held-out predictions for {task_folder} repeat {repeat}: {bad}")
    standard, classes = standardise_prediction_frame(raw, task)
    expected = manifest[
        manifest["task_folder"].eq(task_folder)
        & manifest["repeat"].eq(repeat)
        & manifest["role"].eq("test")
    ][["fold", "sample_id", "participant_id", "label"]].drop_duplicates("sample_id")
    expected["label"] = expected["label"].map(normalise_label)
    actual_ids = set(standard["sample_id"])
    expected_ids = set(expected["sample_id"])
    if actual_ids != expected_ids:
        raise RuntimeError(
            f"Prediction IDs differ from manifest for {task_folder} repeat {repeat}: "
            f"missing={sorted(expected_ids-actual_ids)[:10]}, extra={sorted(actual_ids-expected_ids)[:10]}"
        )
    check = standard.merge(expected, on="sample_id", suffixes=("", "_manifest"), validate="one_to_one")
    if not check["participant_id"].eq(check["participant_id_manifest"]).all():
        raise RuntimeError(f"Participant IDs disagree for {task_folder} repeat {repeat}.")
    if not check["fold"].eq(check["fold_manifest"]).all():
        bad = check.loc[~check["fold"].eq(check["fold_manifest"])].head().to_dict("records")
        raise RuntimeError(f"Fold assignments disagree for {task_folder} repeat {repeat}: {bad}")
    if not check["true_label_standard"].eq(check["label"]).all():
        bad = check.loc[~check["true_label_standard"].eq(check["label"])].head().to_dict("records")
        raise RuntimeError(f"True labels disagree for {task_folder} repeat {repeat}: {bad}")
    check["split_seed"] = split_seed
    check["repeat"] = repeat
    check["task"] = task.report_name
    check["task_folder"] = task_folder
    check["task_order"] = next(
        i for i, value in enumerate(TASK_SPECS, 1) if value.folder == task_folder
    )
    leading = ["task_order", "task_folder", "task", "repeat", "split_seed"]
    check = check[leading + [column for column in check.columns if column not in leading]]
    return check, classes


def _add_run_columns(path: Path, task_folder: str, repeat: int, split_seed: int) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    if "task_folder" in frame.columns:
        subset = frame[frame["task_folder"].astype(str).eq(task_folder)].copy()
        if not subset.empty:
            frame = subset
    task = TASK_BY_FOLDER[task_folder]
    frame.insert(0, "split_seed", split_seed)
    frame.insert(0, "repeat", repeat)
    frame.insert(0, "report_task", task.report_name)
    frame.insert(0, "run_task_folder", task_folder)
    frame.insert(0, "task_order", next(i for i, value in enumerate(TASK_SPECS, 1) if value.folder == task_folder))
    return frame


def _effective_gamma(base_gamma: float, intervals: pd.Series, reference: int) -> pd.Series:
    return base_gamma * (pd.to_numeric(intervals, errors="raise") - 1.0) / (reference - 1.0)


def _state_classes(task_folder: str, beta_rows: int, prediction_classes: tuple[str, ...]) -> tuple[str, ...]:
    task = TASK_BY_FOLDER[task_folder]
    classes = ordered_classes(list(prediction_classes), task)
    if beta_rows == 1 and len(classes) == 2:
        return (classes[1],)
    if beta_rows == len(classes):
        return classes
    raise ValueError(
        f"Cannot map {beta_rows} coefficient rows to prediction classes {classes} for {task_folder}."
    )


def extract_task_coefficient_stability(
    config: RunConfig,
    task_folder: str,
    completed_repeats: list[int],
    prediction_classes: tuple[str, ...],
    output_dir: Path,
) -> dict[str, int]:
    if not completed_repeats:
        return {"fits": 0, "classes": 0, "ricci_features": 0}
    first_root = repeat_dir(config, task_folder, completed_repeats[0])
    metadata = pd.read_csv(first_root / "RICCI_FEATURE_METADATA_USED.csv", low_memory=False)
    if "feature_index" not in metadata.columns:
        metadata.insert(0, "feature_index", np.arange(len(metadata), dtype=int))
    metadata["feature_index"] = pd.to_numeric(metadata["feature_index"], errors="raise").astype(int)
    metadata = metadata.drop_duplicates("feature_index").sort_values("feature_index")
    n_features = int(metadata["feature_index"].max()) + 1
    metadata_indexed = metadata.set_index("feature_index", drop=False)
    process_column = next(
        (column for column in ("process", "process_class", "process_category", "biological_process", "category") if column in metadata.columns),
        None,
    )
    if process_column is None:
        metadata["process"] = "Unannotated"
        metadata_indexed = metadata.set_index("feature_index", drop=False)
        process_column = "process"
    block_column = next(
        (column for column in ("feature_type", "kind", "type", "block") if column in metadata.columns),
        None,
    )

    class_accumulators: dict[str, dict[str, Any]] = {}
    process_fit_rows: list[dict[str, Any]] = []
    h0_rows: list[dict[str, Any]] = []
    n_fits = 0

    for repeat in completed_repeats:
        task_root = repeat_dir(config, task_folder, repeat) / task_folder
        for fold in range(1, config.expected_folds + 1):
            fold_root = task_root / f"fold_{fold}"
            state_path = fold_root / "outer_model.state.npz"
            alpha_path = fold_root / "alpha_profile.csv"
            if not state_path.exists() or not alpha_path.exists():
                raise FileNotFoundError(
                    f"Missing coefficient state for {task_folder} repeat {repeat} fold {fold}: "
                    f"{state_path} / {alpha_path}"
                )
            with np.load(state_path, allow_pickle=False) as payload:
                beta_h0 = np.asarray(payload["beta_h0"], dtype=float)
                if beta_h0.ndim == 1:
                    beta_h0 = beta_h0[None, :]
                indices = np.asarray(payload["ricci_indices"], dtype=int)
                active = np.asarray(payload["beta_ricci_active"], dtype=float)
                if active.ndim == 1:
                    active = active[None, :]
                saved_n_ricci = int(np.asarray(payload["n_ricci"]).reshape(-1)[0])
            if saved_n_ricci != n_features:
                raise ValueError(
                    f"Ricci state has {saved_n_ricci} features but metadata has {n_features}: {state_path}"
                )
            if active.shape[1] != len(indices) or beta_h0.shape[0] != active.shape[0]:
                raise ValueError(f"Coefficient state shape mismatch: {state_path}")
            coefficient_classes = _state_classes(task_folder, beta_h0.shape[0], prediction_classes)
            alpha = pd.read_csv(alpha_path, low_memory=False)
            if not {"log_center", "alpha"}.issubset(alpha.columns):
                raise ValueError(f"Alpha profile lacks log_center/alpha: {alpha_path}")
            if len(alpha) != beta_h0.shape[1]:
                raise ValueError(f"H0 beta/alpha length mismatch: {state_path}")
            n_fits += 1

            for class_index, class_label in enumerate(coefficient_classes):
                accumulator = class_accumulators.setdefault(
                    class_label,
                    {
                        "selected_count": np.zeros(n_features, dtype=np.int32),
                        "positive_count": np.zeros(n_features, dtype=np.int32),
                        "negative_count": np.zeros(n_features, dtype=np.int32),
                        "sum_value": np.zeros(n_features, dtype=float),
                        "sum_square": np.zeros(n_features, dtype=float),
                        "sum_abs": np.zeros(n_features, dtype=float),
                        "selected_values": {},
                    },
                )
                values = active[class_index]
                nonzero = np.abs(values) > 1e-12
                selected_indices = indices[nonzero]
                selected_values = values[nonzero]
                accumulator["selected_count"][selected_indices] += 1
                accumulator["positive_count"][selected_indices] += (selected_values > 0).astype(np.int32)
                accumulator["negative_count"][selected_indices] += (selected_values < 0).astype(np.int32)
                accumulator["sum_value"][selected_indices] += selected_values
                accumulator["sum_square"][selected_indices] += selected_values * selected_values
                accumulator["sum_abs"][selected_indices] += np.abs(selected_values)
                selected_map: dict[int, list[float]] = accumulator["selected_values"]
                for feature_index, value in zip(selected_indices.tolist(), selected_values.tolist()):
                    selected_map.setdefault(int(feature_index), []).append(float(value))

                if len(selected_indices):
                    selected_metadata = metadata_indexed.loc[selected_indices].copy()
                    if isinstance(selected_metadata, pd.Series):
                        selected_metadata = selected_metadata.to_frame().T
                    selected_metadata["coefficient"] = selected_values
                    selected_metadata["absolute_coefficient"] = np.abs(selected_values)
                    selected_metadata["process_value"] = selected_metadata[process_column].fillna("Unannotated").astype(str)
                    selected_metadata["block_value"] = (
                        selected_metadata[block_column].fillna("unknown").astype(str)
                        if block_column is not None else "unknown"
                    )
                    grouped = selected_metadata.groupby(["process_value", "block_value"], as_index=False).agg(
                        selected_features=("feature_index", "count"),
                        total_abs_coefficient=("absolute_coefficient", "sum"),
                        mean_signed_coefficient_selected=("coefficient", "mean"),
                        positive_selected=("coefficient", lambda x: int((x > 0).sum())),
                        negative_selected=("coefficient", lambda x: int((x < 0).sum())),
                    )
                    for row in grouped.to_dict("records"):
                        process_fit_rows.append(
                            {
                                "task_folder": task_folder,
                                "repeat": repeat,
                                "fold": fold,
                                "class": class_label,
                                "process": row["process_value"],
                                "feature_block": row["block_value"],
                                **{key: value for key, value in row.items() if key not in {"process_value", "block_value"}},
                            }
                        )

                for interval_index, alpha_row in alpha.reset_index(drop=True).iterrows():
                    beta = float(beta_h0[class_index, interval_index])
                    alpha_value = float(alpha_row["alpha"])
                    log_center = float(alpha_row["log_center"])
                    h0_rows.append(
                        {
                            "task_folder": task_folder,
                            "repeat": repeat,
                            "fold": fold,
                            "class": class_label,
                            "interval_index": interval_index,
                            "log_center": log_center,
                            "death_center": float(alpha_row.get("death_center", math.exp(log_center))),
                            "alpha": alpha_value,
                            "beta_h0": beta,
                            "alpha_beta": alpha_value * beta,
                            "selected": int(abs(beta) > 1e-12),
                        }
                    )

    coefficient_parts: list[pd.DataFrame] = []
    for class_label, accumulator in class_accumulators.items():
        rows = []
        for feature_index in range(n_features):
            count = int(accumulator["selected_count"][feature_index])
            mean = float(accumulator["sum_value"][feature_index] / n_fits)
            variance = max(float(accumulator["sum_square"][feature_index] / n_fits - mean * mean), 0.0)
            selected_values = np.asarray(accumulator["selected_values"].get(feature_index, []), dtype=float)
            row: dict[str, Any] = {
                "task_folder": task_folder,
                "class": class_label,
                "feature_index": feature_index,
                "n_fits": n_fits,
                "selected_count": count,
                "selection_frequency": count / n_fits,
                "positive_count": int(accumulator["positive_count"][feature_index]),
                "negative_count": int(accumulator["negative_count"][feature_index]),
                "dominant_sign_consistency_selected": (
                    max(
                        int(accumulator["positive_count"][feature_index]),
                        int(accumulator["negative_count"][feature_index]),
                    ) / count if count else float("nan")
                ),
                "coefficient_mean_including_zeros": mean,
                "coefficient_sd_including_zeros": math.sqrt(variance),
                "absolute_coefficient_mean_including_zeros": float(accumulator["sum_abs"][feature_index] / n_fits),
                "coefficient_mean_selected": float(selected_values.mean()) if len(selected_values) else float("nan"),
                "coefficient_median_selected": float(np.median(selected_values)) if len(selected_values) else float("nan"),
                "coefficient_q25_selected": float(np.quantile(selected_values, 0.25)) if len(selected_values) else float("nan"),
                "coefficient_q75_selected": float(np.quantile(selected_values, 0.75)) if len(selected_values) else float("nan"),
            }
            if feature_index in metadata_indexed.index:
                annotation = metadata_indexed.loc[feature_index]
                if isinstance(annotation, pd.DataFrame):
                    annotation = annotation.iloc[0]
                for column, value in annotation.items():
                    if column != "feature_index":
                        row[column] = value
            rows.append(row)
        coefficient_parts.append(pd.DataFrame(rows))
    coefficient_summary = pd.concat(coefficient_parts, ignore_index=True)
    coefficient_summary = coefficient_summary.sort_values(
        ["class", "selection_frequency", "absolute_coefficient_mean_including_zeros"],
        ascending=[True, False, False], kind="mergesort"
    )
    atomic_csv(coefficient_summary, output_dir / "ricci_coefficient_stability.csv", index=False)

    process_fits_selected = pd.DataFrame(process_fit_rows)
    process_keys = metadata.copy()
    process_keys["process"] = process_keys[process_column].fillna("Unannotated").astype(str)
    process_keys["feature_block"] = (
        process_keys[block_column].fillna("unknown").astype(str)
        if block_column is not None else "unknown"
    )
    process_totals = process_keys.groupby(["process", "feature_block"], as_index=False).agg(
        available_features=("feature_index", "count")
    )
    fit_keys = pd.DataFrame(
        [
            {"repeat": repeat, "fold": fold}
            for repeat in completed_repeats
            for fold in range(1, config.expected_folds + 1)
        ]
    )
    class_keys = pd.DataFrame({"class": sorted(class_accumulators)})
    if not process_totals.empty and not fit_keys.empty and not class_keys.empty:
        full = (
            fit_keys.assign(_key=1)
            .merge(class_keys.assign(_key=1), on="_key")
            .merge(process_totals.assign(_key=1), on="_key")
            .drop(columns="_key")
        )
        full["task_folder"] = task_folder
        process_fits = full.merge(
            process_fits_selected,
            on=["task_folder", "repeat", "fold", "class", "process", "feature_block"],
            how="left",
            validate="one_to_one",
        )
        for column in (
            "selected_features", "total_abs_coefficient", "positive_selected", "negative_selected"
        ):
            process_fits[column] = pd.to_numeric(process_fits[column], errors="coerce").fillna(0)
        process_fits["fit_selected_process"] = process_fits["selected_features"].gt(0).astype(int)
        process_summary = process_fits.groupby(
            ["task_folder", "class", "process", "feature_block"], as_index=False
        ).agg(
            available_features=("available_features", "first"),
            n_fits=("repeat", "size"),
            selected_in_fit_frequency=("fit_selected_process", "mean"),
            selected_features_mean_including_zeros=("selected_features", "mean"),
            selected_features_sd_including_zeros=("selected_features", "std"),
            total_abs_coefficient_mean_including_zeros=("total_abs_coefficient", "mean"),
            total_abs_coefficient_sd_including_zeros=("total_abs_coefficient", "std"),
            total_abs_coefficient_median_including_zeros=("total_abs_coefficient", "median"),
            mean_signed_coefficient_selected_mean=("mean_signed_coefficient_selected", "mean"),
            positive_selected_mean_including_zeros=("positive_selected", "mean"),
            negative_selected_mean_including_zeros=("negative_selected", "mean"),
        ).sort_values(
            ["class", "total_abs_coefficient_mean_including_zeros"],
            ascending=[True, False],
            kind="mergesort",
        )
    else:
        process_fits = pd.DataFrame()
        process_summary = pd.DataFrame()
    atomic_csv(process_fits, output_dir / "ricci_process_coefficients_by_fit.csv", index=False)
    atomic_csv(process_summary, output_dir / "ricci_process_coefficient_summary.csv", index=False)

    h0_raw = pd.DataFrame(h0_rows)
    atomic_csv(h0_raw, output_dir / "h0_alpha_beta_profiles_by_fit.csv", index=False)
    h0_summary_parts = []
    if not h0_raw.empty:
        for class_label, class_frame in h0_raw.groupby("class", sort=False):
            grid = np.linspace(float(class_frame["log_center"].min()), float(class_frame["log_center"].max()), 300)
            interpolated_parts = []
            for (repeat, fold), fit in class_frame.groupby(["repeat", "fold"]):
                fit = fit.sort_values("log_center")
                x = fit["log_center"].to_numpy(float)
                death_values = np.interp(grid, x, fit["death_center"].to_numpy(float))
                for quantity in ("alpha", "beta_h0", "alpha_beta"):
                    values = np.interp(grid, x, fit[quantity].to_numpy(float))
                    interpolated_parts.append(
                        pd.DataFrame(
                            {
                                "class": class_label,
                                "repeat": repeat,
                                "fold": fold,
                                "quantity": quantity,
                                "common_log_center": grid,
                                "death_center_interpolated": death_values,
                                "value": values,
                            }
                        )
                    )
            interpolated = pd.concat(interpolated_parts, ignore_index=True)
            summary = interpolated.groupby(
                ["class", "quantity", "common_log_center"], as_index=False
            ).agg(
                death_center=("death_center_interpolated", "mean"),
                death_center_sd=("death_center_interpolated", "std"),
                n_fits=("value", "size"),
                mean=("value", "mean"),
                sd=("value", "std"),
                median=("value", "median"),
                q025=("value", lambda x: float(np.quantile(x, 0.025))),
                q25=("value", lambda x: float(np.quantile(x, 0.25))),
                q75=("value", lambda x: float(np.quantile(x, 0.75))),
                q975=("value", lambda x: float(np.quantile(x, 0.975))),
            )
            h0_summary_parts.append(summary)
    h0_summary = pd.concat(h0_summary_parts, ignore_index=True) if h0_summary_parts else pd.DataFrame()
    atomic_csv(h0_summary, output_dir / "h0_alpha_beta_curve_summary.csv", index=False)
    return {"fits": n_fits, "classes": len(class_accumulators), "ricci_features": n_features}


def aggregate_completed(
    config: RunConfig,
    manifest: pd.DataFrame,
    *,
    strict: bool = False,
    include_coefficients: bool = True,
) -> dict[str, Any]:
    aggregate_dir = Path(config.output_dir) / "aggregate"
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    prediction_parts: list[pd.DataFrame] = []
    participant_parts: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    fold_parts: list[pd.DataFrame] = []
    inner_parts: list[pd.DataFrame] = []
    selected_parts: list[pd.DataFrame] = []
    sparsity_parts: list[pd.DataFrame] = []
    alpha_parts: list[pd.DataFrame] = []
    completed_by_task: dict[str, list[int]] = {}
    classes_by_task: dict[str, tuple[str, ...]] = {}

    for task_order, task in enumerate(TASK_SPECS, start=1):
        completed: list[int] = []
        for repeat in range(1, config.expected_repeats + 1):
            if not core_repeat_complete(config, task.folder, repeat):
                if strict:
                    raise FileNotFoundError(
                        f"Incomplete task/repetition: {task.folder} repeat {repeat}"
                    )
                continue
            predictions, classes = verify_repeat_output(config, manifest, task.folder, repeat)
            completed.append(repeat)
            classes_by_task[task.folder] = classes
            prediction_parts.append(predictions)
            split_seed = int(predictions["split_seed"].iloc[0])
            root = repeat_dir(config, task.folder, repeat)
            fold_parts.append(_add_run_columns(root / "ALL_FOLD_RESULTS.csv", task.folder, repeat, split_seed))
            inner_parts.append(_add_run_columns(root / "ALL_INNER_CONFIG_RESULTS.csv", task.folder, repeat, split_seed))
            selected = _add_run_columns(root / "ALL_SELECTED_CONFIGS.csv", task.folder, repeat, split_seed)
            interval_column = next(
                (column for column in ("selected_intervals", "n_intervals") if column in selected.columns), None
            )
            if interval_column:
                selected["effective_alpha_smoothness_gamma"] = _effective_gamma(
                    config.alpha_smoothness_gamma,
                    selected[interval_column],
                    config.alpha_smoothness_reference_intervals,
                )
            selected_parts.append(selected)
            sparsity_parts.append(_add_run_columns(root / "ALL_SPARSITY_DIAGNOSTICS.csv", task.folder, repeat, split_seed))
            alpha_parts.append(_add_run_columns(root / "ALL_ALPHA_PROFILES.csv", task.folder, repeat, split_seed))

            sample_metrics = classification_metrics(predictions, classes)
            metric_rows.append(
                {
                    "task_order": task_order,
                    "task_folder": task.folder,
                    "task": task.report_name,
                    "repeat": repeat,
                    "split_seed": split_seed,
                    "level": "sample",
                    **sample_metrics,
                }
            )
            participant = participant_average_predictions(predictions, classes)
            participant.insert(0, "split_seed", split_seed)
            participant.insert(0, "repeat", repeat)
            participant.insert(0, "task", task.report_name)
            participant.insert(0, "task_folder", task.folder)
            participant.insert(0, "task_order", task_order)
            participant_parts.append(participant)
            participant_metrics = classification_metrics(participant, classes)
            metric_rows.append(
                {
                    "task_order": task_order,
                    "task_folder": task.folder,
                    "task": task.report_name,
                    "repeat": repeat,
                    "split_seed": split_seed,
                    "level": "participant",
                    **participant_metrics,
                }
            )
        completed_by_task[task.folder] = completed

    concat = lambda parts: pd.concat(parts, ignore_index=True, sort=False) if parts else pd.DataFrame()
    outputs = {
        "all_test_predictions": concat(prediction_parts),
        "participant_average_predictions": concat(participant_parts),
        "repetition_metrics": pd.DataFrame(metric_rows),
        "all_outer_fold_results": concat(fold_parts),
        "all_inner_config_results": concat(inner_parts),
        "all_selected_configs": concat(selected_parts),
        "all_sparsity_diagnostics": concat(sparsity_parts),
        "all_alpha_profiles": concat(alpha_parts),
    }
    for name, frame in outputs.items():
        atomic_csv(frame, aggregate_dir / f"{name}.csv", index=False)
    performance_summary = summary_across_repetitions(outputs["repetition_metrics"])
    atomic_csv(performance_summary, aggregate_dir / "repetition_performance_summary.csv", index=False)

    selected = outputs["all_selected_configs"]
    if not selected.empty:
        selected_interval = next(
            (column for column in ("selected_intervals", "n_intervals") if column in selected.columns), None
        )
        grouping = ["task_order", "run_task_folder", "report_task"]
        for column in (selected_interval, "lambda_h0", "lambda_ricci"):
            if column and column in selected.columns:
                grouping.append(column)
        selected_frequency = selected.groupby(grouping, dropna=False).size().rename("outer_fold_count").reset_index()
        selected_frequency["outer_fold_frequency_within_task"] = selected_frequency.groupby(
            "run_task_folder"
        )["outer_fold_count"].transform(lambda x: x / x.sum())
        selected_frequency = selected_frequency.sort_values(
            ["task_order", "outer_fold_count"], ascending=[True, False], kind="mergesort"
        )
    else:
        selected_frequency = pd.DataFrame()
    atomic_csv(selected_frequency, aggregate_dir / "selected_configuration_frequency.csv", index=False)

    coefficient_status: dict[str, Any] = {}
    if include_coefficients:
        coefficient_root = aggregate_dir / "coefficient_stability"
        for task in TASK_SPECS:
            completed = completed_by_task[task.folder]
            if not completed:
                continue
            task_output = coefficient_root / task.folder
            task_output.mkdir(parents=True, exist_ok=True)
            try:
                coefficient_status[task.folder] = extract_task_coefficient_stability(
                    config,
                    task.folder,
                    completed,
                    classes_by_task[task.folder],
                    task_output,
                )
            except Exception as exc:
                coefficient_status[task.folder] = {
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
                atomic_json(task_output / "coefficient_aggregation_error.json", coefficient_status[task.folder])
                if strict:
                    raise RuntimeError(
                        f"Coefficient aggregation failed for {task.folder}: {type(exc).__name__}: {exc}"
                    ) from exc

    progress_rows = []
    for task_order, task in enumerate(TASK_SPECS, start=1):
        completed = completed_by_task[task.folder]
        progress_rows.append(
            {
                "task_order": task_order,
                "task_folder": task.folder,
                "task": task.report_name,
                "completed_repetitions": len(completed),
                "expected_repetitions": config.expected_repeats,
                "completed_outer_folds": len(completed) * config.expected_folds,
                "expected_outer_folds": config.expected_repeats * config.expected_folds,
            }
        )
    progress_frame = pd.DataFrame(progress_rows)
    atomic_csv(progress_frame, Path(config.output_dir) / "progress_by_task.csv", index=False)
    progress = {
        "updated_utc": now_iso(),
        "task_order": [task.folder for task in TASK_SPECS],
        "completed_repetitions_by_task": completed_by_task,
        "completed_task_repetitions": int(sum(len(values) for values in completed_by_task.values())),
        "expected_task_repetitions": int(config.expected_repeats * len(TASK_SPECS)),
        "completed_outer_folds": int(sum(len(values) for values in completed_by_task.values()) * config.expected_folds),
        "expected_outer_folds": int(config.expected_repeats * config.expected_folds * len(TASK_SPECS)),
        "primary_uncertainty_unit": "pooled out-of-fold metric per repetition within task",
        "coefficient_aggregation": coefficient_status,
    }
    atomic_json(Path(config.output_dir) / "progress.json", progress)
    return progress
