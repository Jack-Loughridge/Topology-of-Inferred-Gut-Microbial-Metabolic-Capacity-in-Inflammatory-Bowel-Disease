from __future__ import annotations

import json
import math
import os
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits

from .config import RunConfig
from .data import LoadedInputs
from .metrics import classification_metrics, confusion, summary_mean_sd
from .model import (
    JointHyperparameters,
    ModelWarmStart,
    fit_joint_sparse_model_prepared,
    load_warm_start,
    model_sparsity,
    prepare_h0_split,
    prepare_ricci_split,
    save_warm_start,
)
from .outputs import save_alpha_plot, save_confusion_artifacts, write_global_latex_tables, write_manifest
from .tasks import TaskSpec, labels_for_task, mask_for_task, select_tasks
from .wkpi import fit_train_only_adaptive_intervals, interpolate_alpha, transform_many


LogFunction = Callable[[str], None]


def _default_log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def _thread_safe_logger(log: LogFunction) -> LogFunction:
    lock = threading.Lock()

    def wrapped(message: str) -> None:
        with lock:
            log(message)

    return wrapped


def _safe_split_count(y: np.ndarray, groups: np.ndarray, requested: int) -> int:
    support = [len(set(groups[y == label])) for label in np.unique(y)]
    return min([requested] + support) if support else 0


def _assert_no_group_overlap(train_indices: np.ndarray, test_indices: np.ndarray, groups: np.ndarray) -> None:
    overlap = set(groups[train_indices]) & set(groups[test_indices])
    if overlap:
        raise RuntimeError(f"Participant leakage detected: {sorted(overlap)[:10]}")


def _penalty_path(config: RunConfig) -> list[tuple[float, float]]:
    """Neighbouring strongest-to-weakest 2-D path for effective warm starts."""
    h_values = sorted((float(value) for value in config.lambda_h0_grid), reverse=True)
    r_values = sorted((float(value) for value in config.lambda_ricci_grid), reverse=True)
    path: list[tuple[float, float]] = []
    for index, lambda_h0 in enumerate(h_values):
        row = r_values if index % 2 == 0 else list(reversed(r_values))
        path.extend((lambda_h0, lambda_ricci) for lambda_ricci in row)
    return path


def _safe_float_name(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def _checkpoint_stem(n_intervals: int, lambda_h0: float, lambda_ricci: float) -> str:
    return (
        f"M{int(n_intervals):03d}_lambdaH{_safe_float_name(lambda_h0)}"
        f"_lambdaR{_safe_float_name(lambda_ricci)}"
    )


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _select_configuration(inner_rows: pd.DataFrame, metric: str) -> tuple[JointHyperparameters, pd.DataFrame]:
    metric_column = f"{metric}_mean"
    grouped = inner_rows.groupby(
        ["n_intervals", "lambda_h0", "lambda_ricci"], as_index=False
    ).agg(
        accuracy_mean=("accuracy", "mean"),
        balanced_accuracy_mean=("balanced_accuracy", "mean"),
        macro_f1_mean=("macro_f1", "mean"),
        roc_auc_mean=("roc_auc", "mean"),
        accuracy_sd=("accuracy", "std"),
        balanced_accuracy_sd=("balanced_accuracy", "std"),
        macro_f1_sd=("macro_f1", "std"),
        roc_auc_sd=("roc_auc", "std"),
    )
    if metric_column not in grouped.columns:
        raise ValueError(f"Unknown selection metric: {metric}")
    grouped["selection_score"] = grouped[metric_column].fillna(-np.inf)
    grouped = grouped.sort_values(
        ["selection_score", "balanced_accuracy_mean", "macro_f1_mean", "n_intervals", "lambda_h0", "lambda_ricci"],
        ascending=[False, False, False, True, False, False],
        kind="mergesort",
    ).reset_index(drop=True)
    if grouped.empty or not math.isfinite(float(grouped.iloc[0]["selection_score"])):
        raise RuntimeError("No finite inner-CV configuration score was available.")
    best = grouped.iloc[0]
    return JointHyperparameters(
        n_intervals=int(best["n_intervals"]),
        lambda_h0=float(best["lambda_h0"]),
        lambda_ricci=float(best["lambda_ricci"]),
    ), grouped


def _top_h0_coefficients(model, intervals, top_n: int) -> pd.DataFrame:
    rows: list[dict] = []
    for class_index, label in enumerate(model.classes):
        beta = model.effective_beta_h0[class_index]
        order = np.argsort(np.abs(beta))[::-1][:top_n]
        for rank, feature_index in enumerate(order, start=1):
            rows.append({
                "class": label,
                "rank": rank,
                "interval_index": int(feature_index),
                "log_left": float(intervals.log_edges[feature_index]),
                "log_right": float(intervals.log_edges[feature_index + 1]),
                "log_center": float(intervals.log_centers[feature_index]),
                "death_center": float(intervals.death_centers[feature_index]),
                "alpha": float(model.alpha[feature_index]),
                "beta_effective": float(beta[feature_index]),
                "absolute_beta": float(abs(beta[feature_index])),
                "alpha_beta": float(model.alpha[feature_index] * beta[feature_index]),
            })
    return pd.DataFrame(rows)


def _top_ricci_coefficients(model, feature_metadata: pd.DataFrame, top_n: int) -> pd.DataFrame:
    rows: list[dict] = []
    metadata = feature_metadata.set_index("feature_index", drop=False)
    for class_index, label in enumerate(model.classes):
        beta = model.effective_beta_ricci[class_index]
        order = np.argsort(np.abs(beta))[::-1][:top_n]
        for rank, feature_index in enumerate(order, start=1):
            row = {
                "class": label,
                "rank": rank,
                "feature_index": int(feature_index),
                "beta_effective": float(beta[feature_index]),
                "absolute_beta": float(abs(beta[feature_index])),
            }
            if feature_index in metadata.index:
                annotation = metadata.loc[feature_index]
                if isinstance(annotation, pd.DataFrame):
                    annotation = annotation.iloc[0]
                for column, value in annotation.items():
                    if column not in row:
                        row[column] = value
            rows.append(row)
    return pd.DataFrame(rows)


def _reference_comparison(
    task: TaskSpec,
    joint_fold_results: pd.DataFrame,
    config: RunConfig,
    task_out: Path,
) -> None:
    reference_path = config.ricci_original_dir / task.folder / "fold_results.csv"
    if not reference_path.exists():
        return
    reference = pd.read_csv(reference_path)
    reference["model"] = "Ricci original-style reference"
    joint = joint_fold_results.copy()
    joint["model"] = "Joint WKPI + Ricci"
    common = [
        column for column in (
            "model", "fold", "accuracy", "balanced_accuracy", "macro_f1", "roc_auc"
        ) if column in reference.columns and column in joint.columns
    ]
    if len(common) >= 3:
        pd.concat([reference[common], joint[common]], ignore_index=True).to_csv(
            task_out / "reference_comparison_original_style_ricci.csv", index=False
        )


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _run_interval_penalty_path(
    *,
    n_intervals: int,
    h0_prepared,
    ricci_prepared,
    intervals,
    y_train: np.ndarray,
    y_validation: np.ndarray,
    task: TaskSpec,
    config: RunConfig,
    checkpoint_dir: Path,
    seed: int,
    log: LogFunction,
) -> list[dict]:
    rows: list[dict] = []
    warm: ModelWarmStart | None = None
    n_classes = len(task.classes)
    n_h0 = h0_prepared.train.shape[1]
    n_ricci = ricci_prepared.train.shape[1]

    for path_index, (lambda_h0, lambda_ricci) in enumerate(_penalty_path(config), start=1):
        hyperparameters = JointHyperparameters(n_intervals, lambda_h0, lambda_ricci)
        stem = _checkpoint_stem(n_intervals, lambda_h0, lambda_ricci)
        result_path = checkpoint_dir / f"{stem}.json"
        state_path = checkpoint_dir / f"{stem}.state.npz"
        partial_path = checkpoint_dir / f"{stem}.partial.npz"
        prefix = (
            f"M={n_intervals} lambda_H={lambda_h0:g} lambda_R={lambda_ricci:g} "
            f"path={path_index}/{len(_penalty_path(config))}"
        )

        if config.resume and result_path.exists() and state_path.exists():
            row = _load_json(result_path)
            rows.append(row)
            warm = load_warm_start(state_path, n_classes, n_h0, n_ricci)
            warm.completed_alternations = 0
            log(f"{prefix} [RESUME] completed checkpoint loaded")
            continue

        initial_state: ModelWarmStart | None = None
        if config.resume and partial_path.exists():
            initial_state = load_warm_start(partial_path, n_classes, n_h0, n_ricci)
            log(
                f"{prefix} [RESUME] partial checkpoint after "
                f"{initial_state.completed_alternations}/{config.n_alternations} alternations"
            )
        elif warm is not None:
            initial_state = warm.copy()
            initial_state.completed_alternations = 0

        started = time.perf_counter()
        log(f"{prefix} [START]")

        def save_partial(state: ModelWarmStart) -> None:
            save_warm_start(partial_path, state)

        model = fit_joint_sparse_model_prepared(
            h0=h0_prepared,
            ricci=ricci_prepared,
            y_train=y_train,
            classes=task.classes,
            hyperparameters=hyperparameters,
            c_value=config.c_value,
            logistic_max_iter=config.logistic_max_iter,
            logistic_tolerance=config.logistic_tolerance,
            n_alternations=config.n_alternations,
            alpha_smoothness_gamma=config.alpha_smoothness_gamma,
            alpha_optimizer_max_iter=config.alpha_optimizer_max_iter,
            seed=seed + path_index,
            initial_active_ricci=config.active_set_initial,
            active_batch_size=config.active_set_batch,
            max_active_rounds=config.active_set_max_rounds,
            warm_start=initial_state,
            progress=lambda message: log(f"{prefix} {message}"),
            state_callback=save_partial,
        )
        if h0_prepared.evaluation is None or ricci_prepared.evaluation is None:
            raise RuntimeError("Inner-CV prepared split lacks validation arrays.")
        probabilities = model.predict_proba_prepared(
            h0_prepared.evaluation, ricci_prepared.evaluation
        )
        metrics = classification_metrics(
            y_validation, probabilities, task.classes, task.positive_class
        )
        elapsed = float(time.perf_counter() - started)
        row = {
            "n_intervals": int(n_intervals),
            "lambda_h0": float(lambda_h0),
            "lambda_ricci": float(lambda_ricci),
            "sigma_log": float(intervals.sigma_log),
            **metrics,
            "elapsed_seconds": elapsed,
            "solver_iterations": model.solver_diagnostics.iterations,
            "solver_active_set_rounds": model.solver_diagnostics.active_set_rounds,
            "solver_active_ricci_features": model.solver_diagnostics.active_ricci_features,
            "solver_max_kkt_violation": model.solver_diagnostics.max_kkt_violation,
            "solver_objective": model.solver_diagnostics.objective,
            "solver_polish_calls": model.solver_diagnostics.polish_calls,
            "solver_polish_iterations": model.solver_diagnostics.polish_iterations,
        }
        final_state = model.warm_start()
        save_warm_start(state_path, final_state)
        _atomic_json(result_path, row)
        partial_path.unlink(missing_ok=True)
        rows.append(row)
        warm = final_state
        warm.completed_alternations = 0
        log(
            f"{prefix} [DONE] roc_auc={metrics['roc_auc']:.4f} "
            f"active_ricci={model.solver_diagnostics.active_ricci_features} "
            f"max_kkt={model.solver_diagnostics.max_kkt_violation:.3e} "
            f"elapsed={elapsed:.1f}s"
        )
    return rows


def _run_all_interval_paths(
    *,
    basis_cache: dict[int, tuple[object, object]],
    ricci_prepared,
    y_train: np.ndarray,
    y_validation: np.ndarray,
    task: TaskSpec,
    config: RunConfig,
    checkpoint_root: Path,
    seed: int,
    log: LogFunction,
) -> list[dict]:
    jobs = max(1, min(int(config.n_jobs), len(config.interval_grid)))
    results: list[dict] = []
    with threadpool_limits(limits=1):
        if jobs == 1:
            for n_intervals in config.interval_grid:
                h0_prepared, intervals = basis_cache[int(n_intervals)]
                results.extend(_run_interval_penalty_path(
                    n_intervals=int(n_intervals),
                    h0_prepared=h0_prepared,
                    ricci_prepared=ricci_prepared,
                    intervals=intervals,
                    y_train=y_train,
                    y_validation=y_validation,
                    task=task,
                    config=config,
                    checkpoint_dir=checkpoint_root / f"M_{int(n_intervals):03d}",
                    seed=seed + int(n_intervals) * 100,
                    log=log,
                ))
        else:
            log(f"Launching {jobs} parallel M paths with one BLAS thread per worker")
            with ThreadPoolExecutor(max_workers=jobs, thread_name_prefix="wkpi-M") as executor:
                futures = {}
                for n_intervals in config.interval_grid:
                    h0_prepared, intervals = basis_cache[int(n_intervals)]
                    future = executor.submit(
                        _run_interval_penalty_path,
                        n_intervals=int(n_intervals),
                        h0_prepared=h0_prepared,
                        ricci_prepared=ricci_prepared,
                        intervals=intervals,
                        y_train=y_train,
                        y_validation=y_validation,
                        task=task,
                        config=config,
                        checkpoint_dir=checkpoint_root / f"M_{int(n_intervals):03d}",
                        seed=seed + int(n_intervals) * 100,
                        log=log,
                    )
                    futures[future] = int(n_intervals)
                for future in as_completed(futures):
                    results.extend(future.result())
    return sorted(results, key=lambda row: (row["n_intervals"], row["lambda_h0"], row["lambda_ricci"]))


def _completed_fold_available(fold_out: Path) -> bool:
    required = (
        "COMPLETE.json",
        "fold_result.csv",
        "test_predictions.csv",
        "alpha_profile.csv",
        "inner_config_results.csv",
        "selected_config.json",
        "sparsity_diagnostics.json",
    )
    return all((fold_out / name).exists() for name in required)


def _load_completed_fold(fold_out: Path) -> dict[str, object]:
    return {
        "fold_rows": pd.read_csv(fold_out / "fold_result.csv").to_dict("records"),
        "predictions": pd.read_csv(fold_out / "test_predictions.csv").to_dict("records"),
        "alpha": pd.read_csv(fold_out / "alpha_profile.csv").to_dict("records"),
        "inner": pd.read_csv(fold_out / "inner_config_results.csv").to_dict("records"),
        "selected": [_load_json(fold_out / "selected_config.json")],
        "sparsity": [_load_json(fold_out / "sparsity_diagnostics.json")],
    }


def run_nested_cv(inputs: LoadedInputs, config: RunConfig, log: LogFunction = _default_log) -> dict[str, pd.DataFrame | bool]:
    config.out_dir.mkdir(parents=True, exist_ok=True)
    log = _thread_safe_logger(log)
    selected_tasks = select_tasks(config.tasks)

    all_fold_rows: list[dict] = []
    all_prediction_rows: list[dict] = []
    all_alpha_rows: list[dict] = []
    all_inner_rows: list[dict] = []
    all_selected_rows: list[dict] = []
    all_sparsity_rows: list[dict] = []

    for task_index, task in enumerate(selected_tasks, start=1):
        log(f"Task {task_index}/{len(selected_tasks)}: {task.display_name}")
        task_mask = mask_for_task(inputs.labels3, task)
        global_indices = np.where(task_mask)[0]
        y = labels_for_task(inputs.labels3[task_mask], task)
        groups = inputs.participant_ids[task_mask]
        sample_ids = inputs.sample_ids[task_mask]
        ricci = inputs.ricci[global_indices]
        deaths = [inputs.h0_deaths[index] for index in global_indices]

        task_out = config.out_dir / task.folder
        task_out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({
            "sample_id": sample_ids,
            "participant_id": groups,
            "label": y,
            "h0_path": [inputs.h0_paths[index] for index in global_indices],
        }).to_csv(task_out / "task_sample_manifest.csv", index=False)

        outer_count = _safe_split_count(y, groups, config.n_outer_splits)
        if outer_count != config.n_outer_splits:
            raise RuntimeError(
                f"Task {task.display_name} supports only {outer_count} participant-grouped folds, "
                f"but {config.n_outer_splits} were requested."
            )
        outer_cv = StratifiedGroupKFold(
            n_splits=config.n_outer_splits,
            shuffle=True,
            random_state=config.seed,
        )

        task_fold_rows: list[dict] = []
        task_prediction_rows: list[dict] = []
        task_alpha_rows: list[dict] = []

        for outer_fold, (outer_train, outer_test) in enumerate(
            outer_cv.split(np.zeros(len(y)), y, groups), start=1
        ):
            _assert_no_group_overlap(outer_train, outer_test, groups)
            fold_out = task_out / f"fold_{outer_fold}"
            fold_out.mkdir(parents=True, exist_ok=True)

            if config.resume and _completed_fold_available(fold_out):
                loaded = _load_completed_fold(fold_out)
                fold_rows = loaded["fold_rows"]
                predictions = loaded["predictions"]
                alpha_rows = loaded["alpha"]
                inner_rows = loaded["inner"]
                selected_rows = loaded["selected"]
                sparsity_rows = loaded["sparsity"]
                task_fold_rows.extend(fold_rows)
                all_fold_rows.extend(fold_rows)
                task_prediction_rows.extend(predictions)
                all_prediction_rows.extend(predictions)
                # alpha_profile.csv does not contain task metadata, restore it.
                for row in alpha_rows:
                    restored = {
                        "task": task.display_name,
                        "task_folder": task.folder,
                        "fold": outer_fold,
                        **row,
                    }
                    task_alpha_rows.append(restored)
                    all_alpha_rows.append(restored)
                all_inner_rows.extend(inner_rows)
                all_selected_rows.extend(selected_rows)
                all_sparsity_rows.extend(sparsity_rows)
                log(f"{task.folder}: outer fold {outer_fold} [RESUME] completed fold loaded")
                continue

            log(
                f"{task.folder}: outer fold {outer_fold}/{config.n_outer_splits}; "
                f"train={len(outer_train)}, test={len(outer_test)}"
            )
            inner_count = _safe_split_count(y[outer_train], groups[outer_train], config.n_inner_splits)
            if inner_count < 2:
                raise RuntimeError(f"Insufficient grouped class support in {task.folder} outer fold {outer_fold}.")
            inner_cv = StratifiedGroupKFold(
                n_splits=inner_count,
                shuffle=True,
                random_state=config.seed + 100 * outer_fold,
            )

            fold_inner_rows: list[dict] = []
            outer_inner_started = time.perf_counter()
            for inner_fold, (relative_train, relative_validation) in enumerate(
                inner_cv.split(np.zeros(len(outer_train)), y[outer_train], groups[outer_train]), start=1
            ):
                inner_train = outer_train[relative_train]
                inner_validation = outer_train[relative_validation]
                _assert_no_group_overlap(inner_train, inner_validation, groups)
                log(
                    f"{task.folder}: outer {outer_fold}, inner {inner_fold}/{inner_count}; "
                    "preparing train-only dense Ricci scaler"
                )
                ricci_prepared = prepare_ricci_split(
                    ricci[inner_train], ricci[inner_validation]
                )
                basis_cache: dict[int, tuple[object, object]] = {}
                for intervals_count in config.interval_grid:
                    intervals = fit_train_only_adaptive_intervals(
                        [deaths[index] for index in inner_train], int(intervals_count)
                    )
                    phi_train = transform_many(
                        [deaths[index] for index in inner_train], intervals, quad_points=config.quad_points
                    )
                    phi_validation = transform_many(
                        [deaths[index] for index in inner_validation], intervals, quad_points=config.quad_points
                    )
                    basis_cache[int(intervals_count)] = (
                        prepare_h0_split(phi_train, phi_validation), intervals
                    )

                rows = _run_all_interval_paths(
                    basis_cache=basis_cache,
                    ricci_prepared=ricci_prepared,
                    y_train=y[inner_train],
                    y_validation=y[inner_validation],
                    task=task,
                    config=config,
                    checkpoint_root=fold_out / "checkpoints" / f"inner_{inner_fold}",
                    seed=config.seed + 10_000 * outer_fold + 1_000 * inner_fold,
                    log=lambda message, p=f"{task.folder}: outer {outer_fold}, inner {inner_fold}: ": log(p + message),
                )
                for row in rows:
                    decorated = {
                        "task": task.display_name,
                        "task_folder": task.folder,
                        "outer_fold": outer_fold,
                        "inner_fold": inner_fold,
                        **row,
                    }
                    fold_inner_rows.append(decorated)
                    all_inner_rows.append(decorated)
                pd.DataFrame(fold_inner_rows).to_csv(
                    fold_out / "inner_config_results.partial.csv", index=False
                )

                if config.benchmark_first_inner_fold:
                    benchmark = pd.DataFrame(fold_inner_rows)
                    benchmark.to_csv(config.out_dir / "BENCHMARK_FIRST_INNER_FOLD.csv", index=False)
                    elapsed = float(time.perf_counter() - outer_inner_started)
                    payload = {
                        "task": task.display_name,
                        "outer_fold": outer_fold,
                        "inner_fold": inner_fold,
                        "configurations": len(rows),
                        "elapsed_seconds": elapsed,
                        "elapsed_hours": elapsed / 3600.0,
                        "mean_configuration_seconds": float(benchmark["elapsed_seconds"].mean()),
                        "maximum_configuration_seconds": float(benchmark["elapsed_seconds"].max()),
                        "note": "Benchmark mode stops after one complete 27-configuration inner fold.",
                    }
                    _atomic_json(config.out_dir / "BENCHMARK_FIRST_INNER_FOLD.json", payload)
                    log(
                        f"[BENCHMARK DONE] {len(rows)} configurations in {elapsed / 60:.1f} minutes. "
                        f"Results: {config.out_dir / 'BENCHMARK_FIRST_INNER_FOLD.csv'}"
                    )
                    return {
                        "benchmark_only": True,
                        "inner_results": benchmark,
                        "summaries": pd.DataFrame(),
                    }

            inner_dataframe = pd.DataFrame(fold_inner_rows)
            expected_rows = inner_count * len(config.interval_grid) * len(config.lambda_h0_grid) * len(config.lambda_ricci_grid)
            if len(inner_dataframe) != expected_rows:
                raise RuntimeError(
                    f"Expected {expected_rows} inner-CV rows, obtained {len(inner_dataframe)}."
                )
            inner_dataframe.to_csv(fold_out / "inner_config_results.csv", index=False)
            (fold_out / "inner_config_results.partial.csv").unlink(missing_ok=True)
            selected, aggregated_inner = _select_configuration(inner_dataframe, config.selection_metric)
            aggregated_inner.to_csv(fold_out / "inner_config_summary.csv", index=False)
            selected_score = float(aggregated_inner.iloc[0]["selection_score"])
            log(
                f"{task.folder}: outer {outer_fold} selected M={selected.n_intervals}, "
                f"lambda_H={selected.lambda_h0:g}, lambda_R={selected.lambda_ricci:g}, "
                f"inner {config.selection_metric}={selected_score:.4f}"
            )

            intervals = fit_train_only_adaptive_intervals(
                [deaths[index] for index in outer_train], selected.n_intervals
            )
            phi_train = transform_many(
                [deaths[index] for index in outer_train], intervals, quad_points=config.quad_points
            )
            phi_test = transform_many(
                [deaths[index] for index in outer_test], intervals, quad_points=config.quad_points
            )
            h0_prepared = prepare_h0_split(phi_train, phi_test)
            ricci_prepared = prepare_ricci_split(ricci[outer_train], ricci[outer_test])
            partial_outer = fold_out / "outer_model.partial.npz"
            initial_outer = None
            if config.resume and partial_outer.exists():
                initial_outer = load_warm_start(
                    partial_outer,
                    len(task.classes),
                    selected.n_intervals,
                    ricci_prepared.train.shape[1],
                )
                log(
                    f"{task.folder}: outer {outer_fold} [RESUME] final model after "
                    f"{initial_outer.completed_alternations}/{config.n_alternations} alternations"
                )
            model = fit_joint_sparse_model_prepared(
                h0=h0_prepared,
                ricci=ricci_prepared,
                y_train=y[outer_train],
                classes=task.classes,
                hyperparameters=selected,
                c_value=config.c_value,
                logistic_max_iter=config.logistic_max_iter,
                logistic_tolerance=config.logistic_tolerance,
                n_alternations=config.n_alternations,
                alpha_smoothness_gamma=config.alpha_smoothness_gamma,
                alpha_optimizer_max_iter=config.alpha_optimizer_max_iter,
                seed=config.seed + 1_000_000 + outer_fold,
                initial_active_ricci=config.active_set_initial,
                active_batch_size=config.active_set_batch,
                max_active_rounds=config.active_set_max_rounds,
                warm_start=initial_outer,
                progress=lambda message: log(f"{task.folder}: outer {outer_fold}: {message}"),
                state_callback=lambda state: save_warm_start(partial_outer, state),
            )
            if h0_prepared.evaluation is None or ricci_prepared.evaluation is None:
                raise RuntimeError("Outer prepared split lacks test arrays.")
            probabilities = model.predict_proba_prepared(
                h0_prepared.evaluation, ricci_prepared.evaluation
            )
            predictions = np.asarray([
                task.classes[index] for index in np.argmax(probabilities, axis=1)
            ], dtype=object)
            metrics = classification_metrics(y[outer_test], probabilities, task.classes, task.positive_class)
            participant_overlap = len(set(groups[outer_train]) & set(groups[outer_test]))

            fold_row = {
                "task": task.display_name,
                "task_folder": task.folder,
                "fold": outer_fold,
                "train_samples": len(outer_train),
                "test_samples": len(outer_test),
                "train_participants": len(set(groups[outer_train])),
                "test_participants": len(set(groups[outer_test])),
                "participant_overlap": participant_overlap,
                "selected_intervals": selected.n_intervals,
                "lambda_h0": selected.lambda_h0,
                "lambda_ricci": selected.lambda_ricci,
                "sigma_log": float(intervals.sigma_log),
                "inner_selection_score": selected_score,
                **metrics,
            }
            task_fold_rows.append(fold_row)
            all_fold_rows.append(fold_row)
            pd.DataFrame([fold_row]).to_csv(fold_out / "fold_result.csv", index=False)

            selected_row = {
                "task": task.display_name,
                "task_folder": task.folder,
                "fold": outer_fold,
                "selected_intervals": selected.n_intervals,
                "lambda_h0": selected.lambda_h0,
                "lambda_ricci": selected.lambda_ricci,
                "inner_selection_metric": config.selection_metric,
                "inner_selection_score": selected_score,
            }
            all_selected_rows.append(selected_row)
            write_manifest(fold_out / "selected_config.json", selected_row)

            split_rows = []
            for split_name, indices in (("train", outer_train), ("test", outer_test)):
                for index in indices:
                    split_rows.append({
                        "split": split_name,
                        "sample_id": sample_ids[index],
                        "participant_id": groups[index],
                        "label": y[index],
                    })
            pd.DataFrame(split_rows).to_csv(fold_out / "participant_group_split.csv", index=False)

            interval_dataframe = pd.DataFrame({
                "interval_index": np.arange(intervals.n_intervals),
                "log_left": intervals.log_edges[:-1],
                "log_right": intervals.log_edges[1:],
                "log_center": intervals.log_centers,
                "death_center": intervals.death_centers,
                "alpha": model.alpha,
            })
            interval_dataframe.to_csv(fold_out / "alpha_profile.csv", index=False)
            for interval_index in range(intervals.n_intervals):
                alpha_row = {
                    "task": task.display_name,
                    "task_folder": task.folder,
                    "fold": outer_fold,
                    "interval_index": interval_index,
                    "log_left": float(intervals.log_edges[interval_index]),
                    "log_right": float(intervals.log_edges[interval_index + 1]),
                    "log_center": float(intervals.log_centers[interval_index]),
                    "death_center": float(intervals.death_centers[interval_index]),
                    "alpha": float(model.alpha[interval_index]),
                }
                task_alpha_rows.append(alpha_row)
                all_alpha_rows.append(alpha_row)

            pd.DataFrame([record.__dict__ for record in model.alternation_history]).to_csv(
                fold_out / "alternation_history.csv", index=False
            )
            _top_h0_coefficients(model, intervals, config.top_coefficients).to_csv(
                fold_out / "top_h0_coefficients.csv", index=False
            )
            _top_ricci_coefficients(model, inputs.feature_metadata, config.top_coefficients).to_csv(
                fold_out / "top_ricci_coefficients.csv", index=False
            )

            sparsity = {
                "task": task.display_name,
                "task_folder": task.folder,
                "fold": outer_fold,
                **model_sparsity(model),
            }
            all_sparsity_rows.append(sparsity)
            write_manifest(fold_out / "sparsity_diagnostics.json", sparsity)

            fold_confusion = confusion(y[outer_test], predictions, task.classes)
            save_confusion_artifacts(
                fold_confusion,
                task.classes,
                fold_out / "test_confusion_matrix",
                title=f"{task.display_name}: outer fold {outer_fold}",
            )

            fold_prediction_rows = []
            for local_index, task_index_value in enumerate(outer_test):
                row = {
                    "task": task.display_name,
                    "task_folder": task.folder,
                    "fold": outer_fold,
                    "sample_id": sample_ids[task_index_value],
                    "participant_id": groups[task_index_value],
                    "true_label": y[task_index_value],
                    "predicted_label": predictions[local_index],
                    "selected_intervals": selected.n_intervals,
                    "lambda_h0": selected.lambda_h0,
                    "lambda_ricci": selected.lambda_ricci,
                }
                for class_index, label in enumerate(task.classes):
                    row[f"probability_{label}"] = float(probabilities[local_index, class_index])
                fold_prediction_rows.append(row)
                task_prediction_rows.append(row)
                all_prediction_rows.append(row)
            pd.DataFrame(fold_prediction_rows).to_csv(fold_out / "test_predictions.csv", index=False)
            save_warm_start(fold_out / "outer_model.state.npz", model.warm_start())
            partial_outer.unlink(missing_ok=True)
            _atomic_json(fold_out / "COMPLETE.json", {
                "completed": True,
                "task": task.display_name,
                "fold": outer_fold,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            })

        task_fold_dataframe = pd.DataFrame(task_fold_rows)
        task_prediction_dataframe = pd.DataFrame(task_prediction_rows)
        task_alpha_dataframe = pd.DataFrame(task_alpha_rows)
        task_fold_dataframe.to_csv(task_out / "fold_results.csv", index=False)
        task_prediction_dataframe.to_csv(task_out / "test_predictions.csv", index=False)
        task_alpha_dataframe.to_csv(task_out / "alpha_profiles_all_folds.csv", index=False)

        aggregate_confusion = confusion(
            task_prediction_dataframe["true_label"].to_numpy(object),
            task_prediction_dataframe["predicted_label"].to_numpy(object),
            task.classes,
        )
        save_confusion_artifacts(
            aggregate_confusion,
            task.classes,
            task_out / "aggregated_confusion_matrix",
            title=f"Aggregated outer-fold confusion matrix: {task.display_name}",
        )

        common_grid = np.linspace(
            task_alpha_dataframe["log_center"].min(),
            task_alpha_dataframe["log_center"].max(),
            config.common_alpha_grid_size,
        )
        interpolated_rows: list[dict] = []
        for fold, fold_alpha in task_alpha_dataframe.groupby("fold"):
            values = interpolate_alpha(
                fold_alpha["log_center"].to_numpy(float),
                fold_alpha["alpha"].to_numpy(float),
                common_grid,
            )
            for grid_value, alpha_value in zip(common_grid, values):
                interpolated_rows.append({
                    "task": task.display_name,
                    "task_folder": task.folder,
                    "fold": int(fold),
                    "common_log_center": float(grid_value),
                    "alpha_interpolated": float(alpha_value),
                })
        interpolated_dataframe = pd.DataFrame(interpolated_rows)
        interpolated_dataframe.to_csv(task_out / "alpha_interpolated_common_grid.csv", index=False)
        save_alpha_plot(
            interpolated_dataframe,
            task_out / "mean_alpha_across_outer_folds",
            title=f"Mean learned alpha across outer folds: {task.display_name}",
        )
        _reference_comparison(task, task_fold_dataframe, config, task_out)

    fold_dataframe = pd.DataFrame(all_fold_rows)
    prediction_dataframe = pd.DataFrame(all_prediction_rows)
    alpha_dataframe = pd.DataFrame(all_alpha_rows)
    inner_dataframe = pd.DataFrame(all_inner_rows)
    selected_dataframe = pd.DataFrame(all_selected_rows)
    sparsity_dataframe = pd.DataFrame(all_sparsity_rows)

    summary_rows: list[dict] = []
    for (task_name, task_folder), rows in fold_dataframe.groupby(["task", "task_folder"], sort=False):
        records = rows.to_dict("records")
        summary = {
            "task": task_name,
            "task_folder": task_folder,
            "n_folds": len(rows),
            **summary_mean_sd(records, ("accuracy", "balanced_accuracy", "macro_f1", "roc_auc")),
            "selected_intervals_mean": float(rows["selected_intervals"].mean()),
            "selected_intervals_sd": float(rows["selected_intervals"].std(ddof=1)),
            "lambda_h0_mean": float(rows["lambda_h0"].mean()),
            "lambda_h0_sd": float(rows["lambda_h0"].std(ddof=1)),
            "lambda_ricci_mean": float(rows["lambda_ricci"].mean()),
            "lambda_ricci_sd": float(rows["lambda_ricci"].std(ddof=1)),
        }
        summary_rows.append(summary)
    summary_dataframe = pd.DataFrame(summary_rows)

    fold_dataframe.to_csv(config.out_dir / "ALL_FOLD_RESULTS.csv", index=False)
    summary_dataframe.to_csv(config.out_dir / "ALL_SUMMARIES.csv", index=False)
    prediction_dataframe.to_csv(config.out_dir / "ALL_TEST_PREDICTIONS.csv", index=False)
    alpha_dataframe.to_csv(config.out_dir / "ALL_ALPHA_PROFILES.csv", index=False)
    inner_dataframe.to_csv(config.out_dir / "ALL_INNER_CONFIG_RESULTS.csv", index=False)
    selected_dataframe.to_csv(config.out_dir / "ALL_SELECTED_CONFIGS.csv", index=False)
    sparsity_dataframe.to_csv(config.out_dir / "ALL_SPARSITY_DIAGNOSTICS.csv", index=False)
    inputs.feature_metadata.to_csv(config.out_dir / "RICCI_FEATURE_METADATA_USED.csv", index=False)

    write_global_latex_tables(summary_dataframe, fold_dataframe, selected_dataframe, config.out_dir)
    write_manifest(config.out_dir / "RUN_CONFIG.json", {
        **config.serialisable(),
        "resolved_h0_pd_dir": str(inputs.h0_pd_dir),
        "ricci_matrix_shape": list(inputs.ricci.shape),
        "source_columns": inputs.source_columns,
        "label_counts": pd.Series(inputs.labels3).value_counts().to_dict(),
        "participant_count": int(len(set(inputs.participant_ids))),
        "sample_count": int(len(inputs.sample_ids)),
        "objective": "weighted mean cross-entropy + lambda_H/(C*sum_w)*||beta_H||_1 + lambda_R/(C*sum_w)*||beta_R||_1 + gamma*||Delta alpha||_2^2",
        "alpha_constraint": "alpha > 0 and mean(alpha) = 1",
        "separate_l1_implementation": "direct blockwise proximal optimization with full KKT-safe Ricci working sets",
        "solver": "warm-started monotone active-set FISTA with exact orthant-constrained L-BFGS-B polishing; binary single-logit and multinomial softmax",
        "parallelism": "independent M paths; shared dense split arrays; one BLAS thread per worker",
    })

    return {
        "benchmark_only": False,
        "fold_results": fold_dataframe,
        "summaries": summary_dataframe,
        "predictions": prediction_dataframe,
        "alpha_profiles": alpha_dataframe,
        "inner_results": inner_dataframe,
        "selected_configs": selected_dataframe,
        "sparsity": sparsity_dataframe,
    }
