from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    roc_auc_score,
)

from .tasks import TaskSpec, normalise_label, ordered_classes


def _normalise_column_token(value: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def discover_probability_columns(frame: pd.DataFrame, classes: tuple[str, ...]) -> dict[str, str]:
    candidates = [
        column for column in frame.columns
        if str(column).lower().startswith(("probability_", "proba_", "prob_", "p_"))
    ]
    mapping: dict[str, str] = {}
    for label in classes:
        token = _normalise_column_token(normalise_label(label))
        matches = []
        for column in candidates:
            column_token = _normalise_column_token(column)
            if column_token.endswith(token):
                matches.append(column)
        if len(matches) == 1:
            mapping[label] = str(matches[0])
        elif len(matches) > 1:
            exact_prefixes = [
                f"probability{token}", f"proba{token}", f"prob{token}", f"p{token}"
            ]
            exact = [column for column in matches if _normalise_column_token(column) in exact_prefixes]
            if len(exact) == 1:
                mapping[label] = str(exact[0])
            else:
                raise ValueError(f"Ambiguous probability columns for class {label}: {matches}")
    if len(classes) == 2 and len(mapping) == 1:
        known_label, known_column = next(iter(mapping.items()))
        missing_label = next(label for label in classes if label != known_label)
        derived = f"__derived_probability_{missing_label}"
        frame[derived] = 1.0 - pd.to_numeric(frame[known_column], errors="raise")
        mapping[missing_label] = derived
    if set(mapping) != set(classes):
        raise ValueError(
            f"Could not map probability columns for classes {classes}. "
            f"Candidates were {candidates}; mapping={mapping}"
        )
    return mapping


def standardise_prediction_frame(frame: pd.DataFrame, task: TaskSpec) -> tuple[pd.DataFrame, tuple[str, ...]]:
    output = frame.copy()
    true_column = next(
        (column for column in ("true_label", "y_true", "label", "actual_label") if column in output.columns),
        None,
    )
    pred_column = next(
        (column for column in ("predicted_label", "pred_label", "prediction", "y_pred") if column in output.columns),
        None,
    )
    if true_column is None:
        raise ValueError(f"Prediction file for {task.folder} lacks a true-label column.")
    output["true_label_standard"] = output[true_column].map(normalise_label)
    classes = ordered_classes(output["true_label_standard"].tolist(), task)
    probability_columns = discover_probability_columns(output, classes)
    probability_matrix = np.column_stack(
        [pd.to_numeric(output[probability_columns[label]], errors="raise").to_numpy(float) for label in classes]
    )
    if not np.isfinite(probability_matrix).all():
        raise ValueError(f"Prediction probabilities for {task.folder} contain NaN or infinity.")
    if (probability_matrix < -1e-9).any() or (probability_matrix > 1 + 1e-9).any():
        raise ValueError(f"Prediction probabilities for {task.folder} lie outside [0,1].")
    row_sums = probability_matrix.sum(axis=1)
    if np.any(row_sums <= 0):
        raise ValueError(f"Prediction probabilities for {task.folder} have nonpositive row sums.")
    if not np.allclose(row_sums, 1.0, atol=1e-6, rtol=1e-6):
        bad = np.flatnonzero(~np.isclose(row_sums, 1.0, atol=1e-6, rtol=1e-6))[:10]
        raise ValueError(
            f"Prediction probabilities for {task.folder} do not sum to one; "
            f"example row sums={row_sums[bad].tolist()}"
        )
    for index, label in enumerate(classes):
        output[f"probability_standard_{label}"] = probability_matrix[:, index]
    argmax_prediction = np.asarray(classes, dtype=object)[np.argmax(probability_matrix, axis=1)]
    if pred_column is not None:
        supplied = output[pred_column].map(normalise_label).to_numpy(object)
        mismatch = supplied != argmax_prediction
        if mismatch.any():
            examples = output.loc[mismatch].head(10)
            raise ValueError(
                f"Supplied predictions disagree with probability argmax for {task.folder}: "
                f"{examples.to_dict('records')}"
            )
    output["predicted_label_standard"] = argmax_prediction
    return output, classes


def classification_metrics(frame: pd.DataFrame, classes: tuple[str, ...]) -> dict[str, Any]:
    y_true = frame["true_label_standard"].to_numpy(object)
    y_pred = frame["predicted_label_standard"].to_numpy(object)
    probabilities = np.column_stack(
        [frame[f"probability_standard_{label}"].to_numpy(float) for label in classes]
    )
    class_to_index = {label: index for index, label in enumerate(classes)}
    try:
        y_true_index = np.asarray([class_to_index[str(label)] for label in y_true], dtype=int)
        y_pred_index = np.asarray([class_to_index[str(label)] for label in y_pred], dtype=int)
    except KeyError as exc:
        raise ValueError(f"Observed label is not in class order {classes}: {exc}") from exc
    numeric_labels = np.arange(len(classes), dtype=int)
    result: dict[str, Any] = {
        "n_observations": int(len(frame)),
        "n_classes": int(len(classes)),
        "accuracy": float(accuracy_score(y_true_index, y_pred_index)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_index, y_pred_index)),
        "macro_f1": float(
            f1_score(
                y_true_index,
                y_pred_index,
                labels=numeric_labels,
                average="macro",
                zero_division=0,
            )
        ),
        "log_loss": float(log_loss(y_true_index, probabilities, labels=numeric_labels)),
    }
    one_hot = np.eye(len(classes), dtype=float)[y_true_index]
    result["multiclass_brier"] = float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))
    try:
        if len(classes) == 2:
            result["roc_auc"] = float(roc_auc_score(y_true_index, probabilities[:, 1]))
            result["positive_class"] = classes[1]
            result["positive_f1"] = float(
                f1_score(y_true_index, y_pred_index, labels=[1], average="macro", zero_division=0)
            )
        else:
            result["roc_auc"] = float(
                roc_auc_score(
                    y_true_index,
                    probabilities,
                    labels=numeric_labels,
                    multi_class="ovr",
                    average="macro",
                )
            )
            result["positive_class"] = ""
            result["positive_f1"] = float("nan")
    except ValueError:
        result["roc_auc"] = float("nan")
        result["positive_class"] = classes[1] if len(classes) == 2 else ""
        result["positive_f1"] = float("nan")
    matrix = confusion_matrix(y_true_index, y_pred_index, labels=numeric_labels)
    for row_index, true_label in enumerate(classes):
        for column_index, pred_label in enumerate(classes):
            result[f"confusion_true_{true_label}_pred_{pred_label}"] = int(matrix[row_index, column_index])
    return result


def participant_average_predictions(frame: pd.DataFrame, classes: tuple[str, ...]) -> pd.DataFrame:
    conflicts = frame.groupby("participant_id")["true_label_standard"].nunique().gt(1)
    if conflicts.any():
        raise ValueError(
            f"Participants have conflicting held-out labels: {conflicts[conflicts].index.tolist()[:10]}"
        )
    aggregations: dict[str, tuple[str, str]] = {
        "true_label_standard": ("true_label_standard", "first"),
        "n_samples": ("sample_id", "count"),
    }
    for label in classes:
        aggregations[f"probability_standard_{label}"] = (f"probability_standard_{label}", "mean")
    grouped = frame.groupby("participant_id", as_index=False).agg(**aggregations)
    probabilities = np.column_stack(
        [grouped[f"probability_standard_{label}"].to_numpy(float) for label in classes]
    )
    grouped["predicted_label_standard"] = np.asarray(classes, dtype=object)[
        np.argmax(probabilities, axis=1)
    ]
    return grouped


def summary_across_repetitions(frame: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        "accuracy", "balanced_accuracy", "macro_f1", "positive_f1", "roc_auc", "multiclass_brier", "log_loss"
    ]
    rows: list[dict[str, Any]] = []
    if frame.empty:
        return pd.DataFrame()
    grouping = ["task_order", "task_folder", "task", "level"]
    for keys, group in frame.groupby(grouping, sort=False):
        row = dict(zip(grouping, keys))
        row["n_repetitions"] = int(group["repeat"].nunique())
        for metric in metric_columns:
            if metric not in group.columns:
                continue
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            row[f"{metric}_mean"] = float(values.mean()) if len(values) else float("nan")
            row[f"{metric}_sd"] = float(values.std(ddof=1)) if len(values) > 1 else float("nan")
            row[f"{metric}_median"] = float(values.median()) if len(values) else float("nan")
            row[f"{metric}_q025"] = float(values.quantile(0.025)) if len(values) else float("nan")
            row[f"{metric}_q25"] = float(values.quantile(0.25)) if len(values) else float("nan")
            row[f"{metric}_q75"] = float(values.quantile(0.75)) if len(values) else float("nan")
            row[f"{metric}_q975"] = float(values.quantile(0.975)) if len(values) else float("nan")
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["task_order", "level"], kind="mergesort")
