from __future__ import annotations

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


def probability_metrics(y_true: np.ndarray, probabilities: np.ndarray, class_names: tuple[str, ...]) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    probabilities = np.clip(probabilities, 0.0, 1.0)
    row_sums = probabilities.sum(axis=1, keepdims=True)
    if np.any(~np.isfinite(row_sums)) or np.any(row_sums <= 0.0):
        raise ValueError("Probabilities contain a non-finite or non-positive row sum")
    probabilities = probabilities / row_sums
    prediction = np.argmax(probabilities, axis=1)
    n_classes = len(class_names)
    one_hot = np.eye(n_classes, dtype=float)[y_true]
    result = {
        "accuracy": float(accuracy_score(y_true, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "macro_f1": float(f1_score(y_true, prediction, average="macro", zero_division=0)),
        "brier_multiclass": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
    }
    try:
        if n_classes == 2:
            result["roc_auc"] = float(roc_auc_score(y_true, probabilities[:, 1]))
        else:
            result["roc_auc"] = float(
                roc_auc_score(
                    y_true,
                    probabilities,
                    labels=np.arange(n_classes),
                    multi_class="ovr",
                    average="macro",
                )
            )
    except Exception:
        result["roc_auc"] = np.nan
    try:
        result["log_loss"] = float(log_loss(y_true, probabilities, labels=np.arange(n_classes)))
    except Exception:
        result["log_loss"] = np.nan
    cm = confusion_matrix(y_true, prediction, labels=np.arange(n_classes))
    for index, name in enumerate(class_names):
        denominator = int(cm[index].sum())
        result[f"recall_{name}"] = float(cm[index, index] / denominator) if denominator else np.nan
    return result


def participant_predictions(sample_predictions: pd.DataFrame, class_names: tuple[str, ...]) -> pd.DataFrame:
    probability_columns = [f"probability_{name}" for name in class_names]
    centroid_columns = [f"centroid_probability_{name}" for name in class_names]
    rows: list[dict[str, Any]] = []
    for participant_id, group in sample_predictions.groupby("participant_id", sort=False):
        labels = group["true_idx"].unique()
        if len(labels) != 1:
            raise ValueError(f"Participant {participant_id} has multiple labels")
        probabilities = group[probability_columns].mean(axis=0).to_numpy(dtype=float)
        centroid = group[centroid_columns].mean(axis=0).to_numpy(dtype=float)
        pred = int(np.argmax(probabilities))
        row: dict[str, Any] = {
            "participant_id": str(participant_id),
            "true_idx": int(labels[0]),
            "true_label": class_names[int(labels[0])],
            "n_samples": int(len(group)),
            "pred_idx": pred,
            "pred_label": class_names[pred],
        }
        row.update({column: float(value) for column, value in zip(probability_columns, probabilities)})
        row.update({column: float(value) for column, value in zip(centroid_columns, centroid)})
        rows.append(row)
    return pd.DataFrame(rows)


def metrics_from_prediction_frame(frame: pd.DataFrame, class_names: tuple[str, ...]) -> dict[str, float]:
    probability_columns = [f"probability_{name}" for name in class_names]
    return probability_metrics(frame["true_idx"].to_numpy(), frame[probability_columns].to_numpy(), class_names)
