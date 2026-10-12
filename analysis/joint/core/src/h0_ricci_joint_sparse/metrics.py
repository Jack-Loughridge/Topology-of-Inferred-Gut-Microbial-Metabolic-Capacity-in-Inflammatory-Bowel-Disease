from __future__ import annotations

import math
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score


def classification_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    classes: tuple[str, ...],
    positive_class: str | None,
) -> dict[str, float]:
    predicted = np.asarray([classes[index] for index in np.argmax(probabilities, axis=1)], dtype=object)
    result = {
        "accuracy": float(accuracy_score(y_true, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
        "macro_f1": float(f1_score(y_true, predicted, labels=list(classes), average="macro", zero_division=0)),
        "roc_auc": math.nan,
    }
    try:
        if len(classes) == 2:
            positive = positive_class or classes[1]
            y_binary = np.asarray([1 if label == positive else 0 for label in y_true], dtype=int)
            result["roc_auc"] = float(roc_auc_score(y_binary, probabilities[:, classes.index(positive)]))
        else:
            # Compute macro one-vs-rest AUC explicitly so the probability columns
            # remain in the scientifically defined class order rather than a
            # lexicographically sorted order imposed by some sklearn versions.
            class_aucs = []
            for class_index, label in enumerate(classes):
                binary_truth = np.asarray([1 if value == label else 0 for value in y_true], dtype=int)
                if np.unique(binary_truth).size < 2:
                    raise ValueError(f"Validation split lacks positive or negative examples for {label}.")
                class_aucs.append(roc_auc_score(binary_truth, probabilities[:, class_index]))
            result["roc_auc"] = float(np.mean(class_aucs))
    except ValueError:
        result["roc_auc"] = math.nan
    return result


def selection_value(metrics: dict[str, float], metric: str) -> float:
    value = float(metrics.get(metric, math.nan))
    return value if math.isfinite(value) else -math.inf


def confusion(y_true: np.ndarray, y_pred: np.ndarray, classes: tuple[str, ...]) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=list(classes))


def summary_mean_sd(rows: list[dict[str, Any]], metric_names: tuple[str, ...]) -> dict[str, float]:
    output: dict[str, float] = {}
    for metric in metric_names:
        values = np.asarray([float(row[metric]) for row in rows], dtype=float)
        finite = values[np.isfinite(values)]
        output[f"{metric}_mean"] = float(np.mean(finite)) if finite.size else math.nan
        output[f"{metric}_sd"] = float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0 if finite.size == 1 else math.nan
    return output
