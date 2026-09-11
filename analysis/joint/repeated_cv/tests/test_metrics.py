import numpy as np
import pandas as pd
import pytest

from joint_repeated_cv.metrics import (
    classification_metrics,
    participant_average_predictions,
    standardise_prediction_frame,
)
from joint_repeated_cv.tasks import TASK_SPECS


def test_binary_metrics_and_participant_averaging() -> None:
    raw = pd.DataFrame(
        {
            "sample_id": ["s1", "s2", "s3", "s4"],
            "participant_id": ["p1", "p1", "p2", "p2"],
            "true_label": ["nonIBD", "nonIBD", "IBD", "IBD"],
            "predicted_label": ["nonIBD", "nonIBD", "IBD", "IBD"],
            "probability_nonIBD": [0.9, 0.8, 0.1, 0.2],
            "probability_IBD": [0.1, 0.2, 0.9, 0.8],
        }
    )
    frame, classes = standardise_prediction_frame(raw, TASK_SPECS[0])
    metrics = classification_metrics(frame, classes)
    assert metrics["accuracy"] == 1.0
    assert metrics["balanced_accuracy"] == 1.0
    assert metrics["roc_auc"] == 1.0
    participants = participant_average_predictions(frame, classes)
    assert len(participants) == 2
    assert set(participants["predicted_label_standard"]) == {"nonIBD", "IBD"}


def test_multiclass_metrics_are_macro_ovr() -> None:
    raw = pd.DataFrame(
        {
            "sample_id": ["a", "b", "c"],
            "participant_id": ["pa", "pb", "pc"],
            "true_label": ["nonIBD", "UC", "CD"],
            "predicted_label": ["nonIBD", "UC", "CD"],
            "probability_nonIBD": [0.9, 0.05, 0.05],
            "probability_UC": [0.05, 0.9, 0.05],
            "probability_CD": [0.05, 0.05, 0.9],
        }
    )
    frame, classes = standardise_prediction_frame(raw, TASK_SPECS[1])
    metrics = classification_metrics(frame, classes)
    assert classes == ("nonIBD", "UC", "CD")
    assert metrics["roc_auc"] == 1.0
    assert np.isclose(metrics["multiclass_brier"], 0.015)


def test_prediction_argmax_disagreement_is_rejected() -> None:
    raw = pd.DataFrame(
        {
            "true_label": ["nonIBD", "IBD"],
            "predicted_label": ["IBD", "IBD"],
            "probability_nonIBD": [0.9, 0.1],
            "probability_IBD": [0.1, 0.9],
        }
    )
    with pytest.raises(ValueError, match="disagree"):
        standardise_prediction_frame(raw, TASK_SPECS[0])


def test_probability_rows_must_sum_to_one() -> None:
    raw = pd.DataFrame(
        {
            "true_label": ["nonIBD", "IBD"],
            "predicted_label": ["nonIBD", "IBD"],
            "probability_nonIBD": [0.8, 0.1],
            "probability_IBD": [0.1, 0.8],
        }
    )
    with pytest.raises(ValueError, match="sum to one"):
        standardise_prediction_frame(raw, TASK_SPECS[0])
