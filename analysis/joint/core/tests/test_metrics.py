import numpy as np

from h0_ricci_joint_sparse.metrics import classification_metrics


def test_multiclass_auc_respects_declared_class_order():
    classes = ("nonIBD", "UC", "CD")
    y = np.array(["nonIBD", "UC", "CD", "nonIBD", "UC", "CD"], dtype=object)
    probabilities = np.array([
        [0.9, 0.05, 0.05],
        [0.05, 0.9, 0.05],
        [0.05, 0.05, 0.9],
        [0.8, 0.1, 0.1],
        [0.1, 0.8, 0.1],
        [0.1, 0.1, 0.8],
    ])
    metrics = classification_metrics(y, probabilities, classes, None)
    assert metrics["roc_auc"] == 1.0
