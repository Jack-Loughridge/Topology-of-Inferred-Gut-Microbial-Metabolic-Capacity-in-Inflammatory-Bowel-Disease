import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


def test_participant_grouped_splits_have_zero_overlap():
    groups = np.repeat(np.arange(12), 3)
    labels_by_group = np.array(["A", "B"] * 6, dtype=object)
    labels = np.repeat(labels_by_group, 3)
    splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=13)
    for train, test in splitter.split(np.zeros(len(labels)), labels, groups):
        assert not (set(groups[train]) & set(groups[test]))
