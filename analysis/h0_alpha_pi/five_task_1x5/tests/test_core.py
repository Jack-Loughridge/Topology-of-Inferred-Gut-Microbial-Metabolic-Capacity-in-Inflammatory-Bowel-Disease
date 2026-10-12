from __future__ import annotations

import numpy as np

from h0_alpha_pi_1x5.tasks import TASKS, map_condition_to_task
from h0_alpha_pi_1x5.metrics import probability_metrics


def test_task_order_and_label_orientation() -> None:
    assert [task.folder for task in TASKS] == [
        "IBD_vs_nonIBD",
        "three_way_nonIBD_UC_CD",
        "nonIBD_vs_UC",
        "nonIBD_vs_CD",
        "CD_vs_UC",
    ]
    assert map_condition_to_task("UC", TASKS[0]) == "IBD"
    assert map_condition_to_task("CD", TASKS[0]) == "IBD"
    assert TASKS[-1].class_order == ("CD", "UC")


def test_binary_and_multiclass_probability_metrics() -> None:
    binary = probability_metrics(
        np.array([0, 1, 0, 1]),
        np.array([[0.8, 0.2], [0.1, 0.9], [0.7, 0.3], [0.2, 0.8]]),
        ("nonIBD", "IBD"),
    )
    assert binary["accuracy"] == 1.0
    multiclass = probability_metrics(
        np.array([0, 1, 2]),
        np.array([[0.8, 0.1, 0.1], [0.1, 0.7, 0.2], [0.1, 0.2, 0.7]]),
        ("nonIBD", "UC", "CD"),
    )
    assert multiclass["balanced_accuracy"] == 1.0


def test_ibd_original_and_shared_manifest_equivalence(tmp_path) -> None:
    import pandas as pd
    import pytest

    from h0_alpha_pi_1x5.orchestrator import verify_ibd_manifest_equivalence

    original_rows = []
    shared_rows = []
    samples = [
        ("S1", "P1", 0, "nonIBD"),
        ("S2", "P2", 1, "IBD"),
        ("S3", "P3", 0, "nonIBD"),
        ("S4", "P4", 1, "IBD"),
    ]
    for fold, test_ids in ((1, {"S1", "S2"}), (2, {"S3", "S4"})):
        for sample_id, participant_id, numeric, text in samples:
            role = "test" if sample_id in test_ids else "train"
            original_rows.append(
                {
                    "repeat": 1,
                    "fold": fold,
                    "split_seed": 13,
                    "role": role,
                    "sample_id": sample_id,
                    "participant_id": participant_id,
                    "label": numeric,
                }
            )
            shared_rows.append(
                {
                    "task_folder": "IBD_vs_nonIBD",
                    "repeat": 1,
                    "fold": fold,
                    "split_seed": 13,
                    "role": role,
                    "sample_id": sample_id,
                    "participant_id": participant_id,
                    "label": text,
                }
            )
    original_path = tmp_path / "original.csv"
    shared_path = tmp_path / "shared.csv"
    pd.DataFrame(original_rows).to_csv(original_path, index=False)
    pd.DataFrame(shared_rows).to_csv(shared_path, index=False)
    audit = verify_ibd_manifest_equivalence(original_path, shared_path)
    assert audit["samples"] == 4
    assert audit["folds"] == 2

    broken = pd.DataFrame(shared_rows)
    broken.loc[(broken["fold"] == 1) & (broken["sample_id"] == "S1"), "role"] = "train"
    broken.to_csv(shared_path, index=False)
    with pytest.raises(RuntimeError):
        verify_ibd_manifest_equivalence(original_path, shared_path)


def test_inner_split_retries_deterministically(monkeypatch) -> None:
    from dataclasses import replace
    from types import SimpleNamespace

    import h0_alpha_pi_1x5.engine as engine

    class FakeSplitter:
        calls: list[int] = []

        def __init__(self, n_splits, shuffle, random_state):
            assert n_splits == 2
            assert shuffle is True
            self.seed = int(random_state)
            FakeSplitter.calls.append(self.seed)

        def split(self, x, labels, groups):
            # The first seed yields a validation side missing class 2.
            if len(FakeSplitter.calls) == 1:
                yield np.array([0, 1, 2, 3, 4]), np.array([5])
            else:
                yield np.array([0, 2, 4]), np.array([1, 3, 5])

    monkeypatch.setattr(engine, "StratifiedGroupKFold", FakeSplitter)
    bundle = SimpleNamespace(
        labels=np.array([0, 0, 1, 1, 2, 2], dtype=int),
        participant_ids=["A0", "A1", "B0", "B1", "C0", "C1"],
    )
    config = replace(
        engine.EngineConfig(
            pd_dir=None,
            label_csv=None,
            metadata_csv=None,
            split_manifest=None,
            output_dir=None,
            task=TASKS[1],
        ),
        inner_splits=2,
        inner_split_attempts=4,
    )
    outer_train = np.arange(6, dtype=int)
    train, validation, accepted_seed = engine._inner_split(bundle, outer_train, 1, config)
    assert len(FakeSplitter.calls) == 2
    assert accepted_seed == FakeSplitter.calls[1]
    assert set(bundle.labels[train]) == {0, 1, 2}
    assert set(bundle.labels[validation]) == {0, 1, 2}
    assert not (set(np.asarray(bundle.participant_ids)[train]) & set(np.asarray(bundle.participant_ids)[validation]))


def test_inner_split_rejects_mathematically_impossible_class_count() -> None:
    from dataclasses import replace
    from types import SimpleNamespace

    import pytest

    import h0_alpha_pi_1x5.engine as engine

    # Class 2 has only one participant in outer training, so no participant-
    # disjoint inner train/validation split can contain all three classes on
    # both sides. This must remain a hard production error.
    bundle = SimpleNamespace(
        labels=np.array([0, 0, 1, 1, 2], dtype=int),
        participant_ids=["A0", "A1", "B0", "B1", "C0"],
    )
    config = replace(
        engine.EngineConfig(
            pd_dir=None,
            label_csv=None,
            metadata_csv=None,
            split_manifest=None,
            output_dir=None,
            task=TASKS[1],
        ),
        inner_splits=2,
        inner_split_attempts=4,
    )
    with pytest.raises(RuntimeError, match="fewer than two outer-training participants"):
        engine._inner_split(bundle, np.arange(5, dtype=int), 1, config)
