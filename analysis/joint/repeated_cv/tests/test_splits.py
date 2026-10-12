from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from joint_repeated_cv.splits import (
    TaskArrays,
    assert_ibd_manifest_equivalence,
    generate_task_manifest,
    validate_task_manifest,
)
from joint_repeated_cv.tasks import TASK_SPECS


def binary_arrays() -> TaskArrays:
    sample_ids = []
    participants = []
    labels = []
    for label in ("nonIBD", "IBD"):
        for participant_index in range(10):
            participant = f"{label}_p{participant_index:02d}"
            for sample_index in range(2):
                sample_ids.append(f"{participant}_s{sample_index}")
                participants.append(participant)
                labels.append(label)
    return TaskArrays(
        sample_ids=np.asarray(sample_ids, dtype=object),
        participant_ids=np.asarray(participants, dtype=object),
        labels=np.asarray(labels, dtype=object),
    )


def test_grouped_manifest_is_complete_and_deterministic(small_config) -> None:
    task = TASK_SPECS[0]
    seeds = {1: 101, 2: 202}
    first = generate_task_manifest(task, binary_arrays(), seeds, small_config)
    second = generate_task_manifest(task, binary_arrays(), seeds, small_config)
    pd.testing.assert_frame_equal(first, second)
    validate_task_manifest(task, first, small_config)
    for repeat in (1, 2):
        test_rows = first[first["repeat"].eq(repeat) & first["role"].eq("test")]
        assert test_rows["sample_id"].nunique() == len(binary_arrays().sample_ids)


def test_participant_leakage_is_rejected(small_config) -> None:
    task = TASK_SPECS[0]
    manifest = generate_task_manifest(task, binary_arrays(), {1: 101, 2: 202}, small_config)
    selector = manifest["repeat"].eq(1) & manifest["fold"].eq(1) & manifest["role"].eq("test")
    row_index = manifest.index[selector][0]
    train_participant = manifest.loc[
        manifest["repeat"].eq(1) & manifest["fold"].eq(1) & manifest["role"].eq("train"),
        "participant_id",
    ].iloc[0]
    manifest.loc[row_index, "participant_id"] = train_participant
    with pytest.raises(ValueError, match="Participant leakage|conflicting sample metadata"):
        validate_task_manifest(task, manifest, small_config)


def test_locked_ibd_equivalence_checks_participant_mapping(small_config) -> None:
    task = TASK_SPECS[0]
    generated = generate_task_manifest(task, binary_arrays(), {1: 101, 2: 202}, small_config)
    locked = generated[["repeat", "fold", "split_seed", "role", "sample_id", "participant_id", "label"]].copy()
    audit = assert_ibd_manifest_equivalence(generated, locked, small_config)
    assert (
        audit["exact_sample_match"]
        & audit["exact_participant_match"]
        & audit["exact_label_match"]
    ).all()
    locked.loc[locked.index[0], "participant_id"] = "wrong_participant"
    with pytest.raises(RuntimeError, match="do not match"):
        assert_ibd_manifest_equivalence(generated, locked, small_config)


def test_locked_ibd_equivalence_infers_numeric_label_orientation(small_config) -> None:
    task = TASK_SPECS[0]
    generated = generate_task_manifest(task, binary_arrays(), {1: 101, 2: 202}, small_config)
    locked = generated[
        ["repeat", "fold", "split_seed", "role", "sample_id", "participant_id", "label"]
    ].copy()
    # Deliberately use an opaque orientation; the implementation must infer it
    # from exact sample identities rather than assuming 0/1 semantics.
    locked["label"] = locked["label"].map({"nonIBD": 7, "IBD": 3})
    audit = assert_ibd_manifest_equivalence(generated, locked, small_config)
    assert audit["exact_label_match"].all()

    corrupted_sample = locked["sample_id"].iloc[0]
    selector = locked["sample_id"].eq(corrupted_sample)
    locked.loc[selector, "label"] = 3 if locked.loc[selector, "label"].iloc[0] == 7 else 7
    with pytest.raises(RuntimeError, match="one-to-one correspondence"):
        assert_ibd_manifest_equivalence(generated, locked, small_config)
