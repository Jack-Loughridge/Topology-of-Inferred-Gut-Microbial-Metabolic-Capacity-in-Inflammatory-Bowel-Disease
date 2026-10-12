from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .tasks import TaskSpec, map_condition_to_task, normalise_condition
from .util import find_col, normalise_sample_id


@dataclass
class TaskBundle:
    task: TaskSpec
    sample_ids: list[str]
    participant_ids: list[str]
    labels: np.ndarray
    label_names: list[str]
    deaths: list[np.ndarray]
    pd_paths: dict[str, Path]
    metadata: pd.DataFrame
    sample_to_index: dict[str, int]


REQUIRED_MANIFEST_COLUMNS = {
    "task_folder",
    "repeat",
    "fold",
    "split_seed",
    "role",
    "sample_id",
    "participant_id",
    "label",
}


def load_task_manifest(path: Path, task: TaskSpec, repeat: int = 1, expected_folds: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, low_memory=False)
    missing = REQUIRED_MANIFEST_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Manifest {path} lacks columns: {sorted(missing)}")
    frame = frame.copy()
    frame["task_folder"] = frame["task_folder"].astype(str).str.strip()
    frame = frame[frame["task_folder"].eq(task.folder)].copy()
    frame["sample_id"] = frame["sample_id"].map(normalise_sample_id)
    frame["participant_id"] = frame["participant_id"].astype(str).str.strip()
    frame["repeat"] = pd.to_numeric(frame["repeat"], errors="raise").astype(int)
    frame["fold"] = pd.to_numeric(frame["fold"], errors="raise").astype(int)
    frame["split_seed"] = pd.to_numeric(frame["split_seed"], errors="raise").astype(int)
    frame["role"] = frame["role"].astype(str).str.strip().str.lower()
    frame["label"] = frame["label"].astype(str).map(lambda x: normalise_condition(x) or str(x).strip())
    frame = frame[frame["repeat"].eq(int(repeat))].copy()
    if frame.empty:
        raise ValueError(f"Manifest {path} has no rows for task={task.folder}, repeat={repeat}")
    if set(frame["role"]) != {"train", "test"}:
        raise ValueError(f"Task {task.folder} repeat {repeat} roles are not exactly train/test")
    folds = sorted(frame["fold"].unique().tolist())
    if folds != list(range(1, expected_folds + 1)):
        raise ValueError(f"Task {task.folder} expected folds 1..{expected_folds}; found {folds}")
    if set(frame["label"]) != set(task.class_order):
        raise ValueError(
            f"Task {task.folder} manifest classes {sorted(set(frame['label']))} do not match {task.class_order}"
        )
    metadata = frame[["sample_id", "participant_id", "label"]].drop_duplicates()
    if metadata["sample_id"].duplicated().any():
        bad = metadata.loc[metadata["sample_id"].duplicated(False), "sample_id"].tolist()[:10]
        raise ValueError(f"Conflicting manifest metadata for samples: {bad}")
    if metadata.groupby("participant_id")["label"].nunique().gt(1).any():
        raise ValueError(f"Task {task.folder} contains participants with mixed labels")
    all_samples = set(metadata["sample_id"])
    all_participants = set(metadata["participant_id"])
    tested: set[str] = set()
    for fold in folds:
        rows = frame[frame["fold"].eq(fold)]
        train = rows[rows["role"].eq("train")]
        test = rows[rows["role"].eq("test")]
        if train.empty or test.empty:
            raise ValueError(f"Task {task.folder} fold {fold} has empty train/test")
        if set(train["sample_id"]) & set(test["sample_id"]):
            raise ValueError(f"Sample leakage in {task.folder} fold {fold}")
        if set(train["sample_id"]) | set(test["sample_id"]) != all_samples:
            raise ValueError(f"Incomplete sample partition in {task.folder} fold {fold}")
        if set(train["participant_id"]) & set(test["participant_id"]):
            raise ValueError(f"Participant leakage in {task.folder} fold {fold}")
        if set(train["participant_id"]) | set(test["participant_id"]) != all_participants:
            raise ValueError(f"Incomplete participant partition in {task.folder} fold {fold}")
        if set(train["label"]) != set(task.class_order) or set(test["label"]) != set(task.class_order):
            raise ValueError(f"Class absent in {task.folder} fold {fold}")
        overlap = tested & set(test["sample_id"])
        if overlap:
            raise ValueError(f"Repeated test samples in {task.folder}: {sorted(overlap)[:10]}")
        tested |= set(test["sample_id"])
    if tested != all_samples:
        raise ValueError(f"Task {task.folder} test folds cover {len(tested)}/{len(all_samples)} samples")
    return frame.reset_index(drop=True), metadata.reset_index(drop=True)


def _load_labels(path: Path, required: set[str]) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    sample_col = find_col(frame, ["sample_id", "sample", "External ID", "external_id", "id"])
    label_col = find_col(frame, ["label", "condition", "diagnosis", "disease", "status"])
    if sample_col is None or label_col is None:
        raise ValueError(f"Could not identify label columns in {path}")
    output = frame[[sample_col, label_col]].copy()
    output.columns = ["sample_id", "condition"]
    output["sample_id"] = output["sample_id"].map(normalise_sample_id)
    output = output[output["sample_id"].isin(required)].copy()
    output["condition"] = output["condition"].map(normalise_condition)
    output = output.dropna(subset=["condition"])
    conflicts = output.groupby("sample_id")["condition"].nunique()
    if conflicts.gt(1).any():
        raise ValueError(f"Conflicting labels for required samples: {conflicts[conflicts.gt(1)].index.tolist()[:10]}")
    return output.drop_duplicates("sample_id")


def _load_participants(path: Path, required: set[str]) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    sample_col = find_col(frame, ["External ID", "external_id", "sample_id", "sample", "id"])
    participant_col = find_col(frame, ["Participant ID", "participant_id", "participant", "subject_id", "host_subject_id"])
    if sample_col is None or participant_col is None:
        raise ValueError(f"Could not identify metadata columns in {path}")
    output = frame[[sample_col, participant_col]].copy()
    output.columns = ["sample_id", "participant_id"]
    output["sample_id"] = output["sample_id"].map(normalise_sample_id)
    output = output[output["sample_id"].isin(required)].copy()
    output["participant_id"] = output["participant_id"].astype("string").str.strip()
    missing = output["participant_id"].isna() | output["participant_id"].isin(["", "nan", "None", "<NA>"])
    if missing.any():
        raise ValueError(f"Missing participants for required samples: {output.loc[missing, 'sample_id'].tolist()[:10]}")
    conflicts = output.groupby("sample_id")["participant_id"].nunique()
    if conflicts.gt(1).any():
        raise ValueError(
            f"Conflicting participants for required samples: {conflicts[conflicts.gt(1)].index.tolist()[:10]}"
        )
    return output.drop_duplicates("sample_id")


def _read_deaths(path: Path) -> np.ndarray:
    array = np.load(path, allow_pickle=False)
    array = np.asarray(array)
    if array.size == 0:
        deaths = np.zeros(0, dtype=np.float32)
    elif array.ndim == 1:
        deaths = array.astype(np.float32)
    elif array.ndim == 2 and array.shape[1] >= 2:
        deaths = array[:, 1].astype(np.float32)
    else:
        deaths = array.ravel().astype(np.float32)
    deaths = deaths[np.isfinite(deaths)]
    deaths = deaths[(deaths > 0.0) & (deaths < 1.0 - 1e-12)]
    return deaths.astype(np.float32, copy=False)


def load_task_bundle(
    pd_dir: Path,
    label_csv: Path,
    metadata_csv: Path,
    manifest_metadata: pd.DataFrame,
    task: TaskSpec,
) -> TaskBundle:
    required = set(manifest_metadata["sample_id"])
    labels = _load_labels(label_csv, required)
    participants = _load_participants(metadata_csv, required)
    check = manifest_metadata.merge(labels, on="sample_id", how="left", validate="one_to_one")
    check = check.merge(participants, on="sample_id", how="left", suffixes=("_manifest", "_source"), validate="one_to_one")
    if check["condition"].isna().any():
        raise ValueError(f"Missing label rows: {check.loc[check['condition'].isna(), 'sample_id'].tolist()[:10]}")
    mapped = check["condition"].map(lambda value: map_condition_to_task(str(value), task))
    label_bad = check.loc[mapped.astype(str) != check["label"].astype(str), "sample_id"].tolist()
    if label_bad:
        raise ValueError(f"Manifest/source label disagreement for samples: {label_bad[:10]}")
    participant_bad = check.loc[
        check["participant_id_manifest"].astype(str) != check["participant_id_source"].astype(str), "sample_id"
    ].tolist()
    if participant_bad:
        raise ValueError(f"Manifest/source participant disagreement for samples: {participant_bad[:10]}")

    candidates: dict[str, Path] = {}
    for path in sorted(pd_dir.glob("*_H0.npy")):
        sample_id = normalise_sample_id(path.name)
        if sample_id in required:
            if sample_id in candidates:
                raise ValueError(f"Multiple H0 files for sample {sample_id}")
            candidates[sample_id] = path
    missing = sorted(required - set(candidates))
    if missing:
        raise FileNotFoundError(f"Missing H0 diagrams for manifest samples: {missing[:10]}")

    label_to_index = {label: index for index, label in enumerate(task.class_order)}
    ordered = manifest_metadata.sort_values("sample_id").reset_index(drop=True)
    sample_ids: list[str] = []
    participant_ids: list[str] = []
    label_names: list[str] = []
    labels_numeric: list[int] = []
    deaths: list[np.ndarray] = []
    paths: dict[str, Path] = {}
    for row in ordered.itertuples(index=False):
        sid = str(row.sample_id)
        sample_ids.append(sid)
        participant_ids.append(str(row.participant_id))
        label_names.append(str(row.label))
        labels_numeric.append(label_to_index[str(row.label)])
        paths[sid] = candidates[sid]
        deaths.append(_read_deaths(candidates[sid]))
    metadata = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "participant_id": participant_ids,
            "label": labels_numeric,
            "label_name": label_names,
        }
    )
    return TaskBundle(
        task=task,
        sample_ids=sample_ids,
        participant_ids=participant_ids,
        labels=np.asarray(labels_numeric, dtype=int),
        label_names=label_names,
        deaths=deaths,
        pd_paths=paths,
        metadata=metadata,
        sample_to_index={sample_id: index for index, sample_id in enumerate(sample_ids)},
    )


def fold_indices(manifest: pd.DataFrame, bundle: TaskBundle, fold: int) -> tuple[np.ndarray, np.ndarray, int]:
    rows = manifest[manifest["fold"].eq(int(fold))]
    train_ids = rows.loc[rows["role"].eq("train"), "sample_id"].tolist()
    test_ids = rows.loc[rows["role"].eq("test"), "sample_id"].tolist()
    return (
        np.asarray([bundle.sample_to_index[value] for value in train_ids], dtype=int),
        np.asarray([bundle.sample_to_index[value] for value in test_ids], dtype=int),
        int(rows["split_seed"].iloc[0]),
    )
