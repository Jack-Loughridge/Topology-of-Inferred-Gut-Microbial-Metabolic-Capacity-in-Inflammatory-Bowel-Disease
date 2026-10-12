from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from .config import RunConfig
from .tasks import TASK_BY_FOLDER, TASK_SPECS, TaskSpec, normalise_label, ordered_classes
from .util import atomic_csv, canonical_frame_sha256, normalise_sample_id


@dataclass(frozen=True)
class TaskArrays:
    sample_ids: np.ndarray
    participant_ids: np.ndarray
    labels: np.ndarray


def _locked_label_token(value: object) -> str:
    """Canonicalise an opaque locked-manifest label without assuming 0/1 orientation."""
    if pd.isna(value):
        raise ValueError("Locked IBD manifest contains a missing label.")
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    text = str(value).strip()
    if not text:
        raise ValueError("Locked IBD manifest contains a blank label.")
    return text


def _infer_locked_ibd_label_mapping(
    generated: pd.DataFrame, locked: pd.DataFrame
) -> dict[str, str]:
    """Infer the locked label encoding from exact shared sample identities.

    Some historical manifests store IBD labels as strings while others store an
    opaque binary code.  The orientation of a numeric code must never be guessed.
    This function derives a one-to-one mapping from locked values to the labels
    loaded by the validated core and rejects any inconsistency.
    """
    generated_master = generated[["sample_id", "label"]].copy()
    generated_master["label"] = generated_master["label"].map(normalise_label)
    generated_master = generated_master.drop_duplicates()
    if generated_master["sample_id"].duplicated().any():
        raise ValueError("Generated IBD manifest has conflicting labels for a sample.")

    locked_master = locked[["sample_id", "label"]].copy()
    locked_master["locked_label_token"] = locked_master["label"].map(_locked_label_token)
    locked_master = locked_master[["sample_id", "locked_label_token"]].drop_duplicates()
    if locked_master["sample_id"].duplicated().any():
        raise ValueError("Locked IBD manifest has conflicting labels for a sample.")

    if set(generated_master["sample_id"]) != set(locked_master["sample_id"]):
        raise RuntimeError("Cannot infer locked IBD label encoding because sample sets differ.")
    joined = generated_master.merge(locked_master, on="sample_id", validate="one_to_one")
    forward_counts = joined.groupby("locked_label_token")["label"].nunique()
    reverse_counts = joined.groupby("label")["locked_label_token"].nunique()
    if forward_counts.gt(1).any() or reverse_counts.gt(1).any():
        raise RuntimeError(
            "Locked IBD label values do not have a one-to-one correspondence with core labels."
        )
    mapping = (
        joined[["locked_label_token", "label"]]
        .drop_duplicates()
        .set_index("locked_label_token")["label"]
        .to_dict()
    )
    if set(mapping.values()) != {"nonIBD", "IBD"} or len(mapping) != 2:
        raise RuntimeError(f"Locked IBD labels do not encode exactly nonIBD and IBD: {mapping}")
    return {str(key): str(value) for key, value in mapping.items()}


def load_ibd_manifest(config: RunConfig) -> pd.DataFrame:
    frame = pd.read_csv(config.ibd_split_manifest, low_memory=False)
    required = {"repeat", "fold", "split_seed", "role", "sample_id", "participant_id", "label"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"IBD split manifest is missing columns: {sorted(missing)}")
    frame = frame.copy()
    if frame[["sample_id", "participant_id", "label"]].isna().any().any():
        raise ValueError("IBD manifest contains missing sample IDs, participant IDs, or labels.")
    frame["sample_id"] = frame["sample_id"].map(normalise_sample_id)
    frame["participant_id"] = frame["participant_id"].astype(str).str.strip()
    frame["repeat"] = pd.to_numeric(frame["repeat"], errors="raise").astype(int)
    frame["fold"] = pd.to_numeric(frame["fold"], errors="raise").astype(int)
    frame["split_seed"] = pd.to_numeric(frame["split_seed"], errors="raise").astype(int)
    frame["role"] = frame["role"].astype(str).str.strip().str.lower()
    if set(frame["role"]) != {"train", "test"}:
        raise ValueError("IBD manifest roles must be exactly train and test.")
    if frame["sample_id"].eq("").any() or frame["participant_id"].eq("").any():
        raise ValueError("IBD manifest contains blank sample or participant IDs.")
    frame["_locked_label_token"] = frame["label"].map(_locked_label_token)
    duplicate_key = ["repeat", "fold", "role", "sample_id"]
    if frame.duplicated(duplicate_key).any():
        bad = frame.loc[frame.duplicated(duplicate_key, keep=False), duplicate_key].head().to_dict("records")
        raise ValueError(f"IBD manifest contains duplicate split rows: {bad}")
    sample_metadata = frame[["sample_id", "participant_id", "_locked_label_token"]].drop_duplicates()
    if sample_metadata["sample_id"].duplicated().any():
        bad = sample_metadata.loc[
            sample_metadata["sample_id"].duplicated(False), "sample_id"
        ].tolist()[:10]
        raise ValueError(f"IBD manifest has conflicting sample metadata: {bad}")
    repeats = sorted(frame["repeat"].unique())
    folds = sorted(frame["fold"].unique())
    if repeats != list(range(1, config.expected_repeats + 1)):
        raise ValueError(f"Expected repeats 1..{config.expected_repeats}; found {repeats}")
    if folds != list(range(1, config.expected_folds + 1)):
        raise ValueError(f"Expected folds 1..{config.expected_folds}; found {folds}")
    for repeat in repeats:
        for fold in folds:
            roles = set(frame.loc[frame["repeat"].eq(repeat) & frame["fold"].eq(fold), "role"])
            if roles != {"train", "test"}:
                raise ValueError(f"IBD manifest repeat {repeat}, fold {fold} lacks train/test roles.")
    return frame.drop(columns="_locked_label_token")


def repeat_seeds(ibd_manifest: pd.DataFrame, config: RunConfig) -> dict[int, int]:
    output: dict[int, int] = {}
    for repeat in range(1, config.expected_repeats + 1):
        values = ibd_manifest.loc[ibd_manifest["repeat"].eq(repeat), "split_seed"].unique()
        if len(values) != 1:
            raise ValueError(f"Repeat {repeat} has {len(values)} split seeds: {values.tolist()}")
        output[repeat] = int(values[0])
    if len(set(output.values())) != len(output):
        duplicates = pd.Series(output).value_counts()
        raise ValueError(
            "The locked repeated-CV manifest reuses split seeds across repetitions: "
            f"{duplicates[duplicates.gt(1)].to_dict()}"
        )
    return output


def validate_task_arrays(
    task: TaskSpec, arrays: TaskArrays, config: RunConfig
) -> tuple[str, ...]:
    n = len(arrays.sample_ids)
    if len(arrays.participant_ids) != n or len(arrays.labels) != n:
        raise ValueError(f"Task {task.folder} arrays have inconsistent lengths.")
    if any(pd.isna(value) for value in arrays.sample_ids):
        raise ValueError(f"Task {task.folder} has missing sample IDs.")
    if any(pd.isna(value) for value in arrays.participant_ids):
        raise ValueError(f"Task {task.folder} has missing participant IDs.")
    if any(pd.isna(value) for value in arrays.labels):
        raise ValueError(f"Task {task.folder} has missing labels.")
    sample_ids = np.asarray([normalise_sample_id(value) for value in arrays.sample_ids], dtype=object)
    if any(not value for value in sample_ids):
        raise ValueError(f"Task {task.folder} has blank sample IDs.")
    if len(set(sample_ids.tolist())) != n:
        raise ValueError(f"Task {task.folder} has duplicate sample IDs.")
    participants = np.asarray([str(value).strip() for value in arrays.participant_ids], dtype=object)
    if any(not value for value in participants):
        raise ValueError(f"Task {task.folder} has blank participant IDs.")
    labels = np.asarray([normalise_label(value) for value in arrays.labels], dtype=object)
    frame = pd.DataFrame({"participant_id": participants, "label": labels})
    conflicting = frame.groupby("participant_id")["label"].nunique().gt(1)
    if conflicting.any():
        raise ValueError(
            f"Task {task.folder} has participants with conflicting labels: "
            f"{conflicting[conflicting].index.tolist()[:10]}"
        )
    classes = ordered_classes(labels.tolist(), task)
    expected_classes = set(task.preferred_class_order)
    if set(classes) != expected_classes:
        raise ValueError(
            f"Task {task.folder} classes are {classes}; expected exactly {task.preferred_class_order}."
        )
    for label in classes:
        participant_count = frame.loc[frame["label"].eq(label), "participant_id"].nunique()
        if participant_count < config.expected_folds:
            raise ValueError(
                f"Task {task.folder} class {label} has only {participant_count} participants; "
                f"{config.expected_folds} grouped folds are impossible."
            )
    return classes


def generate_task_manifest(
    task: TaskSpec,
    arrays: TaskArrays,
    seeds: dict[int, int],
    config: RunConfig,
) -> pd.DataFrame:
    classes = validate_task_arrays(task, arrays, config)
    sample_ids = np.asarray([normalise_sample_id(value) for value in arrays.sample_ids], dtype=object)
    participants = np.asarray([str(value).strip() for value in arrays.participant_ids], dtype=object)
    labels = np.asarray([normalise_label(value) for value in arrays.labels], dtype=object)
    class_to_index = {label: index for index, label in enumerate(classes)}
    rows: list[dict[str, Any]] = []
    for repeat in range(1, config.expected_repeats + 1):
        seed = seeds[repeat]
        splitter = StratifiedGroupKFold(
            n_splits=config.expected_folds, shuffle=True, random_state=seed
        )
        for fold, (train_idx, test_idx) in enumerate(
            splitter.split(np.zeros(len(labels)), labels, groups=participants), start=1
        ):
            for role, indices in (("train", train_idx), ("test", test_idx)):
                for index in indices:
                    label = str(labels[index])
                    rows.append(
                        {
                            "task_order": TASK_ORDER_INDEX[task.folder],
                            "task_folder": task.folder,
                            "task": task.display_name,
                            "report_task": task.report_name,
                            "repeat": repeat,
                            "fold": fold,
                            "split_seed": seed,
                            "role": role,
                            "sample_id": str(sample_ids[index]),
                            "participant_id": str(participants[index]),
                            "label": label,
                            "label_index": class_to_index[label],
                        }
                    )
    manifest = pd.DataFrame(rows)
    validate_task_manifest(task, manifest, config)
    return manifest


TASK_ORDER_INDEX = {task.folder: index for index, task in enumerate(TASK_SPECS, start=1)}


def validate_task_manifest(task: TaskSpec, manifest: pd.DataFrame, config: RunConfig) -> None:
    required = {
        "task_folder", "repeat", "fold", "split_seed", "role", "sample_id", "participant_id", "label"
    }
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest for {task.folder} lacks columns: {sorted(missing)}")
    task_rows = manifest[manifest["task_folder"].eq(task.folder)].copy()
    if task_rows.empty:
        raise ValueError(f"Manifest has no rows for task {task.folder}")
    master = task_rows[["sample_id", "participant_id", "label"]].drop_duplicates()
    if master["sample_id"].duplicated().any():
        bad = master.loc[master["sample_id"].duplicated(False), "sample_id"].tolist()[:10]
        raise ValueError(f"Task {task.folder} has conflicting sample metadata: {bad}")
    if master.groupby("participant_id")["label"].nunique().gt(1).any():
        raise ValueError(f"Task {task.folder} has participant label conflicts.")
    expected_samples = set(master["sample_id"])
    expected_labels = set(master["label"])
    for repeat in range(1, config.expected_repeats + 1):
        repeat_rows = task_rows[task_rows["repeat"].eq(repeat)]
        if repeat_rows.empty:
            raise ValueError(f"Task {task.folder} is missing repeat {repeat}")
        if repeat_rows["split_seed"].nunique() != 1:
            raise ValueError(f"Task {task.folder}, repeat {repeat} has multiple seeds")
        tested: set[str] = set()
        for fold in range(1, config.expected_folds + 1):
            rows = repeat_rows[repeat_rows["fold"].eq(fold)]
            train = rows[rows["role"].eq("train")]
            test = rows[rows["role"].eq("test")]
            if train.empty or test.empty:
                raise ValueError(f"Task {task.folder}, repeat {repeat}, fold {fold} lacks train/test")
            overlap = set(train["participant_id"]) & set(test["participant_id"])
            if overlap:
                raise ValueError(
                    f"Participant leakage in {task.folder}, repeat {repeat}, fold {fold}: "
                    f"{sorted(overlap)[:10]}"
                )
            if set(train["label"]) != expected_labels or set(test["label"]) != expected_labels:
                raise ValueError(
                    f"Task {task.folder}, repeat {repeat}, fold {fold} does not contain all classes."
                )
            test_ids = set(test["sample_id"])
            duplicate = tested & test_ids
            if duplicate:
                raise ValueError(
                    f"Task {task.folder}, repeat {repeat} repeats test samples: {sorted(duplicate)[:10]}"
                )
            tested |= test_ids
        if tested != expected_samples:
            raise ValueError(
                f"Task {task.folder}, repeat {repeat} test folds cover {len(tested)} / "
                f"{len(expected_samples)} samples."
            )


def assert_ibd_manifest_equivalence(
    generated: pd.DataFrame, ibd_manifest: pd.DataFrame, config: RunConfig
) -> pd.DataFrame:
    task = TASK_BY_FOLDER["IBD_vs_nonIBD"]
    generated = generated[generated["task_folder"].eq(task.folder)].copy()
    locked_label_mapping = _infer_locked_ibd_label_mapping(generated, ibd_manifest)
    audit_rows: list[dict[str, Any]] = []
    for repeat in range(1, config.expected_repeats + 1):
        for fold in range(1, config.expected_folds + 1):
            for role in ("train", "test"):
                left = set(
                    generated.loc[
                        generated["repeat"].eq(repeat)
                        & generated["fold"].eq(fold)
                        & generated["role"].eq(role),
                        "sample_id",
                    ]
                )
                right = set(
                    ibd_manifest.loc[
                        ibd_manifest["repeat"].eq(repeat)
                        & ibd_manifest["fold"].eq(fold)
                        & ibd_manifest["role"].eq(role),
                        "sample_id",
                    ]
                )
                generated_rows = generated.loc[
                    generated["repeat"].eq(repeat)
                    & generated["fold"].eq(fold)
                    & generated["role"].eq(role),
                    ["sample_id", "participant_id"],
                ]
                locked_rows = ibd_manifest.loc[
                    ibd_manifest["repeat"].eq(repeat)
                    & ibd_manifest["fold"].eq(fold)
                    & ibd_manifest["role"].eq(role),
                    ["sample_id", "participant_id"],
                ]
                generated_mapping = dict(zip(generated_rows["sample_id"], generated_rows["participant_id"]))
                locked_mapping = dict(zip(locked_rows["sample_id"], locked_rows["participant_id"]))
                exact_participant_match = generated_mapping == locked_mapping
                generated_label_rows = generated.loc[
                    generated["repeat"].eq(repeat)
                    & generated["fold"].eq(fold)
                    & generated["role"].eq(role),
                    ["sample_id", "label"],
                ]
                locked_label_rows = ibd_manifest.loc[
                    ibd_manifest["repeat"].eq(repeat)
                    & ibd_manifest["fold"].eq(fold)
                    & ibd_manifest["role"].eq(role),
                    ["sample_id", "label"],
                ]
                generated_labels = dict(
                    zip(generated_label_rows["sample_id"], generated_label_rows["label"].map(normalise_label))
                )
                locked_labels = dict(
                    zip(
                        locked_label_rows["sample_id"],
                        locked_label_rows["label"].map(_locked_label_token).map(locked_label_mapping),
                    )
                )
                exact_label_match = generated_labels == locked_labels
                audit_rows.append(
                    {
                        "repeat": repeat,
                        "fold": fold,
                        "role": role,
                        "generated_samples": len(left),
                        "locked_samples": len(right),
                        "exact_sample_match": left == right,
                        "exact_participant_match": exact_participant_match,
                        "exact_label_match": exact_label_match,
                        "only_generated": sorted(left - right)[:10],
                        "only_locked": sorted(right - left)[:10],
                    }
                )
    audit = pd.DataFrame(audit_rows)
    exact = (
        audit["exact_sample_match"]
        & audit["exact_participant_match"]
        & audit["exact_label_match"]
    )
    if not exact.all():
        bad = audit.loc[~exact].head().to_dict("records")
        raise RuntimeError(f"Generated IBD folds do not match the locked manifest: {bad}")
    return audit


def build_all_manifests(
    arrays_by_task: dict[str, TaskArrays],
    ibd_manifest: pd.DataFrame,
    config: RunConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    seeds = repeat_seeds(ibd_manifest, config)
    manifests = []
    audit_rows = []
    for task in TASK_SPECS:
        if task.folder not in arrays_by_task:
            raise KeyError(f"Core arrays are missing task {task.folder}")
        manifest = generate_task_manifest(task, arrays_by_task[task.folder], seeds, config)
        manifests.append(manifest)
        master = manifest[["sample_id", "participant_id", "label"]].drop_duplicates()
        audit_rows.append(
            {
                "task_order": TASK_ORDER_INDEX[task.folder],
                "task_folder": task.folder,
                "task": task.report_name,
                "samples": master["sample_id"].nunique(),
                "participants": master["participant_id"].nunique(),
                "classes": "|".join(ordered_classes(master["label"].tolist(), task)),
                "class_sample_counts": master["label"].value_counts().sort_index().to_dict(),
                "class_participant_counts": master.groupby("label")["participant_id"].nunique().to_dict(),
            }
        )
    combined = pd.concat(manifests, ignore_index=True)
    ibd_audit = assert_ibd_manifest_equivalence(combined, ibd_manifest, config)
    audit = pd.DataFrame(audit_rows).sort_values("task_order")
    return combined, audit, ibd_audit


def write_manifests(
    combined: pd.DataFrame,
    audit: pd.DataFrame,
    ibd_audit: pd.DataFrame,
    output_dir: Path,
) -> dict[str, str]:
    split_dir = output_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    atomic_csv(combined, split_dir / "all_task_split_manifest.csv", index=False)
    atomic_csv(audit, split_dir / "task_split_audit.csv", index=False)
    atomic_csv(ibd_audit, split_dir / "ibd_locked_manifest_equivalence.csv", index=False)
    checksums: dict[str, str] = {}
    columns = [
        "task_folder", "repeat", "fold", "split_seed", "role", "sample_id", "participant_id", "label"
    ]
    for task in TASK_SPECS:
        task_manifest = combined[combined["task_folder"].eq(task.folder)].copy()
        path = split_dir / f"{task.folder}_split_manifest.csv"
        atomic_csv(task_manifest, path, index=False)
        checksums[task.folder] = canonical_frame_sha256(task_manifest, columns)
    return checksums
