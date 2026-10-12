from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class TaskSpec:
    folder: str
    display_name: str
    classes: tuple[str, ...]
    positive_class: str | None = None

    @property
    def is_binary(self) -> bool:
        return len(self.classes) == 2


TASKS: tuple[TaskSpec, ...] = (
    TaskSpec(
        folder="three_way_nonIBD_UC_CD",
        display_name="non-IBD vs UC vs CD",
        classes=("nonIBD", "UC", "CD"),
    ),
    TaskSpec(
        folder="IBD_vs_nonIBD",
        display_name="IBD vs non-IBD",
        classes=("nonIBD", "IBD"),
        positive_class="IBD",
    ),
    TaskSpec(
        folder="nonIBD_vs_UC",
        display_name="non-IBD vs UC",
        classes=("nonIBD", "UC"),
        positive_class="UC",
    ),
    TaskSpec(
        folder="nonIBD_vs_CD",
        display_name="non-IBD vs CD",
        classes=("nonIBD", "CD"),
        positive_class="CD",
    ),
    TaskSpec(
        folder="CD_vs_UC",
        display_name="CD vs UC",
        classes=("CD", "UC"),
        positive_class="UC",
    ),
)


def normalise_label(value: object) -> str:
    raw = str(value).strip()
    key = raw.lower().replace("-", "").replace("_", "").replace(" ", "")
    if key in {"nonibd", "healthy", "control", "hc", "noninflammatoryboweldisease"}:
        return "nonIBD"
    if key in {"uc", "ulcerativecolitis"}:
        return "UC"
    if key in {"cd", "crohn", "crohns", "crohnsdisease"}:
        return "CD"
    if key == "ibd":
        return "IBD"
    return raw


def labels_for_task(base_labels: Iterable[object], task: TaskSpec) -> np.ndarray:
    labels = np.asarray([normalise_label(x) for x in base_labels], dtype=object)
    if task.folder == "IBD_vs_nonIBD":
        return np.asarray(["IBD" if x in {"UC", "CD"} else "nonIBD" for x in labels], dtype=object)
    return labels


def mask_for_task(base_labels: Iterable[object], task: TaskSpec) -> np.ndarray:
    labels = labels_for_task(base_labels, task)
    return np.asarray([x in task.classes for x in labels], dtype=bool)


def select_tasks(value: str) -> list[TaskSpec]:
    if value.strip().lower() == "all":
        return list(TASKS)
    requested = {x.strip() for x in value.split(",") if x.strip()}
    by_folder = {task.folder: task for task in TASKS}
    missing = requested - set(by_folder)
    if missing:
        raise ValueError(f"Unknown task folder(s): {sorted(missing)}")
    return [task for task in TASKS if task.folder in requested]
