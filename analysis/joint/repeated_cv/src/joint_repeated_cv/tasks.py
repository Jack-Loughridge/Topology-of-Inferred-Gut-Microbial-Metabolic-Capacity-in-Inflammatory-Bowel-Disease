from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskSpec:
    folder: str
    display_name: str
    report_name: str
    preferred_class_order: tuple[str, ...]


TASK_SPECS: tuple[TaskSpec, ...] = (
    TaskSpec("IBD_vs_nonIBD", "IBD vs non-IBD", "IBD vs non-IBD", ("nonIBD", "IBD")),
    TaskSpec(
        "three_way_nonIBD_UC_CD",
        "non-IBD vs UC vs CD",
        "non-IBD vs UC vs CD",
        ("nonIBD", "UC", "CD"),
    ),
    TaskSpec("nonIBD_vs_UC", "non-IBD vs UC", "non-IBD vs UC", ("nonIBD", "UC")),
    TaskSpec("nonIBD_vs_CD", "non-IBD vs CD", "non-IBD vs CD", ("nonIBD", "CD")),
    TaskSpec("CD_vs_UC", "CD vs UC", "UC vs CD", ("CD", "UC")),
)

TASK_BY_FOLDER = {task.folder: task for task in TASK_SPECS}
TASK_ORDER = tuple(task.folder for task in TASK_SPECS)


def normalise_label(value: object) -> str:
    text = str(value).strip()
    compact = text.lower().replace("_", "-").replace(" ", "")
    aliases = {
        "nonibd": "nonIBD",
        "non-ibd": "nonIBD",
        "nonibdcontrol": "nonIBD",
        "ibd": "IBD",
        "uc": "UC",
        "ulcerativecolitis": "UC",
        "cd": "CD",
        "crohn'sdisease": "CD",
        "crohnsdisease": "CD",
    }
    return aliases.get(compact, text)


def ordered_classes(values: list[str] | tuple[str, ...], task: TaskSpec) -> tuple[str, ...]:
    available = {normalise_label(value) for value in values}
    preferred = [label for label in task.preferred_class_order if label in available]
    extras = sorted(available - set(preferred))
    classes = tuple(preferred + extras)
    if len(classes) < 2:
        raise ValueError(f"Task {task.folder} has fewer than two classes: {classes}")
    return classes
