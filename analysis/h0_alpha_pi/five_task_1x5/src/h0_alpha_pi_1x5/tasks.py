from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskSpec:
    folder: str
    report_name: str
    class_order: tuple[str, ...]


TASKS: tuple[TaskSpec, ...] = (
    TaskSpec("IBD_vs_nonIBD", "IBD vs non-IBD", ("nonIBD", "IBD")),
    TaskSpec("three_way_nonIBD_UC_CD", "non-IBD vs UC vs CD", ("nonIBD", "UC", "CD")),
    TaskSpec("nonIBD_vs_UC", "non-IBD vs UC", ("nonIBD", "UC")),
    TaskSpec("nonIBD_vs_CD", "non-IBD vs CD", ("nonIBD", "CD")),
    TaskSpec("CD_vs_UC", "UC vs CD", ("CD", "UC")),
)
TASK_BY_FOLDER = {task.folder: task for task in TASKS}


def normalise_condition(value: object) -> str | None:
    text = str(value).strip().lower()
    compact = text.replace("_", "").replace("-", "").replace(" ", "")
    if compact == "cd" or "crohn" in compact:
        return "CD"
    if compact == "uc" or "ulcerative" in compact:
        return "UC"
    if compact == "ibd":
        return "IBD"
    if compact in {"nonibd", "control", "healthy"}:
        return "nonIBD"
    return None


def map_condition_to_task(condition: str, task: TaskSpec) -> str | None:
    condition = normalise_condition(condition) or condition
    if task.folder == "IBD_vs_nonIBD":
        if condition == "nonIBD":
            return "nonIBD"
        if condition in {"UC", "CD", "IBD"}:
            return "IBD"
        return None
    if task.folder == "three_way_nonIBD_UC_CD":
        return condition if condition in {"nonIBD", "UC", "CD"} else None
    if task.folder == "nonIBD_vs_UC":
        return condition if condition in {"nonIBD", "UC"} else None
    if task.folder == "nonIBD_vs_CD":
        return condition if condition in {"nonIBD", "CD"} else None
    if task.folder == "CD_vs_UC":
        return condition if condition in {"CD", "UC"} else None
    raise KeyError(task.folder)
