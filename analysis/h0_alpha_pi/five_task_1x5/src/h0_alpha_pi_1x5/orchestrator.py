from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from .data import fold_indices, load_task_bundle, load_task_manifest
from .engine import EngineConfig, _inner_split, run_task
from .ibd_resume import aggregate_ibd_repeat_one, completed_ibd_folds, resume_ibd_until_repeat_one_complete
from .tasks import TASKS
from .util import atomic_csv, atomic_json, normalise_sample_id, now_iso, sha256_file

EXPECTED_IBD_CORRECTED_SCRIPT_SHA256 = "97c0e15e3e01eb2461c640ac083970780f9a4cf88aa6fdcf92ed092976ac8381"


def default_config(base: Path | None = None) -> dict[str, Any]:
    base = base or Path.home() / "Real_Data"
    split_dir = base / "H0_Ricci_JointSparse_RepeatedCV_AllTasks" / "splits"
    return {
        "base": base,
        "pd_dir": base / "out_pds",
        "label_csv": base / "sample_labels.csv",
        "metadata_csv": base / "hmp2_metadata.csv",
        "split_dir": split_dir,
        "output_dir": base / "H0_AlphaPi_1x5_AllTasks",
        "ibd_repo": base / "h0_alpha_pi_repeated_cv",
        "ibd_source_output": base / "H0_AlphaPi_RepeatedCV_TrainOnlyBins_IBD",
        "ibd_locked_manifest": base / "Ricci_IBD_RepeatedCV_CPath" / "splits" / "sample_split_manifest.csv",
    }



def verify_ibd_manifest_equivalence(original_manifest: Path, shared_manifest: Path) -> dict[str, Any]:
    """Prove that repetition 1 of the original binary manifest equals the shared all-task manifest.

    The existing corrected IBD H0 run was initialized with the original Ricci
    repeated-CV manifest, whose labels are numeric.  The all-task manifest uses
    string labels and includes a task_folder column.  We compare the scientific
    split identity directly: every sample's participant, fold, role, binary
    label, and split seed must match in repetition 1.
    """
    original = pd.read_csv(original_manifest, low_memory=False)
    shared = pd.read_csv(shared_manifest, low_memory=False)
    required_original = {"repeat", "fold", "split_seed", "role", "sample_id", "participant_id", "label"}
    required_shared = required_original | {"task_folder"}
    missing_original = required_original - set(original.columns)
    missing_shared = required_shared - set(shared.columns)
    if missing_original or missing_shared:
        raise ValueError(
            f"Cannot compare IBD manifests; missing original={sorted(missing_original)}, "
            f"missing shared={sorted(missing_shared)}"
        )
    original = original[pd.to_numeric(original["repeat"], errors="raise").astype(int).eq(1)].copy()
    shared = shared[
        pd.to_numeric(shared["repeat"], errors="raise").astype(int).eq(1)
        & shared["task_folder"].astype(str).str.strip().eq("IBD_vs_nonIBD")
    ].copy()
    if original.empty or shared.empty:
        raise ValueError("One of the IBD manifests has no repetition-1 rows")

    def canonical(frame: pd.DataFrame, shared_labels: bool) -> pd.DataFrame:
        output = frame[["fold", "split_seed", "role", "sample_id", "participant_id", "label"]].copy()
        output["fold"] = pd.to_numeric(output["fold"], errors="raise").astype(int)
        output["split_seed"] = pd.to_numeric(output["split_seed"], errors="raise").astype(int)
        output["role"] = output["role"].astype(str).str.strip().str.lower()
        output["sample_id"] = output["sample_id"].map(normalise_sample_id)
        output["participant_id"] = output["participant_id"].astype(str).str.strip()
        if shared_labels:
            label_text = output["label"].astype(str).str.strip().str.lower().str.replace("-", "", regex=False).str.replace("_", "", regex=False).str.replace(" ", "", regex=False)
            mapping = {"nonibd": 0, "ibd": 1}
            unknown = sorted(set(label_text) - set(mapping))
            if unknown:
                raise ValueError(f"Unexpected shared IBD labels: {unknown}")
            output["label"] = label_text.map(mapping).astype(int)
        else:
            output["label"] = pd.to_numeric(output["label"], errors="raise").astype(int)
        return output.sort_values(["fold", "role", "sample_id"]).reset_index(drop=True)

    left = canonical(original, False)
    right = canonical(shared, True)
    if len(left) != len(right) or not left.equals(right):
        merged = left.merge(
            right,
            on=["fold", "role", "sample_id"],
            how="outer",
            suffixes=("_original", "_shared"),
            indicator=True,
        )
        mismatch = merged[
            merged["_merge"].ne("both")
            | merged.get("participant_id_original", pd.Series(index=merged.index, dtype=object)).ne(
                merged.get("participant_id_shared", pd.Series(index=merged.index, dtype=object))
            )
            | merged.get("label_original", pd.Series(index=merged.index, dtype=float)).ne(
                merged.get("label_shared", pd.Series(index=merged.index, dtype=float))
            )
            | merged.get("split_seed_original", pd.Series(index=merged.index, dtype=float)).ne(
                merged.get("split_seed_shared", pd.Series(index=merged.index, dtype=float))
            )
        ]
        raise RuntimeError(
            "Original corrected IBD repetition-1 manifest does not match the shared all-task IBD manifest. "
            f"First mismatches: {mismatch.head(10).to_dict(orient='records')}"
        )
    return {
        "rows": int(len(left)),
        "samples": int(left["sample_id"].nunique()),
        "participants": int(left["participant_id"].nunique()),
        "folds": int(left["fold"].nunique()),
        "split_seeds": sorted(left["split_seed"].unique().astype(int).tolist()),
    }

def engine_config_for_task(paths: dict[str, Any], task, **overrides: Any) -> EngineConfig:
    split_manifest = paths["split_dir"] / f"{task.folder}_split_manifest.csv"
    config = EngineConfig(
        pd_dir=paths["pd_dir"],
        label_csv=paths["label_csv"],
        metadata_csv=paths["metadata_csv"],
        split_manifest=split_manifest,
        output_dir=paths["output_dir"],
        task=task,
        repeat=1,
        expected_folds=5,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    return replace(config, **overrides)


def validate_all(paths: dict[str, Any], **engine_overrides: Any) -> pd.DataFrame:
    rows = []
    inner_rows = []
    required_paths = [
        paths["pd_dir"],
        paths["label_csv"],
        paths["metadata_csv"],
        paths["split_dir"],
        paths["ibd_repo"] / "h0_alpha_pi_repeated_cv.py",
        paths["ibd_source_output"],
        paths["ibd_locked_manifest"],
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Required paths are missing: {missing}")
    ibd_script = paths["ibd_repo"] / "h0_alpha_pi_repeated_cv.py"
    ibd_script_sha256 = sha256_file(ibd_script)
    if ibd_script_sha256 != EXPECTED_IBD_CORRECTED_SCRIPT_SHA256:
        raise RuntimeError(
            "Existing IBD H0 script is not the audited corrected v2 source required to resume the partial run. "
            f"Current SHA-256={ibd_script_sha256}; expected={EXPECTED_IBD_CORRECTED_SCRIPT_SHA256}"
        )
    ibd_manifest_equivalence = verify_ibd_manifest_equivalence(
        paths["ibd_locked_manifest"],
        paths["split_dir"] / "IBD_vs_nonIBD_split_manifest.csv",
    )
    for order, task in enumerate(TASKS, start=1):
        config = engine_config_for_task(paths, task, **engine_overrides)
        manifest, metadata = load_task_manifest(config.split_manifest, task, 1, 5)
        bundle = load_task_bundle(config.pd_dir, config.label_csv, config.metadata_csv, metadata, task)
        verified_inner_folds = 0
        if task.folder != "IBD_vs_nonIBD":
            for fold in range(1, config.expected_folds + 1):
                outer_train, outer_test, outer_seed = fold_indices(manifest, bundle, fold)
                inner_train, validation, accepted_inner_seed = _inner_split(bundle, outer_train, fold, config)
                verified_inner_folds += 1
                inner_rows.append(
                    {
                        "task_order": order,
                        "task_folder": task.folder,
                        "fold": fold,
                        "outer_split_seed": int(outer_seed),
                        "accepted_inner_split_seed": int(accepted_inner_seed),
                        "outer_train_samples": int(len(outer_train)),
                        "outer_test_samples": int(len(outer_test)),
                        "inner_train_samples": int(len(inner_train)),
                        "validation_samples": int(len(validation)),
                        "inner_train_participants": int(len(set(bundle.participant_ids[index] for index in inner_train))),
                        "validation_participants": int(len(set(bundle.participant_ids[index] for index in validation))),
                    }
                )
        rows.append(
            {
                "task_order": order,
                "task_folder": task.folder,
                "task": task.report_name,
                "samples": len(bundle.sample_ids),
                "participants": len(set(bundle.participant_ids)),
                "classes": "|".join(task.class_order),
                "folds": manifest["fold"].nunique(),
                "generic_inner_folds_verified": verified_inner_folds,
            }
        )
    audit = pd.DataFrame(rows)
    paths["output_dir"].mkdir(parents=True, exist_ok=True)
    atomic_csv(audit, paths["output_dir"] / "preflight_task_audit.csv", index=False)
    atomic_csv(pd.DataFrame(inner_rows), paths["output_dir"] / "preflight_inner_split_audit.csv", index=False)
    atomic_json(
        paths["output_dir"] / "PREFLIGHT_PASSED.json",
        {
            "completed_utc": now_iso(),
            "task_order": [task.folder for task in TASKS],
            "design": "one locked repetition x five participant-grouped outer folds per task",
            "ibd_completed_folds_at_preflight": completed_ibd_folds(paths["ibd_source_output"]),
            "ibd_corrected_script_sha256": ibd_script_sha256,
            "ibd_original_vs_shared_repeat1_manifest": ibd_manifest_equivalence,
            "audit": rows,
        },
    )
    return audit


def _combined_summary(paths: dict[str, Any]) -> pd.DataFrame:
    frames = []
    for task in TASKS:
        path = paths["output_dir"] / task.folder / "summary_mean_sd_across_folds.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing completed task summary: {path}")
        frames.append(pd.read_csv(path))
    combined = pd.concat(frames, ignore_index=True)
    order = {task.folder: index for index, task in enumerate(TASKS, start=1)}
    combined.insert(0, "task_order", combined["task_folder"].map(order))
    combined = combined.sort_values("task_order").reset_index(drop=True)
    atomic_csv(combined, paths["output_dir"] / "ALL_TASK_SUMMARY.csv", index=False)
    manuscript_columns = [
        "task_order",
        "task",
        "n_samples",
        "n_participants",
        "sample_accuracy_mean",
        "sample_accuracy_sd",
        "sample_balanced_accuracy_mean",
        "sample_balanced_accuracy_sd",
        "sample_macro_f1_mean",
        "sample_macro_f1_sd",
        "sample_roc_auc_mean",
        "sample_roc_auc_sd",
        "selected_q_step_mean",
        "selected_q_step_sd",
        "selected_intervals_mean",
        "selected_intervals_sd",
    ]
    atomic_csv(combined[manuscript_columns], paths["output_dir"] / "MANUSCRIPT_H0_1x5_TABLE.csv", index=False)
    return combined


def run_all(paths: dict[str, Any], **engine_overrides: Any) -> pd.DataFrame:
    validate_all(paths, **engine_overrides)
    paths["output_dir"].mkdir(parents=True, exist_ok=True)
    atomic_json(
        paths["output_dir"] / "RUN_REQUEST.json",
        {
            "started_utc": now_iso(),
            "task_order": [task.folder for task in TASKS],
            "repeat_range": [1, 1],
            "folds_per_task": 5,
            "total_outer_folds": 25,
            "ibd_strategy": "resume corrected existing binary run and stop immediately after repetition 1",
        },
    )

    ibd_task = TASKS[0]
    resume_ibd_until_repeat_one_complete(
        repo=paths["ibd_repo"],
        output_dir=paths["ibd_source_output"],
        split_manifest=paths["ibd_locked_manifest"],
        pd_dir=paths["pd_dir"],
        label_csv=paths["label_csv"],
        metadata_csv=paths["metadata_csv"],
    )
    ibd_config = engine_config_for_task(paths, ibd_task, **engine_overrides)
    aggregate_ibd_repeat_one(
        source_output=paths["ibd_source_output"],
        all_task_output=paths["output_dir"],
        pd_dir=paths["pd_dir"],
        label_csv=paths["label_csv"],
        metadata_csv=paths["metadata_csv"],
        split_manifest=ibd_config.split_manifest,
        generic_config=ibd_config,
    )

    for task in TASKS[1:]:
        config = engine_config_for_task(paths, task, **engine_overrides)
        run_task(config)

    combined = _combined_summary(paths)
    atomic_json(
        paths["output_dir"] / "RUN_COMPLETE.json",
        {
            "completed_utc": now_iso(),
            "task_order": [task.folder for task in TASKS],
            "completed_tasks": 5,
            "completed_outer_folds": 25,
            "repeat_range": [1, 1],
            "summary": str((paths["output_dir"] / "ALL_TASK_SUMMARY.csv").resolve()),
        },
    )
    print("=" * 112, flush=True)
    print("ALL FIVE H0 ALPHA-PI TASKS COMPLETE: ONE LOCKED REPETITION x FIVE FOLDS", flush=True)
    print(paths["output_dir"] / "MANUSCRIPT_H0_1x5_TABLE.csv", flush=True)
    print("=" * 112, flush=True)
    return combined
