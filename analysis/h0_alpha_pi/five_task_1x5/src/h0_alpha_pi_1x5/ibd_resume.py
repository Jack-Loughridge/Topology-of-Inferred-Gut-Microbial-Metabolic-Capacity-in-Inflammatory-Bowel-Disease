from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data import load_task_bundle, load_task_manifest
from .engine import EngineConfig, _aggregate_task
from .tasks import TASK_BY_FOLDER
from .util import atomic_json, now_iso


def completed_ibd_folds(output_dir: Path) -> list[int]:
    folds = []
    for fold in range(1, 6):
        if (output_dir / "folds" / f"repeat_01_fold_{fold}" / "FOLD_COMPLETE.json").exists():
            folds.append(fold)
    return folds


def resume_ibd_until_repeat_one_complete(
    repo: Path,
    output_dir: Path,
    split_manifest: Path,
    pd_dir: Path,
    label_csv: Path,
    metadata_csv: Path,
    poll_seconds: float = 1.0,
) -> None:
    completed = completed_ibd_folds(output_dir)
    if completed == [1, 2, 3, 4, 5]:
        print("[IBD] repeat 1 already has five completed folds; no restart needed.", flush=True)
        return
    script = repo / "h0_alpha_pi_repeated_cv.py"
    if not script.exists():
        raise FileNotFoundError(f"Existing corrected IBD H0 script not found: {script}")
    command = [
        os.environ.get("H0_PYTHON", "python3"),
        "-u",
        str(script),
        "--pd-dir",
        str(pd_dir),
        "--label-csv",
        str(label_csv),
        "--metadata-csv",
        str(metadata_csv),
        "--split-manifest",
        str(split_manifest),
        "--output-dir",
        str(output_dir),
        "--expected-repeats",
        "20",
        "--expected-splits",
        "5",
    ]
    print("[IBD] Resuming the existing corrected binary H0 run until repeat 1 reaches 5/5 folds.", flush=True)
    print("[IBD] Command:", " ".join(command), flush=True)
    process = subprocess.Popen(command, cwd=repo, start_new_session=True)
    try:
        while True:
            completed = completed_ibd_folds(output_dir)
            print(f"[IBD watchdog] completed repetition-1 folds: {len(completed)}/5", flush=True)
            if completed == [1, 2, 3, 4, 5]:
                print("[IBD watchdog] Repetition 1 complete; stopping before continued repeated-CV fitting.", flush=True)
                try:
                    os.killpg(process.pid, signal.SIGINT)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait(timeout=10)
                break
            return_code = process.poll()
            if return_code is not None:
                raise RuntimeError(
                    f"Existing IBD H0 process exited with code {return_code} before 5/5 repetition-1 folds completed."
                )
            time.sleep(float(poll_seconds))
    finally:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    if completed_ibd_folds(output_dir) != [1, 2, 3, 4, 5]:
        raise RuntimeError("IBD watchdog ended without five completed repetition-1 folds")


def _generic_binary_frames(source_output: Path) -> tuple[list[dict[str, Any]], list[pd.DataFrame], list[pd.DataFrame], list[pd.DataFrame], list[pd.DataFrame]]:
    results: list[dict[str, Any]] = []
    samples: list[pd.DataFrame] = []
    participants: list[pd.DataFrame] = []
    curves: list[pd.DataFrame] = []
    candidates: list[pd.DataFrame] = []
    for fold in range(1, 6):
        fold_dir = source_output / "folds" / f"repeat_01_fold_{fold}"
        final_dir = fold_dir / "final_outer_model"
        required = [
            fold_dir / "FOLD_COMPLETE.json",
            final_dir / "final_fold_result.json",
            final_dir / "test_predictions_sample.csv",
            final_dir / "test_predictions_participant.csv",
            final_dir / "interpretation_curves.csv",
            fold_dir / "candidate_summary.csv",
        ]
        if not all(path.exists() for path in required):
            raise RuntimeError(f"Incomplete IBD fold {fold}: {[str(path) for path in required if not path.exists()]}")
        source_result = json.loads(required[1].read_text())
        result = dict(source_result)
        result["task_folder"] = "IBD_vs_nonIBD"
        result["task"] = "IBD vs non-IBD"
        result["repeat"] = 1
        result["fold"] = fold
        result["selected_q_step"] = float(source_result["selected_percentile_step"])
        result["selected_n_intervals_inner"] = int(source_result["selected_inner_n_intervals"])
        result["selected_n_intervals_final"] = int(source_result["final_outer_n_intervals"])
        # The generic multiclass Brier definition sums squared error over both classes.
        result["test_sample_brier_multiclass"] = 2.0 * float(source_result["test_sample_brier"])
        result["test_participant_brier_multiclass"] = 2.0 * float(source_result["test_participant_brier"])
        # Normalize the original binary metric naming to the generic task-class naming.
        result["test_sample_recall_nonIBD"] = float(source_result["test_sample_recall_nonibd"])
        result["test_sample_recall_IBD"] = float(source_result["test_sample_recall_ibd"])
        result["test_participant_recall_nonIBD"] = float(source_result["test_participant_recall_nonibd"])
        result["test_participant_recall_IBD"] = float(source_result["test_participant_recall_ibd"])
        results.append(result)

        sample = pd.read_csv(required[2], dtype={"sample_id": str, "participant_id": str})
        sample["probability_nonIBD"] = 1.0 - sample["probability_ibd"]
        sample["probability_IBD"] = sample["probability_ibd"]
        sample["centroid_probability_nonIBD"] = 1.0 - sample["centroid_probability_ibd"]
        sample["centroid_probability_IBD"] = sample["centroid_probability_ibd"]
        samples.append(sample)

        participant = pd.read_csv(required[3], dtype={"participant_id": str})
        participant["probability_nonIBD"] = 1.0 - participant["probability_ibd"]
        participant["probability_IBD"] = participant["probability_ibd"]
        participant["centroid_probability_nonIBD"] = 1.0 - participant["centroid_probability_ibd"]
        participant["centroid_probability_IBD"] = participant["centroid_probability_ibd"]
        participants.append(participant)

        source_curve = pd.read_csv(required[4])
        signed = source_curve["signed_single_death_logit_influence"].to_numpy(dtype=float)
        curve = pd.DataFrame(
            {
                "death_value": source_curve["death_value"],
                "alpha_interpolated": source_curve["alpha_raw_piecewise"],
                "alpha_normalized_interpolated": source_curve["alpha_normalized_mean1_loggrid"],
                "logit_contribution_nonIBD": -0.5 * signed,
                "logit_contribution_IBD": 0.5 * signed,
                "centered_logit_influence_nonIBD": -0.5 * signed,
                "centered_logit_influence_IBD": 0.5 * signed,
                "signed_logit_influence_class1_minus_class0": signed,
            }
        )
        curves.append(curve)
        candidate = pd.read_csv(required[5])
        if "percentile_step" in candidate and "q_step" not in candidate:
            candidate = candidate.rename(columns={"percentile_step": "q_step"})
        candidates.append(candidate.assign(fold=fold, repeat=1))
    return results, samples, participants, curves, candidates


def aggregate_ibd_repeat_one(
    source_output: Path,
    all_task_output: Path,
    pd_dir: Path,
    label_csv: Path,
    metadata_csv: Path,
    split_manifest: Path,
    generic_config: EngineConfig,
) -> dict[str, Any]:
    task = TASK_BY_FOLDER["IBD_vs_nonIBD"]
    manifest, metadata = load_task_manifest(split_manifest, task, 1, 5)
    bundle = load_task_bundle(pd_dir, label_csv, metadata_csv, metadata, task)
    results, samples, participants, curves, candidates = _generic_binary_frames(source_output)
    # Exact held-out-sample verification against the locked manifest.
    for fold, sample_frame in enumerate(samples, start=1):
        expected = set(
            manifest.loc[manifest["fold"].eq(fold) & manifest["role"].eq("test"), "sample_id"]
        )
        observed = set(sample_frame["sample_id"])
        if observed != expected:
            raise RuntimeError(
                f"IBD fold {fold} completed predictions do not match locked repetition-1 test samples: "
                f"only_observed={sorted(observed - expected)[:10]}, only_expected={sorted(expected - observed)[:10]}"
            )
    task_dir = all_task_output / task.folder
    task_dir.mkdir(parents=True, exist_ok=True)
    stop_note = (
        "The corrected repeated H0 analysis was intentionally limited to repetition 1 after computational-cost review.\n\n"
        "All five locked participant-grouped folds in repetition 1 were completed and verified against the shared manifest. "
        "No results from incomplete later repetitions are used. The remaining four diagnosis tasks are run by the "
        "h0_alpha_pi_all_tasks_1x5 orchestration package on their exact shared repetition-1 manifests.\n"
    )
    (source_output / "STOPPED_BY_DESIGN.txt").write_text(stop_note)
    (source_output / "STOPPED_AFTER_REPETITION_1_BY_DESIGN.txt").write_text(stop_note)
    (task_dir / "SOURCE_OUTPUT.txt").write_text(
        "The five final IBD-vs-non-IBD folds were resumed from the corrected repeated-CV package at:\n"
        f"{source_output.resolve()}\n\n"
        "Only repetition 1 is included in this all-task 1x5 analysis. The original fold-level model packages, "
        "candidate checkpoints, training histories and diagnostics remain in that source directory.\n"
    )
    atomic_json(
        task_dir / "source_fold_paths.json",
        {
            "created_utc": now_iso(),
            "source_output": str(source_output.resolve()),
            "folds": {
                str(fold): str((source_output / "folds" / f"repeat_01_fold_{fold}").resolve())
                for fold in range(1, 6)
            },
        },
    )
    summary = _aggregate_task(task_dir, bundle, results, samples, participants, curves, candidates, generic_config)
    atomic_json(
        task_dir / "IBD_REPETITION_1_IMPORTED_AND_VERIFIED.json",
        {
            "completed_utc": now_iso(),
            "source_output": str(source_output.resolve()),
            "verified_folds": 5,
            "verified_samples": len(bundle.sample_ids),
            "summary": summary,
        },
    )
    return summary
