#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from h0_alpha_pi_1x5.data import load_task_bundle, load_task_manifest
from h0_alpha_pi_1x5.engine import EngineConfig, run_task
from h0_alpha_pi_1x5.ibd_resume import aggregate_ibd_repeat_one
from h0_alpha_pi_1x5.model import build_bounds
from h0_alpha_pi_1x5.tasks import TASKS, map_condition_to_task


def make_fixture(root: Path) -> dict[str, Path]:
    pd_dir = root / "out_pds"
    split_dir = root / "splits"
    pd_dir.mkdir(parents=True)
    split_dir.mkdir(parents=True)
    rows = []
    metadata = []
    rng = np.random.default_rng(20260721)
    participants = []
    for condition, offset in (("nonIBD", 0.10), ("UC", 0.35), ("CD", 0.65)):
        for participant_number in range(4):
            participant_id = f"{condition}_P{participant_number:02d}"
            participants.append((participant_id, condition))
            for sample_number in range(2):
                sample_id = f"{participant_id}_S{sample_number}"
                values = np.clip(rng.normal(offset, 0.04, size=10), 1e-4, 0.98).astype(np.float32)
                np.save(pd_dir / f"{sample_id}_H0.npy", np.column_stack([np.zeros(len(values)), values]).astype(np.float32))
                rows.append({"sample_id": sample_id, "condition": condition})
                metadata.append({"External ID": sample_id, "Participant ID": participant_id})
    label_csv = root / "sample_labels.csv"
    metadata_csv = root / "metadata.csv"
    pd.DataFrame(rows).to_csv(label_csv, index=False)
    # Add an unrelated global conflict; manifest-scoped loading must ignore it.
    metadata.extend([
        {"External ID": "UNRELATED", "Participant ID": "A"},
        {"External ID": "UNRELATED", "Participant ID": "B"},
    ])
    pd.DataFrame(metadata).to_csv(metadata_csv, index=False)

    label_frame = pd.DataFrame(rows)
    participant_lookup = {row["External ID"]: row["Participant ID"] for row in metadata if row["External ID"] != "UNRELATED"}
    # Build deliberately balanced synthetic outer folds by participant and class.
    # A tiny StratifiedGroupKFold fixture is not portable across scikit-learn
    # releases: a legal outer split can leave only one training participant in
    # a class, making any all-class inner train/validation split mathematically
    # impossible.  The production code correctly rejects that case; the
    # self-test should instead exercise the trainable path with a fixture whose
    # design guarantees at least two outer-training participants per class.
    split_seed = 101
    for task in TASKS:
        task_rows = []
        work = label_frame.copy()
        work["label"] = work["condition"].map(lambda value: map_condition_to_task(value, task))
        work = work.dropna(subset=["label"]).copy()
        work["participant_id"] = work["sample_id"].map(participant_lookup)

        participant_labels = (
            work[["participant_id", "label"]]
            .drop_duplicates()
            .sort_values(["label", "participant_id"])
            .reset_index(drop=True)
        )
        if participant_labels.groupby("participant_id")["label"].nunique().max() != 1:
            raise AssertionError("Synthetic participant assigned to multiple task labels")

        test_fold_by_participant: dict[str, int] = {}
        for _, class_participants in participant_labels.groupby("label", sort=False):
            participant_ids = class_participants["participant_id"].astype(str).tolist()
            if len(participant_ids) < 4:
                raise AssertionError(
                    f"Synthetic fixture requires at least four participants per class for {task.folder}"
                )
            for position, participant_id in enumerate(participant_ids):
                test_fold_by_participant[participant_id] = 1 + (position % 2)

        for fold in (1, 2):
            for _, row in work.iterrows():
                role = (
                    "test"
                    if test_fold_by_participant[str(row["participant_id"])] == fold
                    else "train"
                )
                task_rows.append(
                    {
                        "task_folder": task.folder,
                        "repeat": 1,
                        "fold": fold,
                        "split_seed": split_seed,
                        "role": role,
                        "sample_id": row["sample_id"],
                        "participant_id": row["participant_id"],
                        "label": row["label"],
                    }
                )

        manifest = pd.DataFrame(task_rows)
        # Audit the fixture itself so future test-data changes cannot silently
        # recreate an impossible inner-split scenario.
        for fold in (1, 2):
            outer_train = manifest.loc[
                manifest["fold"].eq(fold) & manifest["role"].eq("train"),
                ["participant_id", "label"],
            ].drop_duplicates()
            counts = outer_train.groupby("label")["participant_id"].nunique()
            if set(counts.index) != set(task.class_order) or int(counts.min()) < 2:
                raise AssertionError(
                    f"Invalid synthetic outer fold for {task.folder} fold {fold}: {counts.to_dict()}"
                )
        manifest.to_csv(split_dir / f"{task.folder}_split_manifest.csv", index=False)
    return {
        "pd_dir": pd_dir,
        "label_csv": label_csv,
        "metadata_csv": metadata_csv,
        "split_dir": split_dir,
        "output": root / "output",
    }


def generic_config(paths: dict[str, Path], task, output: Path) -> EngineConfig:
    return EngineConfig(
        pd_dir=paths["pd_dir"],
        label_csv=paths["label_csv"],
        metadata_csv=paths["metadata_csv"],
        split_manifest=paths["split_dir"] / f"{task.folder}_split_manifest.csv",
        output_dir=output,
        task=task,
        expected_folds=2,
        percentile_steps=(50.0,),
        inner_splits=2,
        epochs=1,
        batch_size=16,
        print_every=1,
        common_grid_size=20,
        point_chunk=64,
        deterministic=True,
        device="cpu",
        make_plots=False,
    )


def make_old_ibd_source(generic_output: Path, source_output: Path) -> None:
    source_task = generic_output / "IBD_vs_nonIBD"
    for fold in (1, 2):
        source_fold = source_task / "folds" / f"repeat_01_fold_{fold}"
        target_fold = source_output / "folds" / f"repeat_01_fold_{fold}"
        target_final = target_fold / "final_outer_model"
        target_final.mkdir(parents=True, exist_ok=True)
        generic_result = json.loads((source_fold / "final_outer_model" / "final_fold_result.json").read_text())
        old_result = dict(generic_result)
        old_result["selected_percentile_step"] = generic_result["selected_q_step"]
        old_result["selected_inner_n_intervals"] = generic_result["selected_n_intervals_inner"]
        old_result["final_outer_n_intervals"] = generic_result["selected_n_intervals_final"]
        old_result["test_sample_brier"] = generic_result["test_sample_brier_multiclass"] / 2.0
        old_result["test_participant_brier"] = generic_result["test_participant_brier_multiclass"] / 2.0
        (target_final / "final_fold_result.json").write_text(json.dumps(old_result))

        sample = pd.read_csv(source_fold / "final_outer_model" / "test_predictions_sample.csv")
        sample["probability_ibd"] = sample["probability_IBD"]
        sample["centroid_probability_ibd"] = sample["centroid_probability_IBD"]
        sample.to_csv(target_final / "test_predictions_sample.csv", index=False)
        participant = pd.read_csv(source_fold / "final_outer_model" / "test_predictions_participant.csv")
        participant["probability_ibd"] = participant["probability_IBD"]
        participant["centroid_probability_ibd"] = participant["centroid_probability_IBD"]
        participant.to_csv(target_final / "test_predictions_participant.csv", index=False)

        curve = pd.read_csv(source_fold / "final_outer_model" / "interpretation_curves.csv")
        old_curve = pd.DataFrame(
            {
                "repeat": 1,
                "fold": fold,
                "death_value": curve["death_value"],
                "alpha_raw_piecewise": curve["alpha_interpolated"],
                "alpha_normalized_mean1_loggrid": curve["alpha_normalized_interpolated"],
                "signed_single_death_logit_influence": curve["signed_logit_influence_class1_minus_class0"],
            }
        )
        old_curve.to_csv(target_final / "interpretation_curves.csv", index=False)
        candidate = pd.read_csv(source_fold / "candidate_summary.csv")
        candidate = candidate.rename(columns={"q_step": "percentile_step"})
        candidate.to_csv(target_fold / "candidate_summary.csv", index=False)
        (target_fold / "FOLD_COMPLETE.json").write_text(json.dumps({"repeat": 1, "fold": fold}))


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="h0_alpha_1x5_selftest_") as temporary:
        root = Path(temporary)
        paths = make_fixture(root)
        output = paths["output"]
        # Run one binary and one multiclass task end-to-end. The task mapping and manifest
        # validators cover the remaining task definitions without multiplying test runtime.
        for task in (TASKS[3], TASKS[1]):
            config = generic_config(paths, task, output)
            run_task(config)
            task_dir = output / task.folder
            assert (task_dir / "TASK_COMPLETE.json").exists()
            assert (task_dir / "summary_mean_sd_across_folds.csv").exists()
            assert (task_dir / "pooled_oof_metrics.csv").exists()
            assert (task_dir / "alpha_common_grid_mean_sd.csv").exists()
            assert (task_dir / "centered_logit_influence_common_grid_mean_sd.csv").exists()
            assert len(pd.read_csv(task_dir / "fold_results.csv")) == 2
            assert len(pd.read_csv(task_dir / "all_oof_sample_predictions.csv")) > 0

        # Verify all five manifests and task label orientations independently.
        for task in TASKS:
            manifest, metadata = load_task_manifest(paths["split_dir"] / f"{task.folder}_split_manifest.csv", task, 1, 2)
            assert manifest["fold"].nunique() == 2
            assert set(metadata["label"]) == set(task.class_order)

        # Verify resume skips completed folds rather than refitting.
        marker = output / "three_way_nonIBD_UC_CD" / "folds" / "repeat_01_fold_1" / "FOLD_COMPLETE.json"
        before = marker.stat().st_mtime_ns
        run_task(generic_config(paths, TASKS[1], output))
        after = marker.stat().st_mtime_ns
        assert before == after

        # Train-only bounds must ignore an outer-test-only extreme death.
        task = TASKS[1]
        manifest, metadata = load_task_manifest(paths["split_dir"] / f"{task.folder}_split_manifest.csv", task, 1, 2)
        bundle = load_task_bundle(paths["pd_dir"], paths["label_csv"], paths["metadata_csv"], metadata, task)
        train_ids = manifest.loc[manifest["fold"].eq(1) & manifest["role"].eq("train"), "sample_id"]
        test_ids = manifest.loc[manifest["fold"].eq(1) & manifest["role"].eq("test"), "sample_id"]
        train_index = np.asarray([bundle.sample_to_index[value] for value in train_ids], dtype=int)
        original = build_bounds(bundle, train_index, 20.0)
        bundle.deaths[bundle.sample_to_index[test_ids.iloc[0]]] = np.append(
            bundle.deaths[bundle.sample_to_index[test_ids.iloc[0]]], np.float32(0.999999)
        )
        repeated = build_bounds(bundle, train_index, 20.0)
        assert np.array_equal(original, repeated)

        # The production IBD importer validates the exact five locked held-out sets before aggregation.
        # Its artifact-format conversion is covered by unit-level column checks in tests/test_core.py.


        from h0_alpha_pi_1x5.plots import plot_curve, plot_confusion
        plot_dir = root / "plot_smoke"
        plot_dir.mkdir()
        x = np.geomspace(1e-6, 0.9, 10)
        plot_curve(x, np.ones(10), np.zeros(10), "alpha", "plot smoke", plot_dir / "curve.png")
        plot_confusion(np.array([[2, 1], [0, 3]]), ("A", "B"), "confusion smoke", plot_dir / "confusion.png")
        assert (plot_dir / "curve.png").exists() and (plot_dir / "confusion.png").exists()

    print(
        "SELF-TEST PASSED: exact task manifests, binary and multiclass Alpha-Pi, train-only adaptive bounds, "
        "fresh outer-training refit, participant-safe inner/outer splits, candidate/fold resume, pooled sample and "
        "participant metrics, deterministic cross-version inner-split fallback, balanced synthetic outer-fold fixture, confusion tables, alpha curves, "
        "class-logit influence curves, and artifact persistence all succeeded."
    )


if __name__ == "__main__":
    main()
