#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from install_core_patch import patched_text, synthetic_patcher_test, PATCH_MARKER
from joint_repeated_cv.aggregate import aggregate_completed, verify_repeat_output
from joint_repeated_cv.config import RunConfig
from joint_repeated_cv.core import EXPECTED_CORE_OUTPUTS
from joint_repeated_cv.metrics import classification_metrics, standardise_prediction_frame
from joint_repeated_cv.splits import (
    TaskArrays,
    build_all_manifests,
    validate_task_manifest,
    write_manifests,
)
from joint_repeated_cv.tasks import TASK_SPECS


def build_synthetic_arrays() -> dict[str, TaskArrays]:
    sample_ids: list[str] = []
    participants: list[str] = []
    conditions: list[str] = []
    for condition_index, condition in enumerate(("nonIBD", "UC", "CD")):
        for participant_index in range(6):
            participant = f"{condition[0]}P{participant_index:02d}"
            for sample_index in range(2):
                sample_ids.append(f"S{condition_index}_{participant_index:02d}_{sample_index}")
                participants.append(participant)
                conditions.append(condition)
    sample = np.asarray(sample_ids, dtype=object)
    group = np.asarray(participants, dtype=object)
    label3 = np.asarray(conditions, dtype=object)

    def subset(mask: np.ndarray, labels: np.ndarray) -> TaskArrays:
        return TaskArrays(sample[mask], group[mask], labels)

    is_non = label3 == "nonIBD"
    is_uc = label3 == "UC"
    is_cd = label3 == "CD"
    return {
        "IBD_vs_nonIBD": subset(np.ones(len(sample), dtype=bool), np.where(is_non, "nonIBD", "IBD")),
        "three_way_nonIBD_UC_CD": subset(np.ones(len(sample), dtype=bool), label3.copy()),
        "nonIBD_vs_UC": subset(is_non | is_uc, label3[is_non | is_uc]),
        "nonIBD_vs_CD": subset(is_non | is_cd, label3[is_non | is_cd]),
        "CD_vs_UC": subset(is_cd | is_uc, label3[is_cd | is_uc]),
    }


def build_ibd_manifest(arrays: TaskArrays, repeats: int, folds: int) -> pd.DataFrame:
    rows = []
    seeds = [13, 1022]
    for repeat in range(1, repeats + 1):
        seed = seeds[repeat - 1]
        splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
        for fold, (train_idx, test_idx) in enumerate(
            splitter.split(np.zeros(len(arrays.labels)), arrays.labels, groups=arrays.participant_ids), start=1
        ):
            for role, indices in (("train", train_idx), ("test", test_idx)):
                for index in indices:
                    rows.append(
                        {
                            "repeat": repeat,
                            "fold": fold,
                            "split_seed": seed,
                            "role": role,
                            "sample_id": arrays.sample_ids[index],
                            "participant_id": arrays.participant_ids[index],
                            "label": int(arrays.labels[index] == "IBD"),
                        }
                    )
    return pd.DataFrame(rows)


def config_for(root: Path, manifest_path: Path) -> RunConfig:
    return RunConfig(
        joint_repo=str(root / "joint"),
        ibd_split_manifest=str(manifest_path),
        h0_results_dir=str(root / "h0"),
        h0_pd_dir=str(root / "pds"),
        ricci_original_dir=str(root / "ricci_original"),
        ricci_feature_dir=str(root / "ricci_features"),
        output_dir=str(root / "output"),
        expected_repeats=2,
        expected_folds=2,
        n_inner_splits=2,
        n_jobs=1,
        interval_grid=(64, 96, 160),
        lambda_h0_grid=(0.5, 1.0, 2.0, 4.0),
        lambda_ricci_grid=(0.5, 1.0, 2.0),
        logistic_max_iter=100,
        active_set_max_rounds=5,
        start_repeat=1,
        end_repeat=2,
    )


def task_classes(task_folder: str) -> tuple[str, ...]:
    return {
        "IBD_vs_nonIBD": ("nonIBD", "IBD"),
        "three_way_nonIBD_UC_CD": ("nonIBD", "UC", "CD"),
        "nonIBD_vs_UC": ("nonIBD", "UC"),
        "nonIBD_vs_CD": ("nonIBD", "CD"),
        "CD_vs_UC": ("CD", "UC"),
    }[task_folder]


def write_fake_repeat(config: RunConfig, manifest: pd.DataFrame, task_folder: str, repeat: int) -> None:
    root = Path(config.output_dir) / "runs" / task_folder / f"repeat_{repeat:02d}"
    root.mkdir(parents=True, exist_ok=True)
    classes = task_classes(task_folder)
    test = manifest[
        manifest["task_folder"].eq(task_folder)
        & manifest["repeat"].eq(repeat)
        & manifest["role"].eq("test")
    ][["fold", "sample_id", "participant_id", "label"]].drop_duplicates("sample_id").copy()
    test["task_folder"] = task_folder
    test["true_label"] = test["label"]
    test["predicted_label"] = test["label"]
    for label in classes:
        test[f"probability_{label}"] = np.where(test["label"].eq(label), 0.94, 0.03)
    if len(classes) == 2:
        # Make rows sum exactly to one.
        test[f"probability_{classes[1]}"] = 1.0 - test[f"probability_{classes[0]}"]
    else:
        probability_columns = [f"probability_{label}" for label in classes]
        values = test[probability_columns].to_numpy(float)
        test[probability_columns] = values / values.sum(axis=1, keepdims=True)
    test.to_csv(root / "ALL_TEST_PREDICTIONS.csv", index=False)

    fold_rows = []
    selected_rows = []
    for fold in range(1, config.expected_folds + 1):
        fold_rows.append(
            {
                "task_folder": task_folder,
                "task": task_folder,
                "fold": fold,
                "accuracy": 1.0,
                "balanced_accuracy": 1.0,
                "macro_f1": 1.0,
                "roc_auc": 1.0,
            }
        )
        selected_rows.append(
            {
                "task_folder": task_folder,
                "task": task_folder,
                "fold": fold,
                "selected_intervals": 64 if fold == 1 else 160,
                "lambda_h0": 4.0,
                "lambda_ricci": 1.0,
            }
        )
    pd.DataFrame(fold_rows).to_csv(root / "ALL_FOLD_RESULTS.csv", index=False)
    pd.DataFrame([{"task_folder": task_folder, "task": task_folder}]).to_csv(
        root / "ALL_SUMMARIES.csv", index=False
    )
    pd.DataFrame(selected_rows).to_csv(root / "ALL_SELECTED_CONFIGS.csv", index=False)
    pd.DataFrame(selected_rows).to_csv(root / "ALL_INNER_CONFIG_RESULTS.csv", index=False)
    pd.DataFrame(
        [
            {
                "task_folder": task_folder,
                "fold": 1,
                "h0_nonzero_coefficients": 2,
                "ricci_nonzero_coefficients": 3,
            }
        ]
    ).to_csv(root / "ALL_SPARSITY_DIAGNOSTICS.csv", index=False)
    pd.DataFrame(
        [
            {
                "task_folder": task_folder,
                "fold": 1,
                "interval_index": 0,
                "log_center": -2.0,
                "death_center": np.exp(-2.0),
                "alpha": 1.0,
            }
        ]
    ).to_csv(root / "ALL_ALPHA_PROFILES.csv", index=False)
    pd.DataFrame(
        {
            "feature_index": np.arange(4),
            "process": ["A", "A", "B", "B"],
            "feature_type": ["B", "K0", "B", "K0"],
        }
    ).to_csv(root / "RICCI_FEATURE_METADATA_USED.csv", index=False)
    split_seed = int(
        manifest.loc[
            manifest["task_folder"].eq(task_folder) & manifest["repeat"].eq(repeat),
            "split_seed",
        ].iloc[0]
    )
    run_config = {
        "tasks": task_folder,
        "n_outer_splits": config.expected_folds,
        "n_inner_splits": config.n_inner_splits,
        "seed": split_seed,
        "selection_metric": config.selection_metric,
        "interval_grid": list(config.interval_grid),
        "lambda_h0_grid": list(config.lambda_h0_grid),
        "lambda_ricci_grid": list(config.lambda_ricci_grid),
        "c_value": config.c_value,
        "logistic_max_iter": config.logistic_max_iter,
        "logistic_tolerance": config.logistic_tolerance,
        "alpha_smoothness_gamma": config.alpha_smoothness_gamma,
        "n_alternations": config.alternations,
        "quad_points": config.quad_points,
        "active_set_initial": config.active_set_initial,
        "active_set_batch": config.active_set_batch,
        "active_set_max_rounds": config.active_set_max_rounds,
    }
    (root / "RUN_CONFIG.json").write_text(json.dumps(run_config) + "\n", encoding="utf-8")

    coefficient_rows = len(classes)
    task_root = root / task_folder
    for fold in range(1, config.expected_folds + 1):
        fold_root = task_root / f"fold_{fold}"
        fold_root.mkdir(parents=True, exist_ok=True)
        beta_h0 = np.vstack(
            [np.asarray([0.2, -0.1, 0.0, 0.05]) * (index + 1) for index in range(coefficient_rows)]
        )
        beta_ricci = np.vstack(
            [np.asarray([0.3, -0.2, 0.0, 0.1]) * (index + 1) for index in range(coefficient_rows)]
        )
        np.savez_compressed(
            fold_root / "outer_model.state.npz",
            beta_h0=beta_h0,
            ricci_indices=np.arange(4, dtype=int),
            beta_ricci_active=beta_ricci,
            n_ricci=np.asarray([4], dtype=int),
        )
        pd.DataFrame(
            {
                "interval_index": np.arange(4),
                "log_center": [-4.0, -3.0, -2.0, -1.0],
                "death_center": np.exp([-4.0, -3.0, -2.0, -1.0]),
                "alpha": [0.7, 0.9, 1.1, 1.3],
            }
        ).to_csv(fold_root / "alpha_profile.csv", index=False)
        pd.DataFrame([{"fold": fold, "accuracy": 1.0}]).to_csv(
            fold_root / "fold_result.csv", index=False
        )
        test[test["fold"].eq(fold)].to_csv(fold_root / "test_predictions.csv", index=False)
        pd.DataFrame([{"n_intervals": 64, "roc_auc": 1.0}]).to_csv(
            fold_root / "inner_config_results.csv", index=False
        )
        (fold_root / "selected_config.json").write_text(
            json.dumps({"n_intervals": 64, "lambda_h0": 4.0, "lambda_ricci": 1.0}) + "\n",
            encoding="utf-8",
        )
        (fold_root / "sparsity_diagnostics.json").write_text(
            json.dumps({"h0_nonzero_coefficients": 2, "ricci_nonzero_coefficients": 3}) + "\n",
            encoding="utf-8",
        )
        (fold_root / "COMPLETE.json").write_text(
            json.dumps({"completed": True, "fold": fold}) + "\n", encoding="utf-8"
        )
    assert all((root / name).exists() for name in EXPECTED_CORE_OUTPUTS)


def main() -> None:
    synthetic_patcher_test()
    source = '''def alpha_from_logits(x):\n    return x\ndef _alpha_loss_and_gradient(\n    logits, phi_standardised, ricci_scores, beta_h0, y_indices, sample_weights, smoothness_gamma\n):\n    alpha = alpha_from_logits(logits)\n    h0_scores = (phi_standardised * alpha[None, :]) @ beta_h0.T\n    return h0_scores\n'''
    patched = patched_text(source)
    assert PATCH_MARKER in patched
    compile(patched, "<patched>", "exec")

    with tempfile.TemporaryDirectory(prefix="joint_all_tasks_selftest_") as directory:
        root = Path(directory)
        arrays = build_synthetic_arrays()
        ibd_manifest = build_ibd_manifest(arrays["IBD_vs_nonIBD"], repeats=2, folds=2)
        manifest_path = root / "ibd_manifest.csv"
        ibd_manifest.to_csv(manifest_path, index=False)
        config = config_for(root, manifest_path)
        combined, audit, ibd_equivalence = build_all_manifests(arrays, ibd_manifest, config)
        assert [task.folder for task in TASK_SPECS] == list(config.task_order)
        assert len(audit) == 5
        assert (
            ibd_equivalence["exact_sample_match"]
            & ibd_equivalence["exact_participant_match"]
            & ibd_equivalence["exact_label_match"]
        ).all()
        for task in TASK_SPECS:
            validate_task_manifest(task, combined, config)
        checksums = write_manifests(combined, audit, ibd_equivalence, Path(config.output_dir))
        assert set(checksums) == {task.folder for task in TASK_SPECS}

        # Manifest mismatch rejection.
        broken = combined.copy()
        index = broken.index[
            broken["task_folder"].eq("nonIBD_vs_UC") & broken["role"].eq("test")
        ][0]
        broken.loc[index, "participant_id"] = "LEAKED"
        rejected = False
        try:
            validate_task_manifest(next(task for task in TASK_SPECS if task.folder == "nonIBD_vs_UC"), broken, config)
        except ValueError:
            rejected = True
        assert rejected

        for task in TASK_SPECS:
            for repeat in (1, 2):
                write_fake_repeat(config, combined, task.folder, repeat)
                verified, classes = verify_repeat_output(config, combined, task.folder, repeat)
                metrics = classification_metrics(verified, classes)
                assert metrics["balanced_accuracy"] == 1.0
                assert metrics["roc_auc"] == 1.0

        progress = aggregate_completed(config, combined, strict=True, include_coefficients=True)
        assert progress["completed_task_repetitions"] == 10
        assert progress["completed_outer_folds"] == 20
        summary = pd.read_csv(
            Path(config.output_dir) / "aggregate" / "repetition_performance_summary.csv"
        )
        assert len(summary) == 10
        assert set(summary["n_repetitions"]) == {2}
        assert np.allclose(summary["balanced_accuracy_mean"], 1.0)
        selected = pd.read_csv(Path(config.output_dir) / "aggregate" / "all_selected_configs.csv")
        assert "effective_alpha_smoothness_gamma" in selected.columns
        observed_64 = selected.loc[selected["selected_intervals"].eq(64), "effective_alpha_smoothness_gamma"]
        assert np.allclose(observed_64, 0.01 * 63 / 159)
        multiclass_coeff = (
            Path(config.output_dir)
            / "aggregate"
            / "coefficient_stability"
            / "three_way_nonIBD_UC_CD"
            / "ricci_coefficient_stability.csv"
        )
        coefficient_frame = pd.read_csv(multiclass_coeff)
        assert set(coefficient_frame["class"]) == {"nonIBD", "UC", "CD"}

        # Probability/argmax mismatch rejection.
        frame = pd.DataFrame(
            {
                "true_label": ["nonIBD", "IBD"],
                "predicted_label": ["IBD", "IBD"],
                "probability_nonIBD": [0.9, 0.1],
                "probability_IBD": [0.1, 0.9],
            }
        )
        rejected = False
        try:
            standardise_prediction_frame(frame, TASK_SPECS[0])
        except ValueError:
            rejected = True
        assert rejected

    print(
        "SELF-TEST PASSED: five-task ordering, deterministic task-specific split locking, exact IBD-manifest "
        "reproduction, participant-leakage rejection, binary and multiclass OOF metrics, resolution-normalised "
        "alpha gamma bookkeeping, generic binary/multiclass coefficient stability, resume-compatible aggregation, "
        "and core patcher logic all succeeded."
    )


if __name__ == "__main__":
    main()
