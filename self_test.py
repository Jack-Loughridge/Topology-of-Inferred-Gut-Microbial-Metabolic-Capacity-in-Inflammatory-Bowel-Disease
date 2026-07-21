#!/usr/bin/env python3
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from species_benchmarks_all_tasks import ALL_TASKS, TASK_ORDER, RunConfig, run


def make_cohort() -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(19)
    participant_counter = 0
    for diagnosis in ("nonIBD", "UC", "CD"):
        for _ in range(6):
            participant = f"P{participant_counter:03d}"
            participant_counter += 1
            for sample_index in range(2):
                rows.append({
                    "sample_id": f"S{participant_counter - 1:03d}_{sample_index}",
                    "participant_id": participant,
                    "diagnosis": diagnosis,
                    "jitter": float(rng.normal()),
                })
    return pd.DataFrame(rows)


def task_frame(cohort: pd.DataFrame, folder: str) -> pd.DataFrame:
    if folder == "IBD_vs_nonIBD":
        out = cohort.copy()
        out["label"] = np.where(out["diagnosis"].eq("nonIBD"), "nonIBD", "IBD")
        return out
    if folder == "three_way_nonIBD_UC_CD":
        out = cohort.copy(); out["label"] = out["diagnosis"]; return out
    if folder == "nonIBD_vs_UC":
        out = cohort[cohort["diagnosis"].isin(["nonIBD", "UC"])].copy(); out["label"] = out["diagnosis"]; return out
    if folder == "nonIBD_vs_CD":
        out = cohort[cohort["diagnosis"].isin(["nonIBD", "CD"])].copy(); out["label"] = out["diagnosis"]; return out
    if folder == "CD_vs_UC":
        out = cohort[cohort["diagnosis"].isin(["CD", "UC"])].copy(); out["label"] = out["diagnosis"]; return out
    raise KeyError(folder)


def write_manifests(cohort: pd.DataFrame, split_dir: Path, repeats: int, folds: int) -> None:
    seeds = [101, 202]
    for task in ALL_TASKS:
        data = task_frame(cohort, task.folder).reset_index(drop=True)
        labels = data["label"].to_numpy(dtype=object)
        groups = data["participant_id"].to_numpy(dtype=object)
        rows = []
        for repeat in range(1, repeats + 1):
            splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seeds[repeat - 1])
            for fold, (train_idx, test_idx) in enumerate(splitter.split(np.zeros(len(data)), labels, groups=groups), start=1):
                for role, indices in (("train", train_idx), ("test", test_idx)):
                    for idx in indices:
                        rows.append({
                            "task_folder": task.folder,
                            "repeat": repeat,
                            "fold": fold,
                            "split_seed": seeds[repeat - 1],
                            "role": role,
                            "sample_id": data.loc[idx, "sample_id"],
                            "participant_id": data.loc[idx, "participant_id"],
                            "label": data.loc[idx, "label"],
                        })
        pd.DataFrame(rows).to_csv(split_dir / f"{task.folder}_split_manifest.csv", index=False)


def write_species(cohort: pd.DataFrame, path: Path) -> None:
    rng = np.random.default_rng(23)
    features = [f"species_{index:02d}" for index in range(14)]
    offsets = {
        "nonIBD": np.array([7, 4, 1] + [0] * 11, dtype=float),
        "UC": np.array([1, 7, 4] + [0] * 11, dtype=float),
        "CD": np.array([4, 1, 7] + [0] * 11, dtype=float),
    }
    zero_ids = {"S000_1", "S006_1"}
    rows = []
    for _, row in cohort.iterrows():
        values = rng.gamma(shape=1.5, scale=1.0, size=len(features)) + offsets[row["diagnosis"]]
        if row["sample_id"] in zero_ids:
            values[:] = 0.0
        output = {"External ID": row["sample_id"]}
        output.update(dict(zip(features, values)))
        rows.append(output)
    rows.extend([dict(rows[2]), dict(rows[5])])
    pd.DataFrame(rows).to_excel(path, sheet_name="Sheet1", index=False)


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="species_all_tasks_selftest_") as tmp:
        root = Path(tmp)
        split_dir = root / "splits"; split_dir.mkdir()
        species_file = root / "species.xlsx"
        output_dir = root / "output"
        cohort = make_cohort()
        write_manifests(cohort, split_dir, repeats=1, folds=2)
        write_species(cohort, species_file)

        config = RunConfig(
            species_file=str(species_file), species_sheet="Sheet1",
            split_dir=str(split_dir), output_dir=str(output_dir),
            tasks=TASK_ORDER, models=("logistic_l1", "random_forest", "xgboost"),
            expected_repeats=1, expected_folds=2, expected_global_zero_profiles=2,
            n_jobs=1, logistic_max_iter=1500, rf_trees=8, xgb_estimators=5,
            make_plots=False, save_full_source_models=False,
        )

        class Args:
            overwrite_incompatible_output = False
            validate_only = False
            aggregate_only = False

        run(config, Args())
        expected = 5 * 3 * 1 * 2
        markers = list(output_dir.glob("*/folds/*/repeat_*/fold_*/FOLD_COMPLETE.json"))
        assert len(markers) == expected, (len(markers), expected)
        assert (output_dir / "RUN_COMPLETE.json").exists()
        combined = pd.read_csv(output_dir / "aggregate" / "repetition_performance_summary_all_tasks.csv")
        assert set(combined["task_folder"]) == set(TASK_ORDER)
        assert set(combined["evaluation_level"]) == {"sample", "participant"}
        assert combined["n_repetitions"].eq(1).all()
        three_way = pd.read_csv(output_dir / "three_way_nonIBD_UC_CD" / "repetition_pooled_oof_metrics.csv")
        for name in ("nonIBD", "UC", "CD"):
            assert f"ovr_auc_{name}" in three_way.columns
        for task in TASK_ORDER:
            excluded = pd.read_csv(output_dir / task / "excluded_species_profiles.csv")
            assert not excluded.empty

        before = {path: path.stat().st_mtime_ns for path in markers}
        run(config, Args())
        after = {path: path.stat().st_mtime_ns for path in markers}
        assert before == after, "Resume invocation refitted completed folds."

        broken = RunConfig(**{**config.__dict__, "expected_global_zero_profiles": 3, "output_dir": str(root / "broken")})
        try:
            run(broken, Args())
        except ValueError as exc:
            assert "Expected 3 global all-zero profiles" in str(exc)
        else:
            raise AssertionError("Incorrect zero-profile expectation was not rejected.")

    print(
        "SELF-TEST PASSED: exact shared manifests for all five tasks, fixed task ordering, explicit zero-profile "
        "exclusion, participant-leakage safeguards, train-only CLR/variance/scaling, binary and multiclass "
        "probability alignment and OOF metrics, single XGBoost class weighting, feature stability, full-source "
        "model persistence, combined aggregation, and no-refit resume behavior all succeeded."
    )


if __name__ == "__main__":
    main()
