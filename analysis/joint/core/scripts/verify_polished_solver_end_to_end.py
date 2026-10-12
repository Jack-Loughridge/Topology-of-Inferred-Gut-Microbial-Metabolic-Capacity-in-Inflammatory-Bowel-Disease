#!/usr/bin/env python3
"""Run all five tasks through 5x3 grouped nested CV and resume them.

This is a synthetic pipeline-integrity verification, not a scientific result.
The deliberately tiny FISTA budget forces the orthant-polish fallback to be
exercised repeatedly.
"""
from __future__ import annotations

import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from h0_ricci_joint_sparse.config import RunConfig
from h0_ricci_joint_sparse.cv import run_nested_cv
from h0_ricci_joint_sparse.data import LoadedInputs


def build_inputs(seed: int = 20260716) -> LoadedInputs:
    rng = np.random.default_rng(seed)
    labels_order = ("nonIBD", "UC", "CD")
    participant_ids: list[str] = []
    labels: list[str] = []
    sample_ids: list[str] = []
    for label in labels_order:
        for participant in range(10):
            participant_id = f"{label}_P{participant:02d}"
            for sample in range(2):
                participant_ids.append(participant_id)
                labels.append(label)
                sample_ids.append(f"{participant_id}_S{sample}")

    participant_ids_array = np.asarray(participant_ids, dtype=object)
    labels_array = np.asarray(labels, dtype=object)
    sample_ids_array = np.asarray(sample_ids, dtype=object)
    n = len(labels_array)

    latent = rng.normal(size=(n, 10))
    latent[labels_array == "UC", 0] += 1.1
    latent[labels_array == "UC", 2] += 0.5
    latent[labels_array == "CD", 1] += 1.2
    latent[labels_array == "CD", 2] -= 0.4

    n_ricci = 160
    loadings = rng.normal(size=(10, n_ricci)) / np.sqrt(10)
    ricci_dense = latent @ loadings + 0.15 * rng.normal(size=(n, n_ricci))
    ricci_dense /= np.maximum(ricci_dense.std(axis=0), 1e-8)

    h0_deaths: list[np.ndarray] = []
    for label in labels_array:
        deaths = rng.lognormal(mean=-2.2, sigma=0.7, size=22)
        if label == "UC":
            deaths[:6] *= 1.35
        elif label == "CD":
            deaths[6:12] *= 1.5
        h0_deaths.append(np.sort(deaths.astype(np.float64)))

    metadata = pd.DataFrame({
        "sample_id": sample_ids_array,
        "participant_id": participant_ids_array,
        "diagnosis": labels_array,
        "_sample_id": sample_ids_array,
        "_participant_id": participant_ids_array,
        "_label3": labels_array,
    })
    feature_metadata = pd.DataFrame({
        "feature_index": np.arange(n_ricci),
        "feature_name": [f"R{index}" for index in range(n_ricci)],
    })
    return LoadedInputs(
        ricci=sparse.csr_matrix(ricci_dense),
        metadata=metadata,
        sample_ids=sample_ids_array,
        participant_ids=participant_ids_array,
        labels3=labels_array,
        h0_deaths=h0_deaths,
        h0_paths=[f"/synthetic/{sample_id}.npy" for sample_id in sample_ids_array],
        feature_metadata=feature_metadata,
        source_columns={
            "sample_col": "sample_id",
            "participant_col": "participant_id",
            "label_col": "diagnosis",
        },
        h0_pd_dir=Path("/synthetic/out_pds"),
    )


def main() -> int:
    inputs = build_inputs()
    with tempfile.TemporaryDirectory(prefix="h0_ricci_polish_verify_") as temporary:
        root = Path(temporary)
        for name in ("h0_reference", "ricci_reference", "ricci_features"):
            (root / name).mkdir()
        output = root / "output"
        config = RunConfig(
            h0_results_dir=root / "h0_reference",
            ricci_original_dir=root / "ricci_reference",
            ricci_feature_dir=root / "ricci_features",
            out_dir=output,
            h0_pd_dir=Path("/synthetic/out_pds"),
            tasks="all",
            n_outer_splits=5,
            n_inner_splits=3,
            seed=13,
            n_jobs=2,
            selection_metric="roc_auc",
            interval_grid=(12,),
            lambda_h0_grid=(0.5, 1.0),
            lambda_ricci_grid=(0.5, 1.0),
            c_value=0.08,
            logistic_max_iter=10,
            logistic_tolerance=1e-4,
            n_alternations=2,
            alpha_smoothness_gamma=0.01,
            alpha_optimizer_max_iter=20,
            quad_points=3,
            common_alpha_grid_size=30,
            top_coefficients=10,
            active_set_initial=32,
            active_set_batch=32,
            active_set_max_rounds=20,
            resume=True,
            benchmark_first_inner_fold=False,
        )

        first_logs: list[str] = []
        started = time.perf_counter()
        first = run_nested_cv(inputs, config, log=first_logs.append)
        first_elapsed = time.perf_counter() - started

        required = (
            "ALL_FOLD_RESULTS.csv",
            "ALL_SUMMARIES.csv",
            "ALL_TEST_PREDICTIONS.csv",
            "ALL_INNER_CONFIG_RESULTS.csv",
            "ALL_SELECTED_CONFIGS.csv",
            "ALL_SPARSITY_DIAGNOSTICS.csv",
            "RUN_CONFIG.json",
        )
        assert len(first["fold_results"]) == 25
        assert len(first["summaries"]) == 5
        assert len(list(output.rglob("COMPLETE.json"))) == 25
        assert all((output / filename).is_file() for filename in required)

        second_logs: list[str] = []
        resumed = run_nested_cv(inputs, config, log=second_logs.append)
        resume_markers = sum(
            "[RESUME] completed fold loaded" in message
            for message in second_logs
        )
        assert len(resumed["fold_results"]) == 25
        assert resume_markers == 25

        print(
            "[PASS] Five tasks, 25 outer folds, 5x3 grouped nested CV, "
            f"all global outputs, and 25-fold resume ({first_elapsed:.1f}s)."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
