from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy import sparse

from joint_repeated_cv.aggregate import verify_core_run_config
from joint_repeated_cv.config import RunConfig
from joint_repeated_cv.core import EXPECTED_CORE_OUTPUTS, core_repeat_complete, repeat_dir
from joint_repeated_cv.orchestrator import compute_input_fingerprints, validate_config
from joint_repeated_cv.tasks import TASK_ORDER


def test_input_fingerprints_are_deterministic_and_content_sensitive() -> None:
    def make_inputs() -> SimpleNamespace:
        return SimpleNamespace(
            sample_ids=np.asarray(["s2", "s1"], dtype=object),
            participant_ids=np.asarray(["p2", "p1"], dtype=object),
            labels3=np.asarray(["IBD", "nonIBD"], dtype=object),
            h0_paths=np.asarray(["/tmp/s2.npy", "/tmp/s1.npy"], dtype=object),
            h0_deaths=[np.asarray([0.2, 0.4]), np.asarray([0.1, 0.3])],
            ricci=sparse.csr_matrix(np.asarray([[0.0, 2.0], [1.0, 0.0]])),
        )

    first = compute_input_fingerprints(make_inputs())
    second = compute_input_fingerprints(make_inputs())
    assert first == second

    changed_h0 = make_inputs()
    changed_h0.h0_deaths[0][0] += 1e-6
    changed_h0_fingerprint = compute_input_fingerprints(changed_h0)
    assert changed_h0_fingerprint["h0_diagrams_sha256"] != first["h0_diagrams_sha256"]
    assert changed_h0_fingerprint["ricci_matrix_sha256"] == first["ricci_matrix_sha256"]

    changed_ricci = make_inputs()
    changed_ricci.ricci.data[0] += 1e-6
    changed_ricci_fingerprint = compute_input_fingerprints(changed_ricci)
    assert changed_ricci_fingerprint["ricci_matrix_sha256"] != first["ricci_matrix_sha256"]


def test_publication_configuration_is_locked(small_config: RunConfig) -> None:
    production = replace(
        small_config,
        expected_repeats=20,
        expected_folds=5,
        n_inner_splits=3,
        end_repeat=20,
        task_order=TASK_ORDER,
    )
    validate_config(production)
    with np.testing.assert_raises_regex(ValueError, "20 repetitions"):
        validate_config(replace(production, expected_repeats=19, end_repeat=19))
    with np.testing.assert_raises_regex(ValueError, "three participant-grouped inner folds"):
        validate_config(replace(production, n_inner_splits=4))
    with np.testing.assert_raises_regex(ValueError, "ROC-AUC"):
        validate_config(replace(production, selection_metric="balanced_accuracy"))


def _write_nonempty(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")


def test_repeat_completion_requires_every_fold_artifact(small_config: RunConfig) -> None:
    task_folder = "IBD_vs_nonIBD"
    root = repeat_dir(small_config, task_folder, 1)
    for name in EXPECTED_CORE_OUTPUTS:
        _write_nonempty(root / name)
    fold_required = (
        "COMPLETE.json",
        "fold_result.csv",
        "test_predictions.csv",
        "alpha_profile.csv",
        "inner_config_results.csv",
        "selected_config.json",
        "sparsity_diagnostics.json",
        "outer_model.state.npz",
    )
    for fold in range(1, small_config.expected_folds + 1):
        for name in fold_required:
            _write_nonempty(root / task_folder / f"fold_{fold}" / name)
    assert core_repeat_complete(small_config, task_folder, 1)
    (root / task_folder / "fold_5" / "outer_model.state.npz").unlink()
    assert not core_repeat_complete(small_config, task_folder, 1)


def test_core_run_configuration_is_checked_exactly(small_config: RunConfig) -> None:
    task_folder = "IBD_vs_nonIBD"
    repeat = 1
    split_seed = 123
    root = repeat_dir(small_config, task_folder, repeat)
    root.mkdir(parents=True, exist_ok=True)
    payload = {
        "tasks": task_folder,
        "n_outer_splits": small_config.expected_folds,
        "n_inner_splits": small_config.n_inner_splits,
        "seed": split_seed,
        "selection_metric": small_config.selection_metric,
        "logistic_max_iter": small_config.logistic_max_iter,
        "n_alternations": small_config.alternations,
        "quad_points": small_config.quad_points,
        "active_set_initial": small_config.active_set_initial,
        "active_set_batch": small_config.active_set_batch,
        "active_set_max_rounds": small_config.active_set_max_rounds,
        "c_value": small_config.c_value,
        "logistic_tolerance": small_config.logistic_tolerance,
        "alpha_smoothness_gamma": small_config.alpha_smoothness_gamma,
        "interval_grid": list(small_config.interval_grid),
        "lambda_h0_grid": list(small_config.lambda_h0_grid),
        "lambda_ricci_grid": list(small_config.lambda_ricci_grid),
    }
    (root / "RUN_CONFIG.json").write_text(json.dumps(payload), encoding="utf-8")
    verify_core_run_config(small_config, task_folder, repeat, split_seed)
    payload["lambda_h0_grid"] = [0.5, 1.0]
    (root / "RUN_CONFIG.json").write_text(json.dumps(payload), encoding="utf-8")
    with np.testing.assert_raises_regex(RuntimeError, "incompatible"):
        verify_core_run_config(small_config, task_folder, repeat, split_seed)
