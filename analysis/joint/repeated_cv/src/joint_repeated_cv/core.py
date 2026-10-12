from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from .config import RunConfig
from .splits import TaskArrays
from .tasks import TASK_SPECS, normalise_label
from .util import normalise_sample_id, sha256_file


EXPECTED_CORE_OUTPUTS = (
    "ALL_FOLD_RESULTS.csv",
    "ALL_SUMMARIES.csv",
    "ALL_TEST_PREDICTIONS.csv",
    "ALL_ALPHA_PROFILES.csv",
    "ALL_INNER_CONFIG_RESULTS.csv",
    "ALL_SELECTED_CONFIGS.csv",
    "ALL_SPARSITY_DIAGNOSTICS.csv",
    "RICCI_FEATURE_METADATA_USED.csv",
    "RUN_CONFIG.json",
)

PATCH_MARKER = "ALPHA_SMOOTHNESS_RESOLUTION_NORMALISATION_V1"
PATCH_RECORD = "ALPHA_SMOOTHNESS_RESOLUTION_PATCH.json"
EXPECTED_PATCH_FORMULA = "gamma_eff(M) = gamma * (M - 1) / (160 - 1)"
SUPPORTED_UNPATCHED_MODELS = {
    "4f83b45a04943589dd2887e2d9f41ea488a760f18ffa08438a8215301bbf8e8d":
        "core-v1.1.1-post-FISTA",
    "b0eff25de9bf1400d5d1e3a5d3576ba1233ffdeda3420596cf24de65ed8e8e8e":
        "core-v1.2.0-orthant-polish",
}
EXPECTED_PATCHED_HASHES = {
    "b0eff25de9bf1400d5d1e3a5d3576ba1233ffdeda3420596cf24de65ed8e8e8e":
        "fc6d0a8ec94172900c9ac25e431e5dfb601f0eddf0674c4f389c0993d6d0a230",
}


def core_source_fingerprints(config: RunConfig) -> dict[str, str]:
    root = Path(config.joint_repo)
    relatives = (
        "pyproject.toml",
        "scripts/run_joint_sparse.py",
        "src/h0_ricci_joint_sparse/data.py",
        "src/h0_ricci_joint_sparse/tasks.py",
        "src/h0_ricci_joint_sparse/wkpi.py",
        "src/h0_ricci_joint_sparse/model.py",
        "src/h0_ricci_joint_sparse/cv.py",
        "src/h0_ricci_joint_sparse/metrics.py",
        "src/h0_ricci_joint_sparse/outputs.py",
    )
    output: dict[str, str] = {}
    for relative in relatives:
        path = root / relative
        if not path.exists():
            raise FileNotFoundError(f"Core source file is missing: {path}")
        output[relative] = sha256_file(path)
    return output


def ensure_core_paths(config: RunConfig) -> None:
    paths = {
        "joint repository": Path(config.joint_repo),
        "locked IBD split manifest": Path(config.ibd_split_manifest),
        "H0 reference results": Path(config.h0_results_dir),
        "H0 persistence diagrams": Path(config.h0_pd_dir),
        "Ricci original reference": Path(config.ricci_original_dir),
        "Ricci feature directory": Path(config.ricci_feature_dir),
    }
    missing = [f"{label}: {path}" for label, path in paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError("Required paths are missing:\n  " + "\n  ".join(missing))
    joint = Path(config.joint_repo)
    required = [
        joint / "scripts" / "run_joint_sparse.py",
        joint / "src" / "h0_ricci_joint_sparse" / "model.py",
        joint / "src" / "h0_ricci_joint_sparse" / "data.py",
        joint / "src" / "h0_ricci_joint_sparse" / "tasks.py",
        joint / ".venv" / "bin" / "python",
    ]
    absent = [str(path) for path in required if not path.exists()]
    if absent:
        raise FileNotFoundError("Validated core repository is incomplete:\n  " + "\n  ".join(absent))


def ensure_running_in_core_venv(config: RunConfig) -> None:
    expected_prefix = (Path(config.joint_repo) / ".venv").resolve()
    actual_prefix = Path(sys.prefix).resolve()
    if actual_prefix != expected_prefix:
        expected_python = expected_prefix / "bin" / "python"
        raise RuntimeError(
            "Run the all-task orchestrator with the validated core virtual environment so split "
            "generation, preprocessing and core fitting use the same NumPy/scikit-learn versions.\n"
            f"Expected: {expected_python}\n"
            f"Current sys.prefix: {actual_prefix}"
        )


def core_runtime_versions(config: RunConfig) -> dict[str, str]:
    python = Path(config.joint_repo) / ".venv" / "bin" / "python"
    code = (
        "import json, platform, numpy, pandas, scipy, sklearn; "
        "print(json.dumps({'python': platform.python_version(), 'numpy': numpy.__version__, "
        "'pandas': pandas.__version__, 'scipy': scipy.__version__, "
        "'scikit_learn': sklearn.__version__}, sort_keys=True))"
    )
    result = subprocess.run(
        [str(python), "-c", code], cwd=config.joint_repo, check=True, capture_output=True, text=True
    )
    return json.loads(result.stdout.strip())


def verify_resolution_patch(config: RunConfig) -> dict[str, Any]:
    root = Path(config.joint_repo)
    model = root / "src" / "h0_ricci_joint_sparse" / "model.py"
    text = model.read_text(encoding="utf-8")
    if PATCH_MARKER not in text:
        raise RuntimeError(
            "The core model does not contain the required resolution-normalised alpha smoothness patch. "
            "Run install_core_patch.py before validation or fitting."
        )
    if "resolution_normalised_alpha_smoothness_gamma" not in text:
        raise RuntimeError("Core patch marker exists but its helper function is missing.")
    record_path = root / PATCH_RECORD
    if not record_path.exists():
        raise RuntimeError(f"Core patch record is missing: {record_path}")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    original_hash = str(record.get("original_model_sha256", ""))
    if original_hash not in SUPPORTED_UNPATCHED_MODELS:
        raise RuntimeError(
            "Core patch record names an unaudited original model source: "
            f"{original_hash}"
        )
    source_profile = SUPPORTED_UNPATCHED_MODELS[original_hash]
    expected_record = {
        "patch": PATCH_MARKER,
        "reference_intervals": 160,
        "formula": EXPECTED_PATCH_FORMULA,
        "source_profile": source_profile,
    }
    mismatches = {
        key: {"observed": record.get(key), "expected": value}
        for key, value in expected_record.items()
        if record.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"Core patch record is incompatible: {mismatches}")
    if int(record.get("reference_intervals", -1)) != config.alpha_smoothness_reference_intervals:
        raise RuntimeError(
            "Core patch reference interval count does not match this run: "
            f"{record.get('reference_intervals')} vs {config.alpha_smoothness_reference_intervals}"
        )
    actual_model_sha256 = sha256_file(model)
    recorded_model_sha256 = str(record.get("patched_model_sha256", ""))
    if actual_model_sha256 != recorded_model_sha256:
        raise RuntimeError(
            "Core model.py no longer matches the audited patch record: "
            f"actual={actual_model_sha256}, recorded={recorded_model_sha256}"
        )
    deterministic_hash = EXPECTED_PATCHED_HASHES.get(original_hash)
    if deterministic_hash is not None and actual_model_sha256 != deterministic_hash:
        raise RuntimeError(
            "Core model.py differs from the deterministic audited patch output: "
            f"actual={actual_model_sha256}, expected={deterministic_hash}"
        )
    if config.alpha_smoothness_reference_intervals != 160:
        raise ValueError(
            "This audited core patch is anchored at 160 intervals. Use "
            "--alpha-smoothness-reference-intervals 160."
        )
    return record


def run_core_tests(config: RunConfig) -> None:
    python = Path(config.joint_repo) / ".venv" / "bin" / "python"
    command = [str(python), "-m", "pytest", "-q"]
    print("[core tests] " + " ".join(command), flush=True)
    subprocess.run(command, cwd=config.joint_repo, check=True)


def import_core(config: RunConfig):
    source = str(Path(config.joint_repo) / "src")
    if source not in sys.path:
        sys.path.insert(0, source)
    for name in [
        "h0_ricci_joint_sparse.data",
        "h0_ricci_joint_sparse.tasks",
        "h0_ricci_joint_sparse.model",
    ]:
        if name in sys.modules:
            importlib.reload(sys.modules[name])
    data_module = importlib.import_module("h0_ricci_joint_sparse.data")
    tasks_module = importlib.import_module("h0_ricci_joint_sparse.tasks")
    model_module = importlib.import_module("h0_ricci_joint_sparse.model")
    return data_module, tasks_module, model_module


def load_core_inputs_and_task_arrays(config: RunConfig):
    data_module, tasks_module, model_module = import_core(config)
    inputs = data_module.load_all_inputs(
        h0_results_dir=Path(config.h0_results_dir),
        ricci_original_dir=Path(config.ricci_original_dir),
        ricci_feature_dir=Path(config.ricci_feature_dir),
        h0_pd_dir=Path(config.h0_pd_dir),
    )
    task_objects = {task.folder: task for task in tasks_module.select_tasks("all")}
    arrays: dict[str, TaskArrays] = {}
    for task in TASK_SPECS:
        if task.folder not in task_objects:
            raise RuntimeError(f"Core task selector did not return {task.folder}")
        core_task = task_objects[task.folder]
        mask = np.asarray(tasks_module.mask_for_task(inputs.labels3, core_task), dtype=bool)
        labels = np.asarray(tasks_module.labels_for_task(inputs.labels3[mask], core_task), dtype=object)
        arrays[task.folder] = TaskArrays(
            sample_ids=np.asarray(inputs.sample_ids, dtype=object)[mask],
            participant_ids=np.asarray(inputs.participant_ids, dtype=object)[mask],
            labels=labels,
        )
    helper = getattr(model_module, "resolution_normalised_alpha_smoothness_gamma", None)
    if helper is None:
        raise RuntimeError("Patched core helper resolution_normalised_alpha_smoothness_gamma is unavailable.")
    for intervals in (64, 96, 160):
        expected = config.alpha_smoothness_gamma * (intervals - 1) / 159.0
        observed = float(helper(config.alpha_smoothness_gamma, intervals))
        if not np.isclose(observed, expected, rtol=0, atol=1e-15):
            raise RuntimeError(
                f"Core smoothness normalisation failed for M={intervals}: {observed} vs {expected}"
            )
    return inputs, arrays


def verify_core_generated_splits(
    all_manifest: pd.DataFrame,
    arrays_by_task: dict[str, TaskArrays],
    config: RunConfig,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for task in TASK_SPECS:
        arrays = arrays_by_task[task.folder]
        sample_ids = np.asarray([normalise_sample_id(x) for x in arrays.sample_ids], dtype=object)
        participants = np.asarray([str(x).strip() for x in arrays.participant_ids], dtype=object)
        labels = np.asarray([normalise_label(x) for x in arrays.labels], dtype=object)
        task_manifest = all_manifest[all_manifest["task_folder"].eq(task.folder)]
        for repeat in range(1, config.expected_repeats + 1):
            seed = int(task_manifest.loc[task_manifest["repeat"].eq(repeat), "split_seed"].iloc[0])
            splitter = StratifiedGroupKFold(
                n_splits=config.expected_folds, shuffle=True, random_state=seed
            )
            for fold, (train_idx, test_idx) in enumerate(
                splitter.split(np.zeros(len(labels)), labels, groups=participants), start=1
            ):
                expected_test_frame = task_manifest.loc[
                    task_manifest["repeat"].eq(repeat)
                    & task_manifest["fold"].eq(fold)
                    & task_manifest["role"].eq("test"),
                    ["sample_id", "participant_id"],
                ]
                expected_train_frame = task_manifest.loc[
                    task_manifest["repeat"].eq(repeat)
                    & task_manifest["fold"].eq(fold)
                    & task_manifest["role"].eq("train"),
                    ["sample_id", "participant_id"],
                ]
                expected_test = set(expected_test_frame["sample_id"])
                observed_test = set(sample_ids[test_idx])
                expected_train = set(expected_train_frame["sample_id"])
                observed_train = set(sample_ids[train_idx])
                exact_samples = expected_test == observed_test and expected_train == observed_train

                expected_all_frame = pd.concat([expected_train_frame, expected_test_frame], ignore_index=True)
                expected_mapping = dict(
                    zip(expected_all_frame["sample_id"], expected_all_frame["participant_id"].astype(str))
                )
                observed_mapping = {
                    str(sample_id): str(participant)
                    for sample_id, participant in zip(sample_ids, participants)
                }
                exact_participants = all(
                    expected_mapping.get(str(sample_id)) == observed_mapping.get(str(sample_id))
                    for sample_id in expected_train | expected_test
                )
                expected_label_frame = task_manifest.loc[
                    task_manifest["repeat"].eq(repeat)
                    & task_manifest["fold"].eq(fold),
                    ["sample_id", "label"],
                ].drop_duplicates("sample_id")
                expected_labels = dict(
                    zip(expected_label_frame["sample_id"], expected_label_frame["label"].map(normalise_label))
                )
                observed_labels = {
                    str(sample_id): normalise_label(label)
                    for sample_id, label in zip(sample_ids, labels)
                }
                exact_labels = all(
                    expected_labels.get(str(sample_id)) == observed_labels.get(str(sample_id))
                    for sample_id in expected_train | expected_test
                )
                exact = exact_samples and exact_participants and exact_labels
                rows.append(
                    {
                        "task_order": next(i for i, t in enumerate(TASK_SPECS, 1) if t.folder == task.folder),
                        "task_folder": task.folder,
                        "repeat": repeat,
                        "fold": fold,
                        "split_seed": seed,
                        "expected_train_samples": len(expected_train),
                        "observed_train_samples": len(observed_train),
                        "expected_test_samples": len(expected_test),
                        "observed_test_samples": len(observed_test),
                        "exact_sample_match": exact_samples,
                        "exact_participant_match": exact_participants,
                        "exact_label_match": exact_labels,
                        "test_only_manifest": sorted(expected_test - observed_test)[:10],
                        "test_only_core": sorted(observed_test - expected_test)[:10],
                    }
                )
    audit = pd.DataFrame(rows)
    exact = (
        audit["exact_sample_match"]
        & audit["exact_participant_match"]
        & audit["exact_label_match"]
    )
    if not exact.all():
        bad = audit.loc[~exact].head().to_dict("records")
        raise RuntimeError(f"Core-generated splits differ from locked all-task manifests: {bad}")
    return audit


def repeat_dir(config: RunConfig, task_folder: str, repeat: int) -> Path:
    return Path(config.output_dir) / "runs" / task_folder / f"repeat_{repeat:02d}"


def _core_command(config: RunConfig, task_folder: str, repeat: int, split_seed: int) -> list[str]:
    python = Path(config.joint_repo) / ".venv" / "bin" / "python"
    script = Path(config.joint_repo) / "scripts" / "run_joint_sparse.py"
    return [
        str(python), "-u", str(script),
        "--h0-results-dir", config.h0_results_dir,
        "--h0-pd-dir", config.h0_pd_dir,
        "--ricci-original-dir", config.ricci_original_dir,
        "--ricci-feature-dir", config.ricci_feature_dir,
        "--out-dir", str(repeat_dir(config, task_folder, repeat)),
        "--tasks", task_folder,
        "--n-outer-splits", str(config.expected_folds),
        "--n-inner-splits", str(config.n_inner_splits),
        "--seed", str(split_seed),
        "--n-jobs", str(config.n_jobs),
        "--selection-metric", config.selection_metric,
        "--interval-grid", ",".join(str(value) for value in config.interval_grid),
        "--lambda-h0-grid", ",".join(f"{value:g}" for value in config.lambda_h0_grid),
        "--lambda-ricci-grid", ",".join(f"{value:g}" for value in config.lambda_ricci_grid),
        "--C", f"{config.c_value:g}",
        "--alpha-smoothness-gamma", f"{config.alpha_smoothness_gamma:g}",
        "--logistic-max-iter", str(config.logistic_max_iter),
        "--logistic-tol", f"{config.logistic_tolerance:g}",
        "--active-set-initial", str(config.active_set_initial),
        "--active-set-batch", str(config.active_set_batch),
        "--active-set-max-rounds", str(config.active_set_max_rounds),
        "--alternations", str(config.alternations),
        "--quad-points", str(config.quad_points),
    ]


def run_core_repeat(config: RunConfig, task_folder: str, repeat: int, split_seed: int) -> None:
    root = repeat_dir(config, task_folder, repeat)
    root.parent.mkdir(parents=True, exist_ok=True)
    command = _core_command(config, task_folder, repeat, split_seed)
    command_file = root.parent / f"repeat_{repeat:02d}_command.json"
    command_file.write_text(json.dumps(command, indent=2) + "\n", encoding="utf-8")
    env = os.environ.copy()
    source = str(Path(config.joint_repo) / "src")
    env["PYTHONPATH"] = source + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[key] = "1"
    print("[core run] " + " ".join(command), flush=True)
    subprocess.run(command, cwd=config.joint_repo, env=env, check=True)


def _nonempty_file(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def core_repeat_complete(config: RunConfig, task_folder: str, repeat: int) -> bool:
    root = repeat_dir(config, task_folder, repeat)
    if not all(_nonempty_file(root / name) for name in EXPECTED_CORE_OUTPUTS):
        return False
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
    task_root = root / task_folder
    return all(
        _nonempty_file(task_root / f"fold_{fold}" / name)
        for fold in range(1, config.expected_folds + 1)
        for name in fold_required
    )
