from __future__ import annotations

import hashlib
import json
import platform
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn import __version__ as sklearn_version
from scipy import __version__ as scipy_version
from scipy import sparse

from .aggregate import aggregate_completed, verify_repeat_output
from .config import CODE_VERSION, SCHEMA_VERSION, RunConfig
from .core import (
    core_repeat_complete,
    core_runtime_versions,
    core_source_fingerprints,
    ensure_core_paths,
    ensure_running_in_core_venv,
    load_core_inputs_and_task_arrays,
    repeat_dir,
    run_core_repeat,
    run_core_tests,
    verify_core_generated_splits,
    verify_resolution_patch,
)
from .splits import build_all_manifests, load_ibd_manifest, write_manifests
from .tasks import TASK_SPECS
from .util import atomic_csv, atomic_json, now_iso, sha256_file


def validate_config(config: RunConfig) -> None:
    if config.expected_repeats != 20 or config.expected_folds != 5:
        raise ValueError("The publication analysis is fixed at 20 repetitions and five outer folds.")
    if config.n_inner_splits != 3:
        raise ValueError("The publication analysis is fixed at three participant-grouped inner folds.")
    if config.selection_metric != "roc_auc":
        raise ValueError("The publication analysis selects hyperparameters by ROC-AUC.")
    if tuple(config.task_order) != tuple(task.folder for task in TASK_SPECS):
        raise ValueError("The five-task execution order differs from the audited order.")
    if config.start_repeat < 1 or config.end_repeat > config.expected_repeats or config.start_repeat > config.end_repeat:
        raise ValueError("Invalid repetition range.")
    if config.start_task_index < 1 or config.end_task_index > len(TASK_SPECS) or config.start_task_index > config.end_task_index:
        raise ValueError("Invalid task-index range.")
    if tuple(config.interval_grid) != (64, 96, 160):
        raise ValueError("The final audited interval grid is fixed at 64,96,160.")
    if tuple(config.lambda_h0_grid) != (0.5, 1.0, 2.0, 4.0):
        raise ValueError("The final audited H0 penalty grid is fixed at 0.5,1,2,4.")
    if tuple(config.lambda_ricci_grid) != (0.5, 1.0, 2.0):
        raise ValueError("The final audited Ricci penalty grid is fixed at 0.5,1,2.")
    if not np.isclose(config.c_value, 0.02):
        raise ValueError("The final audited global C is fixed at 0.02.")
    if not np.isclose(config.alpha_smoothness_gamma, 0.01):
        raise ValueError("The final audited base alpha-smoothness gamma is fixed at 0.01.")
    if config.alpha_smoothness_reference_intervals != 160:
        raise ValueError("Alpha smoothness must be anchored at 160 intervals.")
    positive_integers = {
        "n_jobs": config.n_jobs,
        "logistic_max_iter": config.logistic_max_iter,
        "active_set_initial": config.active_set_initial,
        "active_set_batch": config.active_set_batch,
        "active_set_max_rounds": config.active_set_max_rounds,
        "alternations": config.alternations,
        "quad_points": config.quad_points,
    }
    invalid = {key: value for key, value in positive_integers.items() if int(value) < 1}
    if invalid:
        raise ValueError(f"Positive integer solver settings are required: {invalid}")
    if not np.isfinite(config.logistic_tolerance) or config.logistic_tolerance <= 0:
        raise ValueError("logistic_tolerance must be finite and positive.")


def orchestrator_source_fingerprints() -> dict[str, str]:
    """Fingerprint the imported scientific orchestration implementation.

    Keys are package-relative so the lock is stable whether the repository is
    executed directly, installed editable, or installed from a wheel.
    """
    package_root = Path(__file__).resolve().parent
    sources = sorted(package_root.glob("*.py"))
    if not sources:
        raise FileNotFoundError(f"No orchestration source files found under {package_root}")
    return {f"joint_repeated_cv/{path.name}": sha256_file(path) for path in sources}


def _patch_identity(patch_record: dict[str, Any]) -> dict[str, Any]:
    return {
        "patch": patch_record.get("patch"),
        "reference_intervals": int(patch_record.get("reference_intervals", -1)),
        "formula": patch_record.get("formula"),
        "original_model_sha256": patch_record.get("original_model_sha256"),
        "patched_model_sha256": patch_record.get("patched_model_sha256"),
    }


def locked_run_payload(config: RunConfig, patch_record: dict[str, Any]) -> dict[str, Any]:
    """Scientific configuration plus immutable data/code identities for safe resume."""
    return {
        **config.scientific_payload(),
        "source_lock": {
            "locked_ibd_manifest_sha256": sha256_file(Path(config.ibd_split_manifest)),
            "core_source_fingerprints": core_source_fingerprints(config),
            "orchestrator_source_fingerprints": orchestrator_source_fingerprints(),
            "core_patch_identity": _patch_identity(patch_record),
        },
    }


def ensure_compatible_output(
    config: RunConfig, overwrite: bool, patch_record: dict[str, Any]
) -> dict[str, Any]:
    output = Path(config.output_dir)
    expected = locked_run_payload(config, patch_record)
    config_path = output / "orchestrator_config.json"
    if not output.exists():
        output.mkdir(parents=True)
        atomic_json(config_path, expected)
        return expected
    if config_path.exists():
        observed = json.loads(config_path.read_text(encoding="utf-8"))
        if observed == expected:
            return expected
        if not overwrite:
            raise RuntimeError(
                f"Output {output} was created with a different scientific configuration, split manifest, "
                "core source, or orchestrator source. Use a new directory or "
                "--overwrite-incompatible-output."
            )
    elif any(output.iterdir()) and not overwrite:
        raise RuntimeError(
            f"Output {output} is non-empty but has no orchestrator_config.json. "
            "Use a new directory or --overwrite-incompatible-output."
        )
    if any(output.iterdir()):
        archive = output.with_name(output.name + "_incompatible_" + pd.Timestamp.now().strftime("%Y%m%d_%H%M%S"))
        shutil.move(str(output), str(archive))
        print(f"[archive] Incompatible output moved to {archive}", flush=True)
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(config_path, expected)
    return expected


def assert_locked_sources_unchanged(
    config: RunConfig, baseline: dict[str, Any], patch_record: dict[str, Any]
) -> None:
    observed = locked_run_payload(config, patch_record)
    if observed != baseline:
        raise RuntimeError(
            "A locked scientific input, core source file, or orchestrator source file changed after "
            "the run began. Stop and use a new output directory after reviewing the change."
        )


def _update_array_digest(digest: "hashlib._Hash", array: np.ndarray) -> None:
    contiguous = np.ascontiguousarray(array)
    digest.update(str(contiguous.dtype).encode("utf-8"))
    digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
    digest.update(contiguous.view(np.uint8))


def compute_input_fingerprints(inputs: Any) -> dict[str, str]:
    metadata = pd.DataFrame(
        {
            "sample_id": np.asarray(inputs.sample_ids, dtype=str),
            "participant_id": np.asarray(inputs.participant_ids, dtype=str),
            "label": np.asarray(inputs.labels3, dtype=str),
            "h0_path": np.asarray(inputs.h0_paths, dtype=str),
        }
    ).sort_values("sample_id", kind="mergesort")
    metadata_sha = hashlib.sha256(metadata.to_csv(index=False).encode("utf-8")).hexdigest()

    h0_digest = hashlib.sha256()
    for sample_id, values in sorted(
        zip(np.asarray(inputs.sample_ids, dtype=str), inputs.h0_deaths), key=lambda pair: pair[0]
    ):
        encoded = sample_id.encode("utf-8")
        h0_digest.update(len(encoded).to_bytes(8, "little"))
        h0_digest.update(encoded)
        _update_array_digest(h0_digest, np.asarray(values, dtype=np.float64))

    if sparse.issparse(inputs.ricci):
        matrix = inputs.ricci.tocsr(copy=False)
        if not matrix.has_sorted_indices:
            matrix = matrix.copy()
            matrix.sort_indices()
        ricci_digest = hashlib.sha256()
        _update_array_digest(ricci_digest, np.asarray(matrix.shape, dtype=np.int64))
        _update_array_digest(ricci_digest, matrix.indptr)
        _update_array_digest(ricci_digest, matrix.indices)
        _update_array_digest(ricci_digest, matrix.data)
    else:
        ricci_digest = hashlib.sha256()
        _update_array_digest(ricci_digest, np.asarray(inputs.ricci))

    return {
        "sample_metadata_sha256": metadata_sha,
        "h0_diagrams_sha256": h0_digest.hexdigest(),
        "ricci_matrix_sha256": ricci_digest.hexdigest(),
    }


def lock_input_fingerprints(config: RunConfig, inputs: Any) -> dict[str, str]:
    fingerprints = compute_input_fingerprints(inputs)
    path = Path(config.output_dir) / "preflight" / "input_fingerprints.json"
    if path.exists():
        observed = json.loads(path.read_text(encoding="utf-8"))
        if observed != fingerprints:
            raise RuntimeError(
                "The H0 diagrams, Ricci feature matrix, or sample metadata differ from the inputs "
                "already locked to this output directory. Use a new output directory."
            )
    else:
        atomic_json(path, fingerprints)
    return fingerprints


def write_preflight(
    config: RunConfig,
    inputs: Any,
    task_audit: pd.DataFrame,
    ibd_equivalence: pd.DataFrame,
    core_split_audit: pd.DataFrame,
    split_checksums: dict[str, str],
    patch_record: dict[str, Any],
    input_fingerprints: dict[str, str],
    core_versions: dict[str, str],
) -> None:
    root = Path(config.output_dir) / "preflight"
    root.mkdir(parents=True, exist_ok=True)
    atomic_csv(task_audit, root / "task_cohort_audit.csv", index=False)
    atomic_csv(ibd_equivalence, root / "ibd_locked_manifest_equivalence.csv", index=False)
    atomic_csv(core_split_audit, root / "all_core_split_equivalence.csv", index=False)
    sample_audit = pd.DataFrame(
        {
            "sample_id": inputs.sample_ids,
            "participant_id": inputs.participant_ids,
            "diagnosis": inputs.labels3,
            "h0_death_count": [len(values) for values in inputs.h0_deaths],
            "h0_min_death": [float(np.min(values)) for values in inputs.h0_deaths],
            "h0_max_death": [float(np.max(values)) for values in inputs.h0_deaths],
            "h0_path": inputs.h0_paths,
        }
    )
    atomic_csv(sample_audit, root / "input_sample_audit.csv", index=False)
    payload = {
        "timestamp_utc": now_iso(),
        "schema_version": SCHEMA_VERSION,
        "code_version": CODE_VERSION,
        "task_order": [task.folder for task in TASK_SPECS],
        "interval_grid": list(config.interval_grid),
        "lambda_h0_grid": list(config.lambda_h0_grid),
        "lambda_ricci_grid": list(config.lambda_ricci_grid),
        "global_C": config.c_value,
        "alpha_smoothness_base_gamma": config.alpha_smoothness_gamma,
        "alpha_smoothness_reference_intervals": config.alpha_smoothness_reference_intervals,
        "effective_gamma_by_M": {
            str(value): config.alpha_smoothness_gamma * (value - 1) / (config.alpha_smoothness_reference_intervals - 1)
            for value in config.interval_grid
        },
        "outer_design": f"{config.expected_repeats} repetitions x {config.expected_folds} folds per task",
        "inner_folds": config.n_inner_splits,
        "input_samples": int(len(inputs.sample_ids)),
        "input_participants": int(len(set(inputs.participant_ids))),
        "ricci_shape": list(inputs.ricci.shape),
        "split_manifest_checksums": split_checksums,
        "all_core_splits_exact": bool(
            (
                core_split_audit["exact_sample_match"]
                & core_split_audit["exact_participant_match"]
                & core_split_audit["exact_label_match"]
            ).all()
        ),
        "ibd_locked_manifest_exact": bool(
            (
                ibd_equivalence["exact_sample_match"]
                & ibd_equivalence["exact_participant_match"]
                & ibd_equivalence["exact_label_match"]
            ).all()
        ),
        "input_fingerprints": input_fingerprints,
        "core_patch": patch_record,
        "core_source_fingerprints": core_source_fingerprints(config),
        "orchestrator_source_fingerprints": orchestrator_source_fingerprints(),
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy_version,
            "scikit_learn": sklearn_version,
            "core_virtual_environment": core_versions,
        },
    }
    atomic_json(root / "preflight_summary.json", payload)


def print_preflight(config: RunConfig, task_audit: pd.DataFrame, core_split_audit: pd.DataFrame) -> None:
    print("=" * 118)
    print("REPEATED JOINT H0 + RICCI — ALL FIVE TASKS PREFLIGHT")
    print("=" * 118)
    print("Task order:")
    for task_order, task in enumerate(TASK_SPECS, start=1):
        row = task_audit[task_audit["task_folder"].eq(task.folder)].iloc[0]
        print(
            f"  {task_order}. {task.report_name}: {int(row['samples'])} samples, "
            f"{int(row['participants'])} participants, classes={row['classes']}"
        )
    print(f"Design per task: {config.expected_repeats} x {config.expected_folds} outer folds")
    print(f"Total outer folds: {config.expected_repeats * config.expected_folds * len(TASK_SPECS)}")
    print(f"Inner grouped folds: {config.n_inner_splits}")
    print(f"M grid: {config.interval_grid}")
    print(f"lambda_H0 grid: {config.lambda_h0_grid}")
    print(f"lambda_Ricci grid: {config.lambda_ricci_grid}")
    print(f"C: {config.c_value}")
    print(
        "Alpha smoothness: gamma_eff(M) = gamma * (M-1)/(160-1), "
        f"base gamma={config.alpha_smoothness_gamma}"
    )
    exact_core = (
        core_split_audit["exact_sample_match"]
        & core_split_audit["exact_participant_match"]
        & core_split_audit["exact_label_match"]
    )
    print(f"Exact core split checks: {int(exact_core.sum())} / {len(core_split_audit)}")
    print("=" * 118)
    print("REPEATED JOINT H0 + RICCI ALL-TASK PREFLIGHT: PASSED")
    print("=" * 118)


def run(config: RunConfig, args: Any) -> None:
    validate_config(config)
    ensure_core_paths(config)
    ensure_running_in_core_venv(config)
    patch_record = verify_resolution_patch(config)
    core_versions = core_runtime_versions(config)
    locked_payload = ensure_compatible_output(
        config, bool(args.overwrite_incompatible_output), patch_record
    )

    if args.aggregate_only:
        manifest_path = Path(config.output_dir) / "splits" / "all_task_split_manifest.csv"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Cannot aggregate before preflight has written the manifest: {manifest_path}"
            )
        manifest = pd.read_csv(manifest_path, low_memory=False)
        progress = aggregate_completed(config, manifest, strict=False, include_coefficients=True)
        print(json.dumps(progress, indent=2))
        return

    if not args.skip_core_tests:
        run_core_tests(config)
    print("[preflight] Loading all 1,317 H0 diagrams and faithful Ricci features through the core...", flush=True)
    inputs, arrays_by_task = load_core_inputs_and_task_arrays(config)
    input_fingerprints = lock_input_fingerprints(config, inputs)
    ibd_manifest = load_ibd_manifest(config)
    manifest, task_audit, ibd_equivalence = build_all_manifests(arrays_by_task, ibd_manifest, config)
    split_checksums = write_manifests(
        manifest, task_audit, ibd_equivalence, Path(config.output_dir)
    )
    print("[preflight] Proving all 500 core-generated outer folds match the locked task manifests...", flush=True)
    core_split_audit = verify_core_generated_splits(manifest, arrays_by_task, config)
    write_preflight(
        config,
        inputs,
        task_audit,
        ibd_equivalence,
        core_split_audit,
        split_checksums,
        patch_record,
        input_fingerprints,
        core_versions,
    )
    print_preflight(config, task_audit, core_split_audit)
    if args.validate_only:
        return

    # Write an initial zero-completion progress file before the first expensive fit.
    aggregate_completed(config, manifest, strict=False, include_coefficients=False)

    tasks = TASK_SPECS[config.start_task_index - 1 : config.end_task_index]
    for task_order, task in enumerate(TASK_SPECS, start=1):
        if task not in tasks:
            continue
        print("#" * 118)
        print(f"TASK {task_order}/5: {task.report_name} ({task.folder})")
        print("#" * 118, flush=True)
        for repeat in range(config.start_repeat, config.end_repeat + 1):
            assert_locked_sources_unchanged(config, locked_payload, patch_record)
            split_seed = int(
                manifest.loc[
                    manifest["task_folder"].eq(task.folder) & manifest["repeat"].eq(repeat),
                    "split_seed",
                ].iloc[0]
            )
            if core_repeat_complete(config, task.folder, repeat):
                verify_repeat_output(config, manifest, task.folder, repeat)
                print(f"[resume] {task.folder} repeat {repeat:02d} already complete and verified.", flush=True)
            else:
                run_core_repeat(config, task.folder, repeat, split_seed)
                verify_repeat_output(config, manifest, task.folder, repeat)
                atomic_json(
                    repeat_dir(config, task.folder, repeat) / "ORCHESTRATOR_VERIFIED_COMPLETE.json",
                    {
                        "timestamp_utc": now_iso(),
                        "task_order": task_order,
                        "task_folder": task.folder,
                        "repeat": repeat,
                        "split_seed": split_seed,
                    },
                )
                print(f"[verified] {task.folder} repeat {repeat:02d}", flush=True)
            aggregate_completed(config, manifest, strict=False, include_coefficients=False)

    assert_locked_sources_unchanged(config, locked_payload, patch_record)
    full_request = (
        config.start_task_index == 1
        and config.end_task_index == len(TASK_SPECS)
        and config.start_repeat == 1
        and config.end_repeat == config.expected_repeats
    )
    final = aggregate_completed(
        config, manifest, strict=full_request, include_coefficients=True
    )
    completion_name = "RUN_COMPLETE.json" if full_request else "PARTIAL_REQUEST_COMPLETE.json"
    atomic_json(Path(config.output_dir) / completion_name, {"timestamp_utc": now_iso(), **final})
    print("=" * 118)
    print(
        "ALL REQUESTED JOINT H0 + RICCI TASKS COMPLETE"
        if full_request
        else "REQUESTED TASK/REPETITION RANGE COMPLETE; FULL ANALYSIS IS NOT YET MARKED COMPLETE"
    )
    print(f"Output: {config.output_dir}")
    print(f"Summary: {Path(config.output_dir) / 'aggregate' / 'repetition_performance_summary.csv'}")
    print(json.dumps(final, indent=2))
    print("=" * 118)
