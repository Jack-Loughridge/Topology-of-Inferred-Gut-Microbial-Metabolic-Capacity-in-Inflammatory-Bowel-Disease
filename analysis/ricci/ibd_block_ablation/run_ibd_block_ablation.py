#!/usr/bin/env python3
from __future__ import annotations

"""Run B-only and K0-only ablations for the primary IBD Ricci classifier.

The ablations deliberately reuse the frozen 20 x 5 participant-grouped split
manifests, C=0.02, preprocessing, solver, class weighting, model seeds,
participant aggregation, and reporting code from the completed [B | K0]
IBD C-path analysis.  Only the feature block supplied to the classifier changes.
"""

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np
import pandas as pd


SCHEMA_VERSION = 1
SCRIPT_VERSION = "1.0.0"
PRIMARY_C = 0.02
EXPECTED_ENGINE_SHA256 = (
    "88b4110b482cdde7d1a77803835ca3d5b8b414ee956c5987ce5acddeabb66682"
)
SPLIT_FILES = (
    "sample_split_manifest.csv",
    "participant_split_manifest.csv",
    "split_config.json",
)
BLOCK_LABELS = {
    "B": "B only",
    "K0": "K0 only",
}
METRICS = ("accuracy", "balanced_accuracy", "macro_f1", "roc_auc")
LEVELS = ("sample", "participant")


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, payload: Any) -> None:
    atomic_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def import_locked_engine(path: Path, expected_sha256: str) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"Canonical IBD C-path engine not found: {path}")
    observed = sha256_file(path)
    if observed != expected_sha256:
        raise RuntimeError(
            "Canonical engine hash mismatch. Refusing to run an unreviewed implementation.\n"
            f"  expected: {expected_sha256}\n"
            f"  observed: {observed}\n"
            f"  path: {path}"
        )
    spec = importlib.util.spec_from_file_location("locked_ricci_ibd_cpath", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import canonical engine from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def scalar_equal(observed: Any, expected: Any) -> bool:
    if isinstance(expected, float):
        try:
            return math.isclose(float(observed), expected, rel_tol=0.0, abs_tol=1e-15)
        except (TypeError, ValueError):
            return False
    return observed == expected


def audit_split_manifests(
    split_dir: Path,
    n_repeats: int,
    n_splits: int,
    split_seed: int,
) -> dict[str, Any]:
    paths = {name: split_dir / name for name in SPLIT_FILES}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing frozen split files:\n" + "\n".join(missing))

    config = json.loads(paths["split_config.json"].read_text(encoding="utf-8"))
    expected_config = {
        "n_repeats": n_repeats,
        "n_splits": n_splits,
        "split_seed": split_seed,
    }
    for key, expected in expected_config.items():
        if not scalar_equal(config.get(key), expected):
            raise ValueError(
                f"Frozen split configuration has {key}={config.get(key)!r}; expected {expected!r}"
            )

    sample = pd.read_csv(
        paths["sample_split_manifest.csv"],
        dtype={"sample_id": str, "participant_id": str},
        low_memory=False,
    )
    required = {
        "repeat",
        "fold",
        "split_seed",
        "role",
        "sample_id",
        "participant_id",
        "label",
    }
    missing_columns = required - set(sample.columns)
    if missing_columns:
        raise ValueError(f"Sample split manifest missing columns: {sorted(missing_columns)}")
    for column in ("repeat", "fold", "split_seed", "label"):
        sample[column] = pd.to_numeric(sample[column], errors="raise").astype(int)
    sample["role"] = sample["role"].astype(str).str.strip().str.lower()
    if set(sample["role"]) != {"train", "test"}:
        raise ValueError("Sample split manifest roles must be exactly train/test")
    if sorted(sample["repeat"].unique()) != list(range(1, n_repeats + 1)):
        raise ValueError("Sample split manifest repeat values are incomplete")
    if sorted(sample["fold"].unique()) != list(range(1, n_splits + 1)):
        raise ValueError("Sample split manifest fold values are incomplete")

    master = sample[["sample_id", "participant_id", "label"]].drop_duplicates()
    if master["sample_id"].duplicated().any():
        raise ValueError("A sample has conflicting metadata in the frozen split manifest")
    if master.groupby("participant_id")["label"].nunique().gt(1).any():
        raise ValueError("A participant has conflicting labels in the frozen split manifest")
    all_ids = set(master["sample_id"])

    for repeat in range(1, n_repeats + 1):
        repeated = sample[sample["repeat"].eq(repeat)]
        expected_repeat_seed = split_seed + (repeat - 1) * 1009
        seeds = repeated["split_seed"].unique().tolist()
        if seeds != [expected_repeat_seed]:
            raise ValueError(
                f"Repeat {repeat} has split seeds {seeds}; expected {[expected_repeat_seed]}"
            )
        tested: set[str] = set()
        for fold in range(1, n_splits + 1):
            rows = repeated[repeated["fold"].eq(fold)]
            train = rows[rows["role"].eq("train")]
            test = rows[rows["role"].eq("test")]
            if train.empty or test.empty:
                raise ValueError(f"Repeat {repeat}, fold {fold} lacks train or test rows")
            if set(train["participant_id"]) & set(test["participant_id"]):
                raise ValueError(f"Participant leakage in repeat {repeat}, fold {fold}")
            if set(train["label"]) != {0, 1} or set(test["label"]) != {0, 1}:
                raise ValueError(f"A class is absent in repeat {repeat}, fold {fold}")
            test_ids = set(test["sample_id"])
            if tested & test_ids:
                raise ValueError(f"Repeated test sample in repeat {repeat}")
            tested |= test_ids
        if tested != all_ids:
            raise ValueError(f"Test folds do not cover every sample in repeat {repeat}")

    participant = pd.read_csv(
        paths["participant_split_manifest.csv"],
        dtype={"participant_id": str},
        low_memory=False,
    )
    participant_required = {"repeat", "fold", "split_seed", "role", "participant_id", "label"}
    participant_missing = participant_required - set(participant.columns)
    if participant_missing:
        raise ValueError(
            f"Participant split manifest missing columns: {sorted(participant_missing)}"
        )
    sample_membership = (
        sample[["repeat", "fold", "split_seed", "role", "participant_id", "label"]]
        .drop_duplicates()
        .sort_values(["repeat", "fold", "role", "participant_id"])
        .reset_index(drop=True)
    )
    participant_membership = (
        participant[["repeat", "fold", "split_seed", "role", "participant_id", "label"]]
        .copy()
        .sort_values(["repeat", "fold", "role", "participant_id"])
        .reset_index(drop=True)
    )
    for column in ("repeat", "fold", "split_seed", "label"):
        participant_membership[column] = pd.to_numeric(
            participant_membership[column], errors="raise"
        ).astype(int)
    participant_membership["role"] = (
        participant_membership["role"].astype(str).str.strip().str.lower()
    )
    if not sample_membership.equals(participant_membership):
        raise ValueError("Participant split manifest disagrees with sample-derived memberships")

    return {
        "sample_count": int(len(master)),
        "participant_count": int(master["participant_id"].nunique()),
        "sample_class_counts": {
            str(key): int(value) for key, value in master["label"].value_counts().items()
        },
        "participant_class_counts": {
            str(key): int(value)
            for key, value in master.groupby("label")["participant_id"].nunique().items()
        },
        "sha256": {name: sha256_file(path) for name, path in paths.items()},
    }


def validate_combined_reference(
    combined_output_dir: Path,
    feature_dir: Path,
    n_repeats: int,
    n_splits: int,
    split_seed: int,
    model_seed: int,
    max_iter: int,
    tol: float,
    coef_eps: float,
    engine: Any,
) -> dict[str, Any]:
    config_path = combined_output_dir / "run_config.json"
    completion_path = combined_output_dir / "RUN_COMPLETE.json"
    if not config_path.is_file() or not completion_path.is_file():
        raise FileNotFoundError(
            "The completed [B|K0] reference must contain run_config.json and RUN_COMPLETE.json"
        )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    expected = {
        "n_repeats": n_repeats,
        "n_splits": n_splits,
        "split_seed": split_seed,
        "model_seed": model_seed,
        "max_iter": max_iter,
        "tol": tol,
        "coef_eps": coef_eps,
    }
    for key, value in expected.items():
        if not scalar_equal(config.get(key), value):
            raise ValueError(
                f"Combined [B|K0] reference has {key}={config.get(key)!r}; expected {value!r}"
            )
    configured_feature_dir = Path(
        config.get("feature_dir_resolved", config.get("feature_dir", ""))
    ).expanduser().resolve()
    if configured_feature_dir != feature_dir:
        raise ValueError(
            "Combined [B|K0] reference used a different feature directory:\n"
            f"  reference: {configured_feature_dir}\n"
            f"  requested: {feature_dir}"
        )
    c_values = config.get("C_values", config.get("c_values", []))
    if not any(math.isclose(float(value), PRIMARY_C, rel_tol=0.0, abs_tol=1e-15) for value in c_values):
        raise ValueError(f"Combined [B|K0] reference does not contain C={PRIMARY_C}")

    c_dir = combined_output_dir / engine.c_tag(PRIMARY_C)
    repetition_path = c_dir / "repetition_pooled_oof_results.csv"
    if not repetition_path.is_file():
        raise FileNotFoundError(f"Combined repetition results not found: {repetition_path}")
    split_audit = audit_split_manifests(
        combined_output_dir / "splits", n_repeats, n_splits, split_seed
    )
    return {
        "run_config_sha256": sha256_file(config_path),
        "run_complete_sha256": sha256_file(completion_path),
        "repetition_results_sha256": sha256_file(repetition_path),
        "repetition_results_path": str(repetition_path),
        "split_audit": split_audit,
    }


def feature_input_hashes(feature_dir: Path) -> dict[str, str]:
    paths = {
        "feature_matrix_B_K0": feature_dir / "feature_matrix_B_K0.npz",
        "matched_metadata": feature_dir / "matched_metadata.csv",
        "edge_metadata": feature_dir / "edge_metadata.csv",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing feature inputs:\n" + "\n".join(missing))
    return {name: sha256_file(path) for name, path in paths.items()}


def prepare_block_output(
    output_dir: Path,
    source_split_dir: Path,
    lock_payload: dict[str, Any],
) -> None:
    lock_path = output_dir / "ABLATION_LOCK.json"
    stable_hash = canonical_sha256(lock_payload)
    if output_dir.exists() and any(output_dir.iterdir()):
        if not lock_path.is_file():
            raise RuntimeError(
                f"Non-empty ablation output has no compatibility lock: {output_dir}. "
                "Use a new output directory."
            )
        existing = json.loads(lock_path.read_text(encoding="utf-8"))
        if existing.get("scientific_lock_sha256") != stable_hash:
            raise RuntimeError(
                f"Existing ablation output is scientifically incompatible: {output_dir}. "
                "Use a new output directory."
            )
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        atomic_json(
            lock_path,
            {
                "schema_version": SCHEMA_VERSION,
                "scientific_lock": lock_payload,
                "scientific_lock_sha256": stable_hash,
            },
        )

    target_split_dir = output_dir / "splits"
    target_split_dir.mkdir(parents=True, exist_ok=True)
    for name in SPLIT_FILES:
        source = source_split_dir / name
        target = target_split_dir / name
        if target.exists():
            if sha256_file(target) != sha256_file(source):
                raise RuntimeError(f"Saved ablation split differs from frozen reference: {target}")
        else:
            shutil.copy2(source, target)
        if sha256_file(target) != sha256_file(source):
            raise RuntimeError(f"Failed to reproduce frozen split file exactly: {target}")


def block_sliced_inputs(
    original_load_inputs: Any,
    block: str,
    feature_dir: Path,
    annotation_csv: Path | None,
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    X_full, task_meta, edge_meta, feature_meta = original_load_inputs(
        feature_dir, annotation_csv
    )
    n_edges = len(edge_meta)
    if X_full.shape[1] != 2 * n_edges:
        raise ValueError("Canonical loader did not return a [B|K0] matrix")
    if block == "B":
        start, stop = 0, n_edges
    elif block == "K0":
        start, stop = n_edges, 2 * n_edges
    else:
        raise ValueError(f"Unknown feature block: {block}")

    X_block = np.ascontiguousarray(X_full[:, start:stop], dtype=np.float32)
    selected_meta = feature_meta.iloc[start:stop].copy().reset_index(drop=True)
    selected_meta.insert(
        selected_meta.columns.get_loc("feature_index") + 1,
        "source_feature_index",
        selected_meta["feature_index"].to_numpy(dtype=int),
    )
    selected_meta["feature_index"] = np.arange(n_edges, dtype=int)
    if set(selected_meta["feature_type"]) != {block}:
        raise RuntimeError("Feature metadata block does not match the requested slice")
    if not np.isfinite(X_block).all():
        raise ValueError(f"{block}-only matrix contains non-finite values")
    if block == "B":
        unique = np.unique(X_block)
        if not set(unique.tolist()).issubset({0.0, 1.0}):
            raise ValueError(f"B-only block is not binary; values include {unique[:10].tolist()}")
    elif not np.any(X_block != 0):
        raise ValueError("K0-only block contains no non-zero curvature values")
    return X_block, task_meta, edge_meta, selected_meta


def run_block(
    engine: Any,
    engine_path: Path,
    engine_sha256: str,
    block: str,
    feature_dir: Path,
    combined_output_dir: Path,
    block_output_dir: Path,
    feature_hashes: dict[str, str],
    combined_audit: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    lock_payload = {
        "script_version": SCRIPT_VERSION,
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "canonical_engine_sha256": engine_sha256,
        "feature_block": block,
        "feature_definition": (
            "B_s(e)=1 if edge e is active in sample s, otherwise 0"
            if block == "B"
            else "K0_s(e)=Ricci curvature if edge e is active in sample s, otherwise 0"
        ),
        "C": PRIMARY_C,
        "n_repeats": args.n_repeats,
        "n_splits": args.n_splits,
        "split_seed": args.split_seed,
        "model_seed": args.model_seed,
        "max_iter": args.max_iter,
        "tol": args.tol,
        "coef_eps": args.coef_eps,
        "preprocessing": "StandardScaler fit on each outer-training fold only",
        "classifier": "L1 LogisticRegression(solver=saga,class_weight=balanced)",
        "participant_aggregation": "arithmetic mean of held-out sample probabilities within participant",
        "feature_input_sha256": feature_hashes,
        "frozen_split_sha256": combined_audit["split_audit"]["sha256"],
        "combined_reference_run_config_sha256": combined_audit["run_config_sha256"],
    }
    prepare_block_output(
        block_output_dir, combined_output_dir / "splits", lock_payload
    )

    original_load_inputs = engine.load_inputs
    original_fit_or_load_fold = engine.fit_or_load_fold

    def patched_load_inputs(
        requested_feature_dir: Path,
        annotation_csv: Path | None,
    ) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        return block_sliced_inputs(
            original_load_inputs,
            block,
            requested_feature_dir,
            annotation_csv,
        )

    def patched_fit_or_load_fold(*fit_args: Any, **fit_kwargs: Any) -> Any:
        bundle = original_fit_or_load_fold(*fit_args, **fit_kwargs)
        result = dict(bundle.result)
        total = int(result["selected_total"])
        total_mass = float(result["abs_coef_mass_total"])
        if block == "B":
            result.update(
                selected_B=total,
                selected_K0=0,
                abs_coef_mass_B=total_mass,
                abs_coef_mass_K0=0.0,
            )
        else:
            result.update(
                selected_B=0,
                selected_K0=total,
                abs_coef_mass_B=0.0,
                abs_coef_mass_K0=total_mass,
            )
        result["feature_block"] = block
        c_dir = Path(fit_kwargs["c_dir"])
        key = fit_kwargs["key"]
        _, result_path, _ = engine.fold_paths(c_dir, key.repeat, key.fold)
        engine.write_json(result_path, result)
        bundle.result = result
        return bundle

    engine.load_inputs = patched_load_inputs
    engine.fit_or_load_fold = patched_fit_or_load_fold
    try:
        engine_args = argparse.Namespace(
            feature_dir=str(feature_dir),
            output_dir=str(block_output_dir),
            edge_annotation_csv=(
                str(Path(args.edge_annotation_csv).expanduser().resolve())
                if args.edge_annotation_csv
                else None
            ),
            c_values=[PRIMARY_C],
            n_repeats=args.n_repeats,
            n_splits=args.n_splits,
            split_seed=args.split_seed,
            model_seed=args.model_seed,
            max_iter=args.max_iter,
            tol=args.tol,
            n_jobs=args.n_jobs,
            coef_eps=args.coef_eps,
            top_n=args.top_n,
            resume=args.resume,
            feature_block=block,
        )
        engine.run(engine_args)
    finally:
        engine.load_inputs = original_load_inputs
        engine.fit_or_load_fold = original_fit_or_load_fold

    config_path = block_output_dir / "run_config.json"
    run_config = json.loads(config_path.read_text(encoding="utf-8"))
    run_config.update(
        {
            "feature_block": block,
            "representation_label": BLOCK_LABELS[block],
            "feature_definition": lock_payload["feature_definition"],
            "ablation_role": "prespecified primary-task feature-block ablation",
            "fixed_combined_reference_C": PRIMARY_C,
            "combined_reference_output": str(combined_output_dir),
            "canonical_engine_path": str(engine_path),
            "canonical_engine_sha256": engine_sha256,
            "ablation_driver_sha256": lock_payload["script_sha256"],
        }
    )
    atomic_json(config_path, run_config)

    completion_path = block_output_dir / "RUN_COMPLETE.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    completion.update(
        {
            "feature_block": block,
            "representation_label": BLOCK_LABELS[block],
            "combined_reference_C": PRIMARY_C,
        }
    )
    atomic_json(completion_path, completion)

    c_dir = block_output_dir / engine.c_tag(PRIMARY_C)
    result_paths = sorted((c_dir / "fold_artifacts").glob("repeat_*_fold_*.json"))
    expected_models = args.n_repeats * args.n_splits
    if len(result_paths) != expected_models:
        raise RuntimeError(
            f"{block}: found {len(result_paths)} fold result files; expected {expected_models}"
        )
    fold_results = [json.loads(path.read_text(encoding="utf-8")) for path in result_paths]
    warnings = [row for row in fold_results if bool(row.get("convergence_warning"))]
    iteration_limit = [row for row in fold_results if int(row.get("n_iter", 0)) >= args.max_iter]
    if warnings or iteration_limit:
        raise RuntimeError(
            f"{block}: incomplete convergence audit: {len(warnings)} warning folds, "
            f"{len(iteration_limit)} folds at max_iter"
        )
    for row in fold_results:
        if row.get("feature_block") != block:
            raise RuntimeError(f"{block}: fold result lacks the correct feature-block marker")
        if block == "B" and int(row["selected_K0"]) != 0:
            raise RuntimeError("B-only fold reports selected K0 coefficients")
        if block == "K0" and int(row["selected_B"]) != 0:
            raise RuntimeError("K0-only fold reports selected B coefficients")

    feature_metadata_path = block_output_dir / "feature_metadata_used.csv"
    feature_metadata = pd.read_csv(feature_metadata_path, low_memory=False)
    if set(feature_metadata["feature_type"]) != {block}:
        raise RuntimeError(f"{block}: saved feature metadata contains another block")
    source_split_hashes = combined_audit["split_audit"]["sha256"]
    target_split_hashes = {
        name: sha256_file(block_output_dir / "splits" / name) for name in SPLIT_FILES
    }
    if target_split_hashes != source_split_hashes:
        raise RuntimeError(f"{block}: output split hashes differ from the combined reference")

    provenance = {
        "schema_version": SCHEMA_VERSION,
        "script_version": SCRIPT_VERSION,
        "status": "complete",
        "scientific_lock": lock_payload,
        "feature_count": int(len(feature_metadata)),
        "fold_models": expected_models,
        "convergence_warning_folds": 0,
        "folds_at_iteration_limit": 0,
        "output_sha256": {
            "run_config": sha256_file(config_path),
            "run_complete": sha256_file(completion_path),
            "repetition_results": sha256_file(
                c_dir / "repetition_pooled_oof_results.csv"
            ),
            "performance_summary": sha256_file(c_dir / "performance_summary.csv"),
            "feature_metadata": sha256_file(feature_metadata_path),
        },
    }
    atomic_json(block_output_dir / "ABLATION_PROVENANCE.json", provenance)
    return provenance


def load_repetition_results(path: Path, label: str, n_repeats: int) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    required = {"C", "repeat", "oof_samples", "oof_participants"}
    required |= {f"{level}_{metric}" for level in LEVELS for metric in METRICS}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{label} repetition table missing columns: {sorted(missing)}")
    frame = frame[np.isclose(frame["C"].astype(float), PRIMARY_C)].copy()
    if sorted(frame["repeat"].astype(int).tolist()) != list(range(1, n_repeats + 1)):
        raise ValueError(f"{label} does not contain exactly one row per requested repetition")
    frame.insert(0, "representation", label)
    return frame


def summarise_representations(long: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for representation, group in long.groupby("representation", sort=False):
        for level in LEVELS:
            row: dict[str, Any] = {
                "representation": representation,
                "evaluation_level": level,
                "n_repetitions": int(len(group)),
                "n_observations": int(group[f"oof_{level}s"].iloc[0]),
            }
            for metric in METRICS:
                values = group[f"{level}_{metric}"].to_numpy(dtype=float)
                row[f"{metric}_mean"] = float(np.mean(values))
                row[f"{metric}_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
                row[f"{metric}_q025"] = float(np.quantile(values, 0.025))
                row[f"{metric}_q975"] = float(np.quantile(values, 0.975))
            rows.append(row)
    return pd.DataFrame(rows)


def paired_deltas(long: pd.DataFrame) -> pd.DataFrame:
    available = set(long["representation"])
    comparisons = (
        ("K0 only", "B only"),
        ("[B|K0]", "B only"),
        ("[B|K0]", "K0 only"),
    )
    rows: list[dict[str, Any]] = []
    for left, right in comparisons:
        if left not in available or right not in available:
            continue
        a = long[long["representation"].eq(left)].set_index("repeat")
        b = long[long["representation"].eq(right)].set_index("repeat")
        if list(a.index) != list(b.index):
            raise RuntimeError(f"Paired repetition indices differ for {left} and {right}")
        if not np.array_equal(a["oof_samples"].to_numpy(), b["oof_samples"].to_numpy()):
            raise RuntimeError(f"Sample counts differ for {left} and {right}")
        if not np.array_equal(
            a["oof_participants"].to_numpy(), b["oof_participants"].to_numpy()
        ):
            raise RuntimeError(f"Participant counts differ for {left} and {right}")
        for level in LEVELS:
            for metric in METRICS:
                values = (
                    a[f"{level}_{metric}"].to_numpy(dtype=float)
                    - b[f"{level}_{metric}"].to_numpy(dtype=float)
                )
                rows.append(
                    {
                        "contrast": f"{left} minus {right}",
                        "left_representation": left,
                        "right_representation": right,
                        "evaluation_level": level,
                        "metric": metric,
                        "n_paired_repetitions": int(len(values)),
                        "delta_mean": float(np.mean(values)),
                        "delta_sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                        "delta_median": float(np.median(values)),
                        "delta_q025": float(np.quantile(values, 0.025)),
                        "delta_q975": float(np.quantile(values, 0.975)),
                        "proportion_delta_positive": float(np.mean(values > 0)),
                        "proportion_delta_zero": float(np.mean(values == 0)),
                    }
                )
    return pd.DataFrame(rows)


def latex_table(summary: pd.DataFrame) -> str:
    order = {"B only": 0, "K0 only": 1, "[B|K0]": 2}
    frame = summary.copy()
    frame["_order"] = frame["representation"].map(order)
    frame["_level"] = frame["evaluation_level"].map({"sample": 0, "participant": 1})
    frame = frame.sort_values(["_order", "_level"])
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\caption{Prespecified feature-block ablation for IBD versus non-IBD classification at $C=0.02$.}",
        r"\label{tab:ricci_feature_block_ablation}",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"Representation & Unit & Accuracy & Balanced accuracy & Macro F1 & ROC--AUC \\",
        r"\midrule",
    ]
    for row in frame.itertuples(index=False):
        representation = str(row.representation).replace("[B|K0]", r"$[B\mid K_0]$")
        if representation == "K0 only":
            representation = r"$K_0$ only"
        unit = "Participant" if row.evaluation_level == "participant" else "Sample"
        values = []
        for metric in METRICS:
            values.append(
                f"{getattr(row, metric + '_mean'):.3f} $\\pm$ "
                f"{getattr(row, metric + '_sd'):.3f}"
            )
        lines.append(f"{representation} & {unit} & " + " & ".join(values) + r" \\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\begin{minipage}{0.96\linewidth}",
            r"\footnotesize",
            r"Values are mean $\pm$ standard deviation across 20 pooled out-of-fold repetition estimates. All representations use the same participant-grouped folds, training-fold-only standardisation, L1 logistic regression, class weighting, regularisation value and model seeds. The standard deviations describe sensitivity to the repeated partitions and are not confidence intervals from independent cohorts.",
            r"\end{minipage}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def build_comparison_outputs(
    output_dir: Path,
    combined_output_dir: Path,
    engine: Any,
    n_repeats: int,
) -> dict[str, Any]:
    paths = {
        "B only": output_dir / "B_only" / engine.c_tag(PRIMARY_C) / "repetition_pooled_oof_results.csv",
        "K0 only": output_dir / "K0_only" / engine.c_tag(PRIMARY_C) / "repetition_pooled_oof_results.csv",
        "[B|K0]": combined_output_dir / engine.c_tag(PRIMARY_C) / "repetition_pooled_oof_results.csv",
    }
    available = {label: path for label, path in paths.items() if path.is_file()}
    if "[B|K0]" not in available:
        raise FileNotFoundError("Combined [B|K0] repetition results are unavailable")
    frames = [
        load_repetition_results(path, label, n_repeats)
        for label, path in available.items()
    ]
    long = pd.concat(frames, ignore_index=True)
    summary = summarise_representations(long)
    deltas = paired_deltas(long)
    atomic_csv(long, output_dir / "ablation_repetition_metrics_long.csv")
    atomic_csv(summary, output_dir / "ablation_performance_summary.csv")
    atomic_csv(deltas, output_dir / "ablation_paired_deltas.csv")
    atomic_text(output_dir / "ricci_feature_block_ablation.tex", latex_table(summary))
    return {
        "available_representations": list(available),
        "input_repetition_table_sha256": {
            label: sha256_file(path) for label, path in available.items()
        },
    }


def validate_feature_slices(engine: Any, feature_dir: Path, annotation_csv: Path | None) -> dict[str, Any]:
    X, _, edge_meta, feature_meta = engine.load_inputs(feature_dir, annotation_csv)
    n_edges = len(edge_meta)
    if X.shape[1] != 2 * n_edges:
        raise ValueError("Feature matrix is not ordered as [B|K0]")
    if not (feature_meta.iloc[:n_edges]["feature_type"] == "B").all():
        raise ValueError("The first feature block is not B")
    if not (feature_meta.iloc[n_edges:]["feature_type"] == "K0").all():
        raise ValueError("The second feature block is not K0")
    B = X[:, :n_edges]
    K0 = X[:, n_edges:]
    if not set(np.unique(B).tolist()).issubset({0.0, 1.0}):
        raise ValueError("B block is not binary")
    if not np.isfinite(K0).all() or not np.any(K0 != 0):
        raise ValueError("K0 block is empty or non-finite")
    return {
        "samples": int(X.shape[0]),
        "edges": int(n_edges),
        "B_nonzero": int(np.count_nonzero(B)),
        "K0_nonzero": int(np.count_nonzero(K0)),
    }


def parse_args() -> argparse.Namespace:
    base = Path.home() / "Real_Data"
    default_engine = (
        Path(__file__).resolve().parents[1]
        / "ibd_cpath"
        / "repeated_ricci_ibd_cpath.py"
    )
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Locked B-only and K0-only ablations for the primary IBD Ricci classifier",
    )
    parser.add_argument("--engine", type=Path, default=default_engine)
    parser.add_argument("--expected-engine-sha256", default=EXPECTED_ENGINE_SHA256)
    parser.add_argument(
        "--feature-dir",
        type=Path,
        default=base / "Ricci_Classifier_Faithful_Eps0001_n250_v3",
    )
    parser.add_argument(
        "--combined-output-dir",
        type=Path,
        default=base / "Ricci_IBD_RepeatedCV_CPath",
        help="Completed [B|K0] IBD C-path output; its split manifests are reused byte-for-byte",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=base / "Ricci_IBD_FeatureBlock_Ablation_C002",
    )
    parser.add_argument("--blocks", nargs="+", choices=("B", "K0"), default=("B", "K0"))
    parser.add_argument("--edge-annotation-csv", default=None)
    parser.add_argument("--n-repeats", type=int, default=20)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--split-seed", type=int, default=13)
    parser.add_argument("--model-seed", type=int, default=20260717)
    parser.add_argument("--max-iter", type=int, default=10000)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--coef-eps", type=float, default=1e-12)
    parser.add_argument("--top-n", type=int, default=100)
    parser.add_argument("--validate-only", action="store_true")
    parser.set_defaults(resume=True)
    parser.add_argument("--resume", dest="resume", action="store_true")
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(set(args.blocks)) != len(args.blocks):
        raise ValueError("--blocks contains duplicates")
    if args.n_repeats < 1 or args.n_splits < 2:
        raise ValueError("Invalid repeated-CV dimensions")
    if args.max_iter < 1 or args.tol <= 0 or args.coef_eps < 0:
        raise ValueError("Invalid classifier settings")

    engine_path = args.engine.expanduser().resolve()
    feature_dir = args.feature_dir.expanduser().resolve()
    combined_output_dir = args.combined_output_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    engine = import_locked_engine(engine_path, args.expected_engine_sha256)
    feature_hashes = feature_input_hashes(feature_dir)
    combined_audit = validate_combined_reference(
        combined_output_dir=combined_output_dir,
        feature_dir=feature_dir,
        n_repeats=args.n_repeats,
        n_splits=args.n_splits,
        split_seed=args.split_seed,
        model_seed=args.model_seed,
        max_iter=args.max_iter,
        tol=args.tol,
        coef_eps=args.coef_eps,
        engine=engine,
    )
    feature_audit = validate_feature_slices(
        engine,
        feature_dir,
        Path(args.edge_annotation_csv).expanduser().resolve()
        if args.edge_annotation_csv
        else None,
    )

    print("=" * 108)
    print("RICCI IBD FEATURE-BLOCK ABLATION PREFLIGHT")
    print("=" * 108)
    print(f"Canonical engine: {engine_path}")
    print(f"Engine SHA-256: {args.expected_engine_sha256}")
    print(f"Feature matrix: {feature_dir}")
    print(f"Combined reference: {combined_output_dir}")
    print(f"Output: {output_dir}")
    print(f"Blocks: {', '.join(args.blocks)}")
    print(f"Fixed C: {PRIMARY_C}")
    print(f"Design: {args.n_repeats} repetitions x {args.n_splits} participant-grouped folds")
    print(f"Feature audit: {feature_audit}")
    print("Frozen split-manifest audit: PASSED")
    print("RICCI IBD FEATURE-BLOCK ABLATION PREFLIGHT: PASSED")
    print("=" * 108)
    if args.validate_only:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    block_reports: dict[str, Any] = {}
    for block in args.blocks:
        print("\n" + "#" * 108)
        print(f"RUNNING {BLOCK_LABELS[block].upper()}")
        print("#" * 108)
        block_reports[block] = run_block(
            engine=engine,
            engine_path=engine_path,
            engine_sha256=args.expected_engine_sha256,
            block=block,
            feature_dir=feature_dir,
            combined_output_dir=combined_output_dir,
            block_output_dir=output_dir / f"{block}_only",
            feature_hashes=feature_hashes,
            combined_audit=combined_audit,
            args=args,
        )

    comparison = build_comparison_outputs(
        output_dir, combined_output_dir, engine, args.n_repeats
    )
    complete_blocks = {
        block
        for block in BLOCK_LABELS
        if (output_dir / f"{block}_only" / "ABLATION_PROVENANCE.json").is_file()
    }
    complete = complete_blocks == set(BLOCK_LABELS)
    final_payload = {
        "schema_version": SCHEMA_VERSION,
        "script_version": SCRIPT_VERSION,
        "status": "complete" if complete else "partial",
        "feature_blocks_complete": sorted(complete_blocks),
        "C": PRIMARY_C,
        "n_repeats": args.n_repeats,
        "n_splits": args.n_splits,
        "canonical_engine_sha256": args.expected_engine_sha256,
        "feature_input_sha256": feature_hashes,
        "combined_reference": combined_audit,
        "block_reports": block_reports,
        "comparison": comparison,
    }
    atomic_json(output_dir / "ABLATION_RUN_COMPLETE.json", final_payload)
    released = [
        output_dir / "ablation_repetition_metrics_long.csv",
        output_dir / "ablation_performance_summary.csv",
        output_dir / "ablation_paired_deltas.csv",
        output_dir / "ricci_feature_block_ablation.tex",
        output_dir / "ABLATION_RUN_COMPLETE.json",
    ]
    atomic_json(
        output_dir / "ablation_output_manifest.json",
        {"sha256": {path.name: sha256_file(path) for path in released}},
    )

    print("\n" + "=" * 108)
    print("RICCI IBD FEATURE-BLOCK ABLATION COMPLETE" if complete else "RICCI IBD FEATURE-BLOCK ABLATION PARTIAL")
    print(f"Performance summary: {output_dir / 'ablation_performance_summary.csv'}")
    print(f"Paired deltas: {output_dir / 'ablation_paired_deltas.csv'}")
    print(f"LaTeX table: {output_dir / 'ricci_feature_block_ablation.tex'}")
    print("=" * 108)


if __name__ == "__main__":
    main()
