#!/usr/bin/env python3
"""Apply frozen IBDMDB species benchmarks to Serrano-Gomez external v1.

The deployment is deliberately fail-closed:

* all three full-source artifacts must be complete and agree exactly on the
  frozen source feature axis and fitted preprocessing;
* external abundances are reconstructed from the same MetaPhlAn profiles and
  the exact parser/taxonomic projector in the validated structural-v1 source;
* CLR, variance selection and standardisation are applied from the saved
  source artifacts and are never refitted externally;
* probabilities are computed before the external label column is read;
* the decision threshold remains fixed at 0.5;
* participant probabilities are arithmetic means over host_subject_id.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    recall_score,
    roc_auc_score,
)


BASE = Path.home() / "Real_Data"
SCRIPT_VERSION = "1.1.0"
BENCHMARK_ROOT = BASE / "Species_Benchmarks_RepeatedCV_IBD_LastOccurrence"
TASK_FOLDER = "IBD_vs_nonIBD"
MODEL_ORDER = ("logistic_l1", "random_forest", "xgboost")
MODEL_LABELS = {
    "logistic_l1": "Species L1 logistic",
    "random_forest": "Species random forest",
    "xgboost": "Species XGBoost",
}
CLASS_NAMES = ("nonIBD", "IBD")

EXTERNAL_ROOT = (
    BASE
    / "external_validation"
    / "serrano_gomez_ibd"
    / "ge50_external_validation"
)
STRUCTURAL_V1 = EXTERNAL_ROOT / "structural_frozen_ibdmdb"
EXTERNAL_METADATA = (
    STRUCTURAL_V1 / "ricci_features_frozen_training_order" / "matched_metadata.csv"
)
STRUCTURAL_SOURCE = (
    EXTERNAL_ROOT
    / "scripts"
    / "serrano_gomez_external_structural_frozen_ibdmdb_v1"
    / "external_structural_validation.py"
)
PROFILE_ROOT_CANDIDATES = (
    EXTERNAL_ROOT / "metaphlan2_v260_profiles",
    EXTERNAL_ROOT / "metaphlan_profiles",
)
DEFAULT_OUTPUT_DIR = (
    STRUCTURAL_V1 / "external_species_benchmarks_ibd_vs_nonibd_last_occurrence_v1"
)
LEGACY_ARTIFACT_SCHEMA = "species-benchmarks-repeated-cv-v1.1-complete-case"
LEGACY_CODE_VERSION = "1.1.0"
MODERN_ARTIFACT_SCHEMA = 1
MODERN_CODE_VERSION = "2.0.0"
PARTICIPANT_COLUMN = "host_subject_id"
INVALID_TEXT = {"", "nan", "none", "null", "missing", "unknown", "unmapped"}
EPS = 1e-15


def die(message: str) -> None:
    raise RuntimeError(message)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    raise TypeError(type(value).__name__)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=json_value) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_csv(frame: pd.DataFrame, path: Path, index: bool = False) -> None:
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=index)
    os.replace(temporary, path)


def atomic_text(path: Path, content: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def normalise_sample_id(value: object) -> str:
    text = Path(str(value).strip()).name
    for suffix in (".txt", ".tsv", ".csv", ".gz", ".bz2"):
        while text.lower().endswith(suffix):
            text = text[: -len(suffix)]
    return text


def binary_label(value: object) -> int:
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, (int, np.integer, float, np.floating)):
        numeric = float(value)
        if math.isfinite(numeric) and numeric in {0.0, 1.0}:
            return int(numeric)
        die(f"Unsupported binary label: {value!r}")
    raw = str(value).strip().lower()
    if re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)", raw):
        numeric = float(raw)
        if math.isfinite(numeric) and numeric in {0.0, 1.0}:
            return int(numeric)
        die(f"Unsupported binary label: {value!r}")
    compact = re.sub(r"[\s_-]+", "", raw)
    if compact in {"nonibd", "healthy", "healthycontrol", "control", "hc"}:
        return 0
    if compact in {
        "ibd",
        "uc",
        "ulcerativecolitis",
        "cd",
        "crohn",
        "crohns",
        "crohnsdisease",
    }:
        return 1
    die(f"Unsupported binary label: {value!r}")


def canonical_class_order(value: Any, label: str) -> list[str]:
    if isinstance(value, dict):
        try:
            items = sorted((int(key), str(name)) for key, name in value.items())
        except Exception as exc:
            raise RuntimeError(f"{label} class mapping is invalid: {value!r}") from exc
        if [index for index, _ in items] != [0, 1]:
            die(f"{label} class mapping indices are not [0,1]: {value!r}")
        names = [name for _, name in items]
    else:
        names = [str(name) for name in value]
    canonical = []
    for name in names:
        compact = re.sub(r"[\s_-]+", "", name.strip().lower())
        if compact == "nonibd":
            canonical.append("nonIBD")
        elif compact == "ibd":
            canonical.append("IBD")
        else:
            die(f"{label} contains an unsupported class name: {name!r}")
    return canonical


def required_paths(benchmark_root: Path) -> tuple[dict[str, Path], str]:
    paths: dict[str, Path] = {
        "external_metadata": EXTERNAL_METADATA,
        "source_design_summary": benchmark_root / "input_and_design_summary.json",
        "structural_source": STRUCTURAL_SOURCE,
    }
    legacy_root = benchmark_root / "full_source_models"
    modern_root = benchmark_root / TASK_FOLDER / "full_source_models"
    legacy_models = [legacy_root / name / "full_source_model.joblib" for name in MODEL_ORDER]
    modern_models = [modern_root / name / "full_source_model.joblib" for name in MODEL_ORDER]
    if all(path.is_file() for path in legacy_models):
        layout = "legacy_ibd_complete_case_v1_1"
        model_root = legacy_root
        marker_name = "FULL_MODEL_COMPLETE.json"
    elif all(path.is_file() for path in modern_models):
        layout = "modern_all_tasks_v2"
        model_root = modern_root
        marker_name = "MODEL_COMPLETE.json"
    else:
        layout = "unresolved"
        model_root = modern_root
        marker_name = "MODEL_COMPLETE.json"
    for model_name in MODEL_ORDER:
        directory = model_root / model_name
        paths[f"model_{model_name}"] = directory / "full_source_model.joblib"
        paths[f"marker_{model_name}"] = directory / marker_name
    missing = [f"{name}: {path}" for name, path in paths.items() if not path.is_file()]
    if missing:
        die("Required files are missing:\n" + "\n".join(missing))
    if not any(path.is_dir() for path in PROFILE_ROOT_CANDIDATES):
        die(
            "No MetaPhlAn profile root exists. Checked:\n"
            + "\n".join(str(path) for path in PROFILE_ROOT_CANDIDATES)
        )
    return paths, layout


def validate_preprocessor(pre: dict[str, Any], model_name: str) -> dict[str, Any]:
    required = {
        "support_mask",
        "scaler_mean",
        "scaler_scale",
        "scaler_var",
        "input_feature_names",
        "selected_feature_names",
        "pseudocount",
        "variance_threshold",
    }
    missing = sorted(required - set(pre))
    if missing:
        die(f"{model_name} preprocessor is missing keys: {missing}")

    input_names = [str(value) for value in pre["input_feature_names"]]
    selected_names = [str(value) for value in pre["selected_feature_names"]]
    support = np.asarray(pre["support_mask"], dtype=bool).reshape(-1)
    mean = np.asarray(pre["scaler_mean"], dtype=np.float64).reshape(-1)
    scale = np.asarray(pre["scaler_scale"], dtype=np.float64).reshape(-1)
    var = np.asarray(pre["scaler_var"], dtype=np.float64).reshape(-1)
    pseudocount = float(pre["pseudocount"])
    variance_threshold = float(pre["variance_threshold"])

    if not input_names or len(input_names) != len(set(input_names)):
        die(f"{model_name} input feature names are empty or duplicated")
    if support.shape != (len(input_names),):
        die(f"{model_name} variance mask has the wrong length")
    expected_selected = [input_names[index] for index in np.flatnonzero(support)]
    if selected_names != expected_selected:
        die(f"{model_name} selected feature names disagree with its variance mask")
    n_selected = int(support.sum())
    if mean.shape != (n_selected,) or scale.shape != (n_selected,) or var.shape != (n_selected,):
        die(f"{model_name} fitted scaler dimensions are inconsistent")
    if not all(np.isfinite(array).all() for array in (mean, scale, var)):
        die(f"{model_name} fitted preprocessor contains nonfinite values")
    if np.any(scale <= 0) or pseudocount <= 0 or variance_threshold < 0:
        die(f"{model_name} fitted preprocessing parameters are invalid")

    return {
        "support_mask": support,
        "scaler_mean": mean,
        "scaler_scale": scale,
        "scaler_var": var,
        "input_feature_names": input_names,
        "selected_feature_names": selected_names,
        "pseudocount": pseudocount,
        "variance_threshold": variance_threshold,
    }


def load_frozen_models(
    paths: dict[str, Path], layout: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    loaded: dict[str, Any] = {}
    reference_pre: dict[str, Any] | None = None
    training_ids_reference: list[str] | None = None

    for model_name in MODEL_ORDER:
        marker = json.loads(paths[f"marker_{model_name}"].read_text(encoding="utf-8"))
        if marker.get("model") != model_name:
            die(f"Completion marker does not identify model {model_name}")
        if "class_names" in marker:
            observed = canonical_class_order(marker["class_names"], f"{model_name} marker")
            if observed != list(CLASS_NAMES):
                die(f"Completion marker has unexpected classes for {model_name}")
        if "task_folder" in marker and marker["task_folder"] != TASK_FOLDER:
            die(f"Completion marker has unexpected task folder for {model_name}")

        artifact = joblib.load(paths[f"model_{model_name}"])
        required = {
            "schema_version",
            "code_version",
            "model_name",
            "class_names",
            "model",
            "preprocessor",
        }
        missing = sorted(required - set(artifact))
        if missing:
            die(f"{model_name} artifact is missing keys: {missing}")
        if layout == "legacy_ibd_complete_case_v1_1":
            expected_schema: Any = LEGACY_ARTIFACT_SCHEMA
            expected_code = LEGACY_CODE_VERSION
            sample_key = "sample_ids"
            participant_key = "participant_ids"
        elif layout == "modern_all_tasks_v2":
            expected_schema = MODERN_ARTIFACT_SCHEMA
            expected_code = MODERN_CODE_VERSION
            sample_key = "training_sample_ids"
            participant_key = "training_participant_ids"
        else:
            die(f"Unsupported benchmark layout: {layout}")
        if artifact["schema_version"] != expected_schema:
            die(f"Unexpected artifact schema for {model_name}: {artifact['schema_version']}")
        if str(artifact["code_version"]) != expected_code:
            die(f"Unexpected benchmark code version for {model_name}: {artifact['code_version']}")
        if artifact["model_name"] != model_name:
            die(f"Artifact identity mismatch for {model_name}")
        if "task_folder" in artifact and artifact["task_folder"] != TASK_FOLDER:
            die(f"Artifact task-folder mismatch for {model_name}")
        if layout == "legacy_ibd_complete_case_v1_1":
            task_text = str(artifact.get("task", ""))
            if "IBD" not in task_text or "non-IBD" not in task_text:
                die(f"Legacy artifact task is not IBD versus non-IBD: {task_text!r}")
        observed_classes = canonical_class_order(
            artifact["class_names"], f"{model_name} artifact"
        )
        if observed_classes != list(CLASS_NAMES):
            die(f"Artifact class order mismatch for {model_name}")

        model = artifact["model"]
        model_classes = np.asarray(getattr(model, "classes_", []), dtype=int)
        if not np.array_equal(model_classes, np.asarray([0, 1], dtype=int)):
            die(f"{model_name} estimator class order is not [0,1]: {model_classes.tolist()}")
        if not callable(getattr(model, "predict_proba", None)):
            die(f"{model_name} estimator has no predict_proba method")

        pre = validate_preprocessor(dict(artifact["preprocessor"]), model_name)
        if sample_key not in artifact:
            die(f"{model_name} artifact is missing {sample_key}")
        training_ids = [normalise_sample_id(value) for value in artifact[sample_key]]
        if not training_ids or len(training_ids) != len(set(training_ids)):
            die(f"{model_name} training sample IDs are empty or duplicated")
        if participant_key in artifact and len(artifact[participant_key]) != len(training_ids):
            die(f"{model_name} training participant IDs have the wrong length")
        if "training_labels" in artifact:
            raw_labels = list(artifact["training_labels"])
            unique_labels = sorted(set(str(value) for value in raw_labels))
            labels = {
                canonical_class_order([value], f"{model_name} training labels")[0]
                for value in unique_labels
            }
            if labels != set(CLASS_NAMES) or len(raw_labels) != len(training_ids):
                die(f"{model_name} training labels are inconsistent")
        if "feature_names" in artifact:
            artifact_features = [str(value) for value in artifact["feature_names"]]
            if artifact_features != pre["input_feature_names"]:
                die(f"{model_name} artifact feature_names disagree with its preprocessor")
        if "n_input_features" in marker:
            if int(marker["n_input_features"]) != len(pre["input_feature_names"]):
                die(f"{model_name} marker has the wrong input-feature count")
        if "n_variance_selected_features" in marker:
            if int(marker["n_variance_selected_features"]) != int(pre["support_mask"].sum()):
                die(f"{model_name} marker has the wrong selected-feature count")
        if "n_samples" in marker and int(marker["n_samples"]) != len(training_ids):
            die(f"{model_name} marker has the wrong training-sample count")

        if reference_pre is None:
            reference_pre = pre
            training_ids_reference = training_ids
        else:
            for key in ("input_feature_names", "selected_feature_names"):
                if pre[key] != reference_pre[key]:
                    die(f"Frozen preprocessors disagree on {key}: {model_name}")
            for key in ("support_mask", "scaler_mean", "scaler_scale", "scaler_var"):
                if not np.array_equal(pre[key], reference_pre[key]):
                    die(f"Frozen preprocessors disagree on {key}: {model_name}")
            for key in ("pseudocount", "variance_threshold"):
                if pre[key] != reference_pre[key]:
                    die(f"Frozen preprocessors disagree on {key}: {model_name}")
            if training_ids != training_ids_reference:
                die(f"Frozen models disagree on their source training sample order: {model_name}")

        loaded[model_name] = {
            "model": model,
            "artifact": artifact,
            "model_sha256": sha256_file(paths[f"model_{model_name}"]),
            "marker_sha256": sha256_file(paths[f"marker_{model_name}"]),
        }

    if reference_pre is None:
        die("No frozen benchmark models were loaded")
    return loaded, reference_pre


def source_scale_multiplier(summary_path: Path, override: float | None) -> tuple[float, dict[str, Any]]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    diagnostics = summary.get("species_diagnostics", {})
    source_min = diagnostics.get("row_sum_min", diagnostics.get("row_sum_min_complete_case"))
    source_median_value = diagnostics.get(
        "row_sum_median", diagnostics.get("row_sum_median_complete_case")
    )
    source_max = diagnostics.get("row_sum_max", diagnostics.get("row_sum_max_complete_case"))
    source_median = float(source_median_value) if source_median_value is not None else float("nan")
    if override is not None:
        if not math.isfinite(override) or override <= 0:
            die("--external-abundance-multiplier must be finite and positive")
        multiplier = float(override)
        rule = "explicit command-line override"
    elif math.isfinite(source_median) and 10.0 <= source_median <= 110.0:
        multiplier = 1.0
        rule = "source rows are percentage-scale; retain external abundance_percent"
    elif math.isfinite(source_median) and 0.1 <= source_median <= 1.1:
        multiplier = 0.01
        rule = "source rows are proportion-scale; divide external abundance_percent by 100"
    else:
        die(
            "Could not prove the source abundance scale from input_and_design_summary.json: "
            f"row_sum_median={source_median}. Inspect it and pass --external-abundance-multiplier."
        )
    return multiplier, {
        "source_row_sum_min": source_min,
        "source_row_sum_median": source_median_value,
        "source_row_sum_max": source_max,
        "external_abundance_multiplier": multiplier,
        "scale_rule": rule,
    }


def load_structural_mapper(path: Path) -> ModuleType:
    """Import the validated structural-v1 source without invoking its CLI."""
    spec = importlib.util.spec_from_file_location("serrano_structural_v1_mapper", path)
    if spec is None or spec.loader is None:
        die(f"Could not create an import specification for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise RuntimeError(f"Could not import validated structural source {path}: {exc}") from exc
    required = (
        "parse_metaphlan_species",
        "profile_filename_score",
        "build_species_key_map",
        "project_external_species",
    )
    missing = [name for name in required if not callable(getattr(module, name, None))]
    if missing:
        die(f"Validated structural source is missing required functions: {missing}")
    return module


def verify_structural_training_axis(
    module: ModuleType, benchmark_feature_names: list[str]
) -> dict[str, Any]:
    """Prove that the structural mapper and benchmark use the same 508 coordinates."""
    loader = getattr(module, "load_training_species_axis", None)
    training_path = getattr(module, "TRAIN_SPECIES_PATH", None)
    if not callable(loader) or training_path is None:
        die("Validated structural source does not expose its frozen training species axis")
    training_path = Path(training_path).expanduser().resolve()
    if not training_path.is_file():
        die(f"Structural training species table is missing: {training_path}")
    structural_names, structural_stats = loader(training_path)
    structural_names = [str(value) for value in structural_names]
    if structural_names != benchmark_feature_names:
        mismatch = next(
            (
                index
                for index, (left, right) in enumerate(
                    zip(structural_names, benchmark_feature_names)
                )
                if left != right
            ),
            min(len(structural_names), len(benchmark_feature_names)),
        )
        die(
            "Structural and benchmark species axes are not exactly identical in frozen order: "
            f"structural={len(structural_names)}, benchmark={len(benchmark_feature_names)}, "
            f"first_mismatch_index={mismatch}"
        )
    return {
        "training_species_path": str(training_path),
        "training_species_sha256": sha256_file(training_path),
        "n_species": len(structural_names),
        "exact_order_match_to_benchmark": True,
        "training_scale_statistics": structural_stats,
    }


def _path_mentions_sample(path: Path, root: Path, sample_id: str) -> bool:
    relative_parts = path.relative_to(root).parts
    if any(part == sample_id for part in relative_parts[:-1]):
        return True
    filename = relative_parts[-1]
    return re.search(
        rf"(?<![A-Za-z0-9]){re.escape(sample_id)}(?![A-Za-z0-9])", filename
    ) is not None


def locate_and_parse_profile(
    sample_id: str, module: ModuleType
) -> tuple[Path, pd.Series, dict[str, Any], list[str]]:
    """Use the source parser and its deterministic preference rule for one sample."""
    extensions = set(
        getattr(module, "PROFILE_EXTS", {".tsv", ".txt", ".profile", ".csv"})
    )
    all_diagnostics: list[str] = []
    for root in PROFILE_ROOT_CANDIDATES:
        if not root.is_dir():
            continue
        candidates = [
            path
            for path in root.rglob("*")
            if path.is_file()
            and path.suffix.lower() in extensions
            and _path_mentions_sample(path, root, sample_id)
            and int(module.profile_filename_score(path)) >= 0
        ]
        parsed: list[tuple[Path, pd.Series, dict[str, Any], int]] = []
        for path in sorted(candidates):
            try:
                species, meta = module.parse_metaphlan_species(path)
                parsed.append(
                    (path, species, dict(meta), int(module.profile_filename_score(path)))
                )
            except Exception as exc:
                all_diagnostics.append(f"reject {path}: {type(exc).__name__}: {exc}")
        if parsed:
            parsed.sort(
                key=lambda item: (
                    -item[3],
                    -int(item[2]["n_species_rows"]),
                    -float(item[2]["species_abundance_sum_percent"]),
                    str(item[0]),
                )
            )
            path, species, meta, score = parsed[0]
            meta["filename_score"] = score
            meta["n_valid_profile_candidates"] = len(parsed)
            meta["profile_root"] = str(root)
            return path, species, meta, all_diagnostics
    examples = " | ".join(all_diagnostics[:5])
    die(
        f"No parseable MetaPhlAn profile found for {sample_id} under the approved roots. "
        f"Parser diagnostics: {examples or 'no sample-matching candidate files'}"
    )


def build_external_matrix_without_labels(
    pre: dict[str, Any],
    multiplier: float,
    module: ModuleType,
    min_mapped_fraction: float,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    identity = pd.read_csv(
        EXTERNAL_METADATA,
        usecols=["sample_id", PARTICIPANT_COLUMN],
        dtype=str,
    )
    identity["sample_id"] = identity["sample_id"].map(normalise_sample_id)
    identity[PARTICIPANT_COLUMN] = identity[PARTICIPANT_COLUMN].astype(str).str.strip()
    if identity["sample_id"].duplicated().any():
        die("External metadata sample IDs are duplicated")
    invalid_participant = identity[PARTICIPANT_COLUMN].str.lower().isin(INVALID_TEXT)
    if invalid_participant.any():
        die(
            f"Invalid {PARTICIPANT_COLUMN} values: "
            f"{identity.loc[invalid_participant, 'sample_id'].tolist()}"
        )

    feature_names = pre["input_feature_names"]
    unique_key_map, ambiguous_keys = module.build_species_key_map(feature_names)
    matrix = np.zeros((len(identity), len(feature_names)), dtype=np.float64)
    audit_rows: list[dict[str, Any]] = []
    profile_rows: list[dict[str, Any]] = []
    mapping_frames: list[pd.DataFrame] = []

    for row_index, sample_id in enumerate(identity["sample_id"]):
        path, external_species, profile_meta, parser_diagnostics = locate_and_parse_profile(
            sample_id, module
        )
        projected, mapping_meta, mapping_frame = module.project_external_species(
            external_species, feature_names, unique_key_map, ambiguous_keys
        )
        projected = projected.reindex(feature_names)
        if projected.isna().any() or list(projected.index) != feature_names:
            die(f"{sample_id} projection did not preserve the exact frozen species order")
        mapped_fraction = float(mapping_meta["mapped_fraction_of_species_abundance"])
        if mapped_fraction < min_mapped_fraction:
            die(
                f"{sample_id} mapped only {mapped_fraction:.3%} of species abundance, below "
                f"the prespecified {min_mapped_fraction:.3%} minimum"
            )
        matrix[row_index] = projected.to_numpy(dtype=np.float64) * multiplier
        row_sum = float(matrix[row_index].sum())
        if row_sum <= 0:
            die(f"{sample_id} has an all-zero external benchmark vector")
        mapping_frame.insert(0, "sample_id", sample_id)
        mapping_frames.append(mapping_frame)
        profile_rows.append(
            {
                "sample_id": sample_id,
                "profile_path": str(path),
                "profile_sha256": sha256_file(path),
                "parser_diagnostic_examples": " | ".join(parser_diagnostics[:3]),
                **profile_meta,
            }
        )
        audit_rows.append(
            {
                "sample_id": sample_id,
                "profile_path": str(path),
                **mapping_meta,
                "frozen_raw_vector_sum_after_scale_match": row_sum,
            }
        )

    if not np.isfinite(matrix).all() or (matrix < 0).any():
        die("Constructed external benchmark matrix is nonfinite or negative")
    return (
        identity,
        matrix,
        pd.DataFrame(audit_rows),
        pd.DataFrame(profile_rows),
        pd.concat(mapping_frames, ignore_index=True),
    )


def frozen_transform(matrix: np.ndarray, pre: dict[str, Any]) -> np.ndarray:
    log_matrix = np.log(matrix + pre["pseudocount"])
    clr = log_matrix - log_matrix.mean(axis=1, keepdims=True)
    selected = clr[:, pre["support_mask"]]
    transformed = (selected - pre["scaler_mean"]) / pre["scaler_scale"]
    if not np.isfinite(transformed).all():
        die("Nonfinite values after applying frozen source preprocessing")
    return transformed


def aligned_probabilities(
    model: Any, transformed: np.ndarray, model_name: str
) -> np.ndarray:
    raw_native = np.asarray(model.predict_proba(transformed))
    native_dtype = raw_native.dtype
    if raw_native.shape != (len(transformed), 2):
        die(
            f"{model_name} returned probability shape {raw_native.shape}; "
            f"expected ({len(transformed)}, 2)"
        )
    raw = raw_native.astype(np.float64, copy=False)
    classes = np.asarray(model.classes_, dtype=int)
    if classes.shape != (2,):
        die(f"{model_name} estimator class vector has shape {classes.shape}, expected (2,)")
    output = np.zeros((len(transformed), 2), dtype=np.float64)
    for source_column, class_index in enumerate(classes):
        if class_index not in {0, 1}:
            die(f"Unexpected estimator class index: {class_index}")
        output[:, int(class_index)] = raw[:, source_column]
    if not np.isfinite(output).all() or np.any(output < 0) or np.any(output > 1):
        die(f"{model_name} produced nonfinite or out-of-range probabilities")

    # XGBoost predict_proba commonly returns float32. Its two columns can sum
    # to 1 +/- a few float32 ulps even though the probabilities are valid. Use
    # a precision-aware validation bound; do not renormalise or otherwise alter
    # the estimator's frozen outputs.
    if np.issubdtype(native_dtype, np.floating):
        simplex_tolerance = max(1e-12, 32.0 * float(np.finfo(native_dtype).eps))
    else:
        simplex_tolerance = 1e-12
    row_sum_error = np.abs(output.sum(axis=1) - 1.0)
    max_sum_error = float(row_sum_error.max(initial=0.0))
    if max_sum_error > simplex_tolerance:
        worst_row = int(np.argmax(row_sum_error))
        die(
            f"{model_name} probabilities fail the simplex audit: dtype={native_dtype}, "
            f"max_abs_row_sum_error={max_sum_error:.9g}, tolerance={simplex_tolerance:.9g}, "
            f"worst_row={worst_row}, row_sum={output[worst_row].sum():.17g}"
        )
    print(
        f"Probability simplex audit [{model_name}]: dtype={native_dtype}, "
        f"max_abs_sum_error={max_sum_error:.3g}, tolerance={simplex_tolerance:.3g}: PASS"
    )
    return output


def predict_without_labels(
    identity: pd.DataFrame,
    raw_matrix: np.ndarray,
    pre: dict[str, Any],
    models: dict[str, Any],
) -> pd.DataFrame:
    transformed = frozen_transform(raw_matrix, pre)
    frames = []
    for model_name in MODEL_ORDER:
        probability = aligned_probabilities(
            models[model_name]["model"], transformed, model_name
        )
        frame = identity.rename(columns={PARTICIPANT_COLUMN: "participant_id"}).copy()
        frame.insert(0, "model", model_name)
        frame.insert(1, "model_label", MODEL_LABELS[model_name])
        frame["proba_nonIBD"] = probability[:, 0]
        frame["proba_IBD"] = probability[:, 1]
        frame["pred"] = np.argmax(probability, axis=1)
        frame["pred_label"] = frame["pred"].map({0: "nonIBD", 1: "IBD"})
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def metric_bundle(y_true: np.ndarray, probability_ibd: np.ndarray) -> tuple[dict[str, Any], np.ndarray]:
    y_true = np.asarray(y_true, dtype=int)
    probability_ibd = np.asarray(probability_ibd, dtype=np.float64)
    prediction = (probability_ibd >= 0.5).astype(int)
    probability = np.column_stack([1.0 - probability_ibd, probability_ibd])
    cm = confusion_matrix(y_true, prediction, labels=[0, 1])
    metrics = {
        "n": int(len(y_true)),
        "class_counts": {
            "nonIBD": int(np.sum(y_true == 0)),
            "IBD": int(np.sum(y_true == 1)),
        },
        "threshold": 0.5,
        "accuracy": float(accuracy_score(y_true, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "macro_f1": float(f1_score(y_true, prediction, average="macro", zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probability_ibd)),
        "average_precision_ibd": float(average_precision_score(y_true, probability_ibd)),
        "brier": float(np.mean((probability_ibd - y_true) ** 2)),
        "log_loss": float(
            log_loss(y_true, np.clip(probability, EPS, 1.0 - EPS), labels=[0, 1])
        ),
        "recall_nonIBD": float(
            recall_score(y_true, prediction, pos_label=0, zero_division=0)
        ),
        "recall_IBD": float(
            recall_score(y_true, prediction, pos_label=1, zero_division=0)
        ),
        "confusion_matrix": cm.tolist(),
    }
    return metrics, cm


def attach_labels_and_evaluate(
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], pd.DataFrame]:
    labels = pd.read_csv(
        EXTERNAL_METADATA,
        usecols=["sample_id", "external_label_ibd_binary"],
    )
    labels["sample_id"] = labels["sample_id"].map(normalise_sample_id)
    labels["y_true"] = labels["external_label_ibd_binary"].map(binary_label)
    labelled = predictions.merge(
        labels[["sample_id", "external_label_ibd_binary", "y_true"]],
        on="sample_id",
        how="left",
        validate="many_to_one",
    )
    if labelled["y_true"].isna().any():
        die("External labels are missing after frozen probabilities were computed")
    labelled["y_true"] = labelled["y_true"].astype(int)
    labelled["true_label"] = labelled["y_true"].map({0: "nonIBD", 1: "IBD"})

    conflicts = labelled.groupby("participant_id")["y_true"].nunique().gt(1)
    if conflicts.any():
        die(f"Participants have conflicting labels: {conflicts[conflicts].index.tolist()}")

    participant = labelled.groupby(["model", "model_label", "participant_id"], as_index=False).agg(
        sample_count=("sample_id", "size"),
        y_true=("y_true", "first"),
        true_label=("true_label", "first"),
        proba_nonIBD=("proba_nonIBD", "mean"),
        proba_IBD=("proba_IBD", "mean"),
    )
    participant["pred"] = (participant["proba_IBD"] >= 0.5).astype(int)
    participant["pred_label"] = participant["pred"].map({0: "nonIBD", 1: "IBD"})

    all_metrics: dict[str, Any] = {}
    confusion_rows: list[dict[str, Any]] = []
    for model_name in MODEL_ORDER:
        sample_sub = labelled[labelled["model"].eq(model_name)]
        participant_sub = participant[participant["model"].eq(model_name)]
        sample_metrics, sample_cm = metric_bundle(
            sample_sub["y_true"].to_numpy(dtype=int),
            sample_sub["proba_IBD"].to_numpy(dtype=float),
        )
        participant_metrics, participant_cm = metric_bundle(
            participant_sub["y_true"].to_numpy(dtype=int),
            participant_sub["proba_IBD"].to_numpy(dtype=float),
        )
        all_metrics[model_name] = {
            "model_label": MODEL_LABELS[model_name],
            "sample": sample_metrics,
            "participant_probability_averaged": participant_metrics,
        }
        for level, matrix in (("sample", sample_cm), ("participant", participant_cm)):
            confusion_rows.append(
                {
                    "model": model_name,
                    "model_label": MODEL_LABELS[model_name],
                    "level": level,
                    "true_nonIBD_pred_nonIBD": int(matrix[0, 0]),
                    "true_nonIBD_pred_IBD": int(matrix[0, 1]),
                    "true_IBD_pred_nonIBD": int(matrix[1, 0]),
                    "true_IBD_pred_IBD": int(matrix[1, 1]),
                }
            )
    return labelled, participant, all_metrics, pd.DataFrame(confusion_rows)


def latex_benchmark_table(metrics: dict[str, Any]) -> str:
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{External IBD versus non-IBD performance of the frozen IBDMDB species-abundance benchmark models in the Serrano--G\'omez cohort. Participant-level probabilities are arithmetic means across samples from the same participant. No external refitting, scaling, calibration, or threshold selection was performed.}",
        r"\label{tab:external-species-benchmarks-ibd}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llrrrrrrrr}",
        r"\toprule",
        r"Model & Level & $n$ & Accuracy & Bal. acc. & Macro F1 & ROC--AUC & AP$_{\mathrm{IBD}}$ & Recall$_{\mathrm{nonIBD}}$ & Recall$_{\mathrm{IBD}}$ \\",
        r"\midrule",
    ]
    for model_index, model_name in enumerate(MODEL_ORDER):
        for level_key, level_label in (
            ("sample", "Sample"),
            ("participant_probability_averaged", "Participant"),
        ):
            values = metrics[model_name][level_key]
            model_cell = MODEL_LABELS[model_name] if level_key == "sample" else ""
            lines.append(
                f"{model_cell} & {level_label} & {values['n']} & "
                f"{values['accuracy']:.3f} & {values['balanced_accuracy']:.3f} & "
                f"{values['macro_f1']:.3f} & {values['roc_auc']:.3f} & "
                f"{values['average_precision_ibd']:.3f} & {values['recall_nonIBD']:.3f} & "
                f"{values['recall_IBD']:.3f} \\\\"
            )
        if model_index < len(MODEL_ORDER) - 1:
            lines.append(r"\addlinespace")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def print_metrics(metrics: dict[str, Any]) -> None:
    for model_name in MODEL_ORDER:
        print("\n" + MODEL_LABELS[model_name])
        print("-" * len(MODEL_LABELS[model_name]))
        print(json.dumps(metrics[model_name], indent=2, default=json_value))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-root", type=Path, default=BENCHMARK_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--external-abundance-multiplier", type=float, default=None)
    parser.add_argument(
        "--min-mapped-fraction",
        type=float,
        default=0.0,
        help=(
            "Prespecified minimum fraction of each profile's species abundance that must map "
            "to the frozen 508-species axis (default: 0, while all-zero vectors still fail)."
        ),
    )
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    benchmark_root = args.benchmark_root.expanduser().resolve()
    if not 0.0 <= args.min_mapped_fraction <= 1.0:
        die("--min-mapped-fraction must be between 0 and 1")
    paths, benchmark_layout = required_paths(benchmark_root)
    print("=" * 104)
    print("FROZEN IBDMDB SPECIES BENCHMARKS — SERRANO-GOMEZ EXTERNAL IBD-vs-nonIBD")
    print("=" * 104)
    print("Benchmark root:", benchmark_root)
    print("Benchmark artifact layout:", benchmark_layout)
    print("External metadata:", paths["external_metadata"])

    models, pre = load_frozen_models(paths, benchmark_layout)
    print("Frozen models:", ", ".join(MODEL_ORDER))
    print("Source species coordinates:", len(pre["input_feature_names"]))
    print("Variance-selected coordinates:", int(pre["support_mask"].sum()))
    print("Frozen pseudocount:", pre["pseudocount"])

    structural_module = load_structural_mapper(paths["structural_source"])
    axis_audit = verify_structural_training_axis(
        structural_module, pre["input_feature_names"]
    )
    print("Structural mapper source:", paths["structural_source"])
    print("Structural/benchmark 508-species frozen-order identity: PASS")

    multiplier, scale_audit = source_scale_multiplier(
        paths["source_design_summary"], args.external_abundance_multiplier
    )
    print("Abundance-scale audit:")
    print(json.dumps(scale_audit, indent=2))

    identity, raw_matrix, mapping_audit, profile_manifest, mapping_details = (
        build_external_matrix_without_labels(
            pre,
            multiplier,
            structural_module,
            args.min_mapped_fraction,
        )
    )
    print(
        "External mapped-abundance fraction: "
        f"min={mapping_audit['mapped_fraction_of_species_abundance'].min():.3%}, "
        f"median={mapping_audit['mapped_fraction_of_species_abundance'].median():.3%}, "
        f"max={mapping_audit['mapped_fraction_of_species_abundance'].max():.3%}"
    )
    predictions = predict_without_labels(identity, raw_matrix, pre, models)
    print(f"Frozen probabilities computed without external labels: {len(identity)} samples x 3 models")

    labelled, participants, metrics, confusion = attach_labels_and_evaluate(predictions)
    print_metrics(metrics)

    if args.verify_only:
        print("\nVERIFY-ONLY COMPLETE: no result files written.")
        return

    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        die(f"Output directory is non-empty; inspect it before using --overwrite: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    atomic_csv(labelled, output_dir / "sample_predictions_all_models.csv")
    atomic_csv(participants, output_dir / "participant_predictions_probability_averaged_all_models.csv")
    atomic_csv(confusion, output_dir / "confusion_matrices_all_models.csv")
    atomic_csv(mapping_audit, output_dir / "external_species_vectorization_audit.csv")
    atomic_csv(profile_manifest, output_dir / "metaphlan_profile_manifest.csv")
    atomic_csv(mapping_details, output_dir / "external_species_mapping_details.csv")
    atomic_json(output_dir / "external_metrics_all_models.json", metrics)
    atomic_text(output_dir / "latex_external_species_benchmark_performance.tex", latex_benchmark_table(metrics))

    script_path = Path(__file__).resolve()
    provenance = {
        "script_version": SCRIPT_VERSION,
        "script_path": str(script_path),
        "script_sha256": sha256_file(script_path),
        "completed_utc": utc_now(),
        "analysis": (
            "Serrano-Gomez structural-frozen-IBDMDB v1 external species benchmarks "
            "using last-occurrence duplicate resolution"
        ),
        "benchmark_artifact_layout": benchmark_layout,
        "task": TASK_FOLDER,
        "class_order": list(CLASS_NAMES),
        "threshold": 0.5,
        "external_samples": int(identity["sample_id"].nunique()),
        "external_participants": int(identity[PARTICIPANT_COLUMN].nunique()),
        "participant_identifier": PARTICIPANT_COLUMN,
        "participant_prediction_rule": "arithmetic mean of frozen sample probabilities, then threshold 0.5",
        "source_abundance_scale": scale_audit,
        "minimum_mapped_fraction_required": args.min_mapped_fraction,
        "structural_mapper": {
            "source_path": str(paths["structural_source"]),
            "source_sha256": sha256_file(paths["structural_source"]),
            "functions_reused": [
                "parse_metaphlan_species",
                "profile_filename_score",
                "build_species_key_map",
                "project_external_species",
            ],
            "frozen_axis_audit": axis_audit,
        },
        "frozen_preprocessor": {
            "input_species_coordinates": len(pre["input_feature_names"]),
            "variance_selected_coordinates": int(pre["support_mask"].sum()),
            "pseudocount": pre["pseudocount"],
            "variance_threshold": pre["variance_threshold"],
        },
        "models": {
            model_name: {
                "label": MODEL_LABELS[model_name],
                "artifact_path": str(paths[f"model_{model_name}"]),
                "artifact_sha256": models[model_name]["model_sha256"],
                "marker_path": str(paths[f"marker_{model_name}"]),
                "marker_sha256": models[model_name]["marker_sha256"],
            }
            for model_name in MODEL_ORDER
        },
        "source_design_summary": {
            "path": str(paths["source_design_summary"]),
            "sha256": sha256_file(paths["source_design_summary"]),
        },
        "external_metadata": {
            "path": str(paths["external_metadata"]),
            "sha256": sha256_file(paths["external_metadata"]),
        },
        "leakage_control": (
            "External labels were not read until frozen probabilities from all three models had "
            "been computed. Labels were not used for taxon mapping, vectorization, CLR, variance "
            "selection, scaling, fitting, calibration, model selection, or threshold selection."
        ),
    }
    atomic_json(output_dir / "DEPLOYMENT_PROVENANCE.json", provenance)
    atomic_json(
        output_dir / "RUN_COMPLETE.json",
        {
            "completed_utc": utc_now(),
            "status": "complete",
            "script_version": SCRIPT_VERSION,
            "script_sha256": sha256_file(script_path),
            "task": TASK_FOLDER,
            "models": list(MODEL_ORDER),
            "sample_predictions_per_model": int(identity["sample_id"].nunique()),
            "participant_predictions_per_model": int(identity[PARTICIPANT_COLUMN].nunique()),
        },
    )
    print("\nOUTPUTS WRITTEN:", output_dir)


if __name__ == "__main__":
    main()
