#!/usr/bin/env python3
"""Evaluate frozen IBDMDB species benchmarks on external-v2 cohort membership.

This script applies the unchanged full-source IBDMDB species-abundance models
trained after last-occurrence duplicate resolution (L1 logistic regression,
random forest and XGBoost) to:

1. all 184 Serrano--Gomez samples curated for expanded external v2; and
2. the locked 90-sample intersection with strict structural external v1.

The models still receive only their original frozen 508-species coordinates.
No external refitting, feature selection, scaling, calibration or threshold
selection is performed.  Probabilities are computed before labels are read.

If the completed v1 species-benchmark outputs are available, paired-sample
probabilities are required to reproduce exactly (within numerical tolerance).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn


SCRIPT_VERSION = "1.1.0"
EXPECTED_ALL_SAMPLES = 184
EXPECTED_PAIRED_SAMPLES = 90
EXPECTED_SPECIES = 508
THRESHOLD = 0.5
CLASS_NAMES = ("nonIBD", "IBD")
MODEL_ORDER = ("logistic_l1", "random_forest", "xgboost")
MODEL_LABELS = {
    "logistic_l1": "Species L1 logistic",
    "random_forest": "Species random forest",
    "xgboost": "Species XGBoost",
}
INVALID_IDENTIFIERS = {
    "", "nan", "none", "null", "missing", "unknown", "unmapped",
    "not available", "not_applicable", "not applicable", "healthy_control",
    "healthy control",
}


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


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        die(f"Missing {label}: {path}")
    if path.stat().st_size <= 0:
        die(f"Empty {label}: {path}")


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def import_v1_deployment(path: Path) -> ModuleType:
    """Import the audited v1 implementation and validate its required surface."""
    require_file(path, "v1 species-benchmark deployment source")
    name = "audited_species_benchmark_external_v1"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        die(f"Could not construct import specification for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    required = (
        "required_paths",
        "load_frozen_models",
        "load_structural_mapper",
        "verify_structural_training_axis",
        "source_scale_multiplier",
        "frozen_transform",
        "aligned_probabilities",
        "metric_bundle",
        "binary_label",
    )
    missing = [name for name in required if not callable(getattr(module, name, None))]
    if missing:
        die(f"v1 species-benchmark deployment lacks required functions: {missing}")
    if tuple(getattr(module, "MODEL_ORDER", ())) != MODEL_ORDER:
        die("v1 deployment has an unexpected benchmark model order")
    if dict(getattr(module, "MODEL_LABELS", {})) != MODEL_LABELS:
        die("v1 deployment has unexpected benchmark labels")
    return module


def clean_identifier(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if text.lower() in INVALID_IDENTIFIERS:
        return None
    return text


def resolve_participant_ids(metadata: pd.DataFrame) -> tuple[pd.Series, list[str]]:
    # host_subject_id is the prespecified grouping coordinate used in strict v1.
    candidates = [
        "host_subject_id",
        "participant_id",
        "patient_id",
        "subject_id",
        "PRJEB42155_ge50_clean_metadata_full__host_subject_id",
        "PRJEB42155_ge50_clean_metadata_full__patient_id",
    ]
    available = [column for column in candidates if column in metadata.columns]
    if not available:
        die("No participant identifier exists in v2 metadata")
    values: list[str] = []
    for row in metadata.itertuples(index=False):
        row_dict = row._asdict()
        participant = None
        for column in available:
            participant = clean_identifier(row_dict[column])
            if participant is not None:
                break
        sample_id = str(row_dict["sample_id"])
        values.append(participant if participant is not None else f"sample::{sample_id}")
    return pd.Series(values, index=metadata.index, dtype="object"), available


def load_locked_cohort(
    profile_manifest_path: Path,
    paired_ids_path: Path,
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    """Load label-free identities and exact profile paths for all 184 samples."""
    require_file(profile_manifest_path, "v2 profile manifest")
    require_file(paired_ids_path, "locked paired-v1 sample IDs")
    manifest = pd.read_csv(profile_manifest_path, dtype=str)
    required_columns = {"sample_id", "profile_path", "profile_sha256"}
    if not required_columns.issubset(manifest.columns):
        die(f"v2 profile manifest lacks columns: {sorted(required_columns - set(manifest.columns))}")
    manifest = manifest[["sample_id", "profile_path", "profile_sha256"]].copy()
    manifest["sample_id"] = manifest["sample_id"].astype(str)
    if len(manifest) != EXPECTED_ALL_SAMPLES or manifest["sample_id"].duplicated().any():
        die("v2 profile manifest must contain exactly 184 unique samples")

    for row in manifest.itertuples(index=False):
        path = Path(str(row.profile_path)).expanduser().resolve()
        require_file(path, f"MetaPhlAn profile for {row.sample_id}")
        observed = sha256_file(path)
        if observed != str(row.profile_sha256):
            die(
                f"Profile checksum changed for {row.sample_id}: "
                f"manifest={row.profile_sha256}, observed={observed}"
            )

    paired = pd.read_csv(paired_ids_path, dtype=str)
    if "sample_id" not in paired.columns:
        die("paired_v1_sample_ids.csv lacks sample_id")
    paired_ids = paired["sample_id"].astype(str).tolist()
    if len(paired_ids) != EXPECTED_PAIRED_SAMPLES or len(set(paired_ids)) != len(paired_ids):
        die("Locked v1 intersection must contain exactly 90 unique sample IDs")
    if not set(paired_ids).issubset(set(manifest["sample_id"])):
        die("Locked paired IDs are not a subset of the 184-sample profile manifest")
    audit = {
        "profile_manifest_path": str(profile_manifest_path),
        "profile_manifest_sha256": sha256_file(profile_manifest_path),
        "paired_ids_path": str(paired_ids_path),
        "paired_ids_sha256": sha256_file(paired_ids_path),
    }
    return manifest, paired_ids, audit


def build_frozen_species_matrix_without_labels(
    manifest: pd.DataFrame,
    preprocessor: dict[str, Any],
    multiplier: float,
    structural_mapper: ModuleType,
    min_mapped_fraction: float,
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    """Parse all profiles and project to the exact frozen 508-species axis."""
    feature_names = list(preprocessor["input_feature_names"])
    if len(feature_names) != EXPECTED_SPECIES or len(set(feature_names)) != EXPECTED_SPECIES:
        die(f"Expected 508 unique frozen species coordinates, found {len(feature_names)}")
    unique_key_map, ambiguous_keys = structural_mapper.build_species_key_map(feature_names)
    matrix = np.zeros((len(manifest), len(feature_names)), dtype=np.float64)
    audit_rows: list[dict[str, Any]] = []
    mapping_frames: list[pd.DataFrame] = []

    print(f"Projecting {len(manifest)} profiles to the frozen 508-species axis without labels...")
    for row_index, row in enumerate(manifest.itertuples(index=False), 1):
        sample_id = str(row.sample_id)
        profile_path = Path(str(row.profile_path)).expanduser().resolve()
        external_species, profile_meta = structural_mapper.parse_metaphlan_species(profile_path)
        projected, mapping_meta, mapping_frame = structural_mapper.project_external_species(
            external_species,
            feature_names,
            unique_key_map,
            ambiguous_keys,
        )
        projected = projected.reindex(feature_names)
        if projected.isna().any() or list(projected.index) != feature_names:
            die(f"{sample_id} projection did not preserve frozen species order")
        mapped_fraction = float(mapping_meta["mapped_fraction_of_species_abundance"])
        if mapped_fraction < min_mapped_fraction:
            die(
                f"{sample_id} mapped {mapped_fraction:.3%}, below the prespecified "
                f"minimum {min_mapped_fraction:.3%}"
            )
        matrix[row_index - 1] = projected.to_numpy(dtype=np.float64) * multiplier
        row_sum = float(matrix[row_index - 1].sum())
        if row_sum <= 0:
            die(f"{sample_id} has an all-zero frozen benchmark vector")
        audit_rows.append(
            {
                "sample_id": sample_id,
                "profile_path": str(profile_path),
                "profile_sha256": str(row.profile_sha256),
                **dict(profile_meta),
                **dict(mapping_meta),
                "frozen_raw_vector_sum_after_scale_match": row_sum,
            }
        )
        mapping_frame = mapping_frame.copy()
        mapping_frame.insert(0, "sample_id", sample_id)
        mapping_frames.append(mapping_frame)
        if row_index == 1 or row_index % 25 == 0 or row_index == len(manifest):
            print(f"  profiles {row_index}/{len(manifest)}")

    if not np.isfinite(matrix).all() or (matrix < 0).any():
        die("Constructed external benchmark matrix is nonfinite or negative")
    return matrix, pd.DataFrame(audit_rows), pd.concat(mapping_frames, ignore_index=True)


def predict_without_labels(
    sample_ids: list[str],
    raw_matrix: np.ndarray,
    preprocessor: dict[str, Any],
    models: dict[str, Any],
    v1_module: ModuleType,
) -> pd.DataFrame:
    transformed = v1_module.frozen_transform(raw_matrix, preprocessor)
    frames: list[pd.DataFrame] = []
    for model_name in MODEL_ORDER:
        probability = v1_module.aligned_probabilities(
            models[model_name]["model"], transformed, model_name
        )
        frame = pd.DataFrame(
            {
                "model": model_name,
                "model_label": MODEL_LABELS[model_name],
                "sample_id": sample_ids,
                "proba_nonIBD": probability[:, 0],
                "proba_IBD": probability[:, 1],
            }
        )
        frame["pred"] = (frame["proba_IBD"] >= THRESHOLD).astype(int)
        frame["pred_label"] = frame["pred"].map({0: "nonIBD", 1: "IBD"})
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def attach_metadata_after_prediction(
    predictions: pd.DataFrame,
    metadata_path: Path,
    v1_module: ModuleType,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read external labels only after all 552 probabilities already exist."""
    require_file(metadata_path, "matched v2 metadata")
    metadata = pd.read_csv(metadata_path, sep="\t", low_memory=False)
    if "sample_id_v2" not in metadata.columns:
        die("matched v2 metadata lacks sample_id_v2")
    # Assign rather than rename: the source metadata may independently contain
    # a biological sample_id column that is not the selected run accession.
    metadata["sample_id"] = metadata["sample_id_v2"].astype(str)
    if len(metadata) != EXPECTED_ALL_SAMPLES or metadata["sample_id"].duplicated().any():
        die("matched v2 metadata must contain exactly 184 unique samples")
    label_candidates = [
        "external_label_ibd_binary",
        "PRJEB42155_ge50_clean_metadata_full__external_label_ibd_binary",
    ]
    label_column = next((column for column in label_candidates if column in metadata.columns), None)
    if label_column is None:
        die("Could not locate external IBD binary label in v2 metadata")
    metadata["participant_id"], participant_columns = resolve_participant_ids(metadata)
    metadata["y_true"] = metadata[label_column].map(v1_module.binary_label).astype(int)
    metadata["true_label"] = metadata["y_true"].map({0: "nonIBD", 1: "IBD"})
    labelled = predictions.merge(
        metadata[["sample_id", "participant_id", label_column, "y_true", "true_label"]],
        on="sample_id",
        how="left",
        validate="many_to_one",
    )
    if labelled[["participant_id", "y_true"]].isna().any().any():
        die("Missing metadata after attaching labels to frozen probabilities")
    return labelled, {
        "metadata_path": str(metadata_path),
        "metadata_sha256": sha256_file(metadata_path),
        "sample_id_column": "sample_id_v2",
        "label_column": label_column,
        "participant_columns_considered_in_order": participant_columns,
    }


def evaluate(
    labelled: pd.DataFrame,
    v1_module: ModuleType,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    conflicts = labelled.groupby("participant_id")["y_true"].nunique()
    if (conflicts > 1).any():
        die(
            "Participants have conflicting labels: "
            + ", ".join(conflicts[conflicts > 1].index.astype(str).tolist()[:10])
        )
    participant = labelled.groupby(
        ["model", "model_label", "participant_id"], as_index=False
    ).agg(
        sample_count=("sample_id", "size"),
        y_true=("y_true", "first"),
        true_label=("true_label", "first"),
        proba_nonIBD=("proba_nonIBD", "mean"),
        proba_IBD=("proba_IBD", "mean"),
    )
    participant["pred"] = (participant["proba_IBD"] >= THRESHOLD).astype(int)
    participant["pred_label"] = participant["pred"].map({0: "nonIBD", 1: "IBD"})

    metrics: dict[str, Any] = {}
    confusion_rows: list[dict[str, Any]] = []
    for model_name in MODEL_ORDER:
        sample_sub = labelled[labelled["model"].eq(model_name)]
        participant_sub = participant[participant["model"].eq(model_name)]
        sample_metrics, sample_cm = v1_module.metric_bundle(
            sample_sub["y_true"].to_numpy(int), sample_sub["proba_IBD"].to_numpy(float)
        )
        participant_metrics, participant_cm = v1_module.metric_bundle(
            participant_sub["y_true"].to_numpy(int),
            participant_sub["proba_IBD"].to_numpy(float),
        )
        metrics[model_name] = {
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
    return participant, metrics, pd.DataFrame(confusion_rows)


def optional_v1_probability_crosscheck(
    v1_prediction_path: Path,
    paired_predictions: pd.DataFrame,
) -> dict[str, Any]:
    if not v1_prediction_path.is_file():
        return {
            "status": "not_available",
            "path": str(v1_prediction_path),
            "note": "Run the v1 species-benchmark deployment normally to enable this cross-check.",
        }
    previous = pd.read_csv(v1_prediction_path, low_memory=False)
    required = {"model", "sample_id", "proba_IBD"}
    if not required.issubset(previous.columns):
        die(f"v1 benchmark predictions lack columns: {sorted(required - set(previous.columns))}")
    previous = previous[list(required)].copy()
    previous["sample_id"] = previous["sample_id"].astype(str)
    comparison = paired_predictions[["model", "sample_id", "proba_IBD"]].merge(
        previous.rename(columns={"proba_IBD": "proba_IBD_v1"}),
        on=["model", "sample_id"],
        how="inner",
        validate="one_to_one",
    )
    expected_rows = EXPECTED_PAIRED_SAMPLES * len(MODEL_ORDER)
    if len(comparison) != expected_rows:
        die(f"Expected {expected_rows} paired v1 cross-check rows, found {len(comparison)}")
    comparison["absolute_difference"] = (
        comparison["proba_IBD"] - comparison["proba_IBD_v1"]
    ).abs()
    by_model = comparison.groupby("model")["absolute_difference"].max().to_dict()
    maximum = float(comparison["absolute_difference"].max())
    tolerance = 1e-12
    if maximum > tolerance:
        die(
            "Paired-90 benchmark probabilities do not reproduce strict v1: "
            f"max_abs_difference={maximum}"
        )
    return {
        "status": "pass",
        "path": str(v1_prediction_path),
        "sha256": sha256_file(v1_prediction_path),
        "rows_compared": expected_rows,
        "tolerance": tolerance,
        "max_abs_difference_overall": maximum,
        "max_abs_difference_by_model": {key: float(value) for key, value in by_model.items()},
    }


def latex_table(all_metrics: dict[str, Any], paired_metrics: dict[str, Any]) -> str:
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\caption{Transfer of frozen IBDMDB species-abundance benchmark models trained after last-occurrence duplicate resolution to the complete Serrano--G\'omez external cohort and the subset paired with structural v1.}",
        r"\label{tab:external_species_benchmarks_all184_paired90}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lllrrrrrrrr}",
        r"\toprule",
        r"Model & Cohort & Unit & $n$ & Acc. & Bal. acc. & Macro F1 & ROC--AUC & AP$_{\mathrm{IBD}}$ & Rec. nonIBD & Rec. IBD \\",
        r"\midrule",
    ]
    cohorts = (("All 184", all_metrics), ("Paired 90", paired_metrics))
    for model_index, model_name in enumerate(MODEL_ORDER):
        for cohort_label, cohort_metrics in cohorts:
            for level_key, unit in (
                ("sample", "Sample"),
                ("participant_probability_averaged", "Participant"),
            ):
                metric = cohort_metrics[model_name][level_key]
                model_cell = MODEL_LABELS[model_name] if cohort_label == "All 184" and unit == "Sample" else ""
                lines.append(
                    f"{model_cell} & {cohort_label} & {unit} & {metric['n']} & "
                    f"{metric['accuracy']:.3f} & {metric['balanced_accuracy']:.3f} & "
                    f"{metric['macro_f1']:.3f} & {metric['roc_auc']:.3f} & "
                    f"{metric['average_precision_ibd']:.3f} & {metric['recall_nonIBD']:.3f} & "
                    f"{metric['recall_IBD']:.3f} \\\\"
                )
        if model_index < len(MODEL_ORDER) - 1:
            lines.append(r"\addlinespace")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            r"\begin{minipage}{0.98\linewidth}",
            r"\footnotesize",
            r"All models, their 508-species input order, CLR pseudocount, variance mask, source-fitted scaler and 0.5 decision threshold were frozen. No external refitting, rescaling, calibration or threshold selection was performed. Participant probabilities are arithmetic means across samples. The paired cohort contains exactly the 90 samples used in structural external v1. AP denotes IBD average precision.",
            r"\end{minipage}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def print_metrics(title: str, metrics: dict[str, Any]) -> None:
    print("\n" + title)
    print("=" * len(title))
    for model_name in MODEL_ORDER:
        print("\n" + MODEL_LABELS[model_name])
        print(json.dumps(metrics[model_name], indent=2))


def write_outputs(
    output_dir: Path,
    label_blind: pd.DataFrame,
    all_labelled: pd.DataFrame,
    all_participants: pd.DataFrame,
    all_metrics: dict[str, Any],
    all_confusion: pd.DataFrame,
    paired_labelled: pd.DataFrame,
    paired_participants: pd.DataFrame,
    paired_metrics: dict[str, Any],
    paired_confusion: pd.DataFrame,
    mapping_audit: pd.DataFrame,
    mapping_details: pd.DataFrame,
    provenance: dict[str, Any],
    overwrite: bool,
) -> None:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if output_dir.exists():
        if not overwrite:
            die(f"Output directory exists; use --overwrite to move it to a backup: {output_dir}")
        backup = output_dir.with_name(
            output_dir.name + ".backup_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        if backup.exists():
            die(f"Backup path already exists: {backup}")
        output_dir.rename(backup)
        print(f"Existing output moved to recoverable backup: {backup}")

    temporary = Path(tempfile.mkdtemp(prefix=output_dir.name + ".tmp_", dir=output_dir.parent))
    try:
        label_blind.to_csv(temporary / "label_blind_probabilities_all_184.csv", index=False)
        mapping_audit.to_csv(temporary / "external_species_vectorization_audit_all_184.csv", index=False)
        mapping_details.to_csv(
            temporary / "external_species_mapping_details_all_184.csv.gz",
            index=False,
            compression="gzip",
        )
        all_dir = temporary / "all_184"
        paired_dir = temporary / "paired_90"
        all_dir.mkdir()
        paired_dir.mkdir()
        all_labelled.to_csv(all_dir / "sample_predictions_all_models.csv", index=False)
        all_participants.to_csv(
            all_dir / "participant_predictions_probability_averaged_all_models.csv", index=False
        )
        all_confusion.to_csv(all_dir / "confusion_matrices_all_models.csv", index=False)
        atomic_json(all_dir / "external_metrics_all_models.json", all_metrics)
        paired_labelled.to_csv(paired_dir / "sample_predictions_all_models.csv", index=False)
        paired_participants.to_csv(
            paired_dir / "participant_predictions_probability_averaged_all_models.csv", index=False
        )
        paired_confusion.to_csv(paired_dir / "confusion_matrices_all_models.csv", index=False)
        atomic_json(paired_dir / "external_metrics_all_models.json", paired_metrics)
        (temporary / "latex_external_species_benchmarks_all184_paired90.tex").write_text(
            latex_table(all_metrics, paired_metrics), encoding="utf-8"
        )
        atomic_json(temporary / "DEPLOYMENT_PROVENANCE.json", provenance)
        atomic_json(
            temporary / "RUN_COMPLETE.json",
            {
                "status": "complete",
                "completed_utc": utc_now(),
                "script_version": SCRIPT_VERSION,
                "models": list(MODEL_ORDER),
                "all_samples_per_model": EXPECTED_ALL_SAMPLES,
                "paired_samples_per_model": EXPECTED_PAIRED_SAMPLES,
            },
        )
        os.replace(temporary, output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def parse_args() -> argparse.Namespace:
    home = Path.home()
    external_root = (
        home / "Real_Data/external_validation/serrano_gomez_ibd/ge50_external_validation"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--external-root", type=Path, default=external_root)
    parser.add_argument(
        "--benchmark-root",
        type=Path,
        default=home / "Real_Data/Species_Benchmarks_RepeatedCV_IBD_LastOccurrence",
    )
    parser.add_argument(
        "--v1-deployment-source",
        type=Path,
        default=Path(__file__).resolve().with_name(
            "apply_frozen_species_benchmarks_external_ibd_last_occurrence_v1.py"
        ),
    )
    parser.add_argument(
        "--v1-prediction-path",
        type=Path,
        default=(
            external_root
            / "structural_frozen_ibdmdb"
            / "external_species_benchmarks_ibd_vs_nonibd_last_occurrence_v1"
            / "sample_predictions_all_models.csv"
        ),
        help="Corrected strict-v1 predictions used for the paired-90 identity check.",
    )
    parser.add_argument("--external-abundance-multiplier", type=float, default=None)
    parser.add_argument("--min-mapped-fraction", type=float, default=0.0)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    external_root = args.external_root.expanduser().resolve()
    benchmark_root = args.benchmark_root.expanduser().resolve()
    expanded = external_root / "expanded_community_structural_replication_v2"
    curated = expanded / "01_curated"
    profile_manifest_path = curated / "profile_manifest.csv"
    metadata_path = curated / "matched_external_metadata_184.tsv.gz"
    paired_ids_path = curated / "paired_v1_sample_ids.csv"
    v1_prediction_path = args.v1_prediction_path.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else expanded
        / "08_exploratory_prediction"
        / "frozen_species_benchmarks_last_occurrence_all184_and_paired90"
    )

    if not 0.0 <= args.min_mapped_fraction <= 1.0:
        die("--min-mapped-fraction must lie between 0 and 1")
    print("=" * 112)
    print("FROZEN IBDMDB SPECIES BENCHMARKS — EXTERNAL ALL-184 AND PAIRED-90")
    print("=" * 112)
    print(f"Benchmark root: {benchmark_root}")
    print(f"External-v2 curated cohort: {curated}")
    print("Frozen models: " + ", ".join(MODEL_ORDER))
    print("Label firewall: all 552 probabilities are computed before external labels are read")

    v1_source = args.v1_deployment_source.expanduser().resolve()
    v1_module = import_v1_deployment(v1_source)
    paths, layout = v1_module.required_paths(benchmark_root)
    models, preprocessor = v1_module.load_frozen_models(paths, layout)
    if len(preprocessor["input_feature_names"]) != EXPECTED_SPECIES:
        die("Frozen benchmark preprocessor does not contain 508 species")
    structural_mapper = v1_module.load_structural_mapper(paths["structural_source"])
    axis_audit = v1_module.verify_structural_training_axis(
        structural_mapper, preprocessor["input_feature_names"]
    )
    multiplier, scale_audit = v1_module.source_scale_multiplier(
        paths["source_design_summary"], args.external_abundance_multiplier
    )
    print(f"Artifact layout: {layout}")
    print("Structural/benchmark 508-species frozen-order identity: PASS")
    print("Abundance-scale audit:")
    print(json.dumps(scale_audit, indent=2))

    manifest, paired_ids, cohort_audit = load_locked_cohort(
        profile_manifest_path, paired_ids_path
    )
    raw_matrix, mapping_audit, mapping_details = build_frozen_species_matrix_without_labels(
        manifest,
        preprocessor,
        multiplier,
        structural_mapper,
        args.min_mapped_fraction,
    )
    print(
        "Mapped frozen-species abundance fraction: "
        f"min={mapping_audit['mapped_fraction_of_species_abundance'].min():.3%}, "
        f"median={mapping_audit['mapped_fraction_of_species_abundance'].median():.3%}, "
        f"max={mapping_audit['mapped_fraction_of_species_abundance'].max():.3%}"
    )
    label_blind = predict_without_labels(
        manifest["sample_id"].astype(str).tolist(),
        raw_matrix,
        preprocessor,
        models,
        v1_module,
    )
    if len(label_blind) != EXPECTED_ALL_SAMPLES * len(MODEL_ORDER):
        die("Did not compute exactly 552 label-blind benchmark probabilities")
    print("Frozen probabilities computed without labels: 184 samples x 3 models")

    # Labels and participant identifiers are first read here.
    all_labelled, metadata_audit = attach_metadata_after_prediction(
        label_blind, metadata_path, v1_module
    )
    all_participants, all_metrics, all_confusion = evaluate(all_labelled, v1_module)
    paired_set = set(paired_ids)
    paired_labelled = all_labelled[all_labelled["sample_id"].isin(paired_set)].copy()
    if len(paired_labelled) != EXPECTED_PAIRED_SAMPLES * len(MODEL_ORDER):
        die("Paired subset does not contain exactly 90 samples per model")
    paired_participants, paired_metrics, paired_confusion = evaluate(
        paired_labelled, v1_module
    )
    crosscheck = optional_v1_probability_crosscheck(v1_prediction_path, paired_labelled)
    print(f"Paired strict-v1 probability cross-check: {crosscheck['status'].upper()}")
    print_metrics("ALL 184", all_metrics)
    print_metrics("PAIRED 90", paired_metrics)

    script_path = Path(__file__).resolve()
    provenance = {
        "script_version": SCRIPT_VERSION,
        "script_path": str(script_path),
        "script_sha256": sha256_file(script_path),
        "completed_utc": utc_now(),
        "analysis": (
            "frozen IBDMDB species benchmarks trained after last-occurrence duplicate "
            "resolution on all external-v2 samples and the paired v1 subset"
        ),
        "benchmark_duplicate_policy": "last workbook occurrence per bare External ID",
        "method": {
            "input_representation": "original frozen 508 IBDMDB species-abundance coordinates",
            "model_refitting": False,
            "external_rescaling": False,
            "external_feature_selection": False,
            "external_calibration": False,
            "threshold_selection_on_external_data": False,
            "threshold": THRESHOLD,
            "evaluation_sets": {
                "all_external_v2_samples": EXPECTED_ALL_SAMPLES,
                "paired_with_structural_v1": EXPECTED_PAIRED_SAMPLES,
            },
            "label_firewall": (
                "all 552 sample probabilities were computed before matched external metadata "
                "or labels were read"
            ),
        },
        "benchmark_artifact_layout": layout,
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
        "frozen_preprocessor": {
            "input_species_coordinates": len(preprocessor["input_feature_names"]),
            "variance_selected_coordinates": int(preprocessor["support_mask"].sum()),
            "pseudocount": float(preprocessor["pseudocount"]),
            "variance_threshold": float(preprocessor["variance_threshold"]),
        },
        "source_abundance_scale": scale_audit,
        "minimum_mapped_fraction_required": args.min_mapped_fraction,
        "structural_mapper_axis_audit": axis_audit,
        "v1_deployment_source": {
            "path": str(v1_source),
            "sha256": sha256_file(v1_source),
        },
        "cohort": cohort_audit,
        "metadata": metadata_audit,
        "paired_v1_probability_crosscheck": crosscheck,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "joblib": joblib.__version__,
            "sklearn": sklearn.__version__,
        },
    }

    if args.verify_only:
        print("\nVERIFY-ONLY COMPLETE: no result files written.")
        return
    write_outputs(
        output_dir,
        label_blind,
        all_labelled,
        all_participants,
        all_metrics,
        all_confusion,
        paired_labelled,
        paired_participants,
        paired_metrics,
        paired_confusion,
        mapping_audit,
        mapping_details,
        provenance,
        args.overwrite,
    )
    print(f"\nRUN COMPLETE: {output_dir}")


if __name__ == "__main__":
    main()
