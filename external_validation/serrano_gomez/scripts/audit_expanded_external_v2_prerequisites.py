#!/usr/bin/env python3
"""Read-only preflight for Serrano-Gomez expanded-community structural v2.

The source data are never modified.  The script writes only a dedicated audit
directory containing JSON/CSV inventories.  It does not build graphs, H0, Ricci
curvature, classifiers, or label-informed taxonomic mappings.

Primary questions answered
--------------------------
1. Which of the expected MetaPhlAn profiles are parseable and metadata matched?
2. Is AGORA_reactions_canon.parquet a full AGORA catalyst catalogue or a table
   already pruned to the frozen 508-species IBDMDB axis?
3. What frozen-axis and preliminary full-AGORA abundance coverage is obtainable
   with the exact conservative structural-v1 taxon projector?
4. Which graph, H0 and faithful-active Ricci production sources and frozen
   projection assets exist, and what are their hashes?
5. Are the machine, disk and source artifacts ready for v2 construction?

Disease labels are inspected only in the separate metadata inventory.  Profile
parsing and taxonomic coverage are completed without labels.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import importlib.util
import json
import math
import os
import platform
import re
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable, Iterator, Sequence

import numpy as np
import pandas as pd


SCRIPT_VERSION = "1.0.0"
EXPECTED_PROFILE_COUNT = 184
EXPECTED_V1_PAIRED_COUNT = 90
PROFILE_ID_RE = re.compile(r"(?:ERR|SRR|DRR)\d+", re.IGNORECASE)
PROFILE_EXTENSIONS = {".tsv", ".txt", ".profile", ".csv"}
TABLE_EXTENSIONS = {".csv", ".tsv", ".txt", ".xlsx", ".xls"}
SKIP_DIRECTORY_NAMES = {
    ".git",
    ".cache",
    "__pycache__",
    "node_modules",
    "site-packages",
    "expanded_v2_preflight_audit",
}

BASE_DEFAULT = Path.home() / "Real_Data"
EXTERNAL_ROOT_DEFAULT = (
    BASE_DEFAULT
    / "external_validation"
    / "serrano_gomez_ibd"
    / "ge50_external_validation"
)
STRUCTURAL_SOURCE_RELATIVE = (
    Path("scripts")
    / "serrano_gomez_external_structural_frozen_ibdmdb_v1"
    / "external_structural_validation.py"
)
V1_FEATURE_RELATIVE = (
    Path("structural_frozen_ibdmdb")
    / "ricci_features_frozen_training_order"
)
BENCHMARK_MODEL_RELATIVE = (
    Path("Species_Benchmarks_RepeatedCV_IBD_CompleteCase")
    / "full_source_models"
    / "logistic_l1"
    / "full_source_model.joblib"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def die(message: str) -> None:
    raise RuntimeError(message)


def json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(type(value).__name__)


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=json_value) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def normalise_sample_id(value: object) -> str:
    text = Path(str(value).strip()).name
    match = PROFILE_ID_RE.search(text)
    if match:
        return match.group(0).upper()
    for suffix in (".gz", ".bz2", ".txt", ".tsv", ".csv", ".profile"):
        while text.lower().endswith(suffix):
            text = text[: -len(suffix)]
    return text.strip()


def finite_summary(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return {"n": 0}
    return {
        "n": int(len(array)),
        "min": float(np.min(array)),
        "q05": float(np.quantile(array, 0.05)),
        "median": float(np.median(array)),
        "q95": float(np.quantile(array, 0.95)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
    }


def iter_files(root: Path, suffixes: set[str] | None = None) -> Iterator[Path]:
    if not root.is_dir():
        return
    for directory, subdirs, filenames in os.walk(root):
        subdirs[:] = [
            name
            for name in subdirs
            if name not in SKIP_DIRECTORY_NAMES and not name.startswith(".")
        ]
        parent = Path(directory)
        for filename in filenames:
            path = parent / filename
            if suffixes is None or path.suffix.lower() in suffixes:
                yield path


def find_named_files(root: Path, filename: str) -> list[Path]:
    target = filename.lower()
    return sorted(path for path in iter_files(root) if path.name.lower() == target)


def add_unique_path(paths: list[Path], candidate: Path | None) -> None:
    if candidate is None:
        return
    candidate = candidate.expanduser().resolve()
    if candidate.exists() and candidate not in paths:
        paths.append(candidate)


def load_structural_module(source_path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("expanded_v2_structural_source", source_path)
    if spec is None or spec.loader is None:
        die(f"Could not create import specification for {source_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    required = (
        "parse_metaphlan_species",
        "profile_filename_score",
        "build_species_key_map",
        "project_external_species",
    )
    missing = [name for name in required if not callable(getattr(module, name, None))]
    if missing:
        die(f"Structural source is missing required functions: {missing}")
    return module


def profile_sample_id(path: Path) -> str | None:
    match = PROFILE_ID_RE.search(str(path))
    return match.group(0).upper() if match else None


def discover_profile_roots(external_root: Path, module: ModuleType) -> list[Path]:
    roots: list[Path] = []
    for candidate in (
        external_root / "metaphlan2_v260_profiles",
        external_root / "metaphlan_profiles",
        getattr(module, "HUMANN_DIR", None),
    ):
        if candidate is not None:
            add_unique_path(roots, Path(candidate))
    return roots


def discover_profiles(
    roots: Sequence[Path], module: ModuleType
) -> tuple[dict[str, tuple[Path, pd.Series, dict[str, Any]]], pd.DataFrame]:
    grouped: dict[str, list[tuple[int, Path]]] = defaultdict(list)
    rows: list[dict[str, Any]] = []
    extensions = set(getattr(module, "PROFILE_EXTS", PROFILE_EXTENSIONS))
    for root_rank, root in enumerate(roots):
        for path in iter_files(root, extensions):
            sample_id = profile_sample_id(path)
            if sample_id is None:
                continue
            if int(module.profile_filename_score(path)) < 0:
                continue
            grouped[sample_id].append((root_rank, path))

    selected: dict[str, tuple[Path, pd.Series, dict[str, Any]]] = {}
    for sample_id in sorted(grouped):
        parsed: list[tuple[int, Path, pd.Series, dict[str, Any], int]] = []
        for root_rank, path in sorted(grouped[sample_id], key=lambda item: str(item[1])):
            base_row = {
                "sample_id": sample_id,
                "profile_path": str(path),
                "profile_root_rank": root_rank,
                "filename_score": int(module.profile_filename_score(path)),
            }
            try:
                species, meta = module.parse_metaphlan_species(path)
                parsed.append(
                    (
                        root_rank,
                        path,
                        species,
                        dict(meta),
                        int(module.profile_filename_score(path)),
                    )
                )
                rows.append({**base_row, "parse_status": "valid_candidate", **meta})
            except Exception as exc:
                rows.append(
                    {
                        **base_row,
                        "parse_status": "rejected",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
        if not parsed:
            continue
        parsed.sort(
            key=lambda item: (
                item[0],
                -item[4],
                -int(item[3]["n_species_rows"]),
                -float(item[3]["species_abundance_sum_percent"]),
                str(item[1]),
            )
        )
        root_rank, path, species, meta, score = parsed[0]
        selected[sample_id] = (path, species, meta)
        for row in rows:
            if row["sample_id"] == sample_id and row["profile_path"] == str(path):
                row["parse_status"] = "selected"
                row["profile_sha256"] = sha256_file(path)
                row["selected_root_rank"] = root_rank
                row["selected_filename_score"] = score
                row["n_valid_candidates_for_sample"] = len(parsed)
                break
    return selected, pd.DataFrame(rows)


def load_frozen_species_axis(
    base: Path, module: ModuleType
) -> tuple[list[str], dict[str, Any]]:
    audit: dict[str, Any] = {}
    model_axis: list[str] | None = None
    model_path = base / BENCHMARK_MODEL_RELATIVE
    if model_path.is_file():
        try:
            joblib = importlib.import_module("joblib")
            artifact = joblib.load(model_path)
            model_axis = [
                str(value)
                for value in artifact["preprocessor"]["input_feature_names"]
            ]
            audit["benchmark_model"] = {
                "path": str(model_path),
                "sha256": sha256_file(model_path),
                "n_species": len(model_axis),
                "unique": len(model_axis) == len(set(model_axis)),
            }
        except Exception as exc:
            audit["benchmark_model_error"] = f"{type(exc).__name__}: {exc}"

    structural_axis: list[str] | None = None
    training_path = getattr(module, "TRAIN_SPECIES_PATH", None)
    loader = getattr(module, "load_training_species_axis", None)
    if training_path is not None and callable(loader):
        training_path = Path(training_path).expanduser().resolve()
        try:
            structural_axis, scale_stats = loader(training_path)
            structural_axis = [str(value) for value in structural_axis]
            audit["structural_training_axis"] = {
                "path": str(training_path),
                "sha256": sha256_file(training_path),
                "n_species": len(structural_axis),
                "unique": len(structural_axis) == len(set(structural_axis)),
                "scale_statistics": scale_stats,
            }
        except Exception as exc:
            audit["structural_training_axis_error"] = f"{type(exc).__name__}: {exc}"

    if model_axis is not None and structural_axis is not None:
        audit["axes_exactly_equal_in_order"] = model_axis == structural_axis
        if model_axis != structural_axis:
            die("Benchmark and structural frozen species axes disagree")
    axis = model_axis if model_axis is not None else structural_axis
    if axis is None:
        die("Could not load a frozen IBDMDB species axis")
    return axis, audit


def paths_referenced_in_source(source_path: Path, basename: str) -> list[Path]:
    text = source_path.read_text(encoding="utf-8", errors="replace")
    candidates: list[Path] = []
    pattern = re.compile(r"[\"']([^\"'\n]*" + re.escape(basename) + r")[\"']")
    for raw in pattern.findall(text):
        expanded = Path(os.path.expandvars(raw)).expanduser()
        if not expanded.is_absolute():
            expanded = source_path.parent / expanded
        add_unique_path(candidates, expanded)
    return candidates


def choose_agora_parquet(
    base: Path, structural_source: Path
) -> tuple[Path | None, pd.DataFrame]:
    candidates: list[Path] = []
    for path in paths_referenced_in_source(structural_source, "AGORA_reactions_canon.parquet"):
        add_unique_path(candidates, path)
    for path in find_named_files(base, "AGORA_reactions_canon.parquet"):
        add_unique_path(candidates, path)
    rows = []
    for path in candidates:
        rows.append(
            {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "modified_utc": datetime.fromtimestamp(
                    path.stat().st_mtime, tz=timezone.utc
                ).isoformat(),
                "referenced_by_structural_source": path
                in paths_referenced_in_source(
                    structural_source, "AGORA_reactions_canon.parquet"
                ),
            }
        )
    if not candidates:
        return None, pd.DataFrame(rows)
    referenced = [
        path
        for path in candidates
        if path in paths_referenced_in_source(
            structural_source, "AGORA_reactions_canon.parquet"
        )
    ]
    selected = referenced[0] if len(referenced) == 1 else sorted(candidates)[0]
    return selected, pd.DataFrame(rows)


def parse_catalyst_tokens(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set, np.ndarray)):
        output: list[str] = []
        for item in value:
            output.extend(parse_catalyst_tokens(item))
        return output
    if isinstance(value, dict):
        output = []
        for item in value.keys():
            output.extend(parse_catalyst_tokens(item))
        return output
    try:
        if pd.isna(value):
            return []
    except Exception:
        pass
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "[]", "{}"}:
        return []
    if text[0] in "[({" and text[-1] in "])}":
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(text)
                if parsed is not value and parsed != text:
                    return parse_catalyst_tokens(parsed)
            except Exception:
                continue
    separator = None
    for candidate in ("|", ";"):
        if candidate in text:
            separator = candidate
            break
    if separator is not None:
        return [part.strip() for part in text.split(separator) if part.strip()]
    return [text]


def identify_column(columns: Sequence[str], candidates: Sequence[str]) -> str | None:
    lookup = {str(column).strip().lower(): str(column) for column in columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    return None


def inspect_agora_parquet(
    path: Path,
) -> tuple[dict[str, Any], set[str], pd.DataFrame]:
    parquet_file = None
    fallback_frame = None
    try:
        parquet = importlib.import_module("pyarrow.parquet")
        parquet_file = parquet.ParquetFile(path)
        columns = list(parquet_file.schema_arrow.names)
        engine = "pyarrow-row-groups"
        n_rows = int(parquet_file.metadata.num_rows)
        n_row_groups = int(parquet_file.num_row_groups)
        schema_text = str(parquet_file.schema_arrow)
    except Exception as pyarrow_exc:
        try:
            fallback_frame = pd.read_parquet(path)
            columns = [str(column) for column in fallback_frame.columns]
            engine = "pandas-read-parquet-fallback"
            n_rows = len(fallback_frame)
            n_row_groups = None
            schema_text = json.dumps(
                {str(column): str(dtype) for column, dtype in fallback_frame.dtypes.items()},
                sort_keys=True,
            )
        except Exception as pandas_exc:
            die(
                f"No available parquet reader could inspect {path}. "
                f"pyarrow error: {pyarrow_exc}; pandas fallback error: {pandas_exc}"
            )
    catalyst_column = identify_column(
        columns,
        (
            "catalysts",
            "catalyst",
            "catalyst_species",
            "species",
            "organisms",
            "organism_ids",
            "models",
        ),
    )
    source_column = identify_column(
        columns, ("source", "u", "substrate", "reactant", "from", "metabolite_from")
    )
    target_column = identify_column(
        columns, ("target", "v", "product", "to", "metabolite_to")
    )
    if catalyst_column is None:
        die(f"No catalyst-like column found in {path}; columns={columns}")

    catalysts: set[str] = set()
    catalyst_counter: Counter[str] = Counter()
    example_values: list[str] = []
    nonempty_rows = 0
    distinct_edges: set[tuple[str, str]] | None = (
        set() if source_column is not None and target_column is not None else None
    )
    read_columns = [catalyst_column]
    if source_column is not None:
        read_columns.append(source_column)
    if target_column is not None and target_column not in read_columns:
        read_columns.append(target_column)

    if parquet_file is not None:
        frames: Iterable[pd.DataFrame] = (
            parquet_file.read_row_group(row_group, columns=read_columns).to_pandas()
            for row_group in range(parquet_file.num_row_groups)
        )
    else:
        if fallback_frame is None:
            die("Internal parquet fallback error")
        frames = [fallback_frame[read_columns]]

    for frame in frames:
        for value in frame[catalyst_column]:
            tokens = sorted(set(parse_catalyst_tokens(value)))
            if tokens:
                nonempty_rows += 1
                if len(example_values) < 10:
                    example_values.append(repr(value)[:500])
            catalysts.update(tokens)
            catalyst_counter.update(tokens)
        if distinct_edges is not None:
            for source, target in frame[[source_column, target_column]].itertuples(
                index=False, name=None
            ):
                distinct_edges.add((str(source), str(target)))

    top_catalysts = pd.DataFrame(
        [
            {"catalyst": catalyst, "reaction_rows": count}
            for catalyst, count in catalyst_counter.most_common(100)
        ]
    )
    audit = {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "reader_engine": engine,
        "n_rows": n_rows,
        "n_row_groups": n_row_groups,
        "columns": columns,
        "schema": schema_text,
        "catalyst_column": catalyst_column,
        "source_column": source_column,
        "target_column": target_column,
        "rows_with_nonempty_catalysts": nonempty_rows,
        "unique_catalyst_tokens": len(catalysts),
        "distinct_directed_edges": len(distinct_edges) if distinct_edges is not None else None,
        "catalyst_value_examples": example_values,
        "catalyst_token_examples": sorted(catalysts)[:30],
    }
    return audit, catalysts, top_catalysts


def preliminary_coverage(
    profiles: dict[str, tuple[Path, pd.Series, dict[str, Any]]],
    frozen_axis: Sequence[str],
    agora_catalysts: Sequence[str],
    module: ModuleType,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    frozen_axis = [str(value) for value in frozen_axis]
    agora_axis = sorted(set(str(value) for value in agora_catalysts if str(value).strip()))
    frozen_unique, frozen_ambiguous = module.build_species_key_map(frozen_axis)
    agora_unique, agora_ambiguous = module.build_species_key_map(agora_axis)
    frozen_keys = {module.species_key(value) for value in frozen_axis}
    rows: list[dict[str, Any]] = []

    for sample_id in sorted(profiles):
        profile_path, external_species, profile_meta = profiles[sample_id]
        frozen_vector, frozen_meta, frozen_map = module.project_external_species(
            external_species, frozen_axis, frozen_unique, frozen_ambiguous
        )
        agora_vector, agora_meta, agora_map = module.project_external_species(
            external_species, agora_axis, agora_unique, agora_ambiguous
        )
        mapped_targets = set(
            agora_map.loc[agora_map["status"].eq("mapped"), "training_species"].astype(str)
        )
        novel_targets = {
            target
            for target in mapped_targets
            if module.species_key(target) not in frozen_keys
        }
        rows.append(
            {
                "sample_id": sample_id,
                "profile_path": str(profile_path),
                "detected_species": int(len(external_species)),
                "species_abundance_total_percent": float(external_species.sum()),
                "frozen_mapped_abundance_percent": float(
                    frozen_meta["mapped_abundance_total_percent"]
                ),
                "frozen_coverage_fraction": float(
                    frozen_meta["mapped_fraction_of_species_abundance"]
                ),
                "frozen_species_nonzero": int((frozen_vector > 0).sum()),
                "agora_mapped_abundance_percent_preliminary": float(
                    agora_meta["mapped_abundance_total_percent"]
                ),
                "agora_coverage_fraction_preliminary": float(
                    agora_meta["mapped_fraction_of_species_abundance"]
                ),
                "agora_catalysts_nonzero_preliminary": int((agora_vector > 0).sum()),
                "novel_agora_catalysts_nonzero_preliminary": len(novel_targets),
                "agora_ambiguous_abundance_percent_preliminary": float(
                    agora_meta["ambiguous_abundance_total_percent"]
                ),
                "agora_unmatched_abundance_percent_preliminary": float(
                    agora_meta["unmatched_abundance_total_percent"]
                ),
                "profile_species_rows": profile_meta.get("n_species_rows"),
            }
        )

    frame = pd.DataFrame(rows)
    summary = {
        "mapping_rule": (
            "Exact structural-v1 conservative mapper applied directly to catalyst tokens; "
            "this is diagnostic, not a final curated AGORA crosswalk."
        ),
        "n_samples": len(frame),
        "frozen_coverage_fraction": finite_summary(
            frame.get("frozen_coverage_fraction", pd.Series(dtype=float)).tolist()
        ),
        "agora_coverage_fraction_preliminary": finite_summary(
            frame.get(
                "agora_coverage_fraction_preliminary", pd.Series(dtype=float)
            ).tolist()
        ),
        "detected_species": finite_summary(
            frame.get("detected_species", pd.Series(dtype=float)).tolist()
        ),
        "agora_catalysts_nonzero_preliminary": finite_summary(
            frame.get(
                "agora_catalysts_nonzero_preliminary", pd.Series(dtype=float)
            ).tolist()
        ),
        "novel_agora_catalysts_nonzero_preliminary": finite_summary(
            frame.get(
                "novel_agora_catalysts_nonzero_preliminary", pd.Series(dtype=float)
            ).tolist()
        ),
    }
    return frame, summary


def metadata_candidate_paths(external_root: Path) -> list[Path]:
    pattern = re.compile(
        r"metadata|manifest|phenotype|sample[_ -]?(?:info|sheet|table)|cohort",
        re.IGNORECASE,
    )
    return sorted(
        path
        for path in iter_files(external_root, TABLE_EXTENSIONS)
        if pattern.search(path.name) and path.stat().st_size <= 100 * 1024 * 1024
    )


def agora_crosswalk_candidates(base: Path) -> pd.DataFrame:
    name_pattern = re.compile(
        r"agora.*(?:crosswalk|mapping|metadata|organism|model|species|taxon)|"
        r"(?:crosswalk|mapping|metadata|organism|model|species|taxon).*agora",
        re.IGNORECASE,
    )
    suffixes = TABLE_EXTENSIONS | {".json", ".parquet"}
    rows = []
    for path in iter_files(base, suffixes):
        if not name_pattern.search(path.name):
            continue
        row: dict[str, Any] = {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "suffix": path.suffix.lower(),
        }
        try:
            if path.suffix.lower() in TABLE_EXTENSIONS:
                row["columns_json"] = json.dumps(
                    [str(column) for column in read_tabular(path, nrows=0).columns]
                )
            elif path.suffix.lower() == ".parquet":
                try:
                    parquet = importlib.import_module("pyarrow.parquet")
                    row["columns_json"] = json.dumps(
                        list(parquet.ParquetFile(path).schema_arrow.names)
                    )
                except Exception:
                    row["columns_json"] = ""
        except Exception as exc:
            row["inspection_error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
    return pd.DataFrame(rows)


def read_tabular(
    path: Path,
    usecols: list[str] | None = None,
    nrows: int | None = None,
) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, sheet_name=0, usecols=usecols, nrows=nrows)
    if suffix == ".tsv":
        return pd.read_csv(
            path, sep="\t", usecols=usecols, nrows=nrows, low_memory=False
        )
    try:
        return pd.read_csv(path, usecols=usecols, nrows=nrows, low_memory=False)
    except Exception:
        return pd.read_csv(
            path,
            sep=None,
            engine="python",
            usecols=usecols,
            nrows=nrows,
        )


def find_semantic_column(columns: Sequence[str], candidates: Sequence[str]) -> str | None:
    normalised = {
        re.sub(r"[^a-z0-9]+", "", str(column).lower()): str(column)
        for column in columns
    }
    for candidate in candidates:
        key = re.sub(r"[^a-z0-9]+", "", candidate.lower())
        if key in normalised:
            return normalised[key]
    return None


def inspect_metadata_candidates(
    external_root: Path,
    profile_ids: set[str],
    v1_ids: set[str],
) -> tuple[pd.DataFrame, dict[str, set[str]]]:
    rows: list[dict[str, Any]] = []
    ids_by_path: dict[str, set[str]] = {}
    for path in metadata_candidate_paths(external_root):
        row: dict[str, Any] = {
            "path": str(path),
            "size_bytes": path.stat().st_size,
        }
        try:
            header = read_tabular(path, nrows=0)
            columns = [str(column) for column in header.columns]
            sample_column = find_semantic_column(
                columns,
                (
                    "sample_id",
                    "sample",
                    "run_accession",
                    "run",
                    "accession",
                    "sample_name",
                    "external_sample_id",
                ),
            )
            participant_column = find_semantic_column(
                columns,
                (
                    "host_subject_id",
                    "participant_id",
                    "patient_id",
                    "subject_id",
                    "individual_id",
                ),
            )
            label_column = find_semantic_column(
                columns,
                (
                    "external_label_ibd_binary",
                    "ibd_binary",
                    "condition",
                    "cond",
                    "diagnosis",
                    "disease",
                    "phenotype",
                    "label",
                ),
            )
            row.update(
                {
                    "columns_json": json.dumps(columns),
                    "sample_column": sample_column or "",
                    "participant_column": participant_column or "",
                    "label_column": label_column or "",
                }
            )
            if sample_column is None:
                row["status"] = "no_sample_id_column"
                rows.append(row)
                continue
            selected_columns = [sample_column]
            for column in (participant_column, label_column):
                if column and column not in selected_columns:
                    selected_columns.append(column)
            frame = read_tabular(path, usecols=selected_columns)
            ids = {
                normalise_sample_id(value)
                for value in frame[sample_column]
                if str(value).strip().lower() not in {"", "nan", "none", "null"}
            }
            ids_by_path[str(path)] = ids
            overlap = ids & profile_ids
            row.update(
                {
                    "status": "readable",
                    "n_rows": len(frame),
                    "unique_sample_ids": len(ids),
                    "profile_overlap": len(overlap),
                    "profile_overlap_fraction": len(overlap) / len(profile_ids)
                    if profile_ids
                    else float("nan"),
                    "v1_overlap": len(ids & v1_ids),
                    "unique_participants": int(frame[participant_column].nunique())
                    if participant_column
                    else None,
                    "label_counts_json": json.dumps(
                        {
                            str(key): int(value)
                            for key, value in frame[label_column]
                            .astype(str)
                            .value_counts(dropna=False)
                            .head(30)
                            .items()
                        },
                        sort_keys=True,
                    )
                    if label_column
                    else "",
                    "selection_score": len(overlap) * 1000
                    + (100 if label_column else 0)
                    + (10 if participant_column else 0),
                }
            )
        except Exception as exc:
            row.update(
                {
                    "status": "read_error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "selection_score": -1,
                }
            )
        rows.append(row)
    frame = pd.DataFrame(rows)
    if not frame.empty and "selection_score" in frame:
        frame = frame.sort_values(
            ["selection_score", "path"], ascending=[False, True], na_position="last"
        ).reset_index(drop=True)
    return frame, ids_by_path


SOURCE_MARKERS = {
    "agora_reaction_table": "AGORA_reactions_canon",
    "catalyst_support": "catalyst",
    "geometric_positive_support": "geometric",
    "graph_seed": "GRAPH_SEED",
    "tau_threshold": "TAU_THRESHOLD",
    "roots_per_input": "ROOTS_PER_INPUT",
    "h0_union_find": "union",
    "h0_deaths": "h0",
    "ricci_pairwise": "pairwise",
    "ricci_faithful_active": "faithful_active",
    "epsilon_dist": "epsilon_dist",
    "c_single_out": "c_single_out",
    "ricci_beta": "beta",
}


def source_inventory(base: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in iter_files(base, {".py"}):
        try:
            if path.stat().st_size > 5 * 1024 * 1024:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        lower = text.lower()
        hits = [
            name
            for name, marker in SOURCE_MARKERS.items()
            if marker.lower() in lower
        ]
        if not hits:
            continue
        constant_lines = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if re.search(
                r"(?:GRAPH_SEED|TAU_THRESHOLD|ROOTS_PER_INPUT|epsilon_dist|c_single_out|beta)\s*=",
                line,
                flags=re.IGNORECASE,
            ):
                constant_lines.append(f"{line_number}:{line.strip()}")
        rows.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
                "line_count": len(text.splitlines()),
                "matched_markers": " | ".join(hits),
                "marker_count": len(hits),
                "constant_assignments": " | ".join(constant_lines[:50]),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["marker_count", "path"], ascending=[False, True]
    ) if rows else pd.DataFrame(rows)


def inspect_projection_assets(base: Path, external_root: Path) -> dict[str, Any]:
    feature_dir = external_root / V1_FEATURE_RELATIVE
    assets: dict[str, Any] = {}
    for name in (
        "edge_metadata.csv",
        "feature_matrix_B_K0.npz",
        "feature_names.txt",
        "matched_metadata.csv",
    ):
        path = feature_dir / name
        assets[name] = {
            "path": str(path),
            "exists": path.is_file(),
            "size_bytes": path.stat().st_size if path.is_file() else None,
            "sha256": sha256_file(path) if path.is_file() else None,
        }
    edge_path = feature_dir / "edge_metadata.csv"
    if edge_path.is_file():
        frame = pd.read_csv(edge_path, low_memory=False)
        assets["edge_metadata.csv"].update(
            {
                "rows": len(frame),
                "columns": [str(column) for column in frame.columns],
                "duplicate_rows": int(frame.duplicated().sum()),
            }
        )

    interval_rows = []
    for path in find_named_files(base, "adaptive_intervals.npy"):
        row = {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        try:
            array = np.load(path, allow_pickle=False)
            row.update(
                {
                    "shape": list(array.shape),
                    "dtype": str(array.dtype),
                    "finite": bool(np.isfinite(array).all()),
                    "min": float(np.min(array)) if array.size else None,
                    "max": float(np.max(array)) if array.size else None,
                }
            )
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        interval_rows.append(row)
    assets["adaptive_intervals_candidates"] = interval_rows
    return assets


def v1_sample_ids(external_root: Path) -> set[str]:
    path = external_root / V1_FEATURE_RELATIVE / "matched_metadata.csv"
    if not path.is_file():
        return set()
    frame = pd.read_csv(path, usecols=["sample_id"], dtype=str)
    return {normalise_sample_id(value) for value in frame["sample_id"]}


def find_sbml_directories(base: Path) -> pd.DataFrame:
    counts: Counter[str] = Counter()
    for path in iter_files(base):
        lower = path.name.lower()
        if lower.endswith((".xml", ".sbml", ".xml.gz", ".sbml.gz")):
            counts[str(path.parent)] += 1
    return pd.DataFrame(
        [
            {"directory": directory, "sbml_like_files": count}
            for directory, count in counts.most_common(30)
        ]
    )


def system_audit(base: Path, external_root: Path) -> dict[str, Any]:
    disk = shutil.disk_usage(external_root)
    memory_kb: dict[str, int] = {}
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        for line in meminfo.read_text().splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            match = re.search(r"\d+", value)
            if match:
                memory_kb[key] = int(match.group(0))
    versions = {}
    for package in ("numpy", "pandas", "scipy", "networkx", "joblib", "pyarrow"):
        try:
            module = importlib.import_module(package)
            versions[package] = getattr(module, "__version__", "unknown")
        except Exception as exc:
            versions[package] = f"UNAVAILABLE: {type(exc).__name__}: {exc}"
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "memory_total_bytes": memory_kb.get("MemTotal", 0) * 1024,
        "memory_available_bytes": memory_kb.get("MemAvailable", 0) * 1024,
        "disk_total_bytes": disk.total,
        "disk_used_bytes": disk.used,
        "disk_free_bytes": disk.free,
        "base_path": str(base),
        "external_root": str(external_root),
        "package_versions": versions,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=BASE_DEFAULT)
    parser.add_argument("--external-root", type=Path, default=EXTERNAL_ROOT_DEFAULT)
    parser.add_argument("--expected-profiles", type=int, default=EXPECTED_PROFILE_COUNT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: <external-root>/expanded_v2_preflight_audit",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    base = args.base.expanduser().resolve()
    external_root = args.external_root.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else external_root / "expanded_v2_preflight_audit"
    )
    if not base.is_dir():
        die(f"Real_Data root not found: {base}")
    if not external_root.is_dir():
        die(f"External root not found: {external_root}")
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        die(f"Audit directory is non-empty; inspect it or rerun with --overwrite: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    blockers: list[str] = []
    warnings: list[str] = []
    report: dict[str, Any] = {
        "audit_version": SCRIPT_VERSION,
        "created_utc": utc_now(),
        "base": str(base),
        "external_root": str(external_root),
        "output_dir": str(output_dir),
        "expected_profile_count": args.expected_profiles,
        "methodological_role": (
            "Expanded-community structural replication; not strict frozen external validation."
        ),
        "normalisation_rule": (
            "For every expanded edge, use the geometric mean of positive raw supports across "
            "the locked external cohort; apply one external normalisation regime to all edges."
        ),
        "label_firewall": (
            "Profile parsing and taxonomic coverage are computed without metadata labels. "
            "Labels are inspected only in the separate metadata inventory."
        ),
    }

    print("=" * 100)
    print("EXPANDED EXTERNAL V2 — FAIL-CLOSED PREREQUISITE AUDIT")
    print("=" * 100)
    print("External root:", external_root)
    print("Audit outputs:", output_dir)

    structural_source = external_root / STRUCTURAL_SOURCE_RELATIVE
    if not structural_source.is_file():
        die(f"Validated structural-v1 source is missing: {structural_source}")
    module = load_structural_module(structural_source)
    report["structural_source"] = {
        "path": str(structural_source),
        "sha256": sha256_file(structural_source),
    }

    print("[1/9] Discovering and parsing MetaPhlAn profiles...", flush=True)
    profile_roots = discover_profile_roots(external_root, module)
    profiles, profile_inventory = discover_profiles(profile_roots, module)
    atomic_csv(profile_inventory, output_dir / "profile_inventory.csv")
    selected_profile_ids = set(profiles)
    profile_summary = {
        "roots": [str(path) for path in profile_roots],
        "selected_profiles": len(profiles),
        "candidate_files": len(profile_inventory),
        "rejected_candidate_files": int(
            profile_inventory.get("parse_status", pd.Series(dtype=str)).eq("rejected").sum()
        ),
        "species_abundance_sum_percent": finite_summary(
            [float(meta["species_abundance_sum_percent"]) for _, _, meta in profiles.values()]
        ),
        "species_rows": finite_summary(
            [float(meta["n_species_rows"]) for _, _, meta in profiles.values()]
        ),
    }
    report["profiles"] = profile_summary
    if len(profiles) != args.expected_profiles:
        blockers.append(
            f"Expected {args.expected_profiles} parseable profiles but selected {len(profiles)}."
        )
    print(f"Parseable selected profiles: {len(profiles):,} / expected {args.expected_profiles:,}")

    print("[2/9] Verifying the frozen 508-species reference axis...", flush=True)
    frozen_axis, frozen_axis_audit = load_frozen_species_axis(base, module)
    report["frozen_species_axis"] = frozen_axis_audit
    if len(frozen_axis) != 508:
        blockers.append(f"Frozen species axis has {len(frozen_axis)} coordinates, expected 508.")
    print(f"Frozen IBDMDB species coordinates: {len(frozen_axis):,}")

    v1_ids = v1_sample_ids(external_root)
    paired_missing = sorted(v1_ids - selected_profile_ids)
    report["paired_v1_manifest"] = {
        "v1_samples": len(v1_ids),
        "present_in_selected_profiles": len(v1_ids & selected_profile_ids),
        "missing_profile_ids": paired_missing,
    }
    if len(v1_ids) != EXPECTED_V1_PAIRED_COUNT:
        blockers.append(
            f"Version-1 paired manifest has {len(v1_ids)} samples, expected {EXPECTED_V1_PAIRED_COUNT}."
        )
    if paired_missing:
        blockers.append(f"{len(paired_missing)} version-1 samples lack a selected profile.")
    print(
        f"Paired v1 samples with profiles: {len(v1_ids & selected_profile_ids):,} / {len(v1_ids):,}"
    )

    print("[3/9] Locating and inspecting the AGORA reaction parquet...", flush=True)
    parquet_path, parquet_candidates = choose_agora_parquet(base, structural_source)
    atomic_csv(parquet_candidates, output_dir / "agora_parquet_candidates.csv")
    agora_catalysts: set[str] = set()
    if parquet_path is None:
        blockers.append("AGORA_reactions_canon.parquet was not found.")
        report["agora_parquet"] = {"found": False}
        top_catalysts = pd.DataFrame()
    else:
        try:
            agora_audit, agora_catalysts, top_catalysts = inspect_agora_parquet(parquet_path)
            frozen_keys = {module.species_key(value) for value in frozen_axis}
            catalyst_keys = {module.species_key(value) for value in agora_catalysts}
            agora_audit.update(
                {
                    "catalyst_tokens_exactly_on_frozen_axis": len(
                        set(agora_catalysts) & set(frozen_axis)
                    ),
                    "catalyst_normalised_keys_on_frozen_axis": len(
                        catalyst_keys & frozen_keys
                    ),
                    "catalyst_normalised_keys_outside_frozen_axis": len(
                        catalyst_keys - frozen_keys
                    ),
                    "appears_broader_than_frozen_axis": len(catalyst_keys - frozen_keys)
                    > 0,
                }
            )
            report["agora_parquet"] = agora_audit
            if len(agora_catalysts) <= len(frozen_axis):
                blockers.append(
                    "AGORA reaction catalyst universe is not larger than the frozen species axis; "
                    "the parquet may already be IBDMDB-pruned."
                )
            print("AGORA parquet:", parquet_path)
            print(f"AGORA reaction rows: {agora_audit['n_rows']:,}")
            print(f"Unique catalyst tokens: {len(agora_catalysts):,}")
            print(
                "Distinct directed edges:",
                f"{agora_audit['distinct_directed_edges']:,}"
                if agora_audit["distinct_directed_edges"] is not None
                else "not identifiable from schema",
            )
        except Exception as exc:
            blockers.append(f"AGORA parquet inspection failed: {type(exc).__name__}: {exc}")
            report["agora_parquet"] = {
                "found": True,
                "path": str(parquet_path),
                "error": f"{type(exc).__name__}: {exc}",
            }
            top_catalysts = pd.DataFrame()
    atomic_csv(top_catalysts, output_dir / "agora_top_catalysts.csv")
    crosswalks = agora_crosswalk_candidates(base)
    atomic_csv(crosswalks, output_dir / "agora_crosswalk_candidates.csv")
    report["agora_crosswalk_candidates"] = {
        "candidate_files": len(crosswalks),
        "paths": crosswalks["path"].tolist() if not crosswalks.empty else [],
    }

    if profiles and agora_catalysts:
        print("[4/9] Computing label-free preliminary taxonomic coverage...", flush=True)
        coverage, coverage_summary = preliminary_coverage(
            profiles, frozen_axis, sorted(agora_catalysts), module
        )
        atomic_csv(coverage, output_dir / "preliminary_taxonomic_coverage.csv")
        report["preliminary_taxonomic_coverage"] = coverage_summary
        agora_median = coverage_summary["agora_coverage_fraction_preliminary"].get(
            "median", float("nan")
        )
        if math.isfinite(agora_median) and agora_median < 0.05:
            blockers.append(
                "Direct conservative mapping to AGORA catalyst tokens has median coverage below 5%; "
                "a separate AGORA organism crosswalk is required."
            )
        elif math.isfinite(agora_median) and agora_median < 0.50:
            warnings.append(
                "Preliminary median AGORA coverage is below 50%; inspect catalyst identifiers and crosswalk files."
            )
        print(
            "Frozen coverage median:",
            f"{coverage_summary['frozen_coverage_fraction'].get('median', float('nan')):.3%}",
        )
        print(
            "Preliminary AGORA coverage median:",
            f"{coverage_summary['agora_coverage_fraction_preliminary'].get('median', float('nan')):.3%}",
        )
    else:
        coverage = pd.DataFrame()
        report["preliminary_taxonomic_coverage"] = {"completed": False}

    print("[5/9] Auditing metadata candidates and cohort overlap...", flush=True)
    metadata_inventory, metadata_ids = inspect_metadata_candidates(
        external_root, selected_profile_ids, v1_ids
    )
    atomic_csv(metadata_inventory, output_dir / "metadata_candidates.csv")
    best_metadata_path = None
    if not metadata_inventory.empty:
        readable = metadata_inventory[metadata_inventory["status"].eq("readable")]
        if not readable.empty:
            best_metadata_path = str(readable.iloc[0]["path"])
    best_metadata_ids = metadata_ids.get(best_metadata_path or "", set())
    metadata_summary = {
        "candidate_files": len(metadata_inventory),
        "best_candidate_path": best_metadata_path,
        "best_candidate_profile_overlap": len(best_metadata_ids & selected_profile_ids),
        "profiles_without_best_metadata_match": sorted(selected_profile_ids - best_metadata_ids),
        "metadata_ids_without_selected_profile": sorted(best_metadata_ids - selected_profile_ids),
    }
    report["metadata"] = metadata_summary
    if best_metadata_path is None:
        blockers.append("No readable metadata candidate with a sample-ID column was found.")
    elif len(best_metadata_ids & selected_profile_ids) != len(selected_profile_ids):
        blockers.append(
            "The best metadata candidate does not cover every selected MetaPhlAn profile."
        )
    print("Best metadata candidate:", best_metadata_path or "NONE")
    print(
        f"Selected profiles matched by best metadata: {len(best_metadata_ids & selected_profile_ids):,} "
        f"/ {len(selected_profile_ids):,}"
    )

    print("[6/9] Inventorying graph, H0 and Ricci production sources...", flush=True)
    sources = source_inventory(base)
    atomic_csv(sources, output_dir / "production_source_inventory.csv")
    source_markers_found = set()
    if not sources.empty:
        for value in sources["matched_markers"]:
            source_markers_found.update(str(value).split(" | "))
    required_source_markers = {
        "agora_reaction_table",
        "catalyst_support",
        "h0_deaths",
        "ricci_pairwise",
        "epsilon_dist",
        "c_single_out",
    }
    missing_source_markers = sorted(required_source_markers - source_markers_found)
    report["production_sources"] = {
        "candidate_files": len(sources),
        "markers_found": sorted(source_markers_found),
        "required_markers_missing": missing_source_markers,
    }
    if missing_source_markers:
        blockers.append(f"Production source markers missing: {missing_source_markers}")
    print(f"Production source candidates: {len(sources):,}")

    print("[7/9] Verifying frozen projection assets...", flush=True)
    projection_assets = inspect_projection_assets(base, external_root)
    report["projection_assets"] = projection_assets
    for name in ("edge_metadata.csv", "feature_matrix_B_K0.npz", "feature_names.txt"):
        if not projection_assets[name]["exists"]:
            blockers.append(f"Frozen projection asset missing: {name}")
    if not projection_assets["adaptive_intervals_candidates"]:
        warnings.append(
            "No adaptive_intervals.npy was found; raw expanded H0 remains possible, but frozen-interval projection is not yet resolved."
        )

    print("[8/9] Locating AGORA SBML model directories...", flush=True)
    sbml_directories = find_sbml_directories(base)
    atomic_csv(sbml_directories, output_dir / "agora_sbml_directory_candidates.csv")
    report["agora_sbml"] = {
        "candidate_directories": len(sbml_directories),
        "largest_directory_count": int(sbml_directories.iloc[0]["sbml_like_files"])
        if not sbml_directories.empty
        else 0,
    }
    if not sbml_directories.empty:
        print(
            "Largest SBML-like directory:",
            sbml_directories.iloc[0]["directory"],
            f"({int(sbml_directories.iloc[0]['sbml_like_files']):,} files)",
        )
    else:
        warnings.append("No AGORA SBML-like model directory was found under Real_Data.")

    print("[9/9] Recording system and storage readiness...", flush=True)
    report["system"] = system_audit(base, external_root)
    free_gib = report["system"]["disk_free_bytes"] / 1024**3
    if free_gib < 20:
        blockers.append(f"Only {free_gib:.1f} GiB disk space is free; v2 expansion is not safe to launch.")
    elif free_gib < 50:
        warnings.append(f"Disk headroom is limited ({free_gib:.1f} GiB free).")

    report["blockers"] = blockers
    report["warnings"] = warnings
    report["ready_for_pipeline_construction"] = len(blockers) == 0
    report["audit_completed_utc"] = utc_now()
    atomic_json(output_dir / "V2_EXPANDED_PREFLIGHT_AUDIT.json", report)
    atomic_json(
        output_dir / "AUDIT_COMPLETE.json",
        {
            "completed_utc": utc_now(),
            "audit_version": SCRIPT_VERSION,
            "status": "complete",
            "ready_for_pipeline_construction": len(blockers) == 0,
            "blocker_count": len(blockers),
            "warning_count": len(warnings),
            "selected_profiles": len(profiles),
            "paired_v1_profiles": len(v1_ids & selected_profile_ids),
        },
    )

    print("\n" + "=" * 100)
    print("AUDIT COMPLETE")
    print("=" * 100)
    print("Ready for pipeline construction:", "YES" if not blockers else "NO")
    print(f"Blockers: {len(blockers)}")
    for index, message in enumerate(blockers, start=1):
        print(f"  B{index}. {message}")
    print(f"Warnings: {len(warnings)}")
    for index, message in enumerate(warnings, start=1):
        print(f"  W{index}. {message}")
    print("Full audit:", output_dir / "V2_EXPANDED_PREFLIGHT_AUDIT.json")
    print("Audit completed successfully; no graphs or features were built.")


if __name__ == "__main__":
    main()
