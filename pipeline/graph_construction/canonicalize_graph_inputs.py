#!/usr/bin/env python3
"""Reproduce the canonical species-abundance and AGORA reaction inputs.

The rules in this module were recovered by privacy-preserving comparison of the
historical source and canonical tables. They are intentionally explicit and
covered by synthetic regression tests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_VERSION = "1.0.0"

PRODUCTION_INPUT_SHA256 = {
    "species": "991087eb7105b85ad60b0917fae895bb966ab4463709de092f7297fda6d22cb6",
    "reactions": "799fe755c0c9ede07572742442a799a8de1ff4e7165f6ecb2812b6699955cf77",
}

HISTORICAL_CANONICAL_SHA256 = {
    "species": "087a5bd0dd001906b348fdedd433d911edad73fef3bde5af47ca0a0fef5cee69",
    "reactions": "ec906cd5f08448991b3958112fd4367f17141e0bbbe7d0333c8382a5c36af755",
}

PRODUCTION_EXPECTATIONS = {
    "species": {
        "rows": 1360,
        "source_columns": 933,
        "selected_source_columns": 591,
        "species_rank_columns": 578,
        "multitoken_genus_columns": 13,
        "output_columns": 509,
    },
    "reactions": {
        "rows": 12_582_264,
        "columns": 6,
        "unique_model_ids": 7302,
        "unique_canonical_catalysts": 1369,
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def first_two_tokens(value: str, *, remove_model_prefix: bool = False) -> str:
    value = str(value).strip()
    if remove_model_prefix and value.startswith("M_"):
        value = value[2:]
    tokens = value.split("_")
    if len(tokens) < 2 or not tokens[0] or not tokens[1]:
        raise ValueError(f"cannot derive a two-token organism name from {value!r}")
    return "_".join(tokens[:2])


def terminal_taxon(column: str) -> tuple[str, str]:
    terminal = str(column).rsplit("|", 1)[-1]
    if len(terminal) >= 3 and terminal[1:3] == "__":
        return terminal[0], terminal[3:]
    return "other", terminal


def selected_abundance_column(column: str) -> tuple[bool, str, str]:
    """Return selection state, rank, and canonical two-token name.

    Historical rule: retain every terminal species-rank label and every
    terminal genus-rank label containing an underscore. Strip the rank prefix
    and retain the first two underscore-delimited tokens.
    """
    rank, label = terminal_taxon(column)
    selected = rank == "s" or (rank == "g" and "_" in label)
    canonical = first_two_tokens(label) if selected else ""
    return selected, rank, canonical


def ensure_output_target(input_path: Path, output_path: Path, force: bool) -> None:
    if input_path.resolve() == output_path.resolve():
        raise ValueError("input and output paths must differ")
    if output_path.exists() and not force:
        raise FileExistsError(f"output already exists: {output_path}; pass --force to replace it")
    output_path.parent.mkdir(parents=True, exist_ok=True)


def temporary_output_path(output_path: Path) -> Path:
    return output_path.with_name(
        f".{output_path.stem}.{uuid.uuid4().hex}.tmp{output_path.suffix}"
    )


def verify_production_input(kind: str, input_path: Path, require_hash: bool) -> str:
    observed = sha256_file(input_path)
    if require_hash and observed != PRODUCTION_INPUT_SHA256[kind]:
        raise ValueError(
            f"{kind} input SHA-256 mismatch: observed {observed}; "
            f"expected {PRODUCTION_INPUT_SHA256[kind]}"
        )
    return observed


def validate_expected_counts(kind: str, observed: dict[str, int], enabled: bool) -> None:
    if not enabled:
        return
    expected = PRODUCTION_EXPECTATIONS[kind]
    mismatches = {
        key: {"observed": observed.get(key), "expected": value}
        for key, value in expected.items()
        if observed.get(key) != value
    }
    if mismatches:
        raise ValueError(f"{kind} production-count mismatch: {mismatches}")


def runtime_record() -> dict[str, str]:
    record = {
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
    }
    try:
        import pyarrow

        record["pyarrow"] = pyarrow.__version__
    except ImportError:
        pass
    return record


def canonicalize_species(
    input_path: Path,
    output_path: Path,
    *,
    force: bool = False,
    require_production_hash: bool = False,
) -> dict:
    input_path = Path(input_path)
    output_path = Path(output_path)
    ensure_output_target(input_path, output_path, force)
    input_hash = verify_production_input("species", input_path, require_production_hash)

    source = pd.read_excel(input_path, sheet_name=0)
    if source.shape[1] < 2:
        raise ValueError("species workbook must contain an identifier and abundance columns")
    identifier_name = str(source.columns[0])
    if require_production_hash and identifier_name != "External ID":
        raise ValueError(f"expected production identifier column 'External ID', got {identifier_name!r}")

    groups: dict[str, list[int]] = defaultdict(list)
    selected_rank_counts: Counter[str] = Counter()
    for position, column in enumerate(source.columns[1:], start=1):
        selected, rank, canonical = selected_abundance_column(str(column))
        if selected:
            groups[canonical].append(position)
            selected_rank_counts[rank] += 1

    if not groups:
        raise ValueError("no abundance columns matched the historical canonicalization rule")

    output_columns: dict[str, object] = {
        identifier_name: source.iloc[:, 0].to_numpy(copy=True)
    }
    for canonical in sorted(groups):
        frame = source.iloc[:, groups[canonical]].apply(pd.to_numeric, errors="raise")
        if frame.isna().any().any():
            raise ValueError(f"missing abundance encountered while constructing {canonical}")
        if frame.shape[1] == 1:
            output_columns[canonical] = frame.iloc[:, 0].to_numpy(copy=True)
        else:
            output_columns[canonical] = frame.sum(axis=1, skipna=False).to_numpy()
    output = pd.DataFrame(output_columns)

    counts = {
        "rows": int(source.shape[0]),
        "source_columns": int(source.shape[1]),
        "selected_source_columns": int(sum(len(value) for value in groups.values())),
        "species_rank_columns": int(selected_rank_counts["s"]),
        "multitoken_genus_columns": int(selected_rank_counts["g"]),
        "output_columns": int(output.shape[1]),
    }
    validate_expected_counts("species", counts, require_production_hash)

    temporary = temporary_output_path(output_path)
    try:
        output.to_excel(temporary, index=False, engine="openpyxl")
        os.replace(temporary, output_path)
    finally:
        if temporary.exists():
            temporary.unlink()

    output_hash = sha256_file(output_path)
    return {
        "script_version": SCRIPT_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "operation": "species_canonicalization",
        "rule": {
            "select": "terminal rank s, plus terminal rank g when its label contains an underscore",
            "rename": "remove the rank prefix and retain the first two underscore-delimited tokens",
            "aggregate": "sum columns sharing a canonical name",
            "order": "sort canonical names lexicographically",
            "rows": "preserve the first identifier column and row order",
        },
        "input": {"path": str(input_path), "sha256": input_hash},
        "output": {
            "path": str(output_path),
            "sha256": output_hash,
            "historical_reference_sha256": HISTORICAL_CANONICAL_SHA256["species"],
            "byte_identical_to_historical_reference": output_hash
            == HISTORICAL_CANONICAL_SHA256["species"],
        },
        "counts": counts,
        "runtime": runtime_record(),
    }


def canonicalize_reactions(
    input_path: Path,
    output_path: Path,
    *,
    force: bool = False,
    require_production_hash: bool = False,
    batch_size: int = 262_144,
) -> dict:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required for reaction canonicalization") from exc

    input_path = Path(input_path)
    output_path = Path(output_path)
    ensure_output_target(input_path, output_path, force)
    input_hash = verify_production_input("reactions", input_path, require_production_hash)

    parquet = pq.ParquetFile(input_path)
    required = ["model_id", "species_file", "reaction_id", "inputs", "outputs", "catalysts"]
    if parquet.schema_arrow.names != required:
        raise ValueError(
            f"reaction columns must be exactly {required}; observed {parquet.schema_arrow.names}"
        )

    temporary = temporary_output_path(output_path)
    writer = None
    row_count = 0
    source_models: set[str] = set()
    canonical_catalysts: set[str] = set()
    try:
        for batch in parquet.iter_batches(batch_size=batch_size):
            frame = batch.to_pandas()
            if frame["model_id"].isna().any() or frame["catalysts"].isna().any():
                raise ValueError("reaction model_id and catalysts must not contain missing values")
            model_ids = frame["model_id"].astype(str)
            source_catalysts = frame["catalysts"].astype(str)
            if not source_catalysts.equals(model_ids):
                mismatch_count = int((source_catalysts != model_ids).sum())
                raise ValueError(
                    f"historical invariant failed: catalysts differs from model_id in {mismatch_count} rows"
                )
            mapping = {
                model_id: first_two_tokens(model_id, remove_model_prefix=True)
                for model_id in model_ids.unique()
            }
            frame["catalysts"] = model_ids.map(mapping)
            source_models.update(mapping)
            canonical_catalysts.update(mapping.values())
            row_count += len(frame)

            table = pa.Table.from_pandas(frame, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(
                    temporary,
                    table.schema,
                    compression="snappy",
                    use_dictionary=True,
                )
            writer.write_table(table)
        if writer is None:
            raise ValueError("reaction table contains no rows")
        writer.close()
        writer = None
        os.replace(temporary, output_path)
    finally:
        if writer is not None:
            writer.close()
        if temporary.exists():
            temporary.unlink()

    counts = {
        "rows": int(row_count),
        "columns": len(required),
        "unique_model_ids": len(source_models),
        "unique_canonical_catalysts": len(canonical_catalysts),
    }
    validate_expected_counts("reactions", counts, require_production_hash)
    output_hash = sha256_file(output_path)
    return {
        "script_version": SCRIPT_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "operation": "reaction_canonicalization",
        "rule": {
            "preserve": "row order and every column except catalysts",
            "catalysts": "strip leading M_ from model_id and retain the first two underscore-delimited tokens",
            "source_invariant": "source catalysts must equal model_id in every row",
        },
        "input": {"path": str(input_path), "sha256": input_hash},
        "output": {
            "path": str(output_path),
            "sha256": output_hash,
            "historical_reference_sha256": HISTORICAL_CANONICAL_SHA256["reactions"],
            "byte_identical_to_historical_reference": output_hash
            == HISTORICAL_CANONICAL_SHA256["reactions"],
        },
        "counts": counts,
        "runtime": runtime_record(),
    }


def write_provenance(record: dict, path: Path | None) -> None:
    text = json.dumps(record, indent=2, sort_keys=True) + "\n"
    if path is None:
        print(text, end="")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    print(f"Wrote provenance: {path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="operation", required=True)

    species = subparsers.add_parser("species", help="canonicalize the species-abundance workbook")
    species.add_argument("--input", type=Path, required=True)
    species.add_argument("--output", type=Path, required=True)
    species.add_argument("--provenance", type=Path)
    species.add_argument("--force", action="store_true")
    species.add_argument("--require-production-hash", action="store_true")

    reactions = subparsers.add_parser("reactions", help="canonicalize the AGORA reaction table")
    reactions.add_argument("--input", type=Path, required=True)
    reactions.add_argument("--output", type=Path, required=True)
    reactions.add_argument("--provenance", type=Path)
    reactions.add_argument("--batch-size", type=int, default=262_144)
    reactions.add_argument("--force", action="store_true")
    reactions.add_argument("--require-production-hash", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.operation == "species":
        record = canonicalize_species(
            args.input,
            args.output,
            force=args.force,
            require_production_hash=args.require_production_hash,
        )
    else:
        if args.batch_size < 1:
            raise ValueError("--batch-size must be positive")
        record = canonicalize_reactions(
            args.input,
            args.output,
            force=args.force,
            require_production_hash=args.require_production_hash,
            batch_size=args.batch_size,
        )
    write_provenance(record, args.provenance)
    print(f"Completed {args.operation} canonicalization")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
