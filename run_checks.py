#!/usr/bin/env python3
"""Repository-level integrity, unit-test, and synthetic-test entry point."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Iterable


ROOT = Path(__file__).resolve().parent
EXPECTED_SOURCE_FILES = 191
SUPPLEMENTARY_MANIFESTS = (
    (
        ROOT / "provenance/ricci_ibd_feature_block_ablation/source_manifest_20260911.sha256",
        5,
        "Ricci IBD feature-block ablation source manifest",
    ),
    (
        ROOT / "provenance/ricci_ibd_feature_block_ablation/result_manifest_20260911.sha256",
        28,
        "Ricci IBD feature-block ablation result manifest",
    ),
)
CANONICAL_SOURCE_ROOTS = (
    ROOT / "pipeline",
    ROOT / "analysis",
    ROOT / "external_validation",
)

UNIT_SUITES = (
    ("graph-input canonicalization unit tests", ROOT / "pipeline/graph_construction", ("-m", "pytest", "-q", "tests")),
    ("graph-diagnostics unit tests", ROOT / "analysis/graph_diagnostics", ("-m", "pytest", "-q", "tests")),
    ("H0 Alpha-Pi five-task unit tests", ROOT / "analysis/h0_alpha_pi/five_task_1x5", ("-m", "pytest", "-q", "tests")),
    ("joint H0-Ricci core unit tests", ROOT / "analysis/joint/core", ("-m", "pytest", "-q", "tests")),
    ("joint repeated-CV unit tests", ROOT / "analysis/joint/repeated_cv", ("-m", "pytest", "-q", "tests")),
)

SYNTHETIC_SUITES = (
    ("H0 Alpha-Pi five-task self-test", ROOT / "analysis/h0_alpha_pi/five_task_1x5", ("self_test.py",)),
    ("H0 Alpha-Pi IBD repeated-CV self-test", ROOT / "analysis/h0_alpha_pi/ibd_repeated_cv", ("self_test.py",)),
    ("joint repeated-CV self-test", ROOT / "analysis/joint/repeated_cv", ("self_test.py",)),
    ("Ricci IBD feature-block ablation self-test", ROOT / "analysis/ricci/ibd_block_ablation", ("self_test.py",)),
    ("Ricci five-task repeated-CV self-test", ROOT / "analysis/ricci/five_task_repeated_cv", ("self_test.py",)),
    ("species IBD complete-case self-test", ROOT / "analysis/species/ibd_complete_case", ("self_test.py",)),
    ("species remaining-task self-test", ROOT / "analysis/species/remaining_tasks", ("self_test.py",)),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json_without_duplicate_keys(path: Path) -> object:
    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates)


def verify_source_manifest() -> None:
    manifest = ROOT / "provenance/source_import_manifest.sha256"
    lines = [line for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) != EXPECTED_SOURCE_FILES:
        raise RuntimeError(f"source manifest has {len(lines)} entries; expected {EXPECTED_SOURCE_FILES}")

    seen: set[str] = set()
    failures: list[str] = []
    for line_number, line in enumerate(lines, start=1):
        try:
            expected, relative = line.split(maxsplit=1)
        except ValueError as exc:
            raise RuntimeError(f"malformed manifest line {line_number}") from exc
        relative = relative.removeprefix("*")
        if relative in seen:
            failures.append(f"duplicate manifest path: {relative}")
            continue
        seen.add(relative)
        path = ROOT / relative
        if not path.is_file():
            failures.append(f"missing: {relative}")
        elif sha256(path) != expected:
            failures.append(f"checksum mismatch: {relative}")

    if failures:
        raise RuntimeError("source manifest verification failed:\n  " + "\n  ".join(failures))
    print(f"OK: source manifest ({len(lines)} files)")


def verify_supplementary_manifests() -> None:
    for manifest, expected_entries, label in SUPPLEMENTARY_MANIFESTS:
        lines = [
            line
            for line in manifest.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(lines) != expected_entries:
            raise RuntimeError(
                f"{label} has {len(lines)} entries; expected {expected_entries}"
            )

        seen: set[str] = set()
        failures: list[str] = []
        for line_number, line in enumerate(lines, start=1):
            try:
                expected, relative = line.split(maxsplit=1)
            except ValueError as exc:
                raise RuntimeError(
                    f"malformed {label} line {line_number}"
                ) from exc
            relative = relative.removeprefix("*")
            relative_path = Path(relative)
            if relative_path.is_absolute() or ".." in relative_path.parts:
                failures.append(f"unsafe manifest path: {relative}")
                continue
            if relative in seen:
                failures.append(f"duplicate manifest path: {relative}")
                continue
            seen.add(relative)
            path = ROOT / relative_path
            if not path.is_file():
                failures.append(f"missing: {relative}")
            elif sha256(path) != expected:
                failures.append(f"checksum mismatch: {relative}")

        if failures:
            raise RuntimeError(
                f"{label} verification failed:\n  " + "\n  ".join(failures)
            )
        print(f"OK: {label} ({len(lines)} files)")


def validate_structured_metadata() -> None:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required for metadata validation") from exc

    citation = yaml.safe_load((ROOT / "CITATION.cff").read_text(encoding="utf-8"))
    if not isinstance(citation, dict):
        raise RuntimeError("CITATION.cff must contain a YAML mapping")
    required = {"cff-version", "message", "title", "authors"}
    missing = required.difference(citation)
    if missing:
        raise RuntimeError(f"CITATION.cff missing fields: {sorted(missing)}")

    environment_files = (
        ROOT / "environment/environment.yml",
        ROOT / "environment/external-profiling-environment.yml",
        ROOT / "environment/metaphlan2_v260_environment.yml",
    )
    for path in environment_files:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict) or not value.get("name") or not value.get("dependencies"):
            raise RuntimeError(f"invalid environment specification: {path.relative_to(ROOT)}")

    configs = sorted((ROOT / "provenance/locked_run_configs").glob("*.json"))
    if not configs:
        raise RuntimeError("no locked run configurations found")
    for path in configs:
        load_json_without_duplicate_keys(path)
    print(f"OK: citation, 3 environment YAML files, and {len(configs)} locked JSON configurations")


def validate_python_syntax() -> None:
    files: list[Path] = [ROOT / "run_checks.py", ROOT / "environment/validate_analysis_environment.py"]
    for source_root in CANONICAL_SOURCE_ROOTS:
        files.extend(sorted(source_root.rglob("*.py")))
    for path in files:
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (SyntaxError, UnicodeDecodeError) as exc:
            raise RuntimeError(f"Python syntax/encoding failure: {path.relative_to(ROOT)}: {exc}") from exc
    print(f"OK: Python syntax ({len(files)} files)")


def validate_shell_syntax() -> None:
    bash = shutil.which("bash")
    if bash is None:
        print("SKIP: Bash syntax checks (bash not installed)")
        return
    files: list[Path] = []
    for source_root in CANONICAL_SOURCE_ROOTS:
        files.extend(sorted(source_root.rglob("*.sh")))
    for path in files:
        subprocess.run([bash, "-n", str(path)], cwd=ROOT, check=True)
    print(f"OK: Bash syntax ({len(files)} files)")


def validate_tracked_hygiene() -> None:
    if shutil.which("git") is None or not (ROOT / ".git").exists():
        print("SKIP: tracked-file hygiene (not a Git worktree)")
        return
    result = subprocess.run(
        ["git", "ls-files", "-z"], cwd=ROOT, check=True, capture_output=True
    )
    tracked = [Path(item.decode("utf-8")) for item in result.stdout.split(b"\0") if item]
    prohibited = [
        path for path in tracked
        if "__pycache__" in path.parts
        or any(part.endswith(".egg-info") for part in path.parts)
        or path.suffix in {".pyc", ".pyo"}
    ]
    oversized = [path for path in tracked if (ROOT / path).is_file() and (ROOT / path).stat().st_size > 10 * 1024 * 1024]
    if prohibited:
        raise RuntimeError("generated Python files are tracked: " + ", ".join(map(str, prohibited)))
    if oversized:
        raise RuntimeError("tracked files exceed 10 MiB: " + ", ".join(map(str, oversized)))
    print(f"OK: tracked-file hygiene ({len(tracked)} files)")


def run_static_checks() -> None:
    verify_source_manifest()
    verify_supplementary_manifests()
    validate_structured_metadata()
    validate_python_syntax()
    validate_shell_syntax()
    validate_tracked_hygiene()


def run_suites(suites: Iterable[tuple[str, Path, tuple[str, ...]]]) -> None:
    environment = os.environ.copy()
    environment.update(
        {
            "MPLBACKEND": "Agg",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "PYTHONHASHSEED": "0",
        }
    )
    for name, directory, arguments in suites:
        print(f"\n=== {name} ===", flush=True)
        subprocess.run([sys.executable, *arguments], cwd=directory, env=environment, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "level",
        choices=("static", "unit", "full"),
        nargs="?",
        default="static",
        help="static integrity checks; unit adds component pytest suites; full adds synthetic end-to-end self-tests",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_static_checks()
    if args.level in {"unit", "full"}:
        run_suites(UNIT_SUITES)
    if args.level == "full":
        run_suites(SYNTHETIC_SUITES)
    print(f"\nPASS: repository {args.level} checks")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as exc:
        print(f"FAIL: command exited with status {exc.returncode}: {exc.cmd}", file=sys.stderr)
        raise SystemExit(exc.returncode)
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)
