#!/usr/bin/env python3
"""Validate that the direct modern-analysis dependencies are importable."""

from __future__ import annotations

import importlib
import importlib.metadata
import platform
import sys


REQUIRED = (
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("scipy", "scipy"),
    ("scikit-learn", "sklearn"),
    ("matplotlib", "matplotlib"),
    ("torch", "torch"),
    ("networkx", "networkx"),
    ("xgboost", "xgboost"),
    ("pyarrow", "pyarrow"),
    ("openpyxl", "openpyxl"),
    ("fastparquet", "fastparquet"),
    ("tqdm", "tqdm"),
    ("joblib", "joblib"),
    ("threadpoolctl", "threadpoolctl"),
    ("PyYAML", "yaml"),
    ("pytest", "pytest"),
)


def main() -> int:
    print(f"Python: {platform.python_version()}")
    if sys.version_info < (3, 10):
        print("ERROR: Python >=3.10 is required.", file=sys.stderr)
        return 1

    failed: list[str] = []
    for distribution, module in REQUIRED:
        try:
            importlib.import_module(module)
            version = importlib.metadata.version(distribution)
        except Exception as exc:  # report every missing/broken direct dependency
            failed.append(distribution)
            print(f"FAIL: {distribution}: {exc}")
        else:
            print(f"OK:   {distribution} {version}")

    if failed:
        print("Missing or broken dependencies: " + ", ".join(failed), file=sys.stderr)
        return 1

    print("Modern analysis environment: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
