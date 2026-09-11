#!/usr/bin/env python3
from __future__ import annotations

"""Data-free end-to-end test for the IBD Ricci feature-block ablation."""

import argparse
import hashlib
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, save_npz


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def import_engine(path: Path) -> object:
    spec = importlib.util.spec_from_file_location("synthetic_ricci_ibd_cpath", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def create_synthetic_features(feature_dir: Path) -> None:
    rng = np.random.default_rng(20260903)
    n_samples = 40
    n_edges = 8
    y = np.repeat([0, 1], n_samples // 2)

    probability = np.where(y[:, None] == 1, 0.65, 0.35)
    B = rng.binomial(1, probability, size=(n_samples, n_edges)).astype(np.float32)
    K0 = B * (
        rng.normal(0.0, 0.25, size=(n_samples, n_edges))
        + y[:, None] * np.linspace(0.05, 0.35, n_edges)[None, :]
    )
    K0 = K0.astype(np.float32)
    X = np.hstack([B, K0]).astype(np.float32)

    feature_dir.mkdir(parents=True)
    save_npz(feature_dir / "feature_matrix_B_K0.npz", csr_matrix(X))
    pd.DataFrame(
        {
            "sample_id": [f"sample_{index:03d}" for index in range(n_samples)],
            "participant_id": [f"participant_{index:03d}" for index in range(n_samples)],
            "cond": ["nonibd" if value == 0 else "uc" for value in y],
        }
    ).to_csv(feature_dir / "matched_metadata.csv", index=False)
    pd.DataFrame(
        {
            "edge": [f"metabolite_{index} -> metabolite_{index + 1}" for index in range(n_edges)],
            "process": ["synthetic_A" if index < n_edges // 2 else "synthetic_B" for index in range(n_edges)],
        }
    ).to_csv(feature_dir / "edge_metadata.csv", index=False)


def run_combined_reference(engine: object, feature_dir: Path, output_dir: Path) -> None:
    args = argparse.Namespace(
        feature_dir=str(feature_dir),
        output_dir=str(output_dir),
        edge_annotation_csv=None,
        c_values=[0.02],
        n_repeats=2,
        n_splits=2,
        split_seed=13,
        model_seed=20260717,
        max_iter=10000,
        tol=1e-4,
        n_jobs=1,
        coef_eps=1e-12,
        top_n=10,
        resume=True,
    )
    engine.run(args)


def main() -> None:
    directory = Path(__file__).resolve().parent
    driver = directory / "run_ibd_block_ablation.py"
    default_engine = (
        directory.parent / "ibd_cpath" / "repeated_ricci_ibd_cpath.py"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", type=Path, default=default_engine)
    args = parser.parse_args()
    engine_path = args.engine.expanduser().resolve()
    if not engine_path.is_file():
        raise FileNotFoundError(
            f"Canonical engine not found: {engine_path}. Pass --engine for a standalone test."
        )

    with tempfile.TemporaryDirectory(prefix="ricci_block_ablation_self_test_") as temporary:
        root = Path(temporary)
        feature_dir = root / "features"
        combined_output = root / "combined"
        ablation_output = root / "ablation"
        create_synthetic_features(feature_dir)
        engine = import_engine(engine_path)
        run_combined_reference(engine, feature_dir, combined_output)

        command = [
            sys.executable,
            str(driver),
            "--engine",
            str(engine_path),
            "--expected-engine-sha256",
            sha256_file(engine_path),
            "--feature-dir",
            str(feature_dir),
            "--combined-output-dir",
            str(combined_output),
            "--output-dir",
            str(ablation_output),
            "--n-repeats",
            "2",
            "--n-splits",
            "2",
            "--top-n",
            "10",
        ]
        environment = dict(**__import__("os").environ)
        environment.update(
            OMP_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
            NUMEXPR_NUM_THREADS="1",
            PYTHONHASHSEED="0",
        )
        subprocess.run(command, check=True, env=environment)
        first_summary_sha256 = sha256_file(
            ablation_output / "ablation_performance_summary.csv"
        )
        subprocess.run(command, check=True, env=environment)
        if sha256_file(ablation_output / "ablation_performance_summary.csv") != first_summary_sha256:
            raise RuntimeError("Resuming changed the ablation performance summary")

        summary = pd.read_csv(ablation_output / "ablation_performance_summary.csv")
        if set(summary["representation"]) != {"B only", "K0 only", "[B|K0]"}:
            raise RuntimeError("A representation is missing from the ablation summary")
        if set(summary["evaluation_level"]) != {"sample", "participant"}:
            raise RuntimeError("A reporting level is missing from the ablation summary")
        if len(summary) != 6:
            raise RuntimeError("Unexpected number of ablation summary rows")

        deltas = pd.read_csv(ablation_output / "ablation_paired_deltas.csv")
        if deltas["contrast"].nunique() != 3 or len(deltas) != 24:
            raise RuntimeError("Paired ablation contrasts are incomplete")

        for block, expected_source_start in (("B", 0), ("K0", 8)):
            block_dir = ablation_output / f"{block}_only"
            feature_meta = pd.read_csv(block_dir / "feature_metadata_used.csv")
            if len(feature_meta) != 8 or set(feature_meta["feature_type"]) != {block}:
                raise RuntimeError(f"Incorrect {block} feature slice")
            expected_indices = np.arange(expected_source_start, expected_source_start + 8)
            if not np.array_equal(
                feature_meta["source_feature_index"].to_numpy(dtype=int), expected_indices
            ):
                raise RuntimeError(f"Incorrect {block} source feature indices")
            source_split = combined_output / "splits" / "sample_split_manifest.csv"
            block_split = block_dir / "splits" / "sample_split_manifest.csv"
            if sha256_file(source_split) != sha256_file(block_split):
                raise RuntimeError(f"{block} split manifest is not byte-identical")

        completion = __import__("json").loads(
            (ablation_output / "ABLATION_RUN_COMPLETE.json").read_text(encoding="utf-8")
        )
        if completion.get("status") != "complete":
            raise RuntimeError("Ablation did not record complete status")

    print("RICCI IBD FEATURE-BLOCK ABLATION SELF-TEST: PASSED")


if __name__ == "__main__":
    main()
