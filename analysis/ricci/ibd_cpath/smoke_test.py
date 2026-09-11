#!/usr/bin/env python3
"""Generate a tiny synthetic [B|K0] dataset and run a quick end-to-end smoke test."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, save_npz


def main() -> None:
    root = Path(__file__).resolve().parent / "_smoke"
    if root.exists():
        shutil.rmtree(root)
    feature_dir = root / "features"
    output_dir = root / "output"
    feature_dir.mkdir(parents=True)

    rng = np.random.default_rng(20260717)
    n_participants = 20
    n_edges = 12
    rows = []
    matrix_rows = []

    for participant_idx in range(n_participants):
        label = 0 if participant_idx < n_participants // 2 else 1
        cond = "nonibd" if label == 0 else ("uc" if participant_idx % 2 == 0 else "cd")
        n_samples = 2 + (participant_idx % 2)
        participant_effect = rng.normal(0, 0.15, size=n_edges)
        for sample_idx in range(n_samples):
            sample_id = f"S{participant_idx:02d}_{sample_idx:02d}"
            rows.append(
                {
                    "sample_id": sample_id,
                    "participant_id": f"P{participant_idx:02d}",
                    "cond": cond,
                }
            )
            B = (rng.random(n_edges) < (0.55 + 0.10 * label)).astype(float)
            K = rng.normal(-0.5, 0.5, size=n_edges) + participant_effect
            K[0:3] += 1.0 * label
            K[3:5] -= 0.7 * label
            K *= B
            matrix_rows.append(np.concatenate([B, K]))

    X = np.asarray(matrix_rows, dtype=np.float32)
    save_npz(feature_dir / "feature_matrix_B_K0.npz", csr_matrix(X))
    pd.DataFrame(rows).to_csv(feature_dir / "matched_metadata.csv", index=False)
    pd.DataFrame(
        {
            "edge": [f"M_E{i}[c] -> M_P{i}[c]" for i in range(n_edges)],
            "process": ["Transport" if i < 4 else "Redox" if i < 8 else "Nucleotide" for i in range(n_edges)],
        }
    ).to_csv(feature_dir / "edge_metadata.csv", index=False)

    command = [
        sys.executable,
        str(Path(__file__).resolve().parent / "repeated_ricci_ibd_cpath.py"),
        "--feature-dir",
        str(feature_dir),
        "--output-dir",
        str(output_dir),
        "--c-values",
        "0.01",
        "0.1",
        "--n-repeats",
        "2",
        "--n-splits",
        "2",
        "--max-iter",
        "2000",
        "--top-n",
        "10",
    ]
    subprocess.run(command, check=True)

    required = [
        output_dir / "RUN_COMPLETE.json",
        output_dir / "all_C_performance_compact.csv",
        output_dir / "paired_C_performance_comparisons.csv",
        output_dir / "splits" / "participant_split_manifest.csv",
        output_dir / "C_0p01" / "reaction_coefficient_distributions.csv",
        output_dir / "C_0p01" / "process_summary_distributions.csv",
        output_dir / "C_0p01" / "full_source_model" / "full_source_model.npz",
        output_dir / "C_0p1" / "repetition_pooled_oof_results.csv",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError("Smoke test outputs missing:\n" + "\n".join(missing))

    # Run again to exercise resume logic.
    subprocess.run(command, check=True)
    print(f"[READY] Smoke test passed. Outputs: {output_dir}")


if __name__ == "__main__":
    main()
