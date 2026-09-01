#!/usr/bin/env python3
"""Fast validation of the faithful [B|K0] feature directory before the long run."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from scipy.sparse import load_npz


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--feature-dir",
        default=str(Path.home() / "Real_Data" / "Ricci_Classifier_Faithful_Eps0001_n250_v3"),
    )
    parser.add_argument("--edge-annotation-csv", default=None)
    args = parser.parse_args()

    feature_dir = Path(args.feature_dir).expanduser().resolve()
    matrix_path = feature_dir / "feature_matrix_B_K0.npz"
    metadata_path = feature_dir / "matched_metadata.csv"
    edge_path = feature_dir / "edge_metadata.csv"

    for path in [matrix_path, metadata_path, edge_path]:
        if not path.exists():
            raise FileNotFoundError(path)
        print(f"[OK] {path}")

    X = load_npz(matrix_path).tocsr()
    metadata = pd.read_csv(metadata_path, low_memory=False)
    edges = pd.read_csv(edge_path, low_memory=False)

    for column in ["sample_id", "participant_id", "cond"]:
        if column not in metadata.columns:
            raise ValueError(f"matched_metadata.csv missing {column}")
    if "edge" not in edges.columns:
        raise ValueError("edge_metadata.csv missing edge")
    if X.shape[0] != len(metadata):
        raise ValueError(f"matrix rows={X.shape[0]} metadata rows={len(metadata)}")
    if X.shape[1] != 2 * len(edges):
        raise ValueError(f"matrix columns={X.shape[1]} but 2*n_edges={2*len(edges)}")
    if metadata["sample_id"].astype(str).duplicated().any():
        raise ValueError("sample_id is not unique")

    metadata = metadata.copy()
    metadata["cond_norm"] = metadata["cond"].astype(str).str.strip().str.lower().str.replace(r"[\s_-]+", "", regex=True)
    task = metadata[metadata["cond_norm"].isin(["nonibd", "uc", "cd"])].copy()
    task["label"] = task["cond_norm"].map({"nonibd": 0, "uc": 1, "cd": 1})
    if task.groupby("participant_id")["label"].nunique().max() > 1:
        raise ValueError("At least one participant has both IBD and non-IBD labels")

    print("\n[Matrix]")
    print("shape:", X.shape)
    print("nnz:", X.nnz)
    print("n_edges:", len(edges))
    print("dense float32 estimate MiB:", X.shape[0] * X.shape[1] * 4 / 1024**2)

    print("\n[IBD vs non-IBD samples]")
    print(task["cond_norm"].value_counts().to_string())
    print("\n[IBD vs non-IBD participants]")
    participant = task.drop_duplicates("participant_id").copy()
    participant["label_name"] = participant["label"].map({0: "non-IBD", 1: "IBD"})
    print(participant["label_name"].value_counts().to_string())

    if "process" in edges.columns:
        process = edges["process"].fillna("Unannotated").astype(str)
        annotated = (process != "Unannotated").mean()
        print("\n[Process annotations in edge_metadata.csv]")
        print("annotated edge fraction:", f"{annotated:.3f}")
        print("categories:", process.nunique())
        print(process.value_counts().head(20).to_string())
    else:
        print("\n[WARNING] edge_metadata.csv has no process column")

    if args.edge_annotation_csv:
        annotation_path = Path(args.edge_annotation_csv).expanduser().resolve()
        if not annotation_path.exists():
            raise FileNotFoundError(annotation_path)
        annotation = pd.read_csv(annotation_path, low_memory=False)
        print("\n[Optional annotation CSV]")
        print("path:", annotation_path)
        print("rows:", len(annotation))
        print("columns:", list(annotation.columns))

    print("\n[READY] Inputs pass structural validation.")


if __name__ == "__main__":
    main()
