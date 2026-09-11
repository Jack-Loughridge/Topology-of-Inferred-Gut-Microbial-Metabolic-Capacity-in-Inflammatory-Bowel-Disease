#!/usr/bin/env python3
"""Participant-balanced active-outdegree histogram for publication.

Each sample graph is converted to an active-outdegree histogram, the histograms
are averaged within participant, and condition means and population SDs are
then calculated across participants. Vertices with zero active outdegree are
excluded and values at least six share one display bucket.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from .path_backbone_longrange_diagnostics import parse_graphml_active
except ImportError:  # Direct execution from the repository checkout.
    from path_backbone_longrange_diagnostics import parse_graphml_active


BUCKETS = ["1", "2", "3", "4", "5", ">=6"]


def norm_cond(value):
    text = str(value).strip().lower()
    if text == "cd" or "crohn" in text:
        return "cd"
    if text == "uc" or "ulcerative" in text:
        return "uc"
    if "non" in text:
        return "nonibd"
    return text


def active_outdegree_histogram(path: Path, active_tol: float = 1e-12):
    """Return bucket counts and active graph size using the safe GraphML parser."""
    nodes, _adjacency, edge_w = parse_graphml_active(path, active_tol)
    outdegree = Counter(source for source, _target in edge_w)
    hist = {bucket: 0 for bucket in BUCKETS}
    for degree in outdegree.values():
        hist[str(degree) if degree < 6 else ">=6"] += 1
    return hist, len(nodes), len(edge_w)


def load_cohort(metadata_path: Path, graph_dir: Path):
    metadata = pd.read_csv(metadata_path, low_memory=False)
    metadata = metadata[["External ID", "diagnosis", "Participant ID"]].copy()
    metadata.columns = ["sample_id", "diagnosis", "participant_id"]
    metadata["cond"] = metadata["diagnosis"].map(norm_cond)
    metadata = metadata[metadata["cond"].isin(["cd", "uc", "nonibd"])]
    metadata = metadata.drop_duplicates("sample_id").copy()
    graph_files = {path.stem: path for path in graph_dir.glob("*.graphml")}
    metadata = metadata[metadata["sample_id"].isin(graph_files)].copy()
    if metadata.empty:
        raise RuntimeError("No sample graphs matched the metadata.")
    return metadata, graph_files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-dir", default=str(Path.home() / "Real_Data" / "out_graphs"))
    parser.add_argument("--metadata", default=str(Path.home() / "Real_Data" / "hmp2_metadata.csv"))
    parser.add_argument(
        "--output-dir",
        default=str(Path.home() / "Real_Data" / "Outdegree_ParticipantAverage_Corrected"),
    )
    parser.add_argument("--active-tol", type=float, default=1e-12)
    parser.add_argument("--sd-ddof", type=int, default=0)
    parser.add_argument("--expected-samples", type=int, default=None)
    parser.add_argument("--expected-participants", type=int, default=None)
    args = parser.parse_args()

    graph_dir = Path(args.graph_dir).expanduser()
    metadata_path = Path(args.metadata).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata, graph_files = load_cohort(metadata_path, graph_dir)
    n_samples = metadata["sample_id"].nunique()
    n_participants = metadata["participant_id"].nunique()
    if args.expected_samples is not None and n_samples != args.expected_samples:
        raise RuntimeError(f"Expected {args.expected_samples} samples, found {n_samples}.")
    if args.expected_participants is not None and n_participants != args.expected_participants:
        raise RuntimeError(f"Expected {args.expected_participants} participants, found {n_participants}.")

    sample_rows = []
    for index, row in enumerate(metadata.itertuples(index=False), start=1):
        hist, n_vertices, n_edges = active_outdegree_histogram(
            graph_files[row.sample_id], args.active_tol
        )
        sample_rows.append(
            {
                "sample_id": row.sample_id,
                "participant_id": row.participant_id,
                "cond": row.cond,
                "n_active_vertices": n_vertices,
                "n_active_edges": n_edges,
                **{bucket: float(hist[bucket]) for bucket in BUCKETS},
            }
        )
        if index == 1 or index % 100 == 0 or index == n_samples:
            print(f"Processed {index}/{n_samples} sample graphs", flush=True)

    sample_df = pd.DataFrame(sample_rows)
    participant_rows = []
    for participant_id, sub in sample_df.groupby("participant_id", sort=True):
        participant_rows.append(
            {
                "participant_id": participant_id,
                "cond": sub["cond"].value_counts().idxmax(),
                "n_samples": int(len(sub)),
                "n_nonempty_samples": int(sub["n_active_edges"].gt(0).sum()),
                **{bucket: float(sub[bucket].mean()) for bucket in BUCKETS},
            }
        )
    participant_df = pd.DataFrame(participant_rows)

    summary_rows = []
    for cond in ["cd", "uc", "nonibd"]:
        participants = participant_df[participant_df["cond"].eq(cond)]
        samples = sample_df[sample_df["cond"].eq(cond)]
        summary_rows.append(
            {
                "cond": cond,
                "n_participants": int(len(participants)),
                "n_sample_graphs": int(len(samples)),
                "n_nonempty_sample_graphs": int(samples["n_active_edges"].gt(0).sum()),
                "mean_samples_per_participant": float(participants["n_samples"].mean()),
                **{
                    key: value
                    for bucket in BUCKETS
                    for key, value in (
                        (f"{bucket}_mean", float(participants[bucket].mean())),
                        (f"{bucket}_sd", float(participants[bucket].std(ddof=args.sd_ddof))),
                    )
                },
            }
        )
    summary_df = pd.DataFrame(summary_rows)

    stem = "outdegree_hist_active_participant_mean_cap6_excl0"
    sample_df.to_csv(output_dir / f"{stem}_sample_level.csv", index=False)
    participant_df.to_csv(output_dir / f"{stem}_participant_level.csv", index=False)
    summary_df.to_csv(output_dir / f"{stem}_summary.csv", index=False)

    x = np.arange(len(BUCKETS))
    width = 0.22
    figure, axis = plt.subplots(figsize=(10, 5.5))
    for offset, cond, label in [
        (-width, "cd", "Average CD participant"),
        (0.0, "uc", "Average UC participant"),
        (width, "nonibd", "Average non-IBD participant"),
    ]:
        row = summary_df[summary_df["cond"].eq(cond)].iloc[0]
        axis.bar(
            x + offset,
            [row[f"{bucket}_mean"] for bucket in BUCKETS],
            width,
            yerr=[row[f"{bucket}_sd"] for bucket in BUCKETS],
            capsize=4,
            label=label,
        )
    axis.set_xlabel("Active outdegree")
    axis.set_ylabel("Average number of vertices per participant sample")
    axis.set_title("Active outdegree histogram (zero excluded; six or more pooled)")
    axis.set_xticks(x)
    axis.set_xticklabels(["1", "2", "3", "4", "5", r"$\geq 6$"])
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / f"{stem}_with_sd.png", dpi=200)
    figure.savefig(output_dir / f"{stem}_with_sd.pdf")
    plt.close(figure)

    run_config = {
        "graph_dir": str(graph_dir),
        "metadata": str(metadata_path),
        "output_dir": str(output_dir),
        "active_tol": args.active_tol,
        "sd_ddof": args.sd_ddof,
        "samples": n_samples,
        "participants": n_participants,
        "empty_active_graphs": int(sample_df["n_active_edges"].eq(0).sum()),
        "aggregation": "sample histogram -> participant mean -> condition mean and population SD",
        "empty_graph_policy": "an empty active graph contributes a zero histogram",
    }
    (output_dir / f"{stem}_run_config.json").write_text(json.dumps(run_config, indent=2) + "\n")

    print(summary_df.to_string(index=False))
    print(f"Empty active graphs: {run_config['empty_active_graphs']}")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
