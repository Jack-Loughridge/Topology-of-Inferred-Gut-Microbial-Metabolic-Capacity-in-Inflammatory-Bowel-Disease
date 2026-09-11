#!/usr/bin/env python3
"""Participant-balanced active-edge weight-density curves for publication.

The hierarchy is sample density -> participant mean density -> condition mean
density. Active graphs with no retained edge contribute a zero curve, matching
the historical publication figure; the output records their counts and the
resulting integrated curve mass explicitly.
"""

from __future__ import annotations

import argparse
import json
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


def norm_cond(value):
    text = str(value).strip().lower()
    if text == "cd" or "crohn" in text:
        return "cd"
    if text == "uc" or "ulcerative" in text:
        return "uc"
    if "non" in text:
        return "nonibd"
    return text


def clipped_weights(path: Path, active_tol: float, clip_max: float):
    """Read deduplicated active directed-edge weights with the safe parser."""
    _nodes, _adjacency, edge_w = parse_graphml_active(path, active_tol)
    values = np.fromiter(edge_w.values(), dtype=np.float64)
    if not values.size:
        return values
    return values[np.isfinite(values) & (values > 0.0) & (values <= clip_max)]


def density(values: np.ndarray, bins: np.ndarray):
    if not values.size:
        return np.zeros(len(bins) - 1, dtype=np.float64)
    result, _ = np.histogram(values, bins=bins, density=True)
    result = np.asarray(result, dtype=np.float64)
    result[~np.isfinite(result)] = 0.0
    return result


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
        default=str(Path.home() / "Real_Data" / "Weight_Density_ParticipantAverage_Corrected"),
    )
    parser.add_argument("--active-tol", type=float, default=1e-12)
    parser.add_argument("--clip-max", type=float, default=0.99)
    parser.add_argument("--n-bin-edges", type=int, default=200)
    parser.add_argument("--expected-samples", type=int, default=None)
    parser.add_argument("--expected-participants", type=int, default=None)
    args = parser.parse_args()

    graph_dir = Path(args.graph_dir).expanduser()
    metadata_path = Path(args.metadata).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    bins = np.linspace(0.0, args.clip_max, args.n_bin_edges)
    x = (bins[:-1] + bins[1:]) / 2.0
    bin_columns = [f"bin_{index:03d}" for index in range(len(x))]

    metadata, graph_files = load_cohort(metadata_path, graph_dir)
    n_samples = metadata["sample_id"].nunique()
    n_participants = metadata["participant_id"].nunique()
    if args.expected_samples is not None and n_samples != args.expected_samples:
        raise RuntimeError(f"Expected {args.expected_samples} samples, found {n_samples}.")
    if args.expected_participants is not None and n_participants != args.expected_participants:
        raise RuntimeError(f"Expected {args.expected_participants} participants, found {n_participants}.")

    sample_rows = []
    for index, row in enumerate(metadata.itertuples(index=False), start=1):
        values = clipped_weights(graph_files[row.sample_id], args.active_tol, args.clip_max)
        curve = density(values, bins)
        sample_rows.append(
            {
                "sample_id": row.sample_id,
                "participant_id": row.participant_id,
                "cond": row.cond,
                "n_weights_plotted": int(values.size),
                **{column: float(value) for column, value in zip(bin_columns, curve)},
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
                "n_nonempty_samples": int(sub["n_weights_plotted"].gt(0).sum()),
                "mean_n_weights_plotted": float(sub["n_weights_plotted"].mean()),
                **{column: float(sub[column].mean()) for column in bin_columns},
            }
        )
    participant_df = pd.DataFrame(participant_rows)

    def mean_curve(frame):
        if frame.empty:
            return np.zeros(len(x), dtype=np.float64)
        return frame[bin_columns].mean(axis=0).to_numpy(dtype=np.float64)

    curves = {
        "avg_sample_graph": mean_curve(sample_df),
        "avg_participant": mean_curve(participant_df),
        "cd_participant": mean_curve(participant_df[participant_df["cond"].eq("cd")]),
        "uc_participant": mean_curve(participant_df[participant_df["cond"].eq("uc")]),
        "nonibd_participant": mean_curve(participant_df[participant_df["cond"].eq("nonibd")]),
    }
    curve_df = pd.DataFrame({"bin_mid": x, **curves})

    stem = "weight_density_curves_participant_average_clip099"
    sample_df.to_csv(output_dir / f"{stem}_sample_level.csv", index=False)
    participant_df.to_csv(output_dir / f"{stem}_participant_level.csv", index=False)
    curve_df.to_csv(output_dir / f"{stem}_curves.csv", index=False)

    labels = {
        "avg_sample_graph": "Average sample graph",
        "avg_participant": "Average participant",
        "cd_participant": "Average CD participant",
        "uc_participant": "Average UC participant",
        "nonibd_participant": "Average non-IBD participant",
    }
    for log_scale, figure_stem in [
        (False, "weight_density_curves_participant_average_clip099"),
        (True, "weight_log_density_curves_participant_average_clip099"),
    ]:
        figure, axis = plt.subplots(figsize=(9.5, 5.5))
        for key, values in curves.items():
            plot_values = np.log10(values + 1e-12) if log_scale else values
            axis.plot(x, plot_values, label=labels[key])
        axis.set_xlabel(f"Edge weight (0, {args.clip_max:g}]")
        axis.set_ylabel("log10 density" if log_scale else "Density")
        axis.set_title("Participant-average active-edge weight density")
        axis.legend()
        figure.tight_layout()
        figure.savefig(output_dir / f"{figure_stem}.png", dpi=200)
        figure.savefig(output_dir / f"{figure_stem}.pdf")
        plt.close(figure)

    bin_width = float(bins[1] - bins[0])
    summary = {
        "graph_dir": str(graph_dir),
        "metadata": str(metadata_path),
        "output_dir": str(output_dir),
        "active_tol": args.active_tol,
        "clip_max": args.clip_max,
        "n_bin_edges": args.n_bin_edges,
        "samples": n_samples,
        "nonempty_samples": int(sample_df["n_weights_plotted"].gt(0).sum()),
        "participants": n_participants,
        "aggregation": "sample density -> participant mean -> condition mean",
        "empty_graph_policy": "an empty active graph contributes a zero density curve",
        "curve_integrals": {
            key: float(np.sum(values) * bin_width) for key, values in curves.items()
        },
    }
    (output_dir / f"{stem}_run_config.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
