#!/usr/bin/env python3
"""Directed clustering on active sample graphs with participant averaging.

The estimand is deliberately different from clustering a participant union or
participant-mean graph:

1. retain directed edges with ``w < 1 - active_tol``;
2. retain only vertices incident to at least one retained edge;
3. calculate Fagiolo directed clustering within each sample graph;
4. use raw support strength ``s = 1 - w`` for the weighted coefficient;
5. average sample coefficients within participant; and
6. report population SD (ddof=0) across samples or participants.

Duplicate directed edges retain the smallest w (largest support strength).
Graphs without active edges are recorded explicitly and excluded from the
clustering moments because the active-vertex graph is empty.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import scipy
    from scipy.sparse import csr_matrix
except Exception as exc:  # pragma: no cover - import failure message
    raise RuntimeError("This analysis requires SciPy.") from exc


WEIGHT_KEYS = {"weight", "d1", "Weight", "D1"}
CONDITION_ORDER = ["overall", "cd", "uc", "nonibd"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_condition(value: Any) -> str:
    text = str(value).strip().lower()
    if "crohn" in text or text == "cd":
        return "cd"
    if "ulcerative" in text or text == "uc":
        return "uc"
    if "non" in text:
        return "nonibd"
    return text


def display_condition(value: str) -> str:
    return {"overall": "Overall", "cd": "CD", "uc": "UC", "nonibd": "non-IBD"}.get(value, value)


def parse_active_edge_strengths(path: Path, active_tol: float) -> dict[tuple[str, str], float]:
    """Return active directed edge strengths from a GraphML file.

    ``data`` children must remain intact until their parent ``edge`` end event
    is processed.  In particular, do not clear arbitrary child elements in an
    ``iterparse`` catch-all branch.
    """

    key_id_to_name: dict[str, str] = {}
    weight_key_ids = set(WEIGHT_KEYS)
    edge_strength: dict[tuple[str, str], float] = {}

    for _, elem in ET.iterparse(path, events=("end",)):
        tag = elem.tag.rsplit("}", 1)[-1]

        if tag == "key":
            key_id = elem.attrib.get("id")
            attr_name = elem.attrib.get("attr.name")
            if key_id is not None and attr_name is not None:
                key_id_to_name[key_id] = attr_name
                if key_id in WEIGHT_KEYS or attr_name in WEIGHT_KEYS:
                    weight_key_ids.add(key_id)
            elem.clear()

        elif tag == "node":
            elem.clear()

        elif tag == "edge":
            source = elem.attrib.get("source")
            target = elem.attrib.get("target")
            weight = None

            for child in elem:
                if child.tag.rsplit("}", 1)[-1] != "data":
                    continue
                key_id = child.attrib.get("key")
                key_name = key_id_to_name.get(key_id, key_id)
                if key_id in weight_key_ids or key_name in WEIGHT_KEYS:
                    try:
                        weight = float(child.text)
                    except (TypeError, ValueError):
                        weight = None
                    break

            if (
                source is not None
                and target is not None
                and source != target
                and weight is not None
                and math.isfinite(weight)
                and weight < 1.0 - active_tol
            ):
                strength = min(1.0, max(0.0, 1.0 - weight))
                edge = (source, target)
                if edge not in edge_strength or strength > edge_strength[edge]:
                    edge_strength[edge] = float(strength)

            elem.clear()

    return edge_strength


def exact_directed_clustering(edge_strength: dict[tuple[str, str], float]) -> dict[str, float | int]:
    """Calculate mean Fagiolo coefficients on active vertices.

    For binary adjacency A and cube-root strength matrix R, the local
    numerators are ``diag((A + A.T)^3)`` and ``diag((R + R.T)^3)``.  The
    denominator is ``2 * (d_tot * (d_tot - 1) - 2 * d_recip)``.  The
    numerator must not also be divided by two.
    """

    if not edge_strength:
        return {
            "n_active_vertices": 0,
            "n_active_edges": 0,
            "directed_clustering": np.nan,
            "weighted_directed_clustering": np.nan,
        }

    nodes = sorted({node for edge in edge_strength for node in edge})
    node_index = {node: index for index, node in enumerate(nodes)}
    rows = [node_index[source] for source, _ in edge_strength]
    cols = [node_index[target] for _, target in edge_strength]
    strengths = np.asarray(list(edge_strength.values()), dtype=np.float64)
    shape = (len(nodes), len(nodes))

    adjacency = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=shape, dtype=np.float64)
    rooted = csr_matrix((np.cbrt(strengths), (rows, cols)), shape=shape, dtype=np.float64)

    out_degree = np.asarray(adjacency.sum(axis=1)).ravel()
    in_degree = np.asarray(adjacency.sum(axis=0)).ravel()
    total_degree = out_degree + in_degree
    reciprocal_degree = np.asarray(adjacency.multiply(adjacency.T).sum(axis=1)).ravel()
    denominator = 2.0 * (total_degree * (total_degree - 1.0) - 2.0 * reciprocal_degree)

    sym_binary = adjacency + adjacency.T
    triangle_numerator = np.asarray(
        (sym_binary @ sym_binary).multiply(sym_binary.T).sum(axis=1)
    ).ravel()

    sym_weighted = rooted + rooted.T
    weighted_triangle_numerator = np.asarray(
        (sym_weighted @ sym_weighted).multiply(sym_weighted.T).sum(axis=1)
    ).ravel()

    local = np.zeros(len(nodes), dtype=np.float64)
    local_weighted = np.zeros(len(nodes), dtype=np.float64)
    defined = denominator > 0.0
    local[defined] = triangle_numerator[defined] / denominator[defined]
    local_weighted[defined] = weighted_triangle_numerator[defined] / denominator[defined]

    return {
        "n_active_vertices": int(len(nodes)),
        "n_active_edges": int(len(edge_strength)),
        "directed_clustering": float(np.mean(local)),
        "weighted_directed_clustering": float(np.mean(local_weighted)),
    }


def summarize(df: pd.DataFrame, level: str, sd_ddof: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for condition in CONDITION_ORDER:
        subset = df if condition == "overall" else df[df["cond"].eq(condition)]
        valid = subset[
            np.isfinite(pd.to_numeric(subset["directed_clustering"], errors="coerce"))
            & np.isfinite(pd.to_numeric(subset["weighted_directed_clustering"], errors="coerce"))
        ]

        def mean_sd(column: str, frame: pd.DataFrame = valid) -> tuple[float, float]:
            values = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            if values.empty:
                return np.nan, np.nan
            return float(values.mean()), float(values.std(ddof=sd_ddof))

        mean_c, sd_c = mean_sd("directed_clustering")
        mean_weighted, sd_weighted = mean_sd("weighted_directed_clustering")
        mean_vertices, sd_vertices = mean_sd("n_active_vertices", subset)
        mean_edges, sd_edges = mean_sd("n_active_edges", subset)

        rows.append(
            {
                "level": level,
                "cond": condition,
                "n_total": int(len(subset)),
                "n_valid": int(len(valid)),
                "mean_directed_clustering": mean_c,
                "sd_directed_clustering": sd_c,
                "mean_weighted_directed_clustering": mean_weighted,
                "sd_weighted_directed_clustering": sd_weighted,
                "mean_active_vertices_all": mean_vertices,
                "sd_active_vertices_all": sd_vertices,
                "mean_active_edges_all": mean_edges,
                "sd_active_edges_all": sd_edges,
            }
        )

    return pd.DataFrame(rows)


def write_latex(summary: pd.DataFrame, path: Path) -> None:
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Directed clustering on active metabolic graphs. Sample-graph coefficients are computed on vertices incident to at least one active edge; participant-average coefficients first average valid sample-graph values within participant. Values are mean $\pm$ population SD. Weighted clustering uses raw support strength $s=1-w$. The $n$ column gives valid/total observations; empty active graphs are excluded from coefficient moments but retained in the total.}",
        r"\label{tab:directed_clustering_participant_average}",
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"Level & Condition & $n$ valid/total & Directed clustering & Weighted directed clustering \\",
        r"\midrule",
    ]

    for _, row in summary.iterrows():
        level = "Sample graphwise" if row["level"] == "sample_graphwise" else "Participant average"
        n_text = f"{int(row['n_valid'])}/{int(row['n_total'])}"
        directed = f"{row['mean_directed_clustering']:.3f} $\\pm$ {row['sd_directed_clustering']:.3f}"
        weighted = f"{row['mean_weighted_directed_clustering']:.3f} $\\pm$ {row['sd_weighted_directed_clustering']:.3f}"
        lines.append(
            f"{level} & {display_condition(str(row['cond']))} & {n_text} & "
            f"{directed} & {weighted} " + r"\\"
        )

    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def output_manifest(output_dir: Path) -> None:
    names = [
        "directed_clustering_sample_level.csv",
        "directed_clustering_participant_average_level.csv",
        "directed_clustering_summary.csv",
        "directed_clustering_table.tex",
        "empty_active_graphs.csv",
        "failed_samples.csv",
        "run_config.json",
    ]
    lines = [f"{sha256_file(output_dir / name)}  {name}" for name in names]
    (output_dir / "OUTPUT_MANIFEST.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-dir", type=Path, default=Path.home() / "Real_Data" / "out_graphs")
    parser.add_argument("--metadata", type=Path, default=Path.home() / "Real_Data" / "hmp2_metadata.csv")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.home() / "Real_Data" / "Directed_Clustering_ParticipantAverage_Corrected",
    )
    parser.add_argument("--active-tol", type=float, default=1e-12)
    parser.add_argument("--sd-ddof", type=int, choices=(0, 1), default=0)
    parser.add_argument("--expected-samples", type=int, default=1317)
    parser.add_argument("--expected-participants", type=int, default=106)
    args = parser.parse_args()

    graph_dir = args.graph_dir.expanduser().resolve()
    metadata_path = args.metadata.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(metadata_path, low_memory=False)
    required = {"External ID", "diagnosis", "Participant ID"}
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f"Metadata is missing columns: {sorted(missing)}")

    metadata = metadata[["External ID", "diagnosis", "Participant ID"]].copy()
    metadata.columns = ["sample_id", "diagnosis", "participant_id"]
    metadata["sample_id"] = metadata["sample_id"].astype(str)
    metadata["participant_id"] = metadata["participant_id"].astype(str)
    metadata["cond"] = metadata["diagnosis"].map(normalize_condition)
    metadata = metadata[metadata["cond"].isin(["cd", "uc", "nonibd"])]
    metadata = metadata.drop_duplicates("sample_id", keep="first").copy()

    graph_files = {path.stem: path for path in sorted(graph_dir.glob("*.graphml"))}
    metadata = metadata[metadata["sample_id"].isin(graph_files)].copy()
    metadata = metadata.sort_values("sample_id").reset_index(drop=True)

    if len(metadata) != args.expected_samples:
        raise RuntimeError(f"Expected {args.expected_samples} matched samples; found {len(metadata)}")
    if metadata["participant_id"].nunique() != args.expected_participants:
        raise RuntimeError(
            f"Expected {args.expected_participants} participants; found {metadata['participant_id'].nunique()}"
        )

    sample_rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    for index, row in enumerate(metadata.itertuples(index=False), start=1):
        if index == 1 or index % 25 == 0 or index == len(metadata):
            print(f"[clustering] {index}/{len(metadata)} {row.sample_id}", flush=True)
        try:
            strengths = parse_active_edge_strengths(graph_files[row.sample_id], args.active_tol)
            result = exact_directed_clustering(strengths)
            result.update(
                {
                    "sample_id": row.sample_id,
                    "participant_id": row.participant_id,
                    "cond": row.cond,
                }
            )
            sample_rows.append(result)
        except Exception:
            failures.append({"sample_id": row.sample_id, "error": traceback.format_exc()})

    sample = pd.DataFrame(sample_rows)
    failure_frame = pd.DataFrame(failures, columns=["sample_id", "error"])
    failure_frame.to_csv(output_dir / "failed_samples.csv", index=False)
    if failures:
        raise RuntimeError(f"Parsing/computation failed for {len(failures)} samples; see failed_samples.csv")

    participant_rows: list[dict[str, Any]] = []
    for participant_id, subset in sample.groupby("participant_id", sort=True):
        conditions = subset["cond"].value_counts()
        condition = str(conditions.index[0])
        valid = subset[np.isfinite(pd.to_numeric(subset["directed_clustering"], errors="coerce"))]
        participant_rows.append(
            {
                "participant_id": participant_id,
                "cond": condition,
                "n_samples_total": int(len(subset)),
                "n_samples_valid": int(len(valid)),
                "n_active_vertices": float(subset["n_active_vertices"].mean()),
                "n_active_edges": float(subset["n_active_edges"].mean()),
                "directed_clustering": float(valid["directed_clustering"].mean()) if len(valid) else np.nan,
                "weighted_directed_clustering": (
                    float(valid["weighted_directed_clustering"].mean()) if len(valid) else np.nan
                ),
            }
        )

    participant = pd.DataFrame(participant_rows)
    summary = pd.concat(
        [
            summarize(sample, "sample_graphwise", args.sd_ddof),
            summarize(participant, "participant_average", args.sd_ddof),
        ],
        ignore_index=True,
    )

    sample.to_csv(output_dir / "directed_clustering_sample_level.csv", index=False)
    participant.to_csv(output_dir / "directed_clustering_participant_average_level.csv", index=False)
    summary.to_csv(output_dir / "directed_clustering_summary.csv", index=False)
    empty = sample[sample["n_active_edges"].eq(0)][["sample_id", "participant_id", "cond"]]
    empty.to_csv(output_dir / "empty_active_graphs.csv", index=False)
    write_latex(summary, output_dir / "directed_clustering_table.tex")

    config = {
        "analysis": "directed_clustering_active_participant_average",
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "metadata": {"path": str(metadata_path), "sha256": sha256_file(metadata_path)},
        "graph_dir": str(graph_dir),
        "matched_samples": int(len(sample)),
        "participants": int(len(participant)),
        "empty_active_graphs": int(len(empty)),
        "active_edge_rule": f"w < 1 - {args.active_tol}",
        "active_vertices_only": True,
        "weighted_strength": "s = 1 - w without global max normalization",
        "duplicate_edge_rule": "retain minimum w / maximum support strength",
        "aggregation": "sample graph -> participant average -> condition summary",
        "sd_ddof": int(args.sd_ddof),
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "platform": platform.platform(),
        },
    }
    (output_dir / "run_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    output_manifest(output_dir)

    print("\n" + summary.to_string(index=False), flush=True)
    print(f"\nEmpty active graphs: {len(empty)}", flush=True)
    print(f"Output: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
