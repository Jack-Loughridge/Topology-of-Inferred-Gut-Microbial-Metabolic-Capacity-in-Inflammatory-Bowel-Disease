#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
import re
import tempfile
import time
import traceback
import xml.etree.ElementTree as ET
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


WEIGHT_KEYS = {"weight", "d1", "Weight", "D1"}

DEFAULT_STEPS_FOR_TABLE = [0, 1, 2, 3, 5, 8]


def normalize_condition(x: Any) -> str:
    s = str(x).strip().lower()
    if "crohn" in s or s == "cd":
        return "cd"
    if "ulcerative" in s or s == "uc":
        return "uc"
    if "non" in s:
        return "nonibd"
    return s


def display_condition(c: str) -> str:
    return {
        "overall": "overall",
        "cd": "CD",
        "uc": "UC",
        "nonibd": "non-IBD",
    }.get(str(c), str(c))


def safe_filename(x: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(x))


def stable_seed(global_seed: int, sample_id: str) -> int:
    h = hashlib.sha256(f"{global_seed}::{sample_id}".encode("utf-8")).hexdigest()
    return int(h[:16], 16) % (2**32)


def atomic_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=str(path.parent), suffix=".tmp") as handle:
        json.dump(obj, handle)
        tmp = Path(handle.name)
    tmp.replace(path)


def parse_active_graphml(path: Path, active_tol: float) -> tuple[list[str], dict[str, list[str]], dict[tuple[str, str], float]]:
    """
    Parse a directed GraphML sample graph and keep only active directed edges.

    Active edge:
        w < 1 - active_tol

    If duplicate directed edges occur, the smaller weight is retained.
    """
    key_id_to_name: dict[str, str] = {}
    weight_key_ids = set(WEIGHT_KEYS)
    edge_w: dict[tuple[str, str], float] = {}

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

        elif tag == "edge":
            u = elem.attrib.get("source")
            v = elem.attrib.get("target")
            w = None

            for child in elem:
                child_tag = child.tag.rsplit("}", 1)[-1]
                if child_tag != "data":
                    continue

                key = child.attrib.get("key")
                key_name = key_id_to_name.get(key, key)

                if key in weight_key_ids or key_name in WEIGHT_KEYS:
                    try:
                        w = float(child.text)
                    except Exception:
                        w = None
                    break

            if u is not None and v is not None and w is not None and math.isfinite(w):
                if w < 1.0 - active_tol:
                    edge = (u, v)
                    if edge not in edge_w or w < edge_w[edge]:
                        edge_w[edge] = float(w)

            elem.clear()

    active_nodes = set()
    for u, v in edge_w:
        active_nodes.add(u)
        active_nodes.add(v)

    adjacency: dict[str, list[str]] = {node: [] for node in active_nodes}
    for u, v in edge_w:
        adjacency.setdefault(u, []).append(v)
        adjacency.setdefault(v, adjacency.get(v, []))

    for u in adjacency:
        adjacency[u].sort()

    if not edge_w:
        raise ValueError(
            f"No active edges parsed from {path}; the sample has no edges with "
            f"w < {1.0 - active_tol:.16g}."
        )

    return sorted(active_nodes), adjacency, edge_w
def bfs_reachable_paths(adjacency: dict[str, list[str]], source: str) -> dict[str, list[str]]:
    """
    Minimum-hop reachability paths, used only to identify reachable target candidates.
    """
    predecessor: dict[str, str | None] = {source: None}
    queue = deque([source])

    while queue:
        u = queue.popleft()
        for v in adjacency.get(u, []):
            if v not in predecessor:
                predecessor[v] = u
                queue.append(v)

    paths: dict[str, list[str]] = {}

    for target in predecessor:
        if target == source:
            continue

        rev = [target]
        cur = target

        while predecessor[cur] is not None:
            cur = predecessor[cur]  # type: ignore[assignment]
            rev.append(cur)

        paths[target] = list(reversed(rev))

    return paths


def minimax_paths_multi(
    adjacency: dict[str, list[str]],
    edge_w: dict[tuple[str, str], float],
    source: str,
    targets: list[str],
) -> dict[str, list[str]]:
    """
    Minimax paths from one source to several targets.

    Cost is lexicographic:
        (maximum edge weight on path, path length)
    """
    target_set = set(targets)
    if not target_set:
        return {}

    dist: dict[str, tuple[float, int]] = {source: (0.0, 0)}
    predecessor: dict[str, str | None] = {source: None}
    heap: list[tuple[float, int, str]] = [(0.0, 0, source)]
    settled = set()
    remaining = set(target_set)

    while heap and remaining:
        bottleneck, length, u = heapq.heappop(heap)

        if u in settled:
            continue
        if dist.get(u) != (bottleneck, length):
            continue

        settled.add(u)
        remaining.discard(u)

        for v in adjacency.get(u, []):
            e = (u, v)
            w = edge_w[e]
            new_cost = (max(bottleneck, w), length + 1)
            old = dist.get(v)

            if old is None or new_cost < old:
                dist[v] = new_cost
                predecessor[v] = u
                heapq.heappush(heap, (new_cost[0], new_cost[1], v))

    paths: dict[str, list[str]] = {}

    for target in target_set:
        if target not in predecessor:
            continue

        rev = [target]
        cur = target

        while predecessor[cur] is not None:
            cur = predecessor[cur]  # type: ignore[assignment]
            rev.append(cur)

        paths[target] = list(reversed(rev))

    return paths


def minimax_path_single_blocked(
    adjacency: dict[str, list[str]],
    edge_w: dict[tuple[str, str], float],
    source: str,
    target: str,
    blocked_edges: set[tuple[str, str]],
) -> list[str] | None:
    """
    Minimax path from source to target after deleting blocked_edges.
    """
    if source == target:
        return [source]

    dist: dict[str, tuple[float, int]] = {source: (0.0, 0)}
    predecessor: dict[str, str | None] = {source: None}
    heap: list[tuple[float, int, str]] = [(0.0, 0, source)]
    settled = set()

    while heap:
        bottleneck, length, u = heapq.heappop(heap)

        if u in settled:
            continue
        if dist.get(u) != (bottleneck, length):
            continue

        settled.add(u)

        if u == target:
            break

        for v in adjacency.get(u, []):
            e = (u, v)
            if e in blocked_edges:
                continue

            w = edge_w[e]
            new_cost = (max(bottleneck, w), length + 1)
            old = dist.get(v)

            if old is None or new_cost < old:
                dist[v] = new_cost
                predecessor[v] = u
                heapq.heappush(heap, (new_cost[0], new_cost[1], v))

    if target not in predecessor:
        return None

    rev = [target]
    cur = target

    while predecessor[cur] is not None:
        cur = predecessor[cur]  # type: ignore[assignment]
        rev.append(cur)

    return list(reversed(rev))


def path_edges(path: list[str]) -> list[tuple[str, str]]:
    return [(path[i], path[i + 1]) for i in range(len(path) - 1)]


def path_metrics(path: list[str], edge_w: dict[tuple[str, str], float]) -> dict[str, float]:
    edges = path_edges(path)

    if not edges:
        return {
            "path_length": 0.0,
            "bottleneck_w": 0.0,
            "mean_w2": 0.0,
        }

    weights = np.asarray([edge_w[e] for e in edges], dtype=float)

    return {
        "path_length": float(len(edges)),
        "bottleneck_w": float(np.max(weights)),
        "mean_w2": float(np.mean(weights ** 2)),
    }


def count_path_edges(counter: Counter, path: list[str]) -> None:
    for e in path_edges(path):
        counter[e] += 1


def edge_tie_key(edge: tuple[str, str]) -> str:
    return f"{edge[0]}\x1f{edge[1]}"


def choose_routing_attack_edge(path: list[str], baseline_counts: Counter) -> tuple[str, str] | None:
    """
    Remove the edge in the current path with highest baseline routing frequency.

    Ties are broken deterministically using the edge labels, not the edge weights.
    """
    edges = path_edges(path)
    if not edges:
        return None

    return max(edges, key=lambda e: (baseline_counts.get(e, 0), edge_tie_key(e)))


def append_record(
    acc: dict[int, dict[str, list[float]]],
    step: int,
    reachable: bool,
    metrics: dict[str, float] | None,
    length_failure_value: float,
) -> None:
    acc[step]["n_pairs"].append(1.0)
    acc[step]["reachable"].append(1.0 if reachable else 0.0)

    if reachable and metrics is not None:
        acc[step]["finite_path_length"].append(metrics["path_length"])
        acc[step]["finite_bottleneck_w"].append(metrics["bottleneck_w"])
        acc[step]["finite_mean_w2"].append(metrics["mean_w2"])

        acc[step]["penalized_path_length"].append(metrics["path_length"])
        acc[step]["penalized_bottleneck_w"].append(metrics["bottleneck_w"])
        acc[step]["penalized_mean_w2"].append(metrics["mean_w2"])
    else:
        acc[step]["penalized_path_length"].append(float(length_failure_value))
        acc[step]["penalized_bottleneck_w"].append(1.0)
        acc[step]["penalized_mean_w2"].append(1.0)


def summarise_accumulator(
    acc: dict[int, dict[str, list[float]]],
    base: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []

    for step in sorted(acc):
        d = acc[step]
        n_pairs = len(d["n_pairs"])
        reachable = np.asarray(d["reachable"], dtype=float)

        finite_length = np.asarray(d["finite_path_length"], dtype=float)
        finite_bottleneck = np.asarray(d["finite_bottleneck_w"], dtype=float)
        finite_mean_w2 = np.asarray(d["finite_mean_w2"], dtype=float)

        penalized_length = np.asarray(d["penalized_path_length"], dtype=float)
        penalized_bottleneck = np.asarray(d["penalized_bottleneck_w"], dtype=float)
        penalized_mean_w2 = np.asarray(d["penalized_mean_w2"], dtype=float)

        row = dict(base)
        row["step"] = int(step)
        row["n_pairs"] = int(n_pairs)
        row["reachable_count"] = int(np.sum(reachable))
        row["reachable_fraction"] = float(np.mean(reachable)) if n_pairs else np.nan
        row["failure_fraction"] = 1.0 - row["reachable_fraction"] if n_pairs else np.nan

        row["finite_path_length_mean"] = float(np.mean(finite_length)) if finite_length.size else np.nan
        row["finite_path_length_median"] = float(np.median(finite_length)) if finite_length.size else np.nan
        row["finite_path_length_max"] = float(np.max(finite_length)) if finite_length.size else np.nan

        row["finite_bottleneck_w_mean"] = float(np.mean(finite_bottleneck)) if finite_bottleneck.size else np.nan
        row["finite_bottleneck_w_median"] = float(np.median(finite_bottleneck)) if finite_bottleneck.size else np.nan

        row["finite_mean_w2_mean"] = float(np.mean(finite_mean_w2)) if finite_mean_w2.size else np.nan
        row["finite_mean_w2_median"] = float(np.median(finite_mean_w2)) if finite_mean_w2.size else np.nan

        row["penalized_path_length_mean"] = float(np.mean(penalized_length)) if penalized_length.size else np.nan
        row["penalized_path_length_median"] = float(np.median(penalized_length)) if penalized_length.size else np.nan

        row["penalized_bottleneck_w_mean"] = float(np.mean(penalized_bottleneck)) if penalized_bottleneck.size else np.nan
        row["penalized_bottleneck_w_median"] = float(np.median(penalized_bottleneck)) if penalized_bottleneck.size else np.nan

        row["penalized_mean_w2_mean"] = float(np.mean(penalized_mean_w2)) if penalized_mean_w2.size else np.nan
        row["penalized_mean_w2_median"] = float(np.median(penalized_mean_w2)) if penalized_mean_w2.size else np.nan

        rows.append(row)

    return rows


def process_one_sample(task: dict[str, Any]) -> dict[str, Any]:
    try:
        sample_id = task["sample_id"]
        rng = np.random.default_rng(task["seed"])

        nodes, adjacency, edge_w = parse_active_graphml(Path(task["graph_path"]), task["active_tol"])

        candidate_sources = [u for u in nodes if len(adjacency.get(u, [])) > 0]
        if candidate_sources:
            sampled_sources = list(
                rng.choice(
                    candidate_sources,
                    size=min(task["n_source"], len(candidate_sources)),
                    replace=False,
                )
            )
        else:
            sampled_sources = []

        baseline_pairs: list[dict[str, Any]] = []
        baseline_counts: Counter = Counter()
        reachable_fractions = []

        for source in sampled_sources:
            reachable_paths = bfs_reachable_paths(adjacency, source)
            reachable_targets = list(reachable_paths.keys())

            if not reachable_targets:
                reachable_fractions.append(0.0)
                continue

            reachable_fractions.append(len(reachable_targets) / max(1, len(nodes) - 1))

            targets = list(
                rng.choice(
                    reachable_targets,
                    size=min(task["n_target"], len(reachable_targets)),
                    replace=False,
                )
            )

            baseline_minimax = minimax_paths_multi(adjacency, edge_w, source, targets)

            for target in targets:
                path = baseline_minimax.get(target)
                if path is None:
                    continue

                baseline_pairs.append(
                    {
                        "source": source,
                        "target": target,
                        "path": path,
                    }
                )
                count_path_edges(baseline_counts, path)

        acc: dict[int, dict[str, list[float]]] = {
            step: {
                "n_pairs": [],
                "reachable": [],
                "finite_path_length": [],
                "finite_bottleneck_w": [],
                "finite_mean_w2": [],
                "penalized_path_length": [],
                "penalized_bottleneck_w": [],
                "penalized_mean_w2": [],
            }
            for step in range(task["k_max"] + 1)
        }

        for pair in baseline_pairs:
            source = pair["source"]
            target = pair["target"]
            current_path = pair["path"]
            removed_edges: set[tuple[str, str]] = set()

            # Step 0: baseline path, no removals.
            append_record(
                acc=acc,
                step=0,
                reachable=True,
                metrics=path_metrics(current_path, edge_w),
                length_failure_value=task["length_failure_value"],
            )

            failed = False

            for step in range(1, task["k_max"] + 1):
                if failed or current_path is None:
                    append_record(
                        acc=acc,
                        step=step,
                        reachable=False,
                        metrics=None,
                        length_failure_value=task["length_failure_value"],
                    )
                    failed = True
                    continue

                attack_edge = choose_routing_attack_edge(current_path, baseline_counts)

                if attack_edge is None:
                    append_record(
                        acc=acc,
                        step=step,
                        reachable=False,
                        metrics=None,
                        length_failure_value=task["length_failure_value"],
                    )
                    current_path = None
                    failed = True
                    continue

                removed_edges.add(attack_edge)

                new_path = minimax_path_single_blocked(
                    adjacency=adjacency,
                    edge_w=edge_w,
                    source=source,
                    target=target,
                    blocked_edges=removed_edges,
                )

                if new_path is None:
                    append_record(
                        acc=acc,
                        step=step,
                        reachable=False,
                        metrics=None,
                        length_failure_value=task["length_failure_value"],
                    )
                    current_path = None
                    failed = True
                else:
                    current_path = new_path
                    append_record(
                        acc=acc,
                        step=step,
                        reachable=True,
                        metrics=path_metrics(current_path, edge_w),
                        length_failure_value=task["length_failure_value"],
                    )

        common = {
            "sample_id": sample_id,
            "participant_id": task["participant_id"],
            "cond": task["cond"],
            "path_type": "minimax",
            "attack_rule": "highest_baseline_routing_frequency_edge_in_current_path",
            "k_max": task["k_max"],
            "n_active_vertices": len(nodes),
            "n_active_edges": len(edge_w),
            "n_sources_sampled": len(sampled_sources),
            "n_baseline_pairs": len(baseline_pairs),
            "n_source_requested": task["n_source"],
            "n_target_requested_per_source": task["n_target"],
            "length_failure_value": task["length_failure_value"],
            "mean_reachable_fraction_sampled_sources": (
                float(np.mean(reachable_fractions)) if reachable_fractions else np.nan
            ),
        }

        rows = summarise_accumulator(acc, common)

        return {
            "sample_id": sample_id,
            "rows": rows,
            "error": None,
        }

    except Exception:
        return {
            "sample_id": task.get("sample_id", "unknown"),
            "rows": [],
            "error": traceback.format_exc(),
        }


def build_tasks(args: argparse.Namespace) -> list[dict[str, Any]]:
    metadata = pd.read_csv(args.metadata, low_memory=False)

    required = {"External ID", "diagnosis", "Participant ID"}
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f"Metadata is missing columns: {sorted(missing)}")

    metadata = metadata[["External ID", "diagnosis", "Participant ID"]].copy()
    metadata.columns = ["sample_id", "diagnosis", "participant_id"]
    metadata["cond"] = metadata["diagnosis"].apply(normalize_condition)
    metadata = metadata[metadata["cond"].isin(["cd", "uc", "nonibd"])].drop_duplicates("sample_id").copy()

    graph_files = {p.stem: p for p in Path(args.graph_dir).glob("*.graphml")}
    metadata = metadata[metadata["sample_id"].isin(graph_files)].copy()

    if args.max_samples is not None:
        metadata = metadata.head(args.max_samples).copy()

    print("Metadata samples matched to graphs:", len(metadata), flush=True)
    print("Participants:", metadata["participant_id"].nunique(), flush=True)
    print("Condition counts:", flush=True)
    print(metadata["cond"].value_counts().to_string(), flush=True)

    sample_dir = Path(args.output_dir) / "sample_results"
    sample_dir.mkdir(parents=True, exist_ok=True)

    tasks = []
    for _, row in metadata.iterrows():
        sample_id = str(row["sample_id"])
        out_json = sample_dir / f"{safe_filename(sample_id)}.json"

        if args.resume and out_json.exists() and out_json.stat().st_size > 0:
            continue

        tasks.append(
            {
                "sample_id": sample_id,
                "participant_id": str(row["participant_id"]),
                "cond": str(row["cond"]),
                "graph_path": str(graph_files[sample_id]),
                "n_source": int(args.n_source),
                "n_target": int(args.n_target),
                "k_max": int(args.k_max),
                "active_tol": float(args.active_tol),
                "length_failure_value": float(args.length_failure_value),
                "seed": stable_seed(int(args.seed), sample_id),
                "out_json": str(out_json),
            }
        )

    print("Samples still to process:", len(tasks), flush=True)
    return tasks


def load_sample_jsons(sample_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    errors = []

    for path in sorted(sample_dir.glob("*.json")):
        obj = json.loads(path.read_text())
        if obj.get("error"):
            errors.append({"sample_id": obj.get("sample_id"), "error": obj.get("error")})
        else:
            rows.extend(obj.get("rows", []))

    return pd.DataFrame(rows), pd.DataFrame(errors)


def participant_average(sample_df: pd.DataFrame) -> pd.DataFrame:
    if sample_df.empty:
        return sample_df.copy()

    group_cols = ["participant_id", "step"]
    skip = set(group_cols + ["sample_id", "cond", "path_type", "attack_rule"])
    numeric_cols = [
        c for c in sample_df.columns
        if c not in skip and pd.api.types.is_numeric_dtype(sample_df[c])
    ]

    rows = []
    for keys, sub in sample_df.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys))
        row["cond"] = sub["cond"].astype(str).value_counts().idxmax()
        row["path_type"] = "minimax"
        row["attack_rule"] = "highest_baseline_routing_frequency_edge_in_current_path"
        row["n_samples_for_participant"] = int(sub["sample_id"].nunique())

        for c in numeric_cols:
            vals = pd.to_numeric(sub[c], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            row[c] = float(vals.mean()) if len(vals) else np.nan

        rows.append(row)

    return pd.DataFrame(rows)


def condition_summary(sample_df: pd.DataFrame, participant_df: pd.DataFrame, sd_ddof: int = 0) -> pd.DataFrame:
    rows = []

    for level, df in [("sample graphwise", sample_df), ("participant average", participant_df)]:
        if df.empty:
            continue

        skip = {"sample_id", "participant_id", "cond", "path_type", "attack_rule"}
        numeric_cols = [
            c for c in df.columns
            if c not in skip and pd.api.types.is_numeric_dtype(df[c])
        ]

        for step in sorted(df["step"].dropna().unique()):
            step_df = df[df["step"] == step]

            for cond in ["overall", "cd", "uc", "nonibd"]:
                sub = step_df if cond == "overall" else step_df[step_df["cond"] == cond]
                if len(sub) == 0:
                    continue

                row = {
                    "level": level,
                    "cond": cond,
                    "step": int(step),
                    "n": int(len(sub)),
                }

                for c in numeric_cols:
                    vals = pd.to_numeric(sub[c], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
                    row[f"{c}_mean"] = float(vals.mean()) if len(vals) else np.nan
                    row[f"{c}_sd"] = float(vals.std(ddof=sd_ddof)) if len(vals) else np.nan

                rows.append(row)

    return pd.DataFrame(rows)


def fmt_mean_sd(row: pd.Series, metric: str, digits: int = 3) -> str:
    mean = row.get(f"{metric}_mean", np.nan)
    sd = row.get(f"{metric}_sd", np.nan)

    try:
        mean = float(mean)
        sd = float(sd)
    except Exception:
        return "--"

    if not math.isfinite(mean):
        return "--"
    if not math.isfinite(sd):
        return f"{mean:.{digits}f}"

    return f"{mean:.{digits}f} $\\pm$ {sd:.{digits}f}"


def write_metric_latex_table(
    summary_df: pd.DataFrame,
    metric: str,
    caption: str,
    label: str,
    out_path: Path,
    steps: list[int],
    digits: int = 3,
) -> None:
    df = summary_df[summary_df["level"] == "participant average"].copy()
    cond_order = ["overall", "cd", "uc", "nonibd"]

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        rf"\caption{{{caption} Participant-level values are obtained by averaging sample-level diagnostics within participant; values are mean $\pm$ SD across participants.}}",
        rf"\label{{{label}}}",
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r"Condition & $n$ & $k=0$ & $k=1$ & $k=2$ & $k=3$ & $k=5$ & $k=8$ \\",
        r"\midrule",
    ]

    for cond in cond_order:
        sub = df[df["cond"] == cond].set_index("step")
        if sub.empty:
            continue

        n_val = int(sub["n"].max()) if "n" in sub.columns else len(sub)

        vals = []
        for step in steps:
            if step in sub.index:
                vals.append(fmt_mean_sd(sub.loc[step], metric, digits=digits))
            else:
                vals.append("--")

        lines.append(
            f"{display_condition(cond)} & {n_val} & "
            + " & ".join(vals)
            + r" \\"
        )

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    out_path.write_text("\n".join(lines) + "\n")


def write_all_latex_tables(summary_df: pd.DataFrame, output_dir: Path, k_max: int) -> None:
    steps = [s for s in DEFAULT_STEPS_FOR_TABLE if s <= k_max]

    table_specs = [
        (
            "reachable_fraction",
            "Reachable fraction under sequential routing-attack edge removal.",
            "tab:sequential_attack_reachable_fraction",
            output_dir / "latex_reachable_fraction_table.tex",
        ),
        (
            "penalized_bottleneck_w_mean",
            "Penalised bottleneck weight under sequential routing-attack edge removal. Infeasible source--target pairs are assigned bottleneck weight $1$.",
            "tab:sequential_attack_penalized_bottleneck_w",
            output_dir / "latex_penalized_bottleneck_weight_table.tex",
        ),
        (
            "penalized_path_length_mean",
            "Penalised path length under sequential routing-attack edge removal. Infeasible source--target pairs are assigned the fixed path-length penalty used in the run configuration.",
            "tab:sequential_attack_penalized_path_length",
            output_dir / "latex_penalized_path_length_table.tex",
        ),
        (
            "finite_path_length_mean",
            "Conditional path length under sequential routing-attack edge removal, computed only among source--target pairs that remain reachable.",
            "tab:sequential_attack_finite_path_length",
            output_dir / "latex_conditional_path_length_table.tex",
        ),
    ]

    combined_parts = []

    for metric, caption, label, path in table_specs:
        write_metric_latex_table(
            summary_df=summary_df,
            metric=metric,
            caption=caption,
            label=label,
            out_path=path,
            steps=steps,
            digits=3,
        )
        combined_parts.append(path.read_text())

    (output_dir / "latex_all_sequential_attack_tables.tex").write_text("\n\n".join(combined_parts) + "\n")


def plot_metric(
    summary_df: pd.DataFrame,
    metric: str,
    y_label: str,
    title: str,
    out_base: Path,
    y_min: float | None = None,
    y_max: float | None = None,
) -> None:
    df = summary_df[summary_df["level"] == "participant average"].copy()
    cond_order = ["overall", "cd", "uc", "nonibd"]

    fig, ax = plt.subplots(figsize=(7.0, 4.8))

    for cond in cond_order:
        sub = df[df["cond"] == cond].sort_values("step")
        if sub.empty:
            continue

        x = sub["step"].to_numpy(dtype=float)
        y = sub[f"{metric}_mean"].to_numpy(dtype=float)
        sd = sub[f"{metric}_sd"].to_numpy(dtype=float)

        line, = ax.plot(x, y, marker="o", linewidth=2.0, label=display_condition(cond))
        color = line.get_color()

        lower = y - sd
        upper = y + sd
        ax.fill_between(x, lower, upper, alpha=0.15, color=color, linewidth=0)

    ax.set_xlabel("Number of cumulative edge removals")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.set_xticks(sorted(df["step"].dropna().unique()))
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)

    if y_min is not None or y_max is not None:
        ax.set_ylim(y_min, y_max)

    fig.tight_layout()
    fig.savefig(str(out_base) + ".png", dpi=300)
    fig.savefig(str(out_base) + ".pdf")
    plt.close(fig)


def make_plots(summary_df: pd.DataFrame, output_dir: Path, length_failure_value: float) -> None:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    plot_metric(
        summary_df=summary_df,
        metric="reachable_fraction",
        y_label="Reachable fraction",
        title="Replacement-path survival under routing-attack removals",
        out_base=plot_dir / "reachable_fraction_vs_removals",
        y_min=0.0,
        y_max=1.03,
    )

    plot_metric(
        summary_df=summary_df,
        metric="penalized_bottleneck_w_mean",
        y_label="Penalised bottleneck weight",
        title="Penalised bottleneck weight under routing-attack removals",
        out_base=plot_dir / "penalized_bottleneck_weight_vs_removals",
        y_min=0.0,
        y_max=1.03,
    )

    plot_metric(
        summary_df=summary_df,
        metric="penalized_path_length_mean",
        y_label="Penalised path length",
        title="Penalised path length under routing-attack removals",
        out_base=plot_dir / "penalized_path_length_vs_removals",
        y_min=0.0,
        y_max=float(length_failure_value) * 1.05,
    )

    plot_metric(
        summary_df=summary_df,
        metric="finite_path_length_mean",
        y_label="Path length among reachable pairs",
        title="Conditional path length under routing-attack removals",
        out_base=plot_dir / "conditional_path_length_vs_removals",
        y_min=0.0,
        y_max=None,
    )


def aggregate_outputs(output_dir: Path, sd_ddof: int, k_max: int, length_failure_value: float) -> None:
    sample_dir = output_dir / "sample_results"

    sample_df, error_df = load_sample_jsons(sample_dir)
    participant_df = participant_average(sample_df)
    summary_df = condition_summary(sample_df, participant_df, sd_ddof=sd_ddof)

    sample_csv = output_dir / "sample_sequential_attack.csv"
    participant_csv = output_dir / "participant_sequential_attack.csv"
    summary_csv = output_dir / "condition_sequential_attack_summary.csv"
    error_csv = output_dir / "failed_samples.csv"

    sample_df.to_csv(sample_csv, index=False)
    participant_df.to_csv(participant_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    error_df.to_csv(error_csv, index=False)

    write_all_latex_tables(summary_df, output_dir, k_max=k_max)
    make_plots(summary_df, output_dir, length_failure_value=length_failure_value)

    print("\nWrote outputs to:", output_dir, flush=True)
    print(" sample rows:", len(sample_df), flush=True)
    print(" participant rows:", len(participant_df), flush=True)
    print(" summary rows:", len(summary_df), flush=True)
    print(" failed samples:", len(error_df), flush=True)

    if not sample_df.empty and "finite_path_length_max" in sample_df.columns:
        finite_max = pd.to_numeric(sample_df["finite_path_length_max"], errors="coerce").max()
        print(" max finite path length observed:", finite_max, flush=True)
        print(" path-length failure penalty:", length_failure_value, flush=True)

    print("\nMain plots:", flush=True)
    print(" ", output_dir / "plots" / "reachable_fraction_vs_removals.png", flush=True)
    print(" ", output_dir / "plots" / "penalized_bottleneck_weight_vs_removals.png", flush=True)
    print(" ", output_dir / "plots" / "penalized_path_length_vs_removals.png", flush=True)

    print("\nLaTeX tables:", flush=True)
    print(" ", output_dir / "latex_all_sequential_attack_tables.tex", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sequential routing-attack replacement-path diagnostic on directed metabolic graphs."
    )

    parser.add_argument("--graph-dir", default=str(Path.home() / "Real_Data" / "out_graphs"))
    parser.add_argument("--metadata", default=str(Path.home() / "Real_Data" / "hmp2_metadata.csv"))
    parser.add_argument("--output-dir", default=str(Path.home() / "Real_Data" / "Sequential_Routing_Attack_K8"))

    parser.add_argument("--n-source", type=int, default=10)
    parser.add_argument("--n-target", type=int, default=20)
    parser.add_argument("--k-max", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--active-tol", type=float, default=1e-12)
    parser.add_argument("--length-failure-value", type=float, default=25.0)
    parser.add_argument("--seed", type=int, default=20260708)
    parser.add_argument("--sd-ddof", type=int, default=0)

    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--write-only", action="store_true")

    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    args.resume = not args.no_resume

    config = vars(args).copy()
    config["output_dir"] = str(output_dir)
    config["protocol"] = "sample graph diagnostics -> participant averaging -> condition summary"
    config["path_type"] = "minimax"
    config["attack_rule"] = "At each step, remove the current-path edge with highest baseline path-use frequency."
    config["cumulative_removals"] = True
    config["infeasible_bottleneck_w"] = 1.0
    config["infeasible_path_length"] = args.length_failure_value
    config["notes"] = [
        "Lower edge weight w means stronger inferred microbial support.",
        "Penalised bottleneck weight assigns no-path cases to w=1.",
        "Penalised path length assigns no-path cases to the fixed length_failure_value.",
        "Reachable-fraction plot should be read together with penalised bottleneck/path-length plots.",
    ]

    (output_dir / "run_config.json").write_text(json.dumps(config, indent=2, default=str))

    print("=" * 100, flush=True)
    print("Sequential routing-attack replacement-path diagnostic", flush=True)
    print("=" * 100, flush=True)
    print("Protocol:", config["protocol"], flush=True)
    print("Path type:", config["path_type"], flush=True)
    print("Attack rule:", config["attack_rule"], flush=True)
    print(f"K={args.k_max}, n_source={args.n_source}, n_target={args.n_target}, workers={args.workers}", flush=True)
    print("Length failure value:", args.length_failure_value, flush=True)

    if args.write_only:
        aggregate_outputs(
            output_dir=output_dir,
            sd_ddof=args.sd_ddof,
            k_max=args.k_max,
            length_failure_value=args.length_failure_value,
        )
        return

    tasks = build_tasks(args)

    if tasks:
        start = time.time()
        done = 0
        errors = 0

        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(process_one_sample, task): task for task in tasks}

            for future in as_completed(futures):
                task = futures[future]

                try:
                    obj = future.result()
                except Exception:
                    obj = {
                        "sample_id": task["sample_id"],
                        "rows": [],
                        "error": traceback.format_exc(),
                    }

                if obj.get("error"):
                    errors += 1

                atomic_write_json(Path(task["out_json"]), obj)

                done += 1
                if done == 1 or done % 10 == 0 or done == len(tasks):
                    elapsed = time.time() - start
                    rate = done / elapsed if elapsed > 0 else np.nan
                    eta = (len(tasks) - done) / rate if rate and rate > 0 else np.nan

                    print(
                        f"[progress] {done}/{len(tasks)} | errors={errors} | "
                        f"rate={rate:.4f} samples/s | ETA={eta / 3600:.2f} h | latest={task['sample_id']}",
                        flush=True,
                    )

    print("\nAggregating...", flush=True)
    aggregate_outputs(
        output_dir=output_dir,
        sd_ddof=args.sd_ddof,
        k_max=args.k_max,
        length_failure_value=args.length_failure_value,
    )
    print("\n[READY] Sequential routing-attack diagnostic finished.", flush=True)


if __name__ == "__main__":
    main()
