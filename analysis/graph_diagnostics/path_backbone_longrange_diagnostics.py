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

import numpy as np
import pandas as pd


WEIGHT_KEYS = {"weight", "d1", "Weight", "D1"}
QUANTILES = [1, 5, 10, 20, 30, 50, 70, 80, 90, 95, 99]
TOP_FRACS = [0.01, 0.05, 0.10]


def norm_cond(x):
    s = str(x).strip().lower()
    if "crohn" in s or s == "cd":
        return "cd"
    if "ulcerative" in s or s == "uc":
        return "uc"
    if "non" in s:
        return "nonibd"
    return s


def disp_cond(c):
    return {"cd": "CD", "uc": "UC", "nonibd": "non-IBD", "overall": "overall"}.get(c, c)


def safe_name(x):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(x))


def stable_seed(seed, sample_id):
    h = hashlib.sha256(f"{seed}:{sample_id}".encode()).hexdigest()
    return int(h[:16], 16) % (2**32)


def atomic_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=str(path.parent), suffix=".tmp") as f:
        json.dump(obj, f)
        tmp = Path(f.name)
    tmp.replace(path)


def qdict(vals, prefix):
    arr = np.asarray(vals, dtype=float)
    arr = arr[np.isfinite(arr)]
    out = {}
    for q in QUANTILES:
        out[f"{prefix}_q{q}"] = float(np.percentile(arr, q)) if len(arr) else np.nan
    out[f"{prefix}_mean"] = float(np.mean(arr)) if len(arr) else np.nan
    out[f"{prefix}_sd"] = float(np.std(arr, ddof=0)) if len(arr) else np.nan
    return out


def parse_graphml_active(path: Path, active_tol: float):
    key_id_to_name = {}
    weight_key_ids = set(WEIGHT_KEYS)
    edge_w = {}
    nodes = set()

    for _, elem in ET.iterparse(path, events=("end",)):
        tag = elem.tag.rsplit("}", 1)[-1]

        if tag == "key":
            kid = elem.attrib.get("id")
            aname = elem.attrib.get("attr.name")
            if kid is not None and aname is not None:
                key_id_to_name[kid] = aname
                if kid in WEIGHT_KEYS or aname in WEIGHT_KEYS:
                    weight_key_ids.add(kid)
            elem.clear()

        elif tag == "node":
            nid = elem.attrib.get("id")
            if nid is not None:
                nodes.add(nid)
            elem.clear()

        elif tag == "edge":
            u = elem.attrib.get("source")
            v = elem.attrib.get("target")
            if u is not None:
                nodes.add(u)
            if v is not None:
                nodes.add(v)

            w = None
            for child in elem:
                ctag = child.tag.rsplit("}", 1)[-1]
                if ctag != "data":
                    continue
                k = child.attrib.get("key")
                kname = key_id_to_name.get(k, k)
                if k in weight_key_ids or kname in WEIGHT_KEYS:
                    try:
                        w = float(child.text)
                    except Exception:
                        w = None
                    break

            if u is not None and v is not None and w is not None and math.isfinite(w):
                if w < 1.0 - active_tol:
                    e = (u, v)
                    if e not in edge_w or w < edge_w[e]:
                        edge_w[e] = float(w)

            elem.clear()

    active_nodes = set()
    for u, v in edge_w:
        active_nodes.add(u)
        active_nodes.add(v)

    adj = {n: [] for n in active_nodes}
    for u, v in edge_w:
        adj.setdefault(u, []).append(v)
        adj.setdefault(v, adj.get(v, []))

    for u in adj:
        adj[u].sort()

    return sorted(active_nodes), adj, edge_w


def bfs_paths(adj, source):
    pred = {source: None}
    q = deque([source])

    while q:
        u = q.popleft()
        for v in adj.get(u, []):
            if v not in pred:
                pred[v] = u
                q.append(v)

    paths = {}
    for t in pred:
        if t == source:
            continue
        rev = [t]
        cur = t
        while pred[cur] is not None:
            cur = pred[cur]
            rev.append(cur)
        paths[t] = list(reversed(rev))
    return paths


def minimax_paths(adj, edge_w, source, targets):
    targets = set(targets)
    if not targets:
        return {}

    dist = {source: (0.0, 0)}
    pred = {source: None}
    heap = [(0.0, 0, source)]
    settled = set()
    remaining = set(targets)

    while heap and remaining:
        b, length, u = heapq.heappop(heap)
        if u in settled:
            continue
        if dist.get(u) != (b, length):
            continue

        settled.add(u)
        remaining.discard(u)

        for v in adj.get(u, []):
            w = edge_w[(u, v)]
            nb = max(b, w)
            nl = length + 1
            old = dist.get(v)
            if old is None or (nb, nl) < old:
                dist[v] = (nb, nl)
                pred[v] = u
                heapq.heappush(heap, (nb, nl, v))

    paths = {}
    for t in targets:
        if t not in pred:
            continue
        rev = [t]
        cur = t
        while pred[cur] is not None:
            cur = pred[cur]
            rev.append(cur)
        paths[t] = list(reversed(rev))
    return paths


def metrics_for_path(path, edge_w):
    ws = [edge_w[(path[i], path[i + 1])] for i in range(len(path) - 1)]
    if not ws:
        return None
    b = max(ws)
    return {
        "path_length": float(len(ws)),
        "bottleneck_w": float(b),
        "min_support": float(1.0 - b),
        "mean_w2": float(np.mean(np.asarray(ws) ** 2)),
    }


def count_edges(counter, path):
    for i in range(len(path) - 1):
        counter[(path[i], path[i + 1])] += 1


def summarize_path_metrics(rows, base):
    out = dict(base)
    out["n_paths"] = len(rows)
    for m in ["path_length", "bottleneck_w", "min_support", "mean_w2"]:
        out.update(qdict([r[m] for r in rows], m))
    return out


def summarize_edge_use(counter, edge_w, base):
    out = dict(base)
    active_edges = list(edge_w.keys())
    n_active = len(active_edges)
    total = int(sum(counter.values()))
    used = [e for e, c in counter.items() if c > 0]

    out["total_edge_traversals"] = total
    out["unique_edges_used"] = len(used)
    out["unique_edges_used_fraction_of_active"] = len(used) / n_active if n_active else np.nan

    if total > 0:
        shares = np.asarray([counter[e] / total for e in used], dtype=float)
        hhi = float(np.sum(shares ** 2))
        out["edge_traversal_hhi"] = hhi
        out["effective_edges"] = 1.0 / hhi if hhi > 0 else np.nan
        out["effective_edges_fraction_of_active"] = (1.0 / hhi) / n_active if hhi > 0 and n_active else np.nan
    else:
        out["edge_traversal_hhi"] = np.nan
        out["effective_edges"] = np.nan
        out["effective_edges_fraction_of_active"] = np.nan

    ranked = sorted(active_edges, key=lambda e: counter.get(e, 0), reverse=True)

    for frac in TOP_FRACS:
        pct = int(round(frac * 100))
        k = max(1, int(math.ceil(frac * n_active))) if n_active else 0
        top = ranked[:k]
        top_used = [e for e in top if counter.get(e, 0) > 0]
        prefix = f"top{pct}pct"

        out[f"{prefix}_traversal_share"] = (
            sum(counter.get(e, 0) for e in top) / total if total > 0 else np.nan
        )
        out[f"{prefix}_n_edges"] = k
        out[f"{prefix}_n_used_edges"] = len(top_used)

        if top_used:
            ws = np.asarray([edge_w[e] for e in top_used], dtype=float)
            ss = 1.0 - ws
            counts = np.asarray([counter[e] for e in top_used], dtype=float)
            out[f"{prefix}_median_w_top_used"] = float(np.median(ws))
            out[f"{prefix}_median_support_top_used"] = float(np.median(ss))
            out[f"{prefix}_usage_weighted_mean_w_top_used"] = float(np.average(ws, weights=counts))
            out[f"{prefix}_usage_weighted_mean_support_top_used"] = float(np.average(ss, weights=counts))
        else:
            out[f"{prefix}_median_w_top_used"] = np.nan
            out[f"{prefix}_median_support_top_used"] = np.nan
            out[f"{prefix}_usage_weighted_mean_w_top_used"] = np.nan
            out[f"{prefix}_usage_weighted_mean_support_top_used"] = np.nan

    return out


def process_sample(task):
    try:
        sid = task["sample_id"]
        rng = np.random.default_rng(task["seed"])

        nodes, adj, edge_w = parse_graphml_active(Path(task["graph_path"]), task["active_tol"])
        n_vertices = len(nodes)
        n_edges = len(edge_w)

        base = {
            "sample_id": sid,
            "participant_id": task["participant_id"],
            "cond": task["cond"],
            "n_active_vertices": n_vertices,
            "n_active_edges": n_edges,
            "n_source_requested": task["n_source"],
            "n_target_requested_per_source": task["n_target"],
            "long_min_hop": task["long_min_hop"],
        }

        sources = [u for u in nodes if len(adj.get(u, [])) > 0]
        if sources:
            sources = list(rng.choice(sources, size=min(task["n_source"], len(sources)), replace=False))

        hop_all, hop_long = [], []
        mm_all, mm_long = [], []
        hop_edges_all, hop_edges_long = Counter(), Counter()
        mm_edges_all, mm_edges_long = Counter(), Counter()
        reach_fracs = []
        pair_count = 0

        for s in sources:
            hop_paths = bfs_paths(adj, s)
            targets = list(hop_paths.keys())
            if not targets:
                reach_fracs.append(0.0)
                continue

            reach_fracs.append(len(targets) / max(1, n_vertices - 1))
            targets = list(rng.choice(targets, size=min(task["n_target"], len(targets)), replace=False))
            pair_count += len(targets)

            long_targets = set()

            for t in targets:
                p = hop_paths[t]
                m = metrics_for_path(p, edge_w)
                if m is None:
                    continue
                hop_all.append(m)
                count_edges(hop_edges_all, p)
                if m["path_length"] >= task["long_min_hop"]:
                    hop_long.append(m)
                    count_edges(hop_edges_long, p)
                    long_targets.add(t)

            mm_paths = minimax_paths(adj, edge_w, s, targets)

            for t in targets:
                p = mm_paths.get(t)
                if p is None:
                    continue
                m = metrics_for_path(p, edge_w)
                if m is None:
                    continue
                mm_all.append(m)
                count_edges(mm_edges_all, p)
                if t in long_targets:
                    mm_long.append(m)
                    count_edges(mm_edges_long, p)

        reach_mean = float(np.mean(reach_fracs)) if reach_fracs else np.nan

        path_rows = []
        edge_rows = []

        combos = [
            ("minimum_hop", "all", hop_all, hop_edges_all),
            ("minimum_hop", f"long_hop_ge{task['long_min_hop']}", hop_long, hop_edges_long),
            ("minimax", "all", mm_all, mm_edges_all),
            ("minimax", f"long_hop_ge{task['long_min_hop']}", mm_long, mm_edges_long),
        ]

        for mode, path_set, vals, counter in combos:
            b = dict(base)
            b.update({
                "mode": mode,
                "path_set": path_set,
                "n_sources_sampled": len(sources),
                "n_ordered_source_target_pairs": pair_count,
                "mean_reachable_fraction_sampled_sources": reach_mean,
            })
            path_rows.append(summarize_path_metrics(vals, b))
            edge_rows.append(summarize_edge_use(counter, edge_w, b))

        return {"sample_id": sid, "path_rows": path_rows, "edge_rows": edge_rows, "error": None}

    except Exception:
        return {
            "sample_id": task.get("sample_id", "unknown"),
            "path_rows": [],
            "edge_rows": [],
            "error": traceback.format_exc(),
        }


def load_jsons(sample_dir):
    path_rows, edge_rows, errors = [], [], []
    for p in sorted(sample_dir.glob("*.json")):
        obj = json.loads(p.read_text())
        if obj.get("error"):
            errors.append({"sample_id": obj.get("sample_id"), "error": obj.get("error")})
        else:
            path_rows.extend(obj.get("path_rows", []))
            edge_rows.extend(obj.get("edge_rows", []))
    return pd.DataFrame(path_rows), pd.DataFrame(edge_rows), pd.DataFrame(errors)


def participant_average(df):
    if df.empty:
        return df.copy()

    id_cols = ["participant_id", "mode", "path_set"]
    skip = set(id_cols + ["sample_id", "cond"])
    num_cols = [c for c in df.columns if c not in skip and pd.api.types.is_numeric_dtype(df[c])]

    rows = []
    for keys, sub in df.groupby(id_cols):
        row = dict(zip(id_cols, keys))
        row["cond"] = sub["cond"].astype(str).value_counts().idxmax()
        row["n_samples_for_participant"] = sub["sample_id"].nunique()
        for c in num_cols:
            vals = pd.to_numeric(sub[c], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            row[c] = float(vals.mean()) if len(vals) else np.nan
        rows.append(row)

    return pd.DataFrame(rows)


def condition_summary(sample_df, participant_df):
    rows = []

    for level, df in [("sample graphwise", sample_df), ("participant average", participant_df)]:
        if df.empty:
            continue

        skip = {"sample_id", "participant_id", "cond", "mode", "path_set"}
        num_cols = [c for c in df.columns if c not in skip and pd.api.types.is_numeric_dtype(df[c])]

        for mode in sorted(df["mode"].unique()):
            for path_set in sorted(df["path_set"].unique()):
                base_sub = df[(df["mode"] == mode) & (df["path_set"] == path_set)]
                for cond in ["overall", "cd", "uc", "nonibd"]:
                    sub = base_sub if cond == "overall" else base_sub[base_sub["cond"] == cond]
                    if len(sub) == 0:
                        continue
                    valid_col = "n_paths" if "n_paths" in sub.columns else "total_edge_traversals"
                    valid = pd.to_numeric(sub[valid_col], errors="coerce").fillna(0).gt(0)
                    row = {
                        "level": level,
                        "mode": mode,
                        "path_set": path_set,
                        "cond": cond,
                        "n": int(len(sub)),
                        "n_total": int(len(sub)),
                        "n_valid": int(valid.sum()),
                    }
                    for c in num_cols:
                        vals = pd.to_numeric(sub[c], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
                        row[f"{c}_mean"] = float(vals.mean()) if len(vals) else np.nan
                        row[f"{c}_sd"] = float(vals.std(ddof=0)) if len(vals) else np.nan
                    rows.append(row)

    return pd.DataFrame(rows)


def fmt(row, col):
    m = row.get(f"{col}_mean", np.nan)
    s = row.get(f"{col}_sd", np.nan)
    if not math.isfinite(float(m)):
        return "--"
    if not math.isfinite(float(s)):
        return f"{float(m):.3f}"
    return f"{float(m):.3f} $\\pm$ {float(s):.3f}"


def fmt_n(row):
    valid = row.get("n_valid", row.get("n", np.nan))
    total = row.get("n_total", row.get("n", np.nan))
    return f"{int(valid)}/{int(total)}"


def write_latex_quantile(summary, metric, path_set, out_path, caption, label):
    sub = summary[(summary["path_set"] == path_set) & (summary["mode"].isin(["minimum_hop", "minimax"]))].copy()
    order_level = {"sample graphwise": 0, "participant average": 1}
    order_mode = {"minimum_hop": 0, "minimax": 1}
    order_cond = {"overall": 0, "cd": 1, "uc": 2, "nonibd": 3}
    sub["_a"] = sub["level"].map(order_level)
    sub["_b"] = sub["mode"].map(order_mode)
    sub["_c"] = sub["cond"].map(order_cond)
    sub = sub.sort_values(["_a", "_b", "_c"])

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        caption,
        label,
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lll" + "r" * (1 + len(QUANTILES)) + r"}",
        r"\toprule",
        "Level & Path type & Condition & $n$ valid/total & " + " & ".join([f"q{q}" for q in QUANTILES]) + r" \\",
        r"\midrule",
    ]

    for _, r in sub.iterrows():
        mode = "minimum-hop" if r["mode"] == "minimum_hop" else "minimax"
        vals = [fmt(r, f"{metric}_q{q}") for q in QUANTILES]
        lines.append(f"{r['level']} & {mode} & {disp_cond(r['cond'])} & {fmt_n(r)} & " + " & ".join(vals) + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}"]
    out_path.write_text("\n".join(lines) + "\n")


def write_latex_edge(summary, path_set, out_path):
    sub = summary[(summary["path_set"] == path_set) & (summary["mode"].isin(["minimum_hop", "minimax"]))].copy()
    order_level = {"sample graphwise": 0, "participant average": 1}
    order_mode = {"minimum_hop": 0, "minimax": 1}
    order_cond = {"overall": 0, "cd": 1, "uc": 2, "nonibd": 3}
    sub["_a"] = sub["level"].map(order_level)
    sub["_b"] = sub["mode"].map(order_mode)
    sub["_c"] = sub["cond"].map(order_cond)
    sub = sub.sort_values(["_a", "_b", "_c"])

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Long-range path-edge concentration diagnostics. Edge traversal shares rank active directed edges by how often they are traversed by sampled long-range paths. Participant-average rows are obtained by averaging sample-level concentration diagnostics within participant before summarising across participants.}",
        r"\label{tab:long_range_path_edge_concentration}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lllrrrrrrr}",
        r"\toprule",
        r"Level & Path type & Condition & $n$ valid/total & Top 1\% share & Top 5\% share & Top 10\% share & Top 1\% median support & Top 5\% median support & Top 10\% median support \\",
        r"\midrule",
    ]

    for _, r in sub.iterrows():
        mode = "minimum-hop" if r["mode"] == "minimum_hop" else "minimax"
        vals = [
            fmt(r, "top1pct_traversal_share"),
            fmt(r, "top5pct_traversal_share"),
            fmt(r, "top10pct_traversal_share"),
            fmt(r, "top1pct_median_support_top_used"),
            fmt(r, "top5pct_median_support_top_used"),
            fmt(r, "top10pct_median_support_top_used"),
        ]
        lines.append(f"{r['level']} & {mode} & {disp_cond(r['cond'])} & {fmt_n(r)} & " + " & ".join(vals) + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}"]
    out_path.write_text("\n".join(lines) + "\n")


def aggregate_outputs(out_dir, long_min_hop, expected_samples=None, expected_participants=None):
    sample_dir = out_dir / "sample_results"
    path_df, edge_df, err_df = load_jsons(sample_dir)

    path_df.to_csv(out_dir / "path_backbone_sample_path_quantiles.csv", index=False)
    edge_df.to_csv(out_dir / "path_backbone_sample_edge_concentration.csv", index=False)
    err_df.to_csv(out_dir / "path_backbone_failed_samples.csv", index=False)

    path_part = participant_average(path_df)
    edge_part = participant_average(edge_df)

    n_samples = path_df["sample_id"].nunique() if not path_df.empty else 0
    n_participants = path_part["participant_id"].nunique() if not path_part.empty else 0
    if expected_samples is not None and n_samples != expected_samples:
        raise RuntimeError(f"Expected {expected_samples} cached samples, found {n_samples}.")
    if expected_participants is not None and n_participants != expected_participants:
        raise RuntimeError(
            f"Expected {expected_participants} cached participants, found {n_participants}."
        )

    path_part.to_csv(out_dir / "path_backbone_participant_path_quantiles.csv", index=False)
    edge_part.to_csv(out_dir / "path_backbone_participant_edge_concentration.csv", index=False)

    path_summary = condition_summary(path_df, path_part)
    edge_summary = condition_summary(edge_df, edge_part)

    path_summary.to_csv(out_dir / "path_backbone_condition_path_summary.csv", index=False)
    edge_summary.to_csv(out_dir / "path_backbone_condition_edge_summary.csv", index=False)

    long_set = f"long_hop_ge{long_min_hop}"

    write_latex_quantile(
        path_summary,
        "path_length",
        long_set,
        out_dir / "latex_long_range_path_length_quantiles.tex",
        r"\caption{Long-range directed path-length quantiles. Long-range paths are sampled source--target pairs with minimum-hop distance at least 4. Diagnostics are computed per sample graph, then averaged within participant. The $n$ column reports valid/total graph or participant units; units without a sampled path have undefined path metrics and do not enter the moments. Values are mean $\pm$ SD.}",
        r"\label{tab:long_range_path_length_quantiles}",
    )

    write_latex_quantile(
        path_summary,
        "bottleneck_w",
        long_set,
        out_dir / "latex_long_range_bottleneck_weight_quantiles.tex",
        r"\caption{Long-range directed path bottleneck-weight quantiles. The bottleneck weight of path $p$ is $\max_{e\in p} w(e)$, the weakest-supported edge along the path. Lower values indicate stronger support throughout the path. Values are mean $\pm$ SD.}",
        r"\label{tab:long_range_bottleneck_weight_quantiles}",
    )

    write_latex_quantile(
        path_summary,
        "min_support",
        long_set,
        out_dir / "latex_long_range_minimum_support_quantiles.tex",
        r"\caption{Long-range directed path minimum-support quantiles. Minimum support is $1-\max_{e\in p} w(e)$, so larger values indicate that the path avoids weakly supported transitions. Values are mean $\pm$ SD.}",
        r"\label{tab:long_range_minimum_support_quantiles}",
    )

    write_latex_quantile(
        path_summary,
        "mean_w2",
        long_set,
        out_dir / "latex_long_range_mean_w2_quantiles.tex",
        r"\caption{Long-range directed path mean-$w^2$ quantiles. For each sampled path, mean $w^2$ is computed across the directed edges in the selected path. Values are mean $\pm$ SD.}",
        r"\label{tab:long_range_mean_w2_quantiles}",
    )

    write_latex_edge(edge_summary, long_set, out_dir / "latex_long_range_edge_concentration.tex")

    print("\nWrote outputs to:", out_dir, flush=True)
    print("sample path rows:", len(path_df), flush=True)
    print("sample edge rows:", len(edge_df), flush=True)
    print("participant path rows:", len(path_part), flush=True)
    print("participant edge rows:", len(edge_part), flush=True)
    print("failed samples:", len(err_df), flush=True)


def build_tasks(args):
    meta = pd.read_csv(args.meta, low_memory=False)
    meta = meta[["External ID", "diagnosis", "Participant ID"]].copy()
    meta.columns = ["sample_id", "diagnosis", "participant_id"]
    meta["cond"] = meta["diagnosis"].apply(norm_cond)
    meta = meta[meta["cond"].isin(["cd", "uc", "nonibd"])].drop_duplicates("sample_id").copy()

    graph_files = {p.stem: p for p in Path(args.graph_dir).glob("*.graphml")}
    meta = meta[meta["sample_id"].isin(graph_files)].copy()

    if args.max_samples:
        meta = meta.head(args.max_samples).copy()

    print("Metadata samples matched to graphs:", len(meta), flush=True)
    print("Participants:", meta["participant_id"].nunique(), flush=True)
    print("Condition counts:", flush=True)
    print(meta["cond"].value_counts().to_string(), flush=True)

    tasks = []
    sample_dir = Path(args.out_dir) / "sample_results"
    sample_dir.mkdir(parents=True, exist_ok=True)

    for _, r in meta.iterrows():
        sid = str(r["sample_id"])
        out_json = sample_dir / f"{safe_name(sid)}.json"
        if args.resume and out_json.exists() and out_json.stat().st_size > 0:
            continue
        tasks.append({
            "sample_id": sid,
            "participant_id": str(r["participant_id"]),
            "cond": str(r["cond"]),
            "graph_path": str(graph_files[sid]),
            "n_source": args.n_source,
            "n_target": args.n_target,
            "long_min_hop": args.long_min_hop,
            "active_tol": args.active_tol,
            "seed": stable_seed(args.seed, sid),
            "out_json": str(out_json),
        })

    print("Samples still to process:", len(tasks), flush=True)
    return tasks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph-dir", default=str(Path.home() / "Real_Data" / "out_graphs"))
    ap.add_argument("--meta", default=str(Path.home() / "Real_Data" / "hmp2_metadata.csv"))
    ap.add_argument("--out-dir", default=str(Path.home() / "Real_Data" / "Path_Backbone_LongRange_Diagnostics"))
    ap.add_argument("--n-source", type=int, default=100)
    ap.add_argument("--n-target", type=int, default=100)
    ap.add_argument("--long-min-hop", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--active-tol", type=float, default=1e-12)
    ap.add_argument("--seed", type=int, default=20260707)
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--expected-samples", type=int, default=None)
    ap.add_argument("--expected-participants", type=int, default=None)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--write-only", action="store_true")
    args = ap.parse_args()

    args.out_dir = Path(args.out_dir).expanduser()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.resume = not args.no_resume

    print("=" * 100, flush=True)
    print("Long-range path-backbone diagnostics", flush=True)
    print("=" * 100, flush=True)
    print("Protocol: sample graph diagnostics -> participant averaging -> condition summary", flush=True)
    print(f"n_source={args.n_source}, n_target={args.n_target}, long_min_hop={args.long_min_hop}, workers={args.workers}", flush=True)

    config = vars(args).copy()
    config["out_dir"] = str(args.out_dir)
    (args.out_dir / "run_config.json").write_text(json.dumps(config, indent=2, default=str))

    if args.write_only:
        aggregate_outputs(
            args.out_dir,
            args.long_min_hop,
            args.expected_samples,
            args.expected_participants,
        )
        return

    tasks = build_tasks(args)

    if tasks:
        t0 = time.time()
        done = 0
        errors = 0

        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(process_sample, t): t for t in tasks}

            for fut in as_completed(futs):
                task = futs[fut]
                try:
                    obj = fut.result()
                except Exception:
                    obj = {
                        "sample_id": task["sample_id"],
                        "path_rows": [],
                        "edge_rows": [],
                        "error": traceback.format_exc(),
                    }

                if obj.get("error"):
                    errors += 1

                atomic_json(Path(task["out_json"]), obj)

                done += 1
                if done == 1 or done % 10 == 0 or done == len(tasks):
                    elapsed = time.time() - t0
                    rate = done / elapsed if elapsed > 0 else np.nan
                    eta = (len(tasks) - done) / rate if rate and rate > 0 else np.nan
                    print(
                        f"[progress] {done}/{len(tasks)} | errors={errors} | "
                        f"rate={rate:.4f} samples/s | ETA={eta/3600:.2f} h | latest={task['sample_id']}",
                        flush=True,
                    )

    print("\nAggregating...", flush=True)
    aggregate_outputs(
        args.out_dir,
        args.long_min_hop,
        args.expected_samples,
        args.expected_participants,
    )
    print("\n[READY] Path-backbone diagnostics finished.", flush=True)


if __name__ == "__main__":
    main()
