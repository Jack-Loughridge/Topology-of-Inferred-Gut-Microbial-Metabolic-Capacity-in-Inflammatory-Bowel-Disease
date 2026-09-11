#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import tempfile
import time
import traceback
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd


# Reuse the canonical parser and path helpers from the adjacent publication
# source. This replaces the historical dependency on an untracked home file.
import path_backbone_longrange_diagnostics as pb


TOP_FRACS = [
    (0.001, "top0p1pct", r"Top 0.1\%"),
    (0.010, "top1pct", r"Top 1\%"),
    (0.050, "top5pct", r"Top 5\%"),
    (0.100, "top10pct", r"Top 10\%"),
]


def atomic_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=str(path.parent), suffix=".tmp") as f:
        json.dump(obj, f)
        tmp = Path(f.name)
    tmp.replace(path)


def disp_cond(c):
    return {"cd": "CD", "uc": "UC", "nonibd": "non-IBD", "overall": "overall"}.get(str(c), str(c))


def summarize_counter(counter, edge_w, base):
    """
    Rank active directed edges by sampled path-use count.
    For each top-used percentile set, report:
      - traversal share = fraction of all path-edge traversals using that set
      - median w among positively used edges inside that set

    Lower w means stronger inferred microbial/metabolic support.
    """
    row = dict(base)

    active_edges = list(edge_w.keys())
    n_active = len(active_edges)
    total_traversals = int(sum(counter.values()))

    row["n_active_edges"] = n_active
    row["total_edge_traversals"] = total_traversals

    used_edges = [e for e, c in counter.items() if c > 0]
    row["unique_edges_used"] = int(len(used_edges))
    row["unique_edges_used_fraction_of_active"] = float(len(used_edges) / n_active) if n_active else np.nan

    ranked = sorted(active_edges, key=lambda e: counter.get(e, 0), reverse=True)

    for frac, prefix, _ in TOP_FRACS:
        k = max(1, int(math.ceil(frac * n_active))) if n_active else 0
        top_edges = ranked[:k]
        top_used_edges = [e for e in top_edges if counter.get(e, 0) > 0]

        row[f"{prefix}_n_edges"] = int(k)
        row[f"{prefix}_n_used_edges"] = int(len(top_used_edges))

        row[f"{prefix}_traversal_share"] = (
            float(sum(counter.get(e, 0) for e in top_edges) / total_traversals)
            if total_traversals > 0 else np.nan
        )

        if top_used_edges:
            ws = np.asarray([edge_w[e] for e in top_used_edges], dtype=float)
            counts = np.asarray([counter[e] for e in top_used_edges], dtype=float)

            row[f"{prefix}_median_w_top_used"] = float(np.median(ws))
            row[f"{prefix}_mean_w_top_used"] = float(np.mean(ws))
            row[f"{prefix}_usage_weighted_mean_w_top_used"] = float(np.average(ws, weights=counts))
        else:
            row[f"{prefix}_median_w_top_used"] = np.nan
            row[f"{prefix}_mean_w_top_used"] = np.nan
            row[f"{prefix}_usage_weighted_mean_w_top_used"] = np.nan

    return row


def process_sample(task):
    try:
        sid = task["sample_id"]
        rng = np.random.default_rng(task["seed"])

        nodes, adj, edge_w = pb.parse_graphml_active(Path(task["graph_path"]), task["active_tol"])

        sources = [u for u in nodes if len(adj.get(u, [])) > 0]
        if sources:
            sources = list(rng.choice(sources, size=min(task["n_source"], len(sources)), replace=False))

        hop_counter = Counter()
        minimax_counter = Counter()

        n_pairs = 0
        reach_fracs = []

        for source in sources:
            hop_paths = pb.bfs_paths(adj, source)
            reachable_targets = list(hop_paths.keys())

            if not reachable_targets:
                reach_fracs.append(0.0)
                continue

            reach_fracs.append(len(reachable_targets) / max(1, len(nodes) - 1))

            targets = list(
                rng.choice(
                    reachable_targets,
                    size=min(task["n_target"], len(reachable_targets)),
                    replace=False,
                )
            )
            n_pairs += len(targets)

            # Minimum-hop paths.
            for target in targets:
                pb.count_edges(hop_counter, hop_paths[target])

            # Minimax paths for the same ordered source-target pairs.
            minimax_paths = pb.minimax_paths(adj, edge_w, source, targets)
            for target in targets:
                p = minimax_paths.get(target)
                if p is not None:
                    pb.count_edges(minimax_counter, p)

        common = {
            "sample_id": sid,
            "participant_id": task["participant_id"],
            "cond": task["cond"],
            "n_active_vertices": len(nodes),
            "n_sources_sampled": len(sources),
            "n_ordered_source_target_pairs": n_pairs,
            "mean_reachable_fraction_sampled_sources": float(np.mean(reach_fracs)) if reach_fracs else np.nan,
            "n_source_requested": task["n_source"],
            "n_target_requested_per_source": task["n_target"],
        }

        rows = []
        for mode, counter in [
            ("minimum_hop", hop_counter),
            ("minimax", minimax_counter),
        ]:
            base = dict(common)
            base["mode"] = mode
            rows.append(summarize_counter(counter, edge_w, base))

        return {"sample_id": sid, "rows": rows, "error": None}

    except Exception:
        return {
            "sample_id": task.get("sample_id", "unknown"),
            "rows": [],
            "error": traceback.format_exc(),
        }


def build_tasks(args):
    meta = pd.read_csv(args.meta, low_memory=False)
    meta = meta[["External ID", "diagnosis", "Participant ID"]].copy()
    meta.columns = ["sample_id", "diagnosis", "participant_id"]
    meta["cond"] = meta["diagnosis"].apply(pb.norm_cond)
    meta = meta[meta["cond"].isin(["cd", "uc", "nonibd"])].drop_duplicates("sample_id").copy()

    graph_files = {p.stem: p for p in Path(args.graph_dir).glob("*.graphml")}
    meta = meta[meta["sample_id"].isin(graph_files)].copy()

    if args.max_samples is not None:
        meta = meta.head(args.max_samples).copy()

    print("Metadata samples matched to graphs:", len(meta), flush=True)
    print("Participants:", meta["participant_id"].nunique(), flush=True)
    print("Condition counts:", flush=True)
    print(meta["cond"].value_counts().to_string(), flush=True)

    sample_dir = Path(args.out_dir) / "sample_results"
    sample_dir.mkdir(parents=True, exist_ok=True)

    tasks = []
    for _, r in meta.iterrows():
        sid = str(r["sample_id"])
        out_json = sample_dir / f"{pb.safe_name(sid)}.json"

        if args.resume and out_json.exists() and out_json.stat().st_size > 0:
            continue

        tasks.append({
            "sample_id": sid,
            "participant_id": str(r["participant_id"]),
            "cond": str(r["cond"]),
            "graph_path": str(graph_files[sid]),
            "n_source": args.n_source,
            "n_target": args.n_target,
            "active_tol": args.active_tol,
            "seed": pb.stable_seed(args.seed, sid),
            "out_json": str(out_json),
        })

    print("Samples still to process:", len(tasks), flush=True)
    return tasks


def load_jsons(sample_dir):
    rows = []
    errors = []

    for p in sorted(Path(sample_dir).glob("*.json")):
        obj = json.loads(p.read_text())
        if obj.get("error"):
            errors.append({"sample_id": obj.get("sample_id"), "error": obj.get("error")})
        else:
            rows.extend(obj.get("rows", []))

    return pd.DataFrame(rows), pd.DataFrame(errors)


def participant_average(df):
    if df.empty:
        return df.copy()

    id_cols = ["participant_id", "mode"]
    skip = set(id_cols + ["sample_id", "cond"])
    numeric_cols = [c for c in df.columns if c not in skip and pd.api.types.is_numeric_dtype(df[c])]

    rows = []
    for keys, sub in df.groupby(id_cols, dropna=False):
        row = dict(zip(id_cols, keys))
        row["cond"] = sub["cond"].astype(str).value_counts().idxmax()
        row["n_samples_for_participant"] = sub["sample_id"].nunique()

        for c in numeric_cols:
            vals = pd.to_numeric(sub[c], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            row[c] = float(vals.mean()) if len(vals) else np.nan

        rows.append(row)

    return pd.DataFrame(rows)


def condition_summary(sample_df, participant_df):
    rows = []

    for level, df in [("sample graphwise", sample_df), ("participant average", participant_df)]:
        if df.empty:
            continue

        skip = {"sample_id", "participant_id", "cond", "mode"}
        numeric_cols = [c for c in df.columns if c not in skip and pd.api.types.is_numeric_dtype(df[c])]

        for mode in ["minimum_hop", "minimax"]:
            base_sub = df[df["mode"] == mode]

            for cond in ["overall", "cd", "uc", "nonibd"]:
                sub = base_sub if cond == "overall" else base_sub[base_sub["cond"] == cond]
                if len(sub) == 0:
                    continue

                valid = pd.to_numeric(
                    sub["total_edge_traversals"], errors="coerce"
                ).fillna(0).gt(0)
                row = {
                    "level": level,
                    "mode": mode,
                    "cond": cond,
                    "n": int(len(sub)),
                    "n_total": int(len(sub)),
                    "n_valid": int(valid.sum()),
                }

                for c in numeric_cols:
                    vals = pd.to_numeric(sub[c], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
                    row[f"{c}_mean"] = float(vals.mean()) if len(vals) else np.nan
                    row[f"{c}_sd"] = float(vals.std(ddof=0)) if len(vals) else np.nan

                rows.append(row)

    return pd.DataFrame(rows)


def fmt(row, col, digits=3):
    m = row.get(f"{col}_mean", np.nan)
    s = row.get(f"{col}_sd", np.nan)

    try:
        m = float(m)
        s = float(s)
    except Exception:
        return "--"

    if not math.isfinite(m):
        return "--"
    if not math.isfinite(s):
        return f"{m:.{digits}f}"
    return f"{m:.{digits}f} $\\pm$ {s:.{digits}f}"


def fmt_n(row):
    valid = row.get("n_valid", row.get("n", np.nan))
    total = row.get("n_total", row.get("n", np.nan))
    return f"{int(valid)}/{int(total)}"


def write_latex(summary, out_path):
    level_order = {"sample graphwise": 0, "participant average": 1}
    mode_order = {"minimum_hop": 0, "minimax": 1}
    cond_order = {"overall": 0, "cd": 1, "uc": 2, "nonibd": 3}

    sub = summary.copy()
    sub["_level_order"] = sub["level"].map(level_order)
    sub["_mode_order"] = sub["mode"].map(mode_order)
    sub["_cond_order"] = sub["cond"].map(cond_order)
    sub = sub.sort_values(["_level_order", "_mode_order", "_cond_order"])

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Path-edge concentration and weight diagnostics across all sampled directed source--target paths. "
        r"Active directed edges are ranked within each sample graph according to how often they are traversed by sampled paths. "
        r"Traversal-share columns report the fraction of all path-edge traversals accounted for by the top-used edge set. "
        r"Median-$w$ columns report the median edge weight among positively used edges in the same top-used set; lower $w$ indicates stronger inferred microbial support. "
        r"Participant-average rows are obtained by averaging sample-level diagnostics within participant before summarising across participants. The $n$ column reports valid/total units; units with no sampled edge traversal have undefined concentration metrics and do not enter the moments. Values are mean $\pm$ SD.}",
        r"\label{tab:all_sampled_path_edge_concentration_weight}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lllrrrrrrrr}",
        r"\toprule",
        r"Level & Path type & Condition & $n$ valid/total & Top 0.1\% share & Top 1\% share & Top 5\% share & Top 10\% share "
        r"& Top 0.1\% median $w$ & Top 1\% median $w$ & Top 5\% median $w$ & Top 10\% median $w$ \\",
        r"\midrule",
    ]

    for _, r in sub.iterrows():
        mode = "minimum-hop" if r["mode"] == "minimum_hop" else "minimax"

        vals = [
            fmt(r, "top0p1pct_traversal_share"),
            fmt(r, "top1pct_traversal_share"),
            fmt(r, "top5pct_traversal_share"),
            fmt(r, "top10pct_traversal_share"),
            fmt(r, "top0p1pct_median_w_top_used"),
            fmt(r, "top1pct_median_w_top_used"),
            fmt(r, "top5pct_median_w_top_used"),
            fmt(r, "top10pct_median_w_top_used"),
        ]

        lines.append(
            f"{r['level']} & {mode} & {disp_cond(r['cond'])} & {fmt_n(r)} & "
            + " & ".join(vals)
            + r" \\"
        )

    lines += [
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\end{table}",
    ]

    out_path.write_text("\n".join(lines) + "\n")


def aggregate_outputs(out_dir, expected_samples=None, expected_participants=None):
    sample_dir = Path(out_dir) / "sample_results"

    sample_df, err_df = load_jsons(sample_dir)
    participant_df = participant_average(sample_df)
    n_samples = sample_df["sample_id"].nunique() if not sample_df.empty else 0
    n_participants = participant_df["participant_id"].nunique() if not participant_df.empty else 0
    if expected_samples is not None and n_samples != expected_samples:
        raise RuntimeError(f"Expected {expected_samples} cached samples, found {n_samples}.")
    if expected_participants is not None and n_participants != expected_participants:
        raise RuntimeError(
            f"Expected {expected_participants} cached participants, found {n_participants}."
        )
    summary_df = condition_summary(sample_df, participant_df)

    sample_csv = Path(out_dir) / "edge_concentration_weight_top001_sample.csv"
    participant_csv = Path(out_dir) / "edge_concentration_weight_top001_participant.csv"
    summary_csv = Path(out_dir) / "edge_concentration_weight_top001_condition_summary.csv"
    errors_csv = Path(out_dir) / "edge_concentration_weight_top001_failed_samples.csv"
    latex_path = Path(out_dir) / "latex_edge_concentration_weight_top001.tex"

    sample_df.to_csv(sample_csv, index=False)
    participant_df.to_csv(participant_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    err_df.to_csv(errors_csv, index=False)

    write_latex(summary_df, latex_path)

    print("\nWrote outputs to:", out_dir, flush=True)
    print(" sample rows:", len(sample_df), flush=True)
    print(" participant rows:", len(participant_df), flush=True)
    print(" summary rows:", len(summary_df), flush=True)
    print(" failed samples:", len(err_df), flush=True)
    print("\nLaTeX table:", latex_path, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph-dir", default=str(Path.home() / "Real_Data" / "out_graphs"))
    ap.add_argument("--meta", default=str(Path.home() / "Real_Data" / "hmp2_metadata.csv"))
    ap.add_argument("--out-dir", default=str(Path.home() / "Real_Data" / "Path_Backbone_EdgeConcentration_WeightTop001"))
    ap.add_argument("--n-source", type=int, default=100)
    ap.add_argument("--n-target", type=int, default=100)
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
    print("Path-edge concentration with median weight w", flush=True)
    print("=" * 100, flush=True)
    print("Protocol: sample graph diagnostics -> participant averaging -> condition summary", flush=True)
    print(f"n_source={args.n_source}, n_target={args.n_target}, workers={args.workers}", flush=True)
    print("Top-used edge sets:", [(label, frac) for frac, _, label in TOP_FRACS], flush=True)

    config = vars(args).copy()
    config["out_dir"] = str(args.out_dir)
    config["top_fracs"] = [(frac, prefix, label) for frac, prefix, label in TOP_FRACS]
    config["definition"] = {
        "edge_rank": "rank active directed edges by sampled path traversal count",
        "traversal_share": "fraction of all sampled path-edge traversals accounted for by the top-used edge set",
        "median_w": "median edge weight w among positively used edges in the top-used edge set",
        "interpretation": "lower w means stronger inferred microbial support",
    }
    (args.out_dir / "run_config.json").write_text(json.dumps(config, indent=2, default=str))

    if args.write_only:
        aggregate_outputs(args.out_dir, args.expected_samples, args.expected_participants)
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
                        "rows": [],
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
    aggregate_outputs(args.out_dir, args.expected_samples, args.expected_participants)
    print("\n[READY] Edge concentration weight table finished.", flush=True)


if __name__ == "__main__":
    main()
