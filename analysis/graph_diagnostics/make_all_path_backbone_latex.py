#!/usr/bin/env python3
import argparse
from pathlib import Path
import math
import numpy as np
import pandas as pd

QUANTILES = [1, 5, 10, 20, 30, 50, 70, 80, 90, 95, 99]


def disp_cond(c):
    return {"cd": "CD", "uc": "UC", "nonibd": "non-IBD", "overall": "overall"}.get(str(c), str(c))


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


def ordered(summary):
    sub = summary.copy()
    level_order = {"sample graphwise": 0, "participant average": 1}
    mode_order = {"minimum_hop": 0, "minimax": 1}
    cond_order = {"overall": 0, "cd": 1, "uc": 2, "nonibd": 3}

    sub["_level_order"] = sub["level"].map(level_order)
    sub["_mode_order"] = sub["mode"].map(mode_order)
    sub["_cond_order"] = sub["cond"].map(cond_order)
    return sub.sort_values(["_level_order", "_mode_order", "_cond_order"])


def write_quantile_table(summary, metric, out_path, caption, label):
    sub = summary[
        (summary["path_set"] == "all") &
        (summary["mode"].isin(["minimum_hop", "minimax"]))
    ].copy()

    sub = ordered(sub)

    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(caption)
    lines.append(label)
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{lll" + "r" * (1 + len(QUANTILES)) + r"}")
    lines.append(r"\toprule")
    lines.append("Level & Path type & Condition & $n$ valid/total & " + " & ".join([f"q{q}" for q in QUANTILES]) + r" \\")
    lines.append(r"\midrule")

    for _, r in sub.iterrows():
        mode = "minimum-hop" if r["mode"] == "minimum_hop" else "minimax"
        vals = [fmt(r, f"{metric}_q{q}") for q in QUANTILES]
        lines.append(
            f"{r['level']} & {mode} & {disp_cond(r['cond'])} & {fmt_n(r)} & "
            + " & ".join(vals)
            + r" \\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}%")
    lines.append(r"}")
    lines.append(r"\end{table}")

    out_path.write_text("\n".join(lines) + "\n")


def write_edge_table(summary, out_path):
    sub = summary[
        (summary["path_set"] == "all") &
        (summary["mode"].isin(["minimum_hop", "minimax"]))
    ].copy()

    sub = ordered(sub)

    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Path-edge concentration diagnostics across all sampled directed source--target paths. "
        r"Edge traversal shares are computed by ranking active directed edges within each sample graph according "
        r"to how often they are traversed by sampled paths. Different source--target paths are allowed to reuse "
        r"the same directed edges; this repeated edge use is the signal measured by the concentration diagnostic. "
        r"Participant-average rows are obtained by averaging sample-level diagnostics within participant before "
        r"summarising across participants. Values are mean $\pm$ SD.}"
    )
    lines.append(r"\label{tab:all_sampled_path_edge_concentration}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{lllrrrrrrr}")
    lines.append(r"\toprule")
    lines.append(
        r"Level & Path type & Condition & $n$ valid/total & Top 1\% share & Top 5\% share & Top 10\% share "
        r"& Top 1\% median support & Top 5\% median support & Top 10\% median support \\"
    )
    lines.append(r"\midrule")

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
        lines.append(
            f"{r['level']} & {mode} & {disp_cond(r['cond'])} & {fmt_n(r)} & "
            + " & ".join(vals)
            + r" \\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}%")
    lines.append(r"}")
    lines.append(r"\end{table}")

    out_path.write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description="Build unrestricted path-backbone LaTeX tables.")
    ap.add_argument(
        "--input-dir",
        default=str(Path.home() / "Real_Data" / "Path_Backbone_LongRange_Diagnostics"),
    )
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()

    input_dir = Path(args.input_dir).expanduser()
    out_dir = Path(args.output_dir).expanduser() if args.output_dir else input_dir / "all_sampled_path_tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    path_summary_path = input_dir / "path_backbone_condition_path_summary.csv"
    edge_summary_path = input_dir / "path_backbone_condition_edge_summary.csv"
    path_summary = pd.read_csv(path_summary_path)
    edge_summary = pd.read_csv(edge_summary_path)

    # Basic validation: make sure unrestricted rows exist.
    path_sets = set(path_summary["path_set"].astype(str))
    edge_sets = set(edge_summary["path_set"].astype(str))

    if "all" not in path_sets:
        raise SystemExit(f"Could not find path_set='all' in {path_summary_path}. Found: {sorted(path_sets)}")
    if "all" not in edge_sets:
        raise SystemExit(f"Could not find path_set='all' in {edge_summary_path}. Found: {sorted(edge_sets)}")

    write_quantile_table(
        path_summary,
        "path_length",
        out_dir / "latex_all_sampled_path_length_quantiles.tex",
        r"\caption{Directed path-length quantiles across all sampled reachable source--target pairs. "
        r"For each sample graph, unique directed source--target pairs are sampled, one minimum-hop path and one "
        r"minimax path are computed for each pair, and no minimum path-length restriction is applied. "
        r"Participant-average rows are obtained by averaging sample-level diagnostics within participant before "
        r"summarising across participants. The $n$ column reports valid/total units; units without a sampled path do not enter the moments. Values are mean $\pm$ SD.}",
        r"\label{tab:all_sampled_path_length_quantiles}",
    )

    write_quantile_table(
        path_summary,
        "bottleneck_w",
        out_dir / "latex_all_sampled_bottleneck_weight_quantiles.tex",
        r"\caption{Directed path bottleneck-weight quantiles across all sampled reachable source--target pairs. "
        r"The bottleneck weight of path $p$ is $\max_{e\in p}w(e)$, the weakest-supported edge along the path. "
        r"Lower values indicate stronger support throughout the path. No minimum path-length restriction is applied. "
        r"The $n$ column reports valid/total units. Values are mean $\pm$ SD.}",
        r"\label{tab:all_sampled_bottleneck_weight_quantiles}",
    )

    write_quantile_table(
        path_summary,
        "min_support",
        out_dir / "latex_all_sampled_minimum_support_quantiles.tex",
        r"\caption{Directed path minimum-support quantiles across all sampled reachable source--target pairs. "
        r"Minimum support is defined as $1-\max_{e\in p}w(e)$, so larger values indicate that the path avoids "
        r"weakly supported transitions. No minimum path-length restriction is applied. The $n$ column reports valid/total units. Values are mean $\pm$ SD.}",
        r"\label{tab:all_sampled_minimum_support_quantiles}",
    )

    write_quantile_table(
        path_summary,
        "mean_w2",
        out_dir / "latex_all_sampled_mean_w2_quantiles.tex",
        r"\caption{Directed path mean-$w^2$ quantiles across all sampled reachable source--target pairs. "
        r"For each sampled path, mean $w^2$ is computed across the directed edges in the selected path. "
        r"No minimum path-length restriction is applied. The $n$ column reports valid/total units. Values are mean $\pm$ SD.}",
        r"\label{tab:all_sampled_mean_w2_quantiles}",
    )

    write_edge_table(
        edge_summary,
        out_dir / "latex_all_sampled_edge_concentration.tex",
    )

    # Table 5 is the enhanced top-0.1%/median-w table emitted by
    # edge_concentration_weight_top001.py. Keep the older basic concentration
    # table as an auxiliary output, but do not mix it into the Tables 1--4 file.
    combined = out_dir / "path_backbone_tables_1_to_4_for_overleaf.tex"
    parts = [
        out_dir / "latex_all_sampled_path_length_quantiles.tex",
        out_dir / "latex_all_sampled_bottleneck_weight_quantiles.tex",
        out_dir / "latex_all_sampled_minimum_support_quantiles.tex",
        out_dir / "latex_all_sampled_mean_w2_quantiles.tex",
    ]
    combined.write_text("\n\n".join(p.read_text() for p in parts) + "\n")

    print("Wrote unrestricted/all-path Tables 1--4 to:")
    for p in parts:
        print(" ", p)
    print("\nAuxiliary basic concentration table:")
    print(" ", out_dir / "latex_all_sampled_edge_concentration.tex")
    print("\nCombined Tables 1--4 file:")
    print(" ", combined)


if __name__ == "__main__":
    main()
