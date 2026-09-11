from pathlib import Path
from collections import defaultdict, Counter
import pandas as pd
import numpy as np
import re
import math

ROOT = Path.home() / "Real_Data"

CUR_RICCI = ROOT / "Ricci_Classifier_OriginalStyle_AllTasks_NewVectors"
OUT = ROOT / "manuscript_table_figure_update_20260713_1340"

TABLE_DIR = OUT / "tables" / "ricci_interpretation_mapped"
CSV_DIR = OUT / "source_csv" / "ricci_interpretation_mapped"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
CSV_DIR.mkdir(parents=True, exist_ok=True)

# Highest-priority annotation sources first.
ANNOTATION_SOURCES = [
    ROOT / "Ricci_Interpretability" / "annotated_selected_coefficients.csv",
    ROOT / "Ricci_Interpretability" / "reaction_support_for_selected_edges.csv",
    ROOT / "Ricci_Interpretability_3way_refined_v4" / "annotated_selected_coefficients_refined_v4.csv",
    ROOT / "Ricci_Interpretability_3way_refined_v3" / "annotated_selected_coefficients_refined_v3.csv",
    ROOT / "Ricci_Interpretability_3way" / "annotated_selected_coefficients_3way.csv",
    ROOT / "Ricci_Interpretability" / "top_edges_per_edge_process.csv",
    ROOT / "ricci_edges_interpretable.csv",
    ROOT / "ricci_top_edges_per_pathway.csv",
]

TASKS = [
    {
        "folder": "three_way_nonIBD_UC_CD",
        "name": "non-IBD vs UC vs CD",
        "kind": "multiclass",
        "positive_class": None,
    },
    {
        "folder": "IBD_vs_nonIBD",
        "name": "IBD vs non-IBD",
        "kind": "binary",
        "positive_class": "IBD",
    },
    {
        "folder": "nonIBD_vs_UC",
        "name": "non-IBD vs UC",
        "kind": "binary",
        "positive_class": "UC",
    },
    {
        "folder": "nonIBD_vs_CD",
        "name": "non-IBD vs CD",
        "kind": "binary",
        "positive_class": "CD",
    },
    {
        "folder": "CD_vs_UC",
        "name": "CD vs UC",
        "kind": "binary",
        "positive_class": "UC",
    },
]

# ---------------------------------------------------------------------
# Robust keying / display helpers
# ---------------------------------------------------------------------

def decode_agora(x):
    s = str(x)
    s = s.replace("__91__", "[")
    s = s.replace("__93__", "]")
    s = s.replace("_DASH_", "-")
    return s

def compact_met(x):
    """
    Robust matching key for metabolite IDs.

    Examples:
      M_FOL__91__C__93__ -> folc
      M_FOL[C]           -> folc
      FOL[C]             -> folc
    """
    s = decode_agora(str(x)).strip().lower()
    s = re.sub(r"^m[_-]", "", s)
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s

def split_edge(edge):
    s = decode_agora(str(edge))
    s = s.replace("B__", "").replace("K__", "")
    if " -> " in s:
        a, b = s.split(" -> ", 1)
        return a.strip(), b.strip()
    if "__TO__" in s:
        a, b = s.split("__TO__", 1)
        return a.strip(), b.strip()
    return None, None

def pair_key(a, b):
    a2 = compact_met(a)
    b2 = compact_met(b)
    if not a2 or not b2:
        return None
    return a2 + "->" + b2

def safe_str(x):
    if pd.isna(x):
        return ""
    return str(x)

def clean_process_label(x):
    if pd.isna(x) or str(x).strip() == "":
        return "Unannotated"

    s = str(x).strip()

    if s.lower() in {"nan", "none", "unannotated"}:
        return "Unannotated"

    replacements = {
        "currency_coupled_reaction": "Currency-coupled reaction",
        "transport_or_compartment_exchange": "Transport / compartment exchange",
        "carbohydrate_metabolism": "Carbohydrate metabolism",
        "amino_acid_metabolism": "Amino-acid metabolism",
        "nucleotide_metabolism": "Nucleotide metabolism",
        "lipid_fatty_acid_metabolism": "Lipid / fatty-acid metabolism",
        "scfa_related_metabolism": "SCFA-related metabolism",
        "one_carbon_metabolism": "One-carbon metabolism",
        "redox_metabolism": "Redox metabolism",
        "oxidative_stress": "Oxidative stress",
        "energy_metabolism": "Energy metabolism",
        "phosphate_metabolism": "Phosphate metabolism",
        "glycoside_xenobiotic_or_secondary_metabolism": "Glycoside / xenobiotic / secondary metabolism",
        "bile_sterol_lipid_metabolism": "Bile / sterol / lipid metabolism",
        "other": "Other",
    }

    if " -> " in s:
        parts = [clean_process_label(p) for p in s.split(" -> ")]
        return " / ".join(parts)

    return replacements.get(s, s.replace("_", " ").capitalize())

def latex_escape_cell(x):
    if pd.isna(x):
        return "--"

    s = str(x)

    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }

    for a, b in repl.items():
        s = s.replace(a, b)

    return s

def fmt(x, digits=3):
    try:
        x = float(x)
    except Exception:
        return "--"
    if not math.isfinite(x):
        return "--"
    return f"{x:.{digits}f}"

def write_latex_table(df, path, caption, label, align=None, resize=True):
    if df is None or df.empty:
        path.write_text("% Empty table\n")
        return

    if align is None:
        align = "l" * len(df.columns)

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
    ]

    if resize:
        lines.append(r"\resizebox{\textwidth}{!}{%")

    lines += [
        rf"\begin{{tabular}}{{{align}}}",
        r"\toprule",
        " & ".join(str(c) for c in df.columns) + r" \\",
        r"\midrule",
    ]

    for _, row in df.iterrows():
        lines.append(" & ".join(latex_escape_cell(row[c]) for c in df.columns) + r" \\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}%",
    ]

    if resize:
        lines.append(r"}")

    lines.append(r"\end{table}")

    path.write_text("\n".join(lines) + "\n")

# ---------------------------------------------------------------------
# Build combined annotation lookup
# ---------------------------------------------------------------------

def process_col(df):
    for c in [
        "edge_process_refined_v4",
        "edge_process_refined_v3",
        "edge_process",
        "process",
        "Process",
        "pathway",
        "category",
        "reaction_process",
    ]:
        if c in df.columns:
            return c
    return None

def source_target_cols(df):
    for a, b in [
        ("source_norm", "target_norm"),
        ("source_clean", "target_clean"),
        ("source", "target"),
        ("a", "b"),
    ]:
        if a in df.columns and b in df.columns:
            return a, b
    return None, None

def row_to_annotation(row, proc, source_file, priority):
    source_name = safe_str(row.get("source_name", ""))
    target_name = safe_str(row.get("target_name", ""))

    source_clean = safe_str(row.get("source_clean", row.get("source", "")))
    target_clean = safe_str(row.get("target_clean", row.get("target", "")))

    edge_readable = safe_str(row.get("edge_readable", ""))
    if not edge_readable:
        if source_name or target_name:
            left = source_name if source_name else source_clean
            right = target_name if target_name else target_clean
            edge_readable = f"{left} -> {right}"
        else:
            edge_readable = safe_str(row.get("edge", row.get("edge_id", "")))

    return {
        "edge_process": clean_process_label(proc),
        "source_name": source_name,
        "target_name": target_name,
        "source_class": safe_str(row.get("source_class", "")),
        "target_class": safe_str(row.get("target_class", "")),
        "edge_readable": edge_readable,
        "reaction_ids": safe_str(row.get("reaction_ids", "")),
        "catalysts": safe_str(row.get("catalysts", "")),
        "source_file": str(source_file),
        "priority": priority,
    }

def build_combined_lookup():
    votes = defaultdict(list)

    for priority, src in enumerate(ANNOTATION_SOURCES):
        if not src.exists():
            continue

        print(f"[lookup] reading {src}")
        try:
            df = pd.read_csv(src, low_memory=False)
        except Exception as e:
            print(f"  SKIP read error: {e}")
            continue

        pcol = process_col(df)
        scol, tcol = source_target_cols(df)

        if pcol is None:
            print("  SKIP no process/pathway column")
            continue

        candidate_rows = []

        if scol and tcol:
            for _, r in df.iterrows():
                proc = r.get(pcol)
                k = pair_key(r.get(scol), r.get(tcol))
                if k:
                    candidate_rows.append((k, proc, r))

        for edge_col in ["edge", "edge_id", "edge_display", "feature"]:
            if edge_col in df.columns:
                for _, r in df.iterrows():
                    a, b = split_edge(r.get(edge_col))
                    k = pair_key(a, b) if a and b else None
                    if k:
                        candidate_rows.append((k, r.get(pcol), r))

        added = 0
        for k, proc, r in candidate_rows:
            ann = row_to_annotation(r, proc, src, priority)
            # Skip empty/uninformative labels only if there are no useful fields.
            if ann["edge_process"] == "Unannotated" and not ann["edge_readable"]:
                continue
            votes[k].append(ann)
            added += 1

        print(f"  added annotation rows: {added}")

    lookup = {}

    for k, anns in votes.items():
        # Prefer the first high-priority non-Unannotated process.
        chosen_proc = None
        for ann in sorted(anns, key=lambda d: d["priority"]):
            if ann["edge_process"] != "Unannotated":
                chosen_proc = ann["edge_process"]
                break

        if chosen_proc is None:
            chosen_proc = Counter(a["edge_process"] for a in anns).most_common(1)[0][0]

        # For readable metadata, take the highest-priority record that has a useful readable edge.
        chosen_meta = None
        for ann in sorted(anns, key=lambda d: d["priority"]):
            if ann["edge_readable"]:
                chosen_meta = ann
                break

        if chosen_meta is None:
            chosen_meta = sorted(anns, key=lambda d: d["priority"])[0]

        out = dict(chosen_meta)
        out["edge_process"] = chosen_proc
        out["n_annotation_votes"] = len(anns)
        lookup[k] = out

    return lookup

# ---------------------------------------------------------------------
# Annotate current coefficient files
# ---------------------------------------------------------------------

def load_current_coeffs(task):
    path = CUR_RICCI / task["folder"] / "selected_coefficients_final_trainval_model.csv"
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    keys = []
    srcs = []
    tgts = []

    for edge in df["edge"].astype(str):
        a, b = split_edge(edge)
        srcs.append(a)
        tgts.append(b)
        keys.append(pair_key(a, b))

    df["_source_token"] = srcs
    df["_target_token"] = tgts
    df["_pair_key"] = keys

    return df

def annotate_task(task, lookup):
    df = load_current_coeffs(task)

    anns = []
    for _, r in df.iterrows():
        ann = lookup.get(r["_pair_key"], {})
        process = ann.get("edge_process", "Unannotated")
        edge_readable = ann.get("edge_readable", "")

        if not edge_readable or edge_readable == "nan":
            edge_readable = safe_str(r.get("edge_display", r.get("edge", "")))

        row = r.to_dict()
        row.update({
            "mapped_edge_process": process,
            "mapped_edge_readable": edge_readable,
            "mapped_source_name": ann.get("source_name", ""),
            "mapped_target_name": ann.get("target_name", ""),
            "mapped_source_class": ann.get("source_class", ""),
            "mapped_target_class": ann.get("target_class", ""),
            "mapped_reaction_ids": ann.get("reaction_ids", ""),
            "mapped_catalysts": ann.get("catalysts", ""),
            "mapping_source_file": ann.get("source_file", ""),
            "mapped": bool(ann),
        })
        anns.append(row)

    out = pd.DataFrame(anns)
    out["abs_beta"] = pd.to_numeric(out["beta_standardised"], errors="coerce").abs()
    out["push"] = pd.to_numeric(out["average_realised_push"], errors="coerce").abs()

    return out

# ---------------------------------------------------------------------
# Table builders
# ---------------------------------------------------------------------

def add_binary_direction(df, positive_class):
    beta = pd.to_numeric(df["beta_standardised"], errors="coerce")
    out = df.copy()
    out["direction"] = np.where(
        beta >= 0,
        f"toward {positive_class}",
        f"away from {positive_class}",
    )
    return out

def process_summary(df, task):
    group_cols = ["mapped_edge_process"]
    if task["kind"] == "multiclass" and "class" in df.columns:
        group_cols = ["class", "mapped_edge_process"]

    rows = []

    for keys, sub in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)

        beta = pd.to_numeric(sub["beta_standardised"], errors="coerce")
        push = pd.to_numeric(sub["average_realised_push"], errors="coerce").abs()

        k0 = sub[sub["feature_type"] == "K0"]
        b = sub[sub["feature_type"] == "B"]

        k0_beta = pd.to_numeric(k0["beta_standardised"], errors="coerce")

        row = {}

        if task["kind"] == "multiclass":
            row["Class"] = keys[0]
            row["Process"] = keys[1]
        else:
            row["Process"] = keys[0]

        row.update({
            "Features": len(sub),
            "Unique edges": sub["_pair_key"].nunique(),
            "Total $|\\beta|$": beta.abs().sum(),
            "Mean $\\beta$": beta.mean(),
            "$B$ features": len(b),
            "$K_0$ features": len(k0),
            "$K_0$ positive": (k0_beta > 0).mean() if len(k0_beta) else np.nan,
            "Total push": push.sum(),
        })

        rows.append(row)

    out = pd.DataFrame(rows)
    out = out[out["Process"] != "Unannotated"].copy()

    if out.empty:
        return out

    out = out.sort_values("Total $|\\beta|$", ascending=False)

    for c in ["Total $|\\beta|$", "Mean $\\beta$", "$K_0$ positive", "Total push"]:
        out[c] = out[c].map(fmt)

    return out

def push_summary(df, task):
    group_cols = ["mapped_edge_process"]
    if task["kind"] == "multiclass" and "class" in df.columns:
        group_cols = ["class", "mapped_edge_process"]

    rows = []

    for keys, sub in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)

        beta = pd.to_numeric(sub["beta_standardised"], errors="coerce")
        push = pd.to_numeric(sub["average_realised_push"], errors="coerce").abs()

        b = sub[sub["feature_type"] == "B"]
        k0 = sub[sub["feature_type"] == "K0"]

        row = {}

        if task["kind"] == "multiclass":
            row["Class"] = keys[0]
            row["Process"] = keys[1]
        else:
            row["Process"] = keys[0]

        row.update({
            "Features": len(sub),
            "Unique edges": sub["_pair_key"].nunique(),
            "Total push": push.sum(),
            "$B$ push": pd.to_numeric(b["average_realised_push"], errors="coerce").abs().sum(),
            "$K_0$ push": pd.to_numeric(k0["average_realised_push"], errors="coerce").abs().sum(),
            "Total $|\\beta|$": beta.abs().sum(),
        })

        rows.append(row)

    out = pd.DataFrame(rows)
    out = out[out["Process"] != "Unannotated"].copy()

    if out.empty:
        return out

    out = out.sort_values("Total push", ascending=False)

    for c in ["Total push", "$B$ push", "$K_0$ push", "Total $|\\beta|$"]:
        out[c] = out[c].map(fmt)

    return out

def top_feature_table(df, task, n=25):
    out = df.copy()
    out = out.sort_values("abs_beta", ascending=False).head(n).copy()

    rows = []
    for i, (_, r) in enumerate(out.iterrows(), start=1):
        row = {
            "Rank": i,
        }

        if task["kind"] == "multiclass":
            row["Class"] = safe_str(r.get("class", ""))
        else:
            row["Direction"] = "toward " + task["positive_class"] if float(r["beta_standardised"]) >= 0 else "away from " + task["positive_class"]

        row.update({
            "Edge": safe_str(r.get("mapped_edge_readable", r.get("edge_display", r.get("edge", "")))),
            "Process": safe_str(r.get("mapped_edge_process", "Unannotated")),
            "Type": safe_str(r.get("feature_type", "")),
            "$\\beta$": fmt(r.get("beta_standardised")),
            "Push": fmt(r.get("average_realised_push")),
        })

        rows.append(row)

    return pd.DataFrame(rows)

def coverage_rows(annotated_by_task):
    rows = []
    for task in TASKS:
        df = annotated_by_task[task["folder"]]
        rows.append({
            "Task": task["name"],
            "Selected rows": len(df),
            "Mapped rows": int(df["mapped"].sum()),
            "Mapped row rate": fmt(df["mapped"].mean()),
            "Unique edges": df["_pair_key"].nunique(),
            "Mapped unique edges": df.loc[df["mapped"], "_pair_key"].nunique(),
            "Mapped unique-edge rate": fmt(df.loc[df["mapped"], "_pair_key"].nunique() / df["_pair_key"].nunique()),
            "Unannotated rows after mapping": int((df["mapped_edge_process"] == "Unannotated").sum()),
        })

    return pd.DataFrame(rows)

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

print("[1/5] Building combined annotation lookup...")
lookup = build_combined_lookup()
print(f"[lookup] unique edge keys: {len(lookup)}")

print("[2/5] Annotating current Ricci coefficient files...")
annotated_by_task = {}

for task in TASKS:
    df = annotate_task(task, lookup)

    if task["kind"] == "binary":
        df = add_binary_direction(df, task["positive_class"])

    annotated_by_task[task["folder"]] = df

    out_csv = CSV_DIR / f"{task['folder']}_selected_coefficients_mapped.csv"
    df.to_csv(out_csv, index=False)

    print()
    print(f"--- {task['name']} ---")
    print(f"rows: {len(df)}")
    print(f"mapped rows: {int(df['mapped'].sum())} ({df['mapped'].mean():.3f})")
    print("process counts:")
    print(df["mapped_edge_process"].value_counts().head(20).to_string())

print("\n[3/5] Writing coverage table...")
coverage = coverage_rows(annotated_by_task)
coverage.to_csv(CSV_DIR / "ricci_annotation_coverage.csv", index=False)

write_latex_table(
    coverage,
    TABLE_DIR / "ricci_annotation_coverage.tex",
    "Coverage of the recovered Ricci interpretation lookup when mapped onto the corrected standardised Ricci selected coefficients.",
    "tab:ricci_annotation_coverage",
    align="lrrrrrrr",
)

print("[4/5] Writing process and top-edge interpretation tables...")

all_process_parts = []
all_push_parts = []
all_top_parts = []

for task in TASKS:
    folder = task["folder"]
    df = annotated_by_task[folder]

    coeff = process_summary(df, task).head(25)
    push = push_summary(df, task).head(25)
    top = top_feature_table(df, task, n=25)

    coeff_csv = CSV_DIR / f"{folder}_process_coefficients_top25_mapped.csv"
    push_csv = CSV_DIR / f"{folder}_process_push_top25_mapped.csv"
    top_csv = CSV_DIR / f"{folder}_top_features_top25_mapped.csv"

    coeff.to_csv(coeff_csv, index=False)
    push.to_csv(push_csv, index=False)
    top.to_csv(top_csv, index=False)

    coeff_tex = TABLE_DIR / f"ricci_{folder}_process_coefficients_top25_mapped.tex"
    push_tex = TABLE_DIR / f"ricci_{folder}_process_push_top25_mapped.tex"
    top_tex = TABLE_DIR / f"ricci_{folder}_top_features_top25_mapped.tex"

    write_latex_table(
        coeff,
        coeff_tex,
        f"Top mapped process categories by coefficient magnitude for the standardised Ricci classifier on {task['name']}. Unannotated edges are excluded from this table.",
        f"tab:ricci_{folder}_process_coefficients_mapped",
        resize=True,
    )

    write_latex_table(
        push,
        push_tex,
        f"Top mapped process categories by realised push for the standardised Ricci classifier on {task['name']}. Unannotated edges are excluded from this table.",
        f"tab:ricci_{folder}_process_push_mapped",
        resize=True,
    )

    write_latex_table(
        top,
        top_tex,
        f"Top reaction-level Ricci features for the standardised Ricci classifier on {task['name']}, after remapping edges to recovered process and metabolite annotations.",
        f"tab:ricci_{folder}_top_features_mapped",
        resize=True,
    )

    all_process_parts.append(coeff_tex.read_text())
    all_push_parts.append(push_tex.read_text())
    all_top_parts.append(top_tex.read_text())

print("[5/5] Writing combined LaTeX bundles...")

main_parts = [
    (TABLE_DIR / "ricci_annotation_coverage.tex").read_text(),
    (TABLE_DIR / "ricci_IBD_vs_nonIBD_process_coefficients_top25_mapped.tex").read_text(),
    (TABLE_DIR / "ricci_IBD_vs_nonIBD_process_push_top25_mapped.tex").read_text(),
    (TABLE_DIR / "ricci_IBD_vs_nonIBD_top_features_top25_mapped.tex").read_text(),
]

(TABLE_DIR / "latex_ricci_interpretation_main_recommended.tex").write_text("\n\n".join(main_parts) + "\n")

appendix_parts = [
    (TABLE_DIR / "ricci_annotation_coverage.tex").read_text(),
    *all_process_parts,
    *all_push_parts,
    *all_top_parts,
]

(TABLE_DIR / "latex_ricci_interpretation_appendix_all_mapped.tex").write_text("\n\n".join(appendix_parts) + "\n")

# A compact appendix: only process coefficients and top features, not push tables.
compact_parts = [
    (TABLE_DIR / "ricci_annotation_coverage.tex").read_text(),
    *all_process_parts,
    *all_top_parts,
]

(TABLE_DIR / "latex_ricci_interpretation_appendix_compact_mapped.tex").write_text("\n\n".join(compact_parts) + "\n")

print()
print("[READY] Mapped Ricci interpretation tables built.")
print("Table directory:")
print(TABLE_DIR)
print()
print("Main recommended:")
print(TABLE_DIR / "latex_ricci_interpretation_main_recommended.tex")
print()
print("Appendix compact:")
print(TABLE_DIR / "latex_ricci_interpretation_appendix_compact_mapped.tex")
print()
print("Appendix all mapped:")
print(TABLE_DIR / "latex_ricci_interpretation_appendix_all_mapped.tex")
print()
print("Coverage:")
print(TABLE_DIR / "ricci_annotation_coverage.tex")
