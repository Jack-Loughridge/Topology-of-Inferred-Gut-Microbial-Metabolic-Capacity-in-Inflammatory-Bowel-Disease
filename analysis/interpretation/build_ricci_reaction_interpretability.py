#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Ricci interpretability pipeline.

Adds:
1. exact source/target metabolite IDs
2. cleaned metabolite tokens
3. BiGG/VMH-style names where available
4. heuristic biological classes with no empty labels
5. edge-level pathway labels
6. AGORA reaction IDs/catalysts supporting selected edges
7. pathway/class/task-level coefficient aggregation

Inputs:
    ~/Real_Data/AGORA_reactions_canon.parquet
    ~/Real_Data/metabolite_mapping_FULL.csv
    classifier folders containing selected_coefficients_nonzero.csv

Outputs:
    ~/Real_Data/Ricci_Interpretability/
"""

from __future__ import annotations

import re
import json
import argparse
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import pandas as pd


DEFAULT_BASE = Path.home() / "Real_Data"

DEFAULT_RESULT_DIRS = [
    "Ricci_Classifier_3way",
    "Ricci_Classifier_IBD_vs_nonIBD",
    "Ricci_Classifier_CD_vs_UC",
    "Ricci_Classifier_nonIBD_vs_CD",
    "Ricci_Classifier_nonIBD_vs_UC",
]


# ---------------------------------------------------------------------
# Basic metabolite parsing
# ---------------------------------------------------------------------

def norm_raw_met(x) -> str:
    """Normalise AGORA metabolite ID for matching against inputs/outputs."""
    if pd.isna(x):
        return ""
    return str(x).strip().lower()


def decode_met_id(x) -> str:
    """
    Convert AGORA/COBRA style IDs into readable tokens.

    Example:
        M_FOL__91__C__93__ -> fol[c]
        M_glc_D__91__c__93__ -> glc_d[c]
    """
    if pd.isna(x):
        return ""

    s = str(x).strip()
    s = s.replace("__91__", "[")
    s = s.replace("__93__", "]")
    s = s.replace("_91__", "[")
    s = s.replace("__93", "]")

    if s.lower().startswith("m_"):
        s = s[2:]

    return s.lower()


def strip_compartment(cleaned: str) -> str:
    if not isinstance(cleaned, str):
        return ""
    return re.sub(r"\[.*?\]$", "", cleaned.lower())


def get_compartment(cleaned: str) -> str:
    if not isinstance(cleaned, str):
        return ""
    m = re.search(r"\[([^\]]+)\]$", cleaned.lower())
    return m.group(1) if m else ""


# ---------------------------------------------------------------------
# BiGG/name mapping
# ---------------------------------------------------------------------

MANUAL_NAMES = {
    "glc_d": "D-Glucose",
    "fru": "D-Fructose",
    "fol": "Folate",
    "thf": "5,6,7,8-Tetrahydrofolate",
    "nad": "Nicotinamide adenine dinucleotide",
    "nadh": "NADH",
    "nadp": "Nicotinamide adenine dinucleotide phosphate",
    "nadph": "NADPH",
    "o2": "Oxygen",
    "h2o2": "Hydrogen peroxide",
    "accoa": "Acetyl-CoA",
    "coa": "Coenzyme A",
    "amet": "S-Adenosyl-L-methionine",
    "ahcys": "S-Adenosyl-L-homocysteine",
    "hcys_l": "L-Homocysteine",
    "dgsn": "Deoxyguanosine",
    "gua": "Guanine",
    "adp": "ADP",
    "atp": "ATP",
    "amp": "AMP",
    "pi": "Phosphate",
    "ppi": "Diphosphate",
    "pyr": "Pyruvate",
    "lac_l": "L-Lactate",
    "lac_d": "D-Lactate",
    "inulin": "Inulin",
    "raffin": "Raffinose",
    "galactan": "Galactan",
    "bglc": "Beta-D-glucose",
    "g6p": "D-Glucose 6-phosphate",
    "f6p": "D-Fructose 6-phosphate",
    "xu5p_d": "D-Xylulose 5-phosphate",
    "xylu_d": "D-Xylulose",
    "nh4": "Ammonium",
    "h": "Proton",
    "h2o": "Water",
    "co2": "Carbon dioxide",
    "na1": "Sodium",
}


def load_name_mapping(base: Path) -> dict:
    mapping_path = base / "metabolite_mapping_FULL.csv"
    name_map = {}

    if mapping_path.exists():
        mp = pd.read_csv(mapping_path)
        if "cleaned" in mp.columns and "name" in mp.columns:
            for _, r in mp.dropna(subset=["cleaned"]).iterrows():
                cleaned = str(r["cleaned"]).lower()
                name = r.get("name", np.nan)
                if isinstance(name, str) and name.strip():
                    name_map[cleaned] = name.strip()

    return name_map


def met_name(cleaned: str, name_map: dict) -> str:
    cleaned = str(cleaned).lower()
    base = strip_compartment(cleaned)

    if cleaned in name_map:
        return name_map[cleaned]

    if base in MANUAL_NAMES:
        return MANUAL_NAMES[base]

    return ""


# ---------------------------------------------------------------------
# Biological classification
# ---------------------------------------------------------------------

AA_TOKENS = [
    "ala", "arg", "asn", "asp", "cys", "glu", "gln", "gly", "his", "ile",
    "leu", "lys", "met", "phe", "pro", "ser", "thr", "trp", "tyr", "val",
]

NUCLEOTIDE_TOKENS = [
    "dgsn", "gsn", "gua", "ade", "adn", "thymd", "dcyt", "csn",
    "cmp", "ump", "amp", "gmp", "dtmp", "dttp", "dutp", "dudp",
    "prpp", "5furimp", "r5p",
]

CARB_TOKENS = [
    "glc", "fru", "gal", "xyl", "xylu", "xu5p", "xu1p",
    "g6p", "f6p", "g1p", "s7p", "r5p", "lac", "pyr",
    "glycogen", "starch", "inulin", "raffin", "galactan",
    "acgam", "bglc", "malt", "cellob",
]

ONE_CARBON_TOKENS = [
    "fol", "thf", "5mthf", "10fthf", "amet", "ahcys", "hcys",
    "met_l",
]

REDOX_TOKENS = [
    "nad", "nadh", "nadp", "nadph", "fadh", "fdx", "q8", "mql", "mqn",
]

ENERGY_TOKENS = [
    "atp", "adp", "amp", "gtp", "gdp", "utp", "udp", "ctp", "cdp",
]

LIPID_TOKENS = [
    "acp", "coa", "accoa", "malcoa", "malacp", "pg", "pgp", "pe",
    "ps", "pa", "dgr", "agpe", "agpg", "cdpdag", "lipid",
    "myrs", "palm", "stcoa", "ocdca", "ttdcea", "hdcoa", "hddca",
    "dca", "dcea", "oct", "dec", "butacp", "m3h", "mtd", "oddca",
]

BILE_STEROL_TOKENS = [
    "chol", "cholate", "dca", "ocdca", "dhchol", "steroid",
    "sterol", "bile",
]

SCFA_TOKENS = [
    "acet", "ac", "but", "prop", "ppoh", "ppal", "acald", "butyr",
    "isobut", "valer",
]

OXIDATIVE_TOKENS = [
    "o2", "h2o2", "ros",
]

INORGANIC_TOKENS = [
    "h", "h2o", "co2", "na1", "k", "cl", "zn2", "fe2", "fe3",
    "ni2", "mg2", "mn2", "ca2", "nh4",
]

PHOSPHATE_TOKENS = [
    "pi", "ppi", "phos", "phosphate",
]

XENOBIOTIC_GLYCOSIDE_TOKENS = [
    "digoxin", "sfn", "glc]", "_glc", "gluc", "sulfac", "tmao",
    "pcresol", "cresol", "bzd",
]


def contains_any(s: str, tokens: list[str]) -> bool:
    return any(t in s for t in tokens)


def classify_metabolite(cleaned: str, name: str = "") -> tuple[str, str]:
    """
    Returns (class, confidence).
    No empty classes allowed: fallback is 'other'.
    """
    c = str(cleaned).lower()
    base = strip_compartment(c)
    n = str(name).lower() if isinstance(name, str) else ""
    text = f"{base} {n}"

    # Transport/currency labels handled later at edge level, but we still classify metabolites.
    if base in INORGANIC_TOKENS:
        return "inorganic_currency", "heuristic"

    if contains_any(text, ONE_CARBON_TOKENS):
        return "one_carbon_metabolism", "heuristic"

    if contains_any(text, REDOX_TOKENS):
        return "redox_metabolism", "heuristic"

    if contains_any(text, OXIDATIVE_TOKENS):
        return "oxidative_stress", "heuristic"

    if contains_any(text, ENERGY_TOKENS):
        return "energy_metabolism", "heuristic"

    if contains_any(text, NUCLEOTIDE_TOKENS):
        return "nucleotide_metabolism", "heuristic"

    if contains_any(text, BILE_STEROL_TOKENS):
        return "bile_sterol_lipid_metabolism", "heuristic"

    if contains_any(text, LIPID_TOKENS):
        return "lipid_fatty_acid_metabolism", "heuristic"

    if contains_any(text, SCFA_TOKENS):
        return "scfa_related_metabolism", "heuristic"

    if contains_any(text, CARB_TOKENS):
        return "carbohydrate_metabolism", "heuristic"

    if contains_any(text, AA_TOKENS):
        return "amino_acid_metabolism", "heuristic"

    if contains_any(text, PHOSPHATE_TOKENS):
        return "phosphate_metabolism", "heuristic"

    if contains_any(text, XENOBIOTIC_GLYCOSIDE_TOKENS):
        return "glycoside_xenobiotic_or_secondary_metabolism", "heuristic"

    return "other", "fallback"


def edge_process(row) -> str:
    s_base = row["source_base"]
    t_base = row["target_base"]
    s_comp = row["source_compartment"]
    t_comp = row["target_compartment"]

    if s_base == t_base and s_comp and t_comp and s_comp != t_comp:
        return "transport_or_compartment_exchange"

    if row["source_class"] == "inorganic_currency" or row["target_class"] == "inorganic_currency":
        return "currency_coupled_reaction"

    if row["source_class"] == row["target_class"]:
        return row["source_class"]

    return f"{row['source_class']} -> {row['target_class']}"


# ---------------------------------------------------------------------
# Load classifier coefficient files
# ---------------------------------------------------------------------

def load_selected_coefficients(base: Path, result_dirs: list[str]) -> pd.DataFrame:
    frames = []

    for d in result_dirs:
        p = base / d / "selected_coefficients_nonzero.csv"
        if p.exists():
            tmp = pd.read_csv(p)
            tmp["task"] = d
            frames.append(tmp)
            print(f"[load] {d}: {len(tmp)} nonzero coefficients")
        else:
            print(f"[skip] missing {p}")

    if not frames:
        raise FileNotFoundError("No selected_coefficients_nonzero.csv files found.")

    df = pd.concat(frames, ignore_index=True)
    df["coefficient"] = pd.to_numeric(df["coefficient"], errors="coerce")
    df["abs_coefficient"] = df["coefficient"].abs()
    df = df.dropna(subset=["source", "target", "coefficient"]).copy()
    return df


# ---------------------------------------------------------------------
# AGORA reaction scan
# ---------------------------------------------------------------------

def split_met_list(x) -> list[str]:
    if pd.isna(x):
        return []
    return [norm_raw_met(z) for z in str(x).split(";") if str(z).strip()]


def scan_agora_for_selected_edges(
    agora_path: Path,
    selected_pairs: set[tuple[str, str]],
    max_reactions_per_edge: int = 25,
    batch_size: int = 50_000,
) -> pd.DataFrame:
    """
    Scans AGORA reactions and records which reactions/catalysts support each selected edge.

    Edge is considered supported by a reaction if:
        source in reaction inputs AND target in reaction outputs
    """
    reaction_ids = defaultdict(set)
    catalysts = defaultdict(set)
    n_hits = Counter()

    print(f"[agora] scanning {agora_path}")
    print(f"[agora] selected directed pairs: {len(selected_pairs)}")

    try:
        import pyarrow.parquet as pq

        pf = pq.ParquetFile(agora_path)
        cols = ["reaction_id", "inputs", "outputs", "catalysts"]

        seen_rows = 0
        for batch in pf.iter_batches(batch_size=batch_size, columns=cols):
            part = batch.to_pandas()
            seen_rows += len(part)

            for r in part.itertuples(index=False):
                ins = split_met_list(r.inputs)
                outs = split_met_list(r.outputs)

                if not ins or not outs:
                    continue

                for u in ins:
                    for v in outs:
                        key = (u, v)
                        if key in selected_pairs:
                            n_hits[key] += 1
                            if len(reaction_ids[key]) < max_reactions_per_edge:
                                reaction_ids[key].add(str(r.reaction_id))
                            if len(catalysts[key]) < max_reactions_per_edge:
                                catalysts[key].add(str(r.catalysts))

            if seen_rows % 500_000 < batch_size:
                print(f"[agora] scanned ~{seen_rows:,} rows", flush=True)

    except Exception as e:
        print(f"[warning] pyarrow scan failed: {e}")
        print("[warning] falling back to pandas read_parquet; this may use more memory")

        df = pd.read_parquet(agora_path, columns=["reaction_id", "inputs", "outputs", "catalysts"])

        for i, r in enumerate(df.itertuples(index=False), 1):
            ins = split_met_list(r.inputs)
            outs = split_met_list(r.outputs)

            for u in ins:
                for v in outs:
                    key = (u, v)
                    if key in selected_pairs:
                        n_hits[key] += 1
                        if len(reaction_ids[key]) < max_reactions_per_edge:
                            reaction_ids[key].add(str(r.reaction_id))
                        if len(catalysts[key]) < max_reactions_per_edge:
                            catalysts[key].add(str(r.catalysts))

            if i % 500_000 == 0:
                print(f"[agora] scanned {i:,} rows", flush=True)

    rows = []
    for u, v in selected_pairs:
        key = (u, v)
        rows.append({
            "source_norm": u,
            "target_norm": v,
            "n_agora_reactions": int(n_hits[key]),
            "reaction_ids": ";".join(sorted(reaction_ids[key])),
            "catalysts": ";".join(sorted(catalysts[key])),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=str, default=str(DEFAULT_BASE))
    parser.add_argument("--skip-agora-scan", action="store_true")
    parser.add_argument("--max-reactions-per-edge", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=50_000)
    args = parser.parse_args()

    base = Path(args.base).expanduser()
    out_dir = base / "Ricci_Interpretability"
    out_dir.mkdir(parents=True, exist_ok=True)

    agora_path = base / "AGORA_reactions_canon.parquet"
    name_map = load_name_mapping(base)

    print("=" * 100)
    print("RICCI REACTION-LEVEL INTERPRETABILITY PIPELINE")
    print("=" * 100)

    df = load_selected_coefficients(base, DEFAULT_RESULT_DIRS)

    # Metabolite cleanup and naming
    df["source_norm"] = df["source"].apply(norm_raw_met)
    df["target_norm"] = df["target"].apply(norm_raw_met)

    df["source_clean"] = df["source"].apply(decode_met_id)
    df["target_clean"] = df["target"].apply(decode_met_id)

    df["source_base"] = df["source_clean"].apply(strip_compartment)
    df["target_base"] = df["target_clean"].apply(strip_compartment)

    df["source_compartment"] = df["source_clean"].apply(get_compartment)
    df["target_compartment"] = df["target_clean"].apply(get_compartment)

    df["source_name"] = df["source_clean"].apply(lambda x: met_name(x, name_map))
    df["target_name"] = df["target_clean"].apply(lambda x: met_name(x, name_map))

    src_cls = df.apply(lambda r: classify_metabolite(r["source_clean"], r["source_name"]), axis=1)
    tgt_cls = df.apply(lambda r: classify_metabolite(r["target_clean"], r["target_name"]), axis=1)

    df["source_class"] = [x[0] for x in src_cls]
    df["source_class_confidence"] = [x[1] for x in src_cls]
    df["target_class"] = [x[0] for x in tgt_cls]
    df["target_class_confidence"] = [x[1] for x in tgt_cls]

    df["edge_process"] = df.apply(edge_process, axis=1)
    df["importance"] = df["abs_coefficient"]

    # AGORA reaction support
    if not args.skip_agora_scan:
        if not agora_path.exists():
            raise FileNotFoundError(f"Missing AGORA file: {agora_path}")

        selected_pairs = set(zip(df["source_norm"], df["target_norm"]))
        rxn = scan_agora_for_selected_edges(
            agora_path=agora_path,
            selected_pairs=selected_pairs,
            max_reactions_per_edge=args.max_reactions_per_edge,
            batch_size=args.batch_size,
        )

        df = df.merge(rxn, on=["source_norm", "target_norm"], how="left")
    else:
        df["n_agora_reactions"] = np.nan
        df["reaction_ids"] = ""
        df["catalysts"] = ""

    # Useful readable edge labels
    def readable_edge(row):
        s = row["source_name"] if isinstance(row["source_name"], str) and row["source_name"] else row["source_clean"]
        t = row["target_name"] if isinstance(row["target_name"], str) and row["target_name"] else row["target_clean"]
        return f"{s} -> {t}"

    df["edge_readable"] = df.apply(readable_edge, axis=1)

    # Save edge-level annotated coefficients
    edge_out = out_dir / "annotated_selected_coefficients.csv"
    df.to_csv(edge_out, index=False)
    print(f"\n[saved] {edge_out}")

    # Metabolite mapping summary
    all_mets = pd.concat([
        df[["source_clean", "source_name", "source_class", "source_class_confidence"]].rename(
            columns={
                "source_clean": "cleaned",
                "source_name": "name",
                "source_class": "class",
                "source_class_confidence": "confidence",
            }
        ),
        df[["target_clean", "target_name", "target_class", "target_class_confidence"]].rename(
            columns={
                "target_clean": "cleaned",
                "target_name": "name",
                "target_class": "class",
                "target_class_confidence": "confidence",
            }
        ),
    ]).drop_duplicates()

    met_out = out_dir / "metabolites_seen_in_selected_edges.csv"
    all_mets.to_csv(met_out, index=False)
    print(f"[saved] {met_out}")

    # Aggregations
    pathway_overall = (
        df.groupby("edge_process", as_index=False)
          .agg(
              total_importance=("importance", "sum"),
              n_features=("feature", "count"),
              n_unique_edges=("edge_id", "nunique"),
              mean_abs_coef=("importance", "mean"),
          )
          .sort_values("total_importance", ascending=False)
    )
    pathway_overall.to_csv(out_dir / "edge_process_importance_overall.csv", index=False)

    pathway_by_task_class = (
        df.groupby(["task", "class", "edge_process"], as_index=False)
          .agg(
              total_importance=("importance", "sum"),
              n_features=("feature", "count"),
              n_unique_edges=("edge_id", "nunique"),
              mean_abs_coef=("importance", "mean"),
          )
    )
    pathway_by_task_class["relative_importance_within_task_class"] = (
        pathway_by_task_class["total_importance"] /
        pathway_by_task_class.groupby(["task", "class"])["total_importance"].transform("sum")
    )
    pathway_by_task_class = pathway_by_task_class.sort_values(
        ["task", "class", "total_importance"],
        ascending=[True, True, False]
    )
    pathway_by_task_class.to_csv(out_dir / "edge_process_importance_by_task_class.csv", index=False)

    # Top exact edges per biological process
    top_edges_per_process = (
        df.sort_values("importance", ascending=False)
          .groupby("edge_process")
          .head(10)
          .sort_values(["edge_process", "importance"], ascending=[True, False])
    )
    top_edges_per_process.to_csv(out_dir / "top_edges_per_edge_process.csv", index=False)

    # Reaction support summary
    reaction_support = (
        df.groupby(["source_norm", "target_norm"], as_index=False)
          .agg(
              source_clean=("source_clean", "first"),
              target_clean=("target_clean", "first"),
              source_name=("source_name", "first"),
              target_name=("target_name", "first"),
              edge_readable=("edge_readable", "first"),
              edge_process=("edge_process", "first"),
              n_agora_reactions=("n_agora_reactions", "max"),
              reaction_ids=("reaction_ids", "first"),
              catalysts=("catalysts", "first"),
              max_importance=("importance", "max"),
          )
          .sort_values("max_importance", ascending=False)
    )
    reaction_support.to_csv(out_dir / "reaction_support_for_selected_edges.csv", index=False)

    # Print summary
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)

    print("\nTop biological edge processes overall:")
    print(pathway_overall.head(20).to_string(index=False))

    print("\nMetabolite class counts:")
    print(all_mets["class"].value_counts().head(20).to_string())

    if "n_agora_reactions" in df.columns:
        print("\nAGORA support:")
        print(f"Edges/features with at least one supporting AGORA reaction: {(df['n_agora_reactions'].fillna(0) > 0).mean():.2%}")

    print("\nOutput directory:")
    print(out_dir)

    print("\nKey files:")
    print(" - annotated_selected_coefficients.csv")
    print(" - edge_process_importance_overall.csv")
    print(" - edge_process_importance_by_task_class.csv")
    print(" - top_edges_per_edge_process.csv")
    print(" - reaction_support_for_selected_edges.csv")
    print("=" * 100)


if __name__ == "__main__":
    main()
