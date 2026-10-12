#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import re
import numpy as np
import pandas as pd

BASE = Path.home() / "Real_Data"

THREEWAY_DIR_CANDIDATES = [
    BASE / "Ricci_Classifier_3way",
    BASE / "Ricci_Classifier_3_way",
    BASE / "Ricci_3way",
]

BINARY_DIR_CANDIDATES = [
    BASE / "Ricci_Classifier_IBD_vs_nonIBD",
]

OUTDIR = BASE / "Ricci_Process_Tables_Split_B_K0"
OUTDIR.mkdir(exist_ok=True)

TOP_N = 28

# ============================================================
# Process classifier
# ============================================================

def clean_met(x):
    s = str(x).strip()
    s = re.sub(r"^M_", "", s)
    s = s.replace("__91__", "[").replace("__93__", "]")
    return s.lower()

def base_met(x):
    return re.sub(r"\[[^\]]+\]$", "", clean_met(x))

def is_currency(x):
    return base_met(x) in {
        "h", "h2o", "atp", "adp", "amp", "pi", "ppi",
        "nad", "nadh", "nadp", "nadph", "coa", "accoa",
        "co2", "nh4", "na1", "k", "o2", "h2o2",
        "q8", "q8h2", "mqn7", "mql7", "mqn8", "mql8",
        "fad", "fadh2", "fdxrd", "fdxox", "utp", "udp",
        "ctp", "cdp", "gtp", "gdp",
    }

def met_class(x):
    b = base_met(x)

    fibre = {
        "inulin", "galactan", "arabinan101", "raffin",
        "mannan", "arabinoxyl", "pect", "amylopect900",
        "cellb", "cellttr", "xylottr", "xylnt",
    }

    dietary_simple = {
        "glc_d", "fru", "gal", "xyl_d", "xylu_d",
        "arab_l", "arab_d", "mann", "rib_d",
        "lac_l", "acgal", "bglc",
    }

    central_carbon = {
        "g6p", "f6p", "fdp", "g1p", "g3p", "dhap",
        "pep", "pyr", "r5p", "xu5p_d", "xu1p_d",
        "dxyl5p", "6pgc", "2ddg6p", "tag6p_d",
    }

    complex_non_fibre = {
        "malt", "malt6p", "malttr", "maltttr", "malthx",
        "malthp", "maltpt", "glycogen", "glycogenb",
        "sucr", "suc6p", "tre",
    }

    one_carbon = {
        "fol", "thf", "5mthf", "10fthf", "mlthf",
        "hcys_l", "amet", "ahcys",
    }

    nucleotide = {
        "ade", "adn", "amp", "adp", "atp",
        "gua", "gsn", "gmp", "gdp", "gtp",
        "dgsn", "dgmp", "dgtp",
        "cmp", "cdp", "ctp", "csn",
        "ump", "udp", "utp",
        "dcyt", "dudp", "dttp", "thymd",
        "prpp", "5furimp",
    }

    lipid = {
        "stcoa", "accoa", "2mbcoa", "ibcoa", "ppal", "ppoh",
        "pg180", "pg141", "pgp141", "ocdca", "malacp",
        "13mmyrsacp", "13mtmrs2eacp", "3ocmrs7eacp",
        "2agpg180", "2agpg141",
    }

    redox = {
        "nad", "nadh", "nadp", "nadph", "fad", "fadh2",
        "fdxrd", "fdxox", "q8", "q8h2", "mqn7", "mql7",
        "mqn8", "mql8", "2dmmq8",
    }

    oxidative = {"o2", "h2o2"}
    phosphate = {"pi", "ppi"}

    energy = {
        "atp", "adp", "amp", "gtp", "gdp",
        "ctp", "cdp", "utp", "udp",
    }

    amino = {
        "ala_l", "arg_l", "asn_l", "asp_l", "cys_l", "glu_l",
        "glu_d", "gln_l", "gly", "his_l", "ile_l", "leu_l",
        "lys_l", "met_l", "phe_l", "pro_l", "ser_l", "thr_l",
        "trp_l", "tyr_l", "val_l",
    }

    scfa = {
        "ac", "acald", "actp", "but", "btcoa", "ppa", "ppal",
        "ppoh", "2obut", "succ", "succoa",
    }

    if b in fibre:
        return "Fibre"
    if b in dietary_simple:
        return "Dietary simple carbohydrates"
    if b in central_carbon:
        return "Central carbon"
    if b in complex_non_fibre:
        return "Complex non-fibre carbohydrates"
    if "_glc" in b or b.endswith("glc") or "diglc" in b or b.endswith("glyc"):
        return "Glycosides"
    if b in one_carbon:
        return "One-carbon"
    if b in nucleotide:
        return "Nucleotide metabolism"
    if b in lipid:
        return "Lipid metabolism"
    if b in oxidative:
        return "Oxidative stress"
    if b in redox:
        return "Redox metabolism"
    if b in phosphate:
        return "Phosphate metabolism"
    if b in energy:
        return "Energy metabolism"
    if b in amino:
        return "Amino acid metabolism"
    if b in scfa:
        return "SCFA-related metabolism"

    return "Other"

def process_name(source, target):
    if is_currency(source) or is_currency(target):
        return "Currency-coupled reactions"

    s = met_class(source)
    t = met_class(target)

    if s == t:
        if s == "One-carbon":
            return "One-carbon (internal)"
        if s == "Central carbon":
            return "Central carbon (internal)"
        if s == "Glycosides":
            return "Glycoside (internal)"
        if s == "Energy metabolism":
            return "Energy metabolism (internal)"
        if s == "Amino acid metabolism":
            return "Amino acid metabolism (internal)"
        return s

    return f"{s} -> {t}"

def locate_dir(candidates):
    for d in candidates:
        if d.exists():
            return d
    return None

def locate_coef_file(task_dir):
    candidates = [
        task_dir / "selected_coefficients_nonzero.csv",
        task_dir / "selected_coefficients_nonzero_v4.csv",
        task_dir / "nonzero_coefficients.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    hits = sorted(task_dir.glob("*nonzero*.csv"))
    return hits[0] if hits else None

def detect_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def prepare_coef(df, binary_keep_ibd=False):
    coef_col = detect_col(df, ["coefficient", "coef", "beta"])
    if coef_col is None:
        raise ValueError(f"No coefficient column found. Columns: {df.columns.tolist()}")

    required = ["class", "component", "edge_id", "source", "target"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns {missing}. Columns: {df.columns.tolist()}")

    df = df.copy()
    df["coefficient"] = pd.to_numeric(df[coef_col], errors="coerce")
    df = df.dropna(subset=["coefficient"])
    df = df[df["coefficient"] != 0].copy()

    df["class"] = df["class"].astype(str).str.strip()
    df["component"] = df["component"].astype(str).str.strip()

    if binary_keep_ibd:
        df = df[df["class"] == "IBD"].copy()

    # Remove duplicate class/component/edge rows if any.
    dedupe_cols = ["class", "edge_id", "component"]
    df = (
        df.sort_values("coefficient", key=lambda s: s.abs(), ascending=False)
          .drop_duplicates(dedupe_cols)
          .copy()
    )

    df["Process category"] = [
        process_name(s, t) for s, t in zip(df["source"], df["target"])
    ]

    return df

# ============================================================
# 3-way table
# ============================================================

threeway_dir = locate_dir(THREEWAY_DIR_CANDIDATES)
if threeway_dir is None:
    print("Could not find 3-way directory. Checked:")
    for d in THREEWAY_DIR_CANDIDATES:
        print(" -", d)
else:
    threeway_file = locate_coef_file(threeway_dir)
    if threeway_file is None:
        print("Could not find 3-way coefficient file in:", threeway_dir)
    else:
        print("\nUsing 3-way coefficient file:", threeway_file)
        df3_raw = pd.read_csv(threeway_file)
        df3 = prepare_coef(df3_raw, binary_keep_ibd=False)

        print("\n3-way class counts:")
        print(df3["class"].value_counts())
        print("\n3-way component counts:")
        print(df3["component"].value_counts())

        classes = ["CD", "UC", "nonIBD"]

        base = (
            df3.groupby("Process category")
            .agg(
                **{
                    "Total importance": ("coefficient", lambda x: x.abs().sum()),
                    "Features": ("coefficient", "size"),
                    "Unique edges": ("edge_id", "nunique"),
                    r"Mean $|\beta|$": ("coefficient", lambda x: x.abs().mean()),
                }
            )
            .reset_index()
        )

        pieces = [base]

        for comp in ["B", "K0"]:
            for cls in classes:
                sub = df3[(df3["component"] == comp) & (df3["class"] == cls)]
                tmp = (
                    sub.groupby("Process category")
                    .agg(**{f"Mean {comp} {cls}": ("coefficient", "mean")})
                    .reset_index()
                )
                pieces.append(tmp)

        out3 = pieces[0]
        for p in pieces[1:]:
            out3 = out3.merge(p, on="Process category", how="left")

        out3 = (
            out3.sort_values("Total importance", ascending=False)
                .head(TOP_N)
                .copy()
        )

        out3_path = OUTDIR / "table31_3way_split_B_K0_top28.csv"
        out3.to_csv(out3_path, index=False)

        print("\n" + "=" * 110)
        print("TABLE 31 REBUILD — 3-WAY, B AND K0 SPLIT BY CLASS")
        print("=" * 110)
        print(out3.to_string(index=False))
        print("\nSaved:", out3_path)

# ============================================================
# IBD vs nonIBD table
# ============================================================

binary_dir = locate_dir(BINARY_DIR_CANDIDATES)
if binary_dir is None:
    print("Could not find IBD vs nonIBD directory.")
else:
    binary_file = locate_coef_file(binary_dir)
    if binary_file is None:
        print("Could not find binary coefficient file in:", binary_dir)
    else:
        print("\nUsing IBD vs nonIBD coefficient file:", binary_file)
        df2_raw = pd.read_csv(binary_file)
        df2 = prepare_coef(df2_raw, binary_keep_ibd=True)

        print("\nBinary class counts after keeping IBD:")
        print(df2["class"].value_counts())
        print("\nBinary component counts:")
        print(df2["component"].value_counts())

        base = (
            df2.groupby("Process category")
            .agg(
                **{
                    "Total importance": ("coefficient", lambda x: x.abs().sum()),
                    "Features": ("coefficient", "size"),
                    "Unique edges": ("edge_id", "nunique"),
                    r"Mean $|\beta|$": ("coefficient", lambda x: x.abs().mean()),
                }
            )
            .reset_index()
        )

        mean_B = (
            df2[df2["component"] == "B"]
            .groupby("Process category")
            .agg(**{r"Mean B $\beta$": ("coefficient", "mean")})
            .reset_index()
        )

        mean_K0 = (
            df2[df2["component"] == "K0"]
            .groupby("Process category")
            .agg(
                **{
                    r"Mean $K_0$ $\beta$": ("coefficient", "mean"),
                    r"\% positive $K_0$": ("coefficient", lambda x: 100.0 * np.mean(x > 0)),
                    r"$K_0$ features": ("coefficient", "size"),
                }
            )
            .reset_index()
        )

        out2 = base.merge(mean_B, on="Process category", how="left")
        out2 = out2.merge(mean_K0, on="Process category", how="left")

        out2 = (
            out2.sort_values("Total importance", ascending=False)
                .head(TOP_N)
                .copy()
        )

        bad = out2[out2["Features"] > 2 * out2["Unique edges"]]
        if len(bad):
            print("\nWARNING: feature count violation:")
            print(bad[["Process category", "Features", "Unique edges"]].to_string(index=False))
        else:
            print("\nSanity check passed for binary table: Features <= 2 * Unique edges.")

        out2_path = OUTDIR / "table37_IBD_vs_nonIBD_split_B_K0_top28.csv"
        out2.to_csv(out2_path, index=False)

        print("\n" + "=" * 110)
        print("TABLE 37 REBUILD — IBD vs nonIBD, B AND K0 SPLIT")
        print("=" * 110)
        print(out2.to_string(index=False))
        print("\nSaved:", out2_path)
