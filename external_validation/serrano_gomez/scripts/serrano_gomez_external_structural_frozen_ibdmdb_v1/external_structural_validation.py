#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Strict external structural-validation builder for the Serrano-Gomez GE50 cohort.

Primary design rule
-------------------
The IBDMDB graph representation is frozen.  External samples are projected into
that representation; they are never allowed to redefine it.

Specifically:
  * same AGORA reaction table;
  * same deterministic multi-input gate and accepted directed-edge scaffold;
  * same IBDMDB edge-specific geometric-mean baselines from
        ~/Real_Data/out_graphs/global_baselines.csv;
  * external numerator is species RELATIVE ABUNDANCE on the same percentage
    scale as the IBDMDB abundance table (MetaPhlAn species rows, not HUMAnN
    gene-family signal);
  * no external-cohort baseline and no external-cohort renormalisation;
  * weight = exp(-E_external / (Ehat_IBDMDB + 1e-8));
  * zero support remains a template edge with w=1;
  * H0 is computed on the underlying undirected weighted graph.  Its nontrivial
    death multiset is exactly the Kruskal/union-find merge multiset, equivalent
    to the minimax-path H0 filtration used in the project;
  * Ricci itself is NOT reimplemented here.  The launch helper calls the exact
    existing 14_compute_ricci_faithful_pairwise_active.py with the production
    n_paths=250 and all other production parameters explicit;
  * external [B | K0] vectors are aligned to the exact row order of the frozen
    training edge_metadata.csv.

The script is deliberately fail-closed.  Before it builds external samples,
`preflight` reconstructs the accepted scaffold from the current AGORA table,
checks it against the frozen IBDMDB baseline table, checks existing IBDMDB
GraphML attributes against the graph equation, and regression-tests the H0
implementation against existing training *_H0.npy files.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import itertools
import json
import math
import os
import random
import re
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import save_npz


# =============================================================================
# Frozen project locations
# =============================================================================

REAL = Path(os.environ.get("REAL_DATA_DIR", "/home/Jack/Real_Data")).expanduser().resolve()
GE50 = REAL / "external_validation" / "serrano_gomez_ibd" / "ge50_external_validation"

DEFAULT_OUT = GE50 / "structural_frozen_ibdmdb"
REACTIONS_PATH = REAL / "AGORA_reactions_canon.parquet"
TRAIN_SPECIES_PATH = REAL / "Real_Species_Abundances_canon.xlsx"
TRAIN_GRAPH_DIR = REAL / "out_graphs"
FROZEN_BASELINES_PATH = TRAIN_GRAPH_DIR / "global_baselines.csv"
TRAIN_H0_DIR = REAL / "out_pds"
TRAIN_RICCI_FEATURE_DIR = REAL / "Ricci_Classifier_Faithful_Eps0001_n250_v3"
HUMANN_DIR = GE50 / "humann"
MANIFEST_DIR = GE50 / "manifests"
CURRENT_PROXY_BUILDER = GE50 / "scripts" / "build_ge50_species_proxy_graphs_033.py"

GRAPH_SEED = 13
TAU_THRESHOLD = 0.0001
ROOTS_PER_INPUT = 50
MAX_BETA_POOL = 2500
WEIGHT_EPSILON = 1e-8
THETA_MIN = 0.0
ACTIVE_TOL = 1e-12

# Explicit production Ricci parameters.  The source file's historical default
# n_paths is 1000; the corrected manuscript feature set is the n250 run.
RICCI_N_PATHS = 250
RICCI_EPSILON_DIST = 1e-4
RICCI_C_SINGLE_OUT = 0.001
RICCI_BETA = 1.4
RICCI_MAX_STEPS = 10000
RICCI_SEED = 13
RICCI_MU_PRUNE_THRESHOLD = 1e-6
RICCI_TOPK_SUPPORT = 150
RICCI_MAX_LP_VARS = 200000

# These are intentionally the same strings as the original graph builder.
CURRENCY = {
    "H2O", "WATER", "PROTON", "H+", "ATP", "ADP", "PI", "PPI", "CO2", "CO(2)", "NAD", "NADH",
    "NADP", "NADPH", "FAD", "FADH2", "NAD(P)H", "OXYGEN", "O2", "NH3", "AMMONIA", "HCO3-",
    "BICARBONATE", "CO-A", "COENZYME_A", "COENZYME A", "S-ADENOSYLMETHIONINE", "SAM",
    "S-ADENOSYLHOMOCYSTEINE", "SAH", "UMP", "UDP", "UTP", "AMP", "CMP", "CDP", "CTP",
    "GMP", "GDP", "GTP",
}


# =============================================================================
# Generic helpers
# =============================================================================

def die(msg: str) -> None:
    raise RuntimeError(msg)


def sha256_file(path: Path, block: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            b = fh.read(block)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def json_dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n")
    tmp.replace(path)


def canon(x: str) -> str:
    """EXACT original metabolite canonicalisation."""
    return str(x).strip().upper().replace(" ", "_")


def is_currency(x: str) -> bool:
    """EXACT original currency check (including its compartment-name behaviour)."""
    return canon(x) in CURRENCY


def parse_listish(cell: Any) -> List[str]:
    """EXACT original list parser."""
    if pd.isna(cell):
        return []
    s = str(cell)
    for sep in [";", "|"]:
        s = s.replace(sep, ",")
    return [t.strip() for t in s.split(",") if t.strip()]


def geometric_mean(vals: Iterable[float]) -> float:
    arr = [float(v) for v in vals if float(v) > 0]
    if not arr:
        return 0.0
    return float(np.exp(np.mean(np.log(arr))))


def find_col(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    lower = {str(c).strip().lower(): c for c in df.columns}
    for c in candidates:
        if c in df.columns:
            return c
        if str(c).strip().lower() in lower:
            return lower[str(c).strip().lower()]
    return None


# =============================================================================
# Original graph-builder logic, copied faithfully
# =============================================================================

@dataclass(frozen=True)
class Reaction:
    rid: str
    inputs: Tuple[str, ...]
    outputs: Tuple[str, ...]
    catalysts: Tuple[str, ...]


def load_reactions(path: Path) -> List[Reaction]:
    ext = path.suffix.lower()
    if ext in (".parquet", ".pq"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_excel(path, sheet_name=0)

    need = ["reaction_id", "inputs", "outputs", "catalysts"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        die(f"Reaction table missing columns {missing}: {path}")

    recs: List[Reaction] = []
    for _, row in df.iterrows():
        rid = str(row["reaction_id"]).strip()
        ins = tuple(canon(x) for x in parse_listish(row["inputs"]) if not is_currency(x))
        outs = tuple(canon(x) for x in parse_listish(row["outputs"]) if not is_currency(x))
        cats = tuple(str(x).strip() for x in parse_listish(row["catalysts"]))
        if not ins or not outs or not cats:
            continue
        recs.append(Reaction(rid=rid, inputs=ins, outputs=outs, catalysts=cats))
    return recs


def build_global_substrate_graph(reactions: List[Reaction]) -> nx.Graph:
    G = nx.Graph()
    for r in reactions:
        for x in r.inputs + r.outputs:
            if not is_currency(x):
                G.add_node(canon(x))
        for a, b in itertools.combinations(r.inputs, 2):
            G.add_edge(canon(a), canon(b))
        for a in r.inputs:
            for b in r.outputs:
                if not is_currency(a) and not is_currency(b):
                    G.add_edge(canon(a), canon(b))
    return G


def multi_input_gate(
    inputs: Tuple[str, ...],
    G0: nx.Graph,
    tau: float,
    roots_per_input: int,
    rng: random.Random,
) -> bool:
    """EXACT original deterministic (seeded) gate."""
    if len(inputs) <= 1:
        return True

    nodes = list(G0.nodes)
    if not nodes:
        return False

    D: Dict[str, List[int]] = {a: [] for a in inputs}
    for Ai in inputs:
        trials = 0
        while len(D[Ai]) < roots_per_input and trials < roots_per_input * 4:
            trials += 1
            root = rng.choice(nodes)
            if not nx.has_path(G0, root, Ai):
                continue
            d = nx.shortest_path_length(G0, root, Ai)
            D[Ai].append(int(d))

    if any(len(v) == 0 for v in D.values()):
        return False

    beta_pool = sorted(set(itertools.chain.from_iterable(D.values())))
    if not beta_pool:
        return False
    if len(beta_pool) > MAX_BETA_POOL:
        beta_pool = beta_pool[:MAX_BETA_POOL]

    def s_of_beta(beta: int) -> float:
        return max(min(abs(beta - d) for d in D[Ai]) for Ai in inputs)

    scores = [(beta, s_of_beta(beta)) for beta in beta_pool]
    scores.sort(key=lambda x: x[1])
    top3 = scores[:3] if len(scores) >= 3 else scores
    if not top3:
        return False
    avg_s = float(sum(s for _, s in top3)) / len(top3)
    return avg_s < tau


def accepted_edges_map(
    reactions: List[Reaction], G0: nx.Graph, rng: random.Random
) -> Dict[Tuple[str, str], List[Reaction]]:
    edge2rxns: Dict[Tuple[str, str], List[Reaction]] = defaultdict(list)
    for r in reactions:
        if not multi_input_gate(r.inputs, G0, TAU_THRESHOLD, ROOTS_PER_INPUT, rng):
            continue
        for a in r.inputs:
            for b in r.outputs:
                A, B = canon(a), canon(b)
                if A == B or is_currency(A) or is_currency(B):
                    continue
                edge2rxns[(A, B)].append(r)
    return edge2rxns


# =============================================================================
# Frozen scaffold/reference audit
# =============================================================================

def load_frozen_baselines(path: Path) -> Dict[Tuple[str, str], float]:
    df = pd.read_csv(path)
    need = {"A", "B", "E_hat_geom"}
    if not need.issubset(df.columns):
        die(f"{path} must contain {sorted(need)}; got {list(df.columns)}")
    if df.duplicated(["A", "B"]).any():
        die(f"Duplicate directed edge rows in frozen baseline file: {path}")

    out: Dict[Tuple[str, str], float] = {}
    for _, r in df.iterrows():
        e = (str(r["A"]), str(r["B"]))
        v = float(r["E_hat_geom"])
        if not math.isfinite(v) or v < 0:
            die(f"Invalid frozen baseline for {e}: {v}")
        out[e] = v
    return out


def build_scaffold_table(edge2rxns: Dict[Tuple[str, str], List[Reaction]]) -> pd.DataFrame:
    rows = []
    for (a, b), rxns in sorted(edge2rxns.items()):
        catalysts = sorted(set(itertools.chain.from_iterable(r.catalysts for r in rxns)))
        rows.append(
            {
                "A": a,
                "B": b,
                "n_reactions": len(rxns),
                "catalysts_json": json.dumps(catalysts, separators=(",", ":")),
                "reaction_ids_json": json.dumps([r.rid for r in rxns], separators=(",", ":")),
            }
        )
    return pd.DataFrame(rows)


def verify_training_graphs_against_formula(
    baselines: Dict[Tuple[str, str], float],
    graph_dir: Path,
    n_graphs: int = 3,
) -> List[dict]:
    paths = sorted(graph_dir.glob("*.graphml"))[:n_graphs]
    if len(paths) < 1:
        die(f"No training GraphML files found in {graph_dir}")

    expected_edges = set(baselines)
    reports: List[dict] = []

    for gp in paths:
        G = nx.read_graphml(gp)
        actual = {(str(u), str(v)) for u, v in G.edges()}
        if actual != expected_edges:
            miss = sorted(expected_edges - actual)[:5]
            extra = sorted(actual - expected_edges)[:5]
            die(
                f"Training graph {gp.name} does not have the frozen scaffold exactly. "
                f"expected={len(expected_edges)}, actual={len(actual)}, "
                f"missing_examples={miss}, extra_examples={extra}"
            )

        max_base_err = 0.0
        max_weight_err = 0.0
        bad = 0
        for u, v, d in G.edges(data=True):
            e = (str(u), str(v))
            base = float(d.get("baseline"))
            support = float(d.get("support"))
            weight = float(d.get("weight"))
            frozen = float(baselines[e])
            expected_w = math.exp(-support / (frozen + WEIGHT_EPSILON))

            max_base_err = max(max_base_err, abs(base - frozen))
            max_weight_err = max(max_weight_err, abs(weight - expected_w))
            if not math.isclose(base, frozen, rel_tol=1e-11, abs_tol=1e-13):
                bad += 1
            if not math.isclose(weight, expected_w, rel_tol=1e-10, abs_tol=1e-12):
                bad += 1

        if bad:
            die(
                f"Frozen graph-equation audit failed for {gp.name}: bad_checks={bad}, "
                f"max_baseline_abs_error={max_base_err:.3g}, "
                f"max_weight_abs_error={max_weight_err:.3g}"
            )

        reports.append(
            {
                "graph": gp.name,
                "n_edges": G.number_of_edges(),
                "max_baseline_abs_error": max_base_err,
                "max_weight_abs_error": max_weight_err,
            }
        )

    return reports


# =============================================================================
# H0: exact H0 merge multiset for minimax filtration
# =============================================================================

class UnionFind:
    def __init__(self, nodes: Iterable[str]):
        self.parent = {x: x for x in nodes}
        self.rank = {x: 0 for x in nodes}

    def find(self, x: str) -> str:
        p = self.parent[x]
        if p != x:
            self.parent[x] = self.find(p)
        return self.parent[x]

    def union(self, a: str, b: str) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return True


def h0_pd_from_digraph(G: nx.DiGraph) -> np.ndarray:
    """
    Return Nx2 [birth, death] H0 persistence pairs for finite graph merges.

    The directed graph is made undirected for H0.  If both directions exist,
    the undirected filtration weight is min(w_uv, w_vu), because the vertices
    become connected at the first threshold at which either directed relation
    is present.  Kruskal merge weights equal the H0 death values of the minimax
    path metric filtration.

    Disconnected essential components are deliberately not assigned artificial
    finite deaths.  The downstream H0 code uses only finite 0<death<1 values.
    """
    undirected: Dict[Tuple[str, str], float] = {}
    nodes = [str(x) for x in G.nodes()]

    for u0, v0, d in G.edges(data=True):
        u, v = str(u0), str(v0)
        if u == v:
            continue
        try:
            w = float(d["weight"])
        except Exception as e:
            raise ValueError(f"Missing/invalid weight on {u}->{v}: {d}") from e
        if not math.isfinite(w):
            raise ValueError(f"Nonfinite weight on {u}->{v}: {w}")
        key = (u, v) if u <= v else (v, u)
        old = undirected.get(key)
        if old is None or w < old:
            undirected[key] = w

    uf = UnionFind(nodes)
    deaths: List[float] = []
    for (u, v), w in sorted(undirected.items(), key=lambda kv: (kv[1], kv[0][0], kv[0][1])):
        if uf.union(u, v):
            deaths.append(float(w))

    if not deaths:
        return np.empty((0, 2), dtype=np.float64)
    return np.column_stack([np.zeros(len(deaths), dtype=np.float64), np.asarray(deaths, dtype=np.float64)])


def filtered_h0_deaths(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.size == 0:
        return np.empty(0, dtype=np.float64)
    if arr.ndim == 1:
        d = arr.astype(np.float64)
    elif arr.ndim == 2 and arr.shape[1] >= 2:
        d = arr[:, 1].astype(np.float64)
    else:
        d = arr.ravel().astype(np.float64)
    d = d[np.isfinite(d)]
    d = d[(d > 0.0) & (d < 1.0 - 1e-12)]
    return np.sort(d)


def find_training_h0_for_sid(sid: str, pd_dir: Path) -> Optional[Path]:
    exacts = [pd_dir / f"{sid}_H0.npy", pd_dir / f"{sid}.npy"]
    for p in exacts:
        if p.exists():
            return p
    hits = sorted(pd_dir.glob(f"{sid}*H0*.npy"))
    return hits[0] if hits else None


def regression_test_h0(graph_dir: Path, pd_dir: Path, n_graphs: int = 5) -> List[dict]:
    matches: List[Tuple[Path, Path]] = []
    for gp in sorted(graph_dir.glob("*.graphml")):
        hp = find_training_h0_for_sid(gp.stem, pd_dir)
        if hp is not None:
            matches.append((gp, hp))
        if len(matches) >= n_graphs:
            break
    if not matches:
        die(
            f"Could not find any training GraphML/H0 pairs across {graph_dir} and {pd_dir}; "
            "cannot certify the H0 implementation."
        )

    reports = []
    for gp, hp in matches:
        G = nx.read_graphml(gp)
        calc = filtered_h0_deaths(h0_pd_from_digraph(G))
        ref = filtered_h0_deaths(np.load(hp, allow_pickle=False))

        if len(calc) != len(ref):
            die(
                f"H0 regression failed for {gp.stem}: calculated {len(calc)} nontrivial deaths, "
                f"training file has {len(ref)}."
            )
        if len(calc):
            max_err = float(np.max(np.abs(calc - ref)))
        else:
            max_err = 0.0
        if not np.allclose(calc, ref, rtol=1e-8, atol=1e-10):
            idx = int(np.argmax(np.abs(calc - ref))) if len(calc) else -1
            die(
                f"H0 regression failed for {gp.stem}: max_abs_error={max_err:.6g}, "
                f"worst_index={idx}, calculated={calc[idx] if idx>=0 else None}, "
                f"reference={ref[idx] if idx>=0 else None}."
            )
        reports.append(
            {
                "sample_id": gp.stem,
                "n_nontrivial_deaths": len(ref),
                "max_abs_error": max_err,
                "reference_h0": str(hp),
            }
        )
    return reports


# =============================================================================
# Training species axis and conservative MetaPhlAn mapping
# =============================================================================

def load_training_species_axis(path: Path) -> Tuple[List[str], dict]:
    df = pd.read_excel(path, sheet_name=0)
    if len(df.columns) < 2:
        die(f"Training species file has too few columns: {path}")
    # Exact original loader: first column was pandas index/sample ID.
    df = df.set_index(df.columns[0]).fillna(0.0)
    cols = [str(c) for c in df.columns]
    if len(cols) != len(set(cols)):
        dup = pd.Series(cols)[pd.Series(cols).duplicated()].tolist()[:10]
        die(f"Duplicate species columns in training abundance table: {dup}")

    numeric = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    row_sums = numeric.sum(axis=1).to_numpy(dtype=float)
    stats = {
        "n_samples": int(len(df)),
        "n_species": int(len(cols)),
        "row_sum_min": float(np.min(row_sums)),
        "row_sum_median": float(np.median(row_sums)),
        "row_sum_max": float(np.max(row_sums)),
        "row_sum_q05": float(np.quantile(row_sums, 0.05)),
        "row_sum_q95": float(np.quantile(row_sums, 0.95)),
    }
    return cols, stats


def species_terminal_name(x: str) -> str:
    s = str(x).strip()
    # MetaPhlAn full clade path -> terminal species token.
    if "|" in s:
        species_tokens = [p for p in s.split("|") if p.startswith("s__")]
        if species_tokens:
            s = species_tokens[-1]
        else:
            s = s.split("|")[-1]
    if s.startswith("s__"):
        s = s[3:]
    return s.strip()


def species_key(x: str) -> str:
    """
    Conservative punctuation-insensitive species key.

    Matching is accepted only when this key is unique on the frozen training
    species axis.  No edit-distance/fuzzy taxonomic guesses are made.
    """
    s = species_terminal_name(x).lower()
    return re.sub(r"[^a-z0-9]+", "", s)


def build_species_key_map(training_cols: Sequence[str]) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    buckets: Dict[str, List[str]] = defaultdict(list)
    for c in training_cols:
        buckets[species_key(c)].append(c)
    unique = {k: vals[0] for k, vals in buckets.items() if k and len(vals) == 1}
    ambiguous = {k: vals for k, vals in buckets.items() if k and len(vals) > 1}
    return unique, ambiguous


# =============================================================================
# MetaPhlAn profile discovery/parser
# =============================================================================

PROFILE_EXTS = {".tsv", ".txt", ".profile", ".csv"}


def profile_filename_score(path: Path) -> int:
    n = path.name.lower()
    score = 0
    if "metaphlan" in n:
        score += 100
    if "bugs_list" in n or "bugs-list" in n:
        score += 80
    if "profile" in n:
        score += 30
    if "taxonomic" in n:
        score += 20
    if "genefamil" in n or "pathabundance" in n or "pathcoverage" in n:
        score -= 500
    if "temp" in str(path.parent).lower():
        score += 5
    return score


def _parse_table_guess(path: Path) -> pd.DataFrame:
    # MetaPhlAn files are tab-separated, sometimes with several '#'-header lines.
    # Reading comment lines away is robust to v3/v4 header formats.
    try:
        return pd.read_csv(path, sep="\t", comment="#", low_memory=False)
    except Exception:
        return pd.read_csv(path, sep=None, engine="python", comment="#", low_memory=False)


def parse_metaphlan_species(path: Path) -> Tuple[pd.Series, dict]:
    """Return species-level relative abundance percentages indexed by species name."""
    df = _parse_table_guess(path)
    if df.empty:
        raise ValueError("empty parsed table")

    tax_col = find_col(
        df,
        ["clade_name", "taxonomy", "taxon", "clade", "#clade_name", "name"],
    )
    abund_col = find_col(
        df,
        ["relative_abundance", "abundance", "relative abundance", "rel_abundance"],
    )

    # Some MetaPhlAn/HUMAnN bugs_list-style files have no canonical header after
    # comments; fall back to first text-like and last numeric-like columns.
    if tax_col is None:
        objectish = [c for c in df.columns if df[c].dtype == object]
        if objectish:
            tax_col = objectish[0]
        else:
            tax_col = df.columns[0]
    if abund_col is None:
        numeric_candidates = []
        for c in df.columns:
            if c == tax_col:
                continue
            vals = pd.to_numeric(df[c], errors="coerce")
            frac = float(vals.notna().mean())
            if frac > 0.80:
                numeric_candidates.append(c)
        if numeric_candidates:
            abund_col = numeric_candidates[-1]
        else:
            raise ValueError(f"could not identify relative-abundance column; columns={list(df.columns)}")

    taxa = df[tax_col].astype(str).str.strip()
    vals = pd.to_numeric(df[abund_col], errors="coerce")

    rows: List[Tuple[str, float]] = []
    for tax, val in zip(taxa, vals):
        if not math.isfinite(float(val)) if pd.notna(val) else True:
            continue
        valf = float(val)
        if valf < 0:
            raise ValueError(f"negative abundance {valf} for taxon {tax}")

        # Species level ONLY.  This avoids counting strain rows in addition to
        # their species parent.  Full taxonomy paths use '|s__'; terminal files
        # may begin directly with s__.
        is_species = ("|s__" in tax and "|t__" not in tax) or tax.startswith("s__")
        if not is_species:
            continue
        name = species_terminal_name(tax)
        if name:
            rows.append((name, valf))

    if not rows:
        raise ValueError("no MetaPhlAn species-level rows detected")

    s = pd.Series({})
    tmp = pd.DataFrame(rows, columns=["species", "abundance"])
    s = tmp.groupby("species", sort=True)["abundance"].sum().astype(float)

    total = float(s.sum())
    maxv = float(s.max()) if len(s) else 0.0

    # Unit harmonisation ONLY, not compositional renormalisation.  Standard
    # MetaPhlAn is percentage-scale.  If a profile was explicitly converted to
    # fractions and sums to ~1, convert units back to percent.
    scale_factor = 1.0
    if 0 < total <= 1.5 and maxv <= 1.0 + 1e-9:
        s = s * 100.0
        scale_factor = 100.0
        total = float(s.sum())
        maxv = float(s.max())

    if total <= 0:
        raise ValueError("species abundance sum is zero")
    if total > 110.0:
        raise ValueError(
            f"species-level abundance sum={total:.3f} exceeds 110%; likely wrong file/column or duplicated ranks"
        )
    if maxv > 100.0 + 1e-6:
        raise ValueError(f"species abundance max={maxv:.3f} exceeds 100%")

    meta = {
        "n_species_rows": int(len(s)),
        "species_abundance_sum_percent": total,
        "species_abundance_max_percent": maxv,
        "unit_scale_factor_applied": scale_factor,
        "tax_col": str(tax_col),
        "abundance_col": str(abund_col),
    }
    return s, meta


def discover_metaphlan_profile(sample_dir: Path) -> Tuple[Optional[Path], Optional[pd.Series], Optional[dict], List[str]]:
    candidates = []
    diagnostics: List[str] = []
    for p in sample_dir.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in PROFILE_EXTS:
            continue
        if profile_filename_score(p) < 0:
            continue
        candidates.append(p)

    parsed = []
    for p in sorted(candidates, key=lambda x: (-profile_filename_score(x), str(x))):
        try:
            s, meta = parse_metaphlan_species(p)
            parsed.append((p, s, meta, profile_filename_score(p)))
        except Exception as e:
            diagnostics.append(f"reject {p}: {type(e).__name__}: {e}")

    if not parsed:
        return None, None, None, diagnostics

    # Deterministic preference: filename semantics first, then species count,
    # then total species abundance, then lexical path.
    parsed.sort(
        key=lambda x: (
            -x[3],
            -int(x[2]["n_species_rows"]),
            -float(x[2]["species_abundance_sum_percent"]),
            str(x[0]),
        )
    )
    p, s, meta, score = parsed[0]
    meta = dict(meta)
    meta["filename_score"] = score
    meta["n_valid_profile_candidates"] = len(parsed)
    return p, s, meta, diagnostics


def discover_available_profiles(humann_dir: Path) -> Tuple[Dict[str, Tuple[Path, pd.Series, dict]], pd.DataFrame]:
    if not humann_dir.exists():
        die(f"HUMAnN directory not found: {humann_dir}")

    found: Dict[str, Tuple[Path, pd.Series, dict]] = {}
    audit_rows = []
    for d in sorted(p for p in humann_dir.iterdir() if p.is_dir()):
        sid = d.name
        pp, s, meta, diagnostics = discover_metaphlan_profile(d)
        row = {
            "sample_id": sid,
            "sample_dir": str(d),
            "profile_found": pp is not None,
            "profile_path": str(pp) if pp else "",
            "diagnostic_examples": " | ".join(diagnostics[:3]),
        }
        if pp is not None and s is not None and meta is not None:
            found[sid] = (pp, s, meta)
            row.update(meta)
        audit_rows.append(row)
    return found, pd.DataFrame(audit_rows)


# =============================================================================
# External species projection and graph construction
# =============================================================================

def project_external_species(
    ext: pd.Series,
    training_cols: Sequence[str],
    unique_key_map: Dict[str, str],
    ambiguous_keys: Dict[str, List[str]],
) -> Tuple[pd.Series, dict, pd.DataFrame]:
    out = pd.Series(0.0, index=list(training_cols), dtype=float)
    rows = []
    mapped_total = 0.0
    ambiguous_total = 0.0
    unmatched_total = 0.0

    exact_lookup = {str(c): str(c) for c in training_cols}
    exact_lower: Dict[str, List[str]] = defaultdict(list)
    for c in training_cols:
        exact_lower[str(c).lower()].append(str(c))

    for ext_name, val0 in ext.items():
        val = float(val0)
        target = None
        method = None
        raw = str(ext_name)
        terminal = species_terminal_name(raw)

        if raw in exact_lookup:
            target = exact_lookup[raw]
            method = "exact"
        elif terminal in exact_lookup:
            target = exact_lookup[terminal]
            method = "terminal_exact"
        elif len(exact_lower.get(terminal.lower(), [])) == 1:
            target = exact_lower[terminal.lower()][0]
            method = "case_insensitive_exact"
        else:
            key = species_key(terminal)
            if key in unique_key_map:
                target = unique_key_map[key]
                method = "unique_normalized"
            elif key in ambiguous_keys:
                ambiguous_total += val
                rows.append(
                    {
                        "external_species": raw,
                        "abundance_percent": val,
                        "training_species": "",
                        "status": "ambiguous",
                        "method": "",
                        "candidate_training_species": " | ".join(ambiguous_keys[key]),
                    }
                )
                continue

        if target is None:
            unmatched_total += val
            rows.append(
                {
                    "external_species": raw,
                    "abundance_percent": val,
                    "training_species": "",
                    "status": "unmatched",
                    "method": "",
                    "candidate_training_species": "",
                }
            )
            continue

        out.loc[target] += val
        mapped_total += val
        rows.append(
            {
                "external_species": raw,
                "abundance_percent": val,
                "training_species": target,
                "status": "mapped",
                "method": method,
                "candidate_training_species": "",
            }
        )

    raw_total = float(ext.sum())
    meta = {
        "raw_species_total_percent": raw_total,
        "mapped_abundance_total_percent": mapped_total,
        "unmatched_abundance_total_percent": unmatched_total,
        "ambiguous_abundance_total_percent": ambiguous_total,
        "mapped_fraction_of_species_abundance": (mapped_total / raw_total if raw_total > 0 else 0.0),
        "n_external_species": int(len(ext)),
        "n_training_species_nonzero": int((out > 0).sum()),
        "no_external_renormalisation": True,
    }
    return out, meta, pd.DataFrame(rows)


def compute_external_supports(
    scaffold: pd.DataFrame,
    abundance: pd.Series,
) -> Dict[Tuple[str, str], float]:
    E: Dict[Tuple[str, str], float] = {}
    training_axis = set(abundance.index)
    for _, r in scaffold.iterrows():
        e = (str(r["A"]), str(r["B"]))
        catalysts = json.loads(r["catalysts_json"])
        species = [sp for sp in catalysts if sp in training_axis]
        Es = float(abundance.loc[species].sum()) if species else 0.0
        E[e] = Es
    return E


def build_external_graph(
    sid: str,
    scaffold: pd.DataFrame,
    baselines: Dict[Tuple[str, str], float],
    supports: Dict[Tuple[str, str], float],
) -> Tuple[nx.DiGraph, pd.DataFrame]:
    G = nx.DiGraph(sample_id=sid)
    rows = []
    for _, r in scaffold.iterrows():
        A, B = str(r["A"]), str(r["B"])
        e = (A, B)
        Es = float(supports[e])
        if Es < THETA_MIN:
            continue
        base = float(baselines[e])
        w = math.exp(-Es / (base + WEIGHT_EPSILON))
        if not math.isfinite(w) or w < 0.0 or w > 1.0 + 1e-12:
            die(f"Invalid weight for {sid} {A}->{B}: support={Es}, baseline={base}, w={w}")
        # Guard tiny floating overshoot, though exp(-x) should never exceed 1.
        w = min(1.0, max(0.0, w))
        G.add_edge(
            A,
            B,
            weight=float(w),
            support=float(Es),
            baseline=float(base),
            n_reactions=int(r["n_reactions"]),
        )
        rows.append(
            {
                "A": A,
                "B": B,
                "support_E": Es,
                "baseline_Ehat_IBDMDB": base,
                "weight": w,
                "active": bool(w < 1.0 - ACTIVE_TOL),
                "n_reactions": int(r["n_reactions"]),
            }
        )
    return G, pd.DataFrame(rows)


# =============================================================================
# Frozen reference signature and scaffold cache
# =============================================================================

def find_ricci_source() -> Path:
    direct_candidates = [
        REAL / "14_compute_ricci_faithful_pairwise_active.py",
        REAL / "scripts" / "14_compute_ricci_faithful_pairwise_active.py",
    ]
    for p in direct_candidates:
        if p.exists():
            return p.resolve()
    hits = sorted(REAL.rglob("14_compute_ricci_faithful_pairwise_active.py"))
    if not hits:
        die(
            "Could not find 14_compute_ricci_faithful_pairwise_active.py under ~/Real_Data. "
            "The external pipeline intentionally reuses the exact production Ricci source rather than reimplementing it."
        )
    return hits[0].resolve()


def verify_ricci_source_text(path: Path) -> dict:
    text = path.read_text(errors="replace")
    required = [
        "epsilon_dist: float = 1e-4",
        "c_single_out: float = 0.001",
        "beta: float = 1.4",
        "active_tol: float = 1e-12",
        "w < (1.0 - cfg.active_tol)",
        "def ricci_edge_faithful",
        "precompute_reachability",
        "FAITHFUL_ACTIVE_DONE.marker",
    ]
    missing = [x for x in required if x not in text]
    if missing:
        die(f"Ricci source signature check failed for {path}; missing snippets: {missing}")
    return {"path": str(path), "sha256": sha256_file(path), "required_signature_snippets": "all_present"}


def reference_signature() -> dict:
    paths = {
        "reactions": REACTIONS_PATH,
        "training_species": TRAIN_SPECIES_PATH,
        "frozen_baselines": FROZEN_BASELINES_PATH,
        "ricci_edge_metadata": TRAIN_RICCI_FEATURE_DIR / "edge_metadata.csv",
    }
    result = {}
    for name, p in paths.items():
        if not p.exists():
            die(f"Required frozen training artifact missing: {name}: {p}")
        result[name] = {"path": str(p), "sha256": sha256_file(p)}

    ricci_source = find_ricci_source()
    result["ricci_source"] = verify_ricci_source_text(ricci_source)
    if CURRENT_PROXY_BUILDER.exists():
        result["previous_external_proxy_builder"] = {
            "path": str(CURRENT_PROXY_BUILDER),
            "sha256": sha256_file(CURRENT_PROXY_BUILDER),
            "role": "provenance_only_not_executed_for_primary_validation",
        }
    else:
        result["previous_external_proxy_builder"] = {
            "path": str(CURRENT_PROXY_BUILDER),
            "exists": False,
            "role": "provenance_only_not_executed_for_primary_validation",
        }
    return result


def assert_reference_signature_compatible(out_dir: Path, current: dict) -> None:
    p = out_dir / "provenance" / "REFERENCE_SIGNATURE.json"
    if not p.exists():
        return
    old = json.loads(p.read_text())
    # Compare hashes only for the frozen inputs that exist in both.
    for k in ["reactions", "training_species", "frozen_baselines", "ricci_edge_metadata", "ricci_source"]:
        if old.get(k, {}).get("sha256") != current.get(k, {}).get("sha256"):
            die(
                f"Frozen reference changed for {k}. Existing output directory {out_dir} was built with a different reference. "
                "Use a new output directory rather than mixing representations."
            )


def load_or_create_scaffold(out_dir: Path, sig: dict) -> pd.DataFrame:
    prov = out_dir / "provenance"
    cache = prov / "frozen_edge_scaffold.csv.gz"
    cache_meta = prov / "frozen_edge_scaffold.meta.json"
    baselines = load_frozen_baselines(FROZEN_BASELINES_PATH)

    if cache.exists() and cache_meta.exists():
        m = json.loads(cache_meta.read_text())
        if (
            m.get("reactions_sha256") == sig["reactions"]["sha256"]
            and m.get("baselines_sha256") == sig["frozen_baselines"]["sha256"]
        ):
            df = pd.read_csv(cache, compression="gzip")
            if set(zip(df["A"].astype(str), df["B"].astype(str))) != set(baselines):
                die("Cached scaffold edge set does not equal frozen baseline edge set.")
            return df

    print("[preflight] Reconstructing accepted IBDMDB edge scaffold with the exact original gate...", flush=True)
    reactions = load_reactions(REACTIONS_PATH)
    G0 = build_global_substrate_graph(reactions)
    edge2rxns = accepted_edges_map(reactions, G0, random.Random(GRAPH_SEED))
    scaffold = build_scaffold_table(edge2rxns)

    reconstructed = set(zip(scaffold["A"].astype(str), scaffold["B"].astype(str)))
    frozen = set(baselines)
    if reconstructed != frozen:
        miss = sorted(frozen - reconstructed)[:10]
        extra = sorted(reconstructed - frozen)[:10]
        die(
            "The current AGORA table + original gate do not reconstruct the exact frozen IBDMDB edge scaffold. "
            f"frozen={len(frozen)}, reconstructed={len(reconstructed)}, "
            f"missing_examples={miss}, extra_examples={extra}. "
            "Do NOT run external validation until this provenance mismatch is resolved."
        )

    prov.mkdir(parents=True, exist_ok=True)
    scaffold.to_csv(cache, index=False, compression="gzip")
    json_dump(
        cache_meta,
        {
            "reactions_sha256": sig["reactions"]["sha256"],
            "baselines_sha256": sig["frozen_baselines"]["sha256"],
            "graph_seed": GRAPH_SEED,
            "tau_threshold": TAU_THRESHOLD,
            "roots_per_input": ROOTS_PER_INPUT,
            "max_beta_pool": MAX_BETA_POOL,
            "n_edges": int(len(scaffold)),
        },
    )
    return scaffold


# =============================================================================
# External metadata attachment (best-effort; never used for graph construction)
# =============================================================================

def candidate_manifest_files() -> List[Path]:
    if not MANIFEST_DIR.exists():
        return []
    preferred = [
        MANIFEST_DIR / "PRJEB42155_ge50_labels_for_validation.tsv",
        MANIFEST_DIR / "PRJEB42155_ge50_run_manifest.tsv",
    ]
    rest = sorted(MANIFEST_DIR.glob("*.tsv")) + sorted(MANIFEST_DIR.glob("*.csv"))
    seen = set()
    ans = []
    for p in preferred + rest:
        if p.exists() and p not in seen:
            ans.append(p)
            seen.add(p)
    return ans


def read_manifest(path: Path) -> pd.DataFrame:
    sep = "\t" if path.suffix.lower() == ".tsv" else ","
    return pd.read_csv(path, sep=sep, low_memory=False)


def metadata_for_samples(sample_ids: Sequence[str]) -> pd.DataFrame:
    base = pd.DataFrame({"sample_id": list(sample_ids)})
    ids = set(sample_ids)
    for p in candidate_manifest_files():
        try:
            df = read_manifest(p)
        except Exception:
            continue
        run_col = find_col(
            df,
            ["run_accession", "run", "Run", "sra_run", "run_id", "sample_id", "accession"],
        )
        if run_col is None:
            # Detect a column by overlap with current sample IDs.
            best = None
            best_n = 0
            for c in df.columns:
                vals = set(df[c].dropna().astype(str))
                n = len(vals & ids)
                if n > best_n:
                    best, best_n = c, n
            run_col = best if best_n else None
        if run_col is None:
            continue

        tmp = df.copy()
        tmp[run_col] = tmp[run_col].astype(str)
        tmp = tmp[tmp[run_col].isin(ids)].drop_duplicates(run_col, keep="first")
        if tmp.empty:
            continue

        ren = {run_col: "sample_id"}
        for c in tmp.columns:
            if c == run_col:
                continue
            # Prevent collisions when combining multiple manifests.
            if c in base.columns:
                ren[c] = f"{p.stem}__{c}"
        tmp = tmp.rename(columns=ren)
        base = base.merge(tmp, on="sample_id", how="left")

    # Compatibility conveniences for later external classifier code.  These are
    # metadata only and have zero influence on graph/H0/Ricci construction.
    if "participant_id" not in base.columns:
        pc = find_col(base, ["participant", "subject_id", "host_subject_id", "subject", "individual_id"])
        base["participant_id"] = base[pc].astype(str) if pc else base["sample_id"].astype(str)
    if "cond" not in base.columns:
        cc = find_col(base, ["condition", "diagnosis", "label", "disease", "ibd_subtype"])
        base["cond"] = base[cc].astype(str) if cc else "UNKNOWN"
    return base


# =============================================================================
# Commands
# =============================================================================

def cmd_preflight(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "provenance").mkdir(parents=True, exist_ok=True)

    print("[preflight] Building frozen-reference signature...", flush=True)
    sig = reference_signature()
    assert_reference_signature_compatible(out_dir, sig)

    print("[preflight] Loading/reconstructing exact accepted edge scaffold...", flush=True)
    scaffold = load_or_create_scaffold(out_dir, sig)
    baselines = load_frozen_baselines(FROZEN_BASELINES_PATH)

    print("[preflight] Checking existing IBDMDB GraphML files against frozen baselines and weight equation...", flush=True)
    graph_audit = verify_training_graphs_against_formula(baselines, TRAIN_GRAPH_DIR, n_graphs=args.graph_checks)

    print("[preflight] Regression-testing H0 merge deaths against existing IBDMDB *_H0.npy files...", flush=True)
    h0_audit = regression_test_h0(TRAIN_GRAPH_DIR, TRAIN_H0_DIR, n_graphs=args.h0_checks)

    training_cols, training_stats = load_training_species_axis(TRAIN_SPECIES_PATH)

    edge_meta_path = TRAIN_RICCI_FEATURE_DIR / "edge_metadata.csv"
    edge_meta = pd.read_csv(edge_meta_path)
    if "edge" not in edge_meta.columns:
        die(f"Training Ricci edge metadata lacks required 'edge' column: {edge_meta_path}")
    X_path = TRAIN_RICCI_FEATURE_DIR / "feature_matrix_B_K0.npz"
    if X_path.exists():
        X_train = sparse.load_npz(X_path)
        if X_train.shape[1] != 2 * len(edge_meta):
            die(
                f"Frozen Ricci feature matrix has {X_train.shape[1]} columns but edge_metadata has {len(edge_meta)} rows; "
                "expected exactly [B | K0] = 2*n_edges."
            )
        ricci_feature_shape = [int(X_train.shape[0]), int(X_train.shape[1])]
    else:
        die(f"Frozen Ricci feature matrix missing: {X_path}")

    sig["parameters"] = {
        "graph_seed": GRAPH_SEED,
        "tau_threshold": TAU_THRESHOLD,
        "roots_per_input": ROOTS_PER_INPUT,
        "weight_epsilon": WEIGHT_EPSILON,
        "theta_min": THETA_MIN,
        "active_tol": ACTIVE_TOL,
        "ricci_n_paths": RICCI_N_PATHS,
        "ricci_epsilon_dist": RICCI_EPSILON_DIST,
        "ricci_c_single_out": RICCI_C_SINGLE_OUT,
        "ricci_beta": RICCI_BETA,
        "ricci_max_steps": RICCI_MAX_STEPS,
        "ricci_seed": RICCI_SEED,
        "ricci_mu_prune_threshold": RICCI_MU_PRUNE_THRESHOLD,
        "ricci_topk_support": RICCI_TOPK_SUPPORT,
        "ricci_max_lp_vars": RICCI_MAX_LP_VARS,
    }
    sig["audit"] = {
        "scaffold_n_edges": int(len(scaffold)),
        "frozen_baseline_n_edges": int(len(baselines)),
        "training_graph_formula_checks": graph_audit,
        "h0_regression_checks": h0_audit,
        "training_species_scale": training_stats,
        "frozen_ricci_edge_count": int(len(edge_meta)),
        "frozen_ricci_feature_matrix_shape": ricci_feature_shape,
    }
    json_dump(out_dir / "provenance" / "REFERENCE_SIGNATURE.json", sig)
    json_dump(out_dir / "provenance" / "PREFLIGHT_PASSED.json", sig["audit"])

    print("=" * 88)
    print("STRICT EXTERNAL STRUCTURAL PREFLIGHT: PASSED")
    print("=" * 88)
    print(f"Frozen scaffold edges:     {len(scaffold):,}")
    print(f"Frozen Ricci feature edges:{len(edge_meta):,}")
    print(f"Training species features: {len(training_cols):,}")
    print(f"Training abundance row-sum median: {training_stats['row_sum_median']:.6g}")
    print(f"Reference record: {out_dir / 'provenance' / 'REFERENCE_SIGNATURE.json'}")


def require_preflight(out_dir: Path) -> Tuple[dict, pd.DataFrame, Dict[Tuple[str, str], float]]:
    marker = out_dir / "provenance" / "PREFLIGHT_PASSED.json"
    sig_path = out_dir / "provenance" / "REFERENCE_SIGNATURE.json"
    if not marker.exists() or not sig_path.exists():
        die(f"Preflight has not passed for {out_dir}. Run the preflight command first.")
    current = reference_signature()
    assert_reference_signature_compatible(out_dir, current)
    sig = json.loads(sig_path.read_text())
    scaffold = pd.read_csv(out_dir / "provenance" / "frozen_edge_scaffold.csv.gz", compression="gzip")
    baselines = load_frozen_baselines(FROZEN_BASELINES_PATH)
    return sig, scaffold, baselines


def invalidate_ricci_for_sample(out_dir: Path, sid: str) -> None:
    ricci_dir = out_dir / "ricci_raw"
    patterns = [
        f"{sid}_ricci.csv.gz",
        f"{sid}.FAITHFUL_ACTIVE_DONE.marker",
        f"{sid}_faithful_active_reachability_stats.csv",
        f"{sid}_faithful_active_reachability_cache.pkl.gz",
    ]
    for name in patterns:
        p = ricci_dir / name
        if p.exists():
            p.unlink()


def cmd_build(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir).expanduser().resolve()
    _, scaffold, baselines = require_preflight(out_dir)

    training_cols, training_stats = load_training_species_axis(TRAIN_SPECIES_PATH)
    unique_key_map, ambiguous_keys = build_species_key_map(training_cols)

    profiles, profile_audit = discover_available_profiles(HUMANN_DIR)
    (out_dir / "audits").mkdir(parents=True, exist_ok=True)
    profile_audit.to_csv(out_dir / "audits" / "profile_discovery.csv", index=False)

    if not profiles:
        die(
            f"No parseable MetaPhlAn species profiles found under {HUMANN_DIR}. "
            "This strict primary pipeline does not fall back to HUMAnN gene-family species proxies."
        )

    graphs_dir = out_dir / "graphs"
    edges_dir = out_dir / "edges"
    h0_dir = out_dir / "h0"
    species_dir = out_dir / "species_projection"
    marker_dir = out_dir / "sample_markers"
    for d in [graphs_dir, edges_dir, h0_dir, species_dir, marker_dir, out_dir / "ricci_raw"]:
        d.mkdir(parents=True, exist_ok=True)

    status_rows = []
    built = 0
    skipped = 0
    failed = 0

    for sid in sorted(profiles):
        profile_path, ext_species, pmeta = profiles[sid]
        profile_sha = sha256_file(profile_path)
        marker_path = marker_dir / f"{sid}.json"
        graph_path = graphs_dir / f"{sid}.graphml"
        h0_path = h0_dir / f"{sid}_H0.npy"
        edges_path = edges_dir / f"{sid}_edges.csv.gz"
        proj_path = species_dir / f"{sid}_species_projection.csv.gz"

        old = json.loads(marker_path.read_text()) if marker_path.exists() else None
        unchanged = (
            old is not None
            and old.get("profile_sha256") == profile_sha
            and graph_path.exists()
            and h0_path.exists()
            and edges_path.exists()
            and proj_path.exists()
        )
        if unchanged and not args.overwrite:
            skipped += 1
            status_rows.append({"sample_id": sid, "status": "skip_unchanged", **old})
            continue

        try:
            projected, mmap, mapping_df = project_external_species(
                ext_species, training_cols, unique_key_map, ambiguous_keys
            )
            mapped_frac = float(mmap["mapped_fraction_of_species_abundance"])
            if mapped_frac < args.min_mapped_fraction:
                raise RuntimeError(
                    f"Mapped only {mapped_frac:.3%} of species-level abundance, below required "
                    f"{args.min_mapped_fraction:.3%}. This sample is not safely comparable to the frozen training catalyst axis."
                )

            supports = compute_external_supports(scaffold, projected)
            G, edge_df = build_external_graph(sid, scaffold, baselines, supports)

            if G.number_of_edges() != len(scaffold):
                raise RuntimeError(
                    f"Graph has {G.number_of_edges()} edges but frozen scaffold has {len(scaffold)}."
                )
            active_edges = sum(float(d["weight"]) < 1.0 - ACTIVE_TOL for _, _, d in G.edges(data=True))
            if active_edges <= 0:
                raise RuntimeError("External graph has zero active edges.")

            pd_h0 = h0_pd_from_digraph(G)
            nontrivial_h0 = filtered_h0_deaths(pd_h0)

            # Atomic-ish writes: temporary local paths then replace.
            tmp_graph = graph_path.with_suffix(".graphml.tmp")
            nx.write_graphml(G, tmp_graph)
            tmp_graph.replace(graph_path)

            edge_df.to_csv(edges_path, index=False, compression="gzip")
            mapping_df.to_csv(proj_path, index=False, compression="gzip")
            np.save(h0_path, pd_h0, allow_pickle=False)

            # A changed external profile invalidates any previous Ricci result.
            if old is not None and old.get("profile_sha256") != profile_sha:
                invalidate_ricci_for_sample(out_dir, sid)

            marker = {
                "sample_id": sid,
                "status": "built",
                "profile_path": str(profile_path),
                "profile_sha256": profile_sha,
                **pmeta,
                **mmap,
                "training_abundance_row_sum_median": training_stats["row_sum_median"],
                "n_template_edges": int(G.number_of_edges()),
                "n_active_edges": int(active_edges),
                "n_h0_finite_merges": int(len(pd_h0)),
                "n_h0_nontrivial_deaths_0_lt_d_lt_1": int(len(nontrivial_h0)),
                "graph_path": str(graph_path),
                "h0_path": str(h0_path),
                "no_external_cohort_baseline": True,
                "no_external_compositional_renormalisation": True,
                "numerator_source": "MetaPhlAn species relative abundance percentage",
                "denominator_source": str(FROZEN_BASELINES_PATH),
            }
            json_dump(marker_path, marker)
            status_rows.append(marker)
            built += 1
            print(
                f"[built] {sid}: mapped={mapped_frac:.1%}, active_edges={active_edges:,}, "
                f"H0_nontrivial={len(nontrivial_h0):,}",
                flush=True,
            )
        except Exception as e:
            failed += 1
            row = {
                "sample_id": sid,
                "status": "FAILED",
                "profile_path": str(profile_path),
                "error": f"{type(e).__name__}: {e}",
            }
            status_rows.append(row)
            print(f"[FAILED] {sid}: {row['error']}", file=sys.stderr, flush=True)
            if args.fail_fast:
                pd.DataFrame(status_rows).to_csv(out_dir / "audits" / "build_status.csv", index=False)
                raise

    status_df = pd.DataFrame(status_rows)
    status_df.to_csv(out_dir / "audits" / "build_status.csv", index=False)

    # Build a metadata table only for successfully materialised samples.
    complete_sids = sorted(p.stem.replace("_H0", "") for p in h0_dir.glob("*_H0.npy"))
    metadata_for_samples(complete_sids).to_csv(out_dir / "external_sample_metadata.csv", index=False)

    print("=" * 88)
    print("EXTERNAL GRAPH + H0 BUILD COMPLETE")
    print("=" * 88)
    print(f"Profiles currently available: {len(profiles)}")
    print(f"Built/rebuilt: {built}")
    print(f"Skipped unchanged: {skipped}")
    print(f"Failed: {failed}")
    print(f"Graphs: {graphs_dir}")
    print(f"H0:     {h0_dir}")
    if failed:
        print(f"Review: {out_dir / 'audits' / 'build_status.csv'}")


def parse_edge_identity(edge_meta: pd.DataFrame) -> List[Tuple[str, str]]:
    source_col = find_col(edge_meta, ["source", "src", "u", "a", "from", "tail"])
    target_col = find_col(edge_meta, ["target", "dst", "v", "b", "to", "head"])
    if source_col is not None and target_col is not None:
        return list(zip(edge_meta[source_col].astype(str), edge_meta[target_col].astype(str)))

    if "edge" not in edge_meta.columns:
        die("Training edge_metadata.csv has neither source/target columns nor an 'edge' column.")

    ans = []
    for e in edge_meta["edge"].astype(str):
        if " -> " in e:
            a, b = e.split(" -> ", 1)
        elif "->" in e:
            a, b = e.split("->", 1)
            a, b = a.strip(), b.strip()
        else:
            die(f"Cannot parse frozen training edge identity: {e!r}")
        ans.append((a, b))
    return ans


def cmd_vectorize(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir).expanduser().resolve()
    require_preflight(out_dir)

    ricci_dir = out_dir / "ricci_raw"
    files = sorted(ricci_dir.glob("*_ricci.csv.gz"))
    if not files:
        die(f"No Ricci CSV.GZ outputs found in {ricci_dir}. Run the exact Ricci job first.")

    edge_meta_path = TRAIN_RICCI_FEATURE_DIR / "edge_metadata.csv"
    edge_meta = pd.read_csv(edge_meta_path)
    frozen_edges = parse_edge_identity(edge_meta)
    if len(frozen_edges) != len(set(frozen_edges)):
        die("Frozen Ricci edge_metadata.csv has duplicate edge identities.")
    edge_to_idx = {e: i for i, e in enumerate(frozen_edges)}
    n_edges = len(frozen_edges)

    B_rows = []
    K_rows = []
    sample_ids = []
    audit_rows = []

    for p in files:
        sid = p.name[:-len("_ricci.csv.gz")]
        marker = ricci_dir / f"{sid}.FAITHFUL_ACTIVE_DONE.marker"
        if not marker.exists():
            print(f"[skip] {sid}: Ricci CSV exists but done marker is absent", flush=True)
            continue

        diag = json.loads(marker.read_text())
        if int(diag.get("n_paths", -1)) != RICCI_N_PATHS:
            die(f"{sid}: Ricci done marker says n_paths={diag.get('n_paths')}, expected {RICCI_N_PATHS}.")

        # Training-classifier fidelity: the established faithful [B | K0]
        # loader replaces any non-finite sparse feature value with zero before
        # modelling (h0_ricci_joint_sparse/data.py).  Therefore a present
        # frozen edge with undefined/non-finite curvature must retain B=1
        # while contributing K0=0.  Keep the raw Ricci CSV unchanged and
        # record the zeroing here for auditability.
        marker_nonfinite = int(diag.get("n_nan_k_ab", 0))

        df = pd.read_csv(p, compression="gzip")
        need = {"a", "b", "k_ab"}
        if not need.issubset(df.columns):
            die(f"{p} lacks required columns {sorted(need)}")

        b_idx = []
        b_val = []
        k_idx = []
        k_val = []
        outside = 0
        duplicate = 0
        nonfinite_frozen_k0_zeroed = 0
        nonfinite_outside_frozen = 0
        seen = set()
        for _, r in df.iterrows():
            e = (str(r["a"]), str(r["b"]))
            if e in seen:
                duplicate += 1
                continue
            seen.add(e)
            idx = edge_to_idx.get(e)
            kval = float(r["k_ab"])
            if idx is None:
                # Correct external projection behaviour: a training-unseen active
                # edge may affect graph topology and Ricci neighbourhoods, but it
                # has no coefficient coordinate in a frozen classifier.
                outside += 1
                if not math.isfinite(kval):
                    nonfinite_outside_frozen += 1
                continue

            # Presence is independent of whether curvature is finite.
            b_idx.append(idx)
            b_val.append(1.0)

            # Exact training-loader convention: non-finite sparse feature values
            # are converted to the established zero value.  Thus an active edge
            # remains B=1 but has K0=0 when k_ab is NaN/Inf.
            if not math.isfinite(kval):
                nonfinite_frozen_k0_zeroed += 1
                continue

            if kval != 0.0:
                k_idx.append(idx)
                k_val.append(kval)

        B = sparse.csr_matrix(
            (np.asarray(b_val, dtype=np.float64), (np.zeros(len(b_idx), dtype=int), np.asarray(b_idx, dtype=int))),
            shape=(1, n_edges),
            dtype=np.float64,
        )
        K = sparse.csr_matrix(
            (np.asarray(k_val, dtype=np.float64), (np.zeros(len(k_idx), dtype=int), np.asarray(k_idx, dtype=int))),
            shape=(1, n_edges),
            dtype=np.float64,
        )
        B_rows.append(B)
        K_rows.append(K)
        sample_ids.append(sid)
        audit_rows.append(
            {
                "sample_id": sid,
                "ricci_active_edges": int(len(df)),
                "frozen_feature_edges_present": int(len(b_idx)),
                "active_edges_outside_frozen_feature_universe": int(outside),
                "duplicate_rows_skipped": int(duplicate),
                "raw_marker_nonfinite_k_ab": int(marker_nonfinite),
                "nonfinite_frozen_K0_zeroed": int(nonfinite_frozen_k0_zeroed),
                "nonfinite_outside_frozen_feature_universe": int(nonfinite_outside_frozen),
                "n_paths": int(diag.get("n_paths", -1)),
            }
        )

    if not sample_ids:
        die("No complete Ricci outputs were eligible for vectorization.")

    Bmat = sparse.vstack(B_rows, format="csr")
    Kmat = sparse.vstack(K_rows, format="csr")
    X = sparse.hstack([Bmat, Kmat], format="csr")

    feature_dir = out_dir / "ricci_features_frozen_training_order"
    feature_dir.mkdir(parents=True, exist_ok=True)
    save_npz(feature_dir / "feature_matrix_B_K0.npz", X)
    shutil.copy2(edge_meta_path, feature_dir / "edge_metadata.csv")

    meta = metadata_for_samples(sample_ids)
    meta.to_csv(feature_dir / "matched_metadata.csv", index=False)
    pd.DataFrame(audit_rows).to_csv(feature_dir / "vectorization_audit.csv", index=False)

    feature_names = (
        [f"B__{e}" for e in edge_meta["edge"].astype(str).tolist()]
        + [f"K__{e}" for e in edge_meta["edge"].astype(str).tolist()]
    )
    (feature_dir / "feature_names.txt").write_text("\n".join(feature_names) + "\n")

    json_dump(
        feature_dir / "FEATURE_MATRIX_PROVENANCE.json",
        {
            "shape": [int(X.shape[0]), int(X.shape[1])],
            "n_frozen_edges": n_edges,
            "column_layout": "[B_0..B_(m-1) | K0_0..K0_(m-1)] in exact frozen edge_metadata.csv row order",
            "B_definition": "1 iff frozen edge is active/present in the external sample Ricci output, else 0",
            "K0_definition": "finite k_ab iff frozen edge is active/present; 0 if edge absent or k_ab nonfinite, matching the training classifier loader",
            "nonfinite_policy": "Preserve B=1 for present frozen edges; encode K0=0 for nonfinite k_ab exactly as the training h0_ricci_joint_sparse/data.py loader zeroed nonfinite sparse feature values.",
            "training_edge_metadata_path": str(edge_meta_path),
            "training_edge_metadata_sha256": sha256_file(edge_meta_path),
            "ricci_parameters": {
                "n_paths": RICCI_N_PATHS,
                "epsilon_dist": RICCI_EPSILON_DIST,
                "c_single_out": RICCI_C_SINGLE_OUT,
                "beta": RICCI_BETA,
                "active_tol": ACTIVE_TOL,
            },
        },
    )

    print("=" * 88)
    print("EXTERNAL RICCI [B | K0] VECTORIZATION COMPLETE")
    print("=" * 88)
    print(f"Samples: {X.shape[0]}")
    print(f"Frozen edges: {n_edges}")
    print(f"Feature columns: {X.shape[1]} = 2 x {n_edges}")
    print(f"Output: {feature_dir}")


def cmd_status(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir).expanduser().resolve()
    graphs = {p.stem for p in (out_dir / "graphs").glob("*.graphml")} if (out_dir / "graphs").exists() else set()
    h0 = {p.name[:-len("_H0.npy")] for p in (out_dir / "h0").glob("*_H0.npy")} if (out_dir / "h0").exists() else set()
    ricci = {p.name[:-len("_ricci.csv.gz")] for p in (out_dir / "ricci_raw").glob("*_ricci.csv.gz")} if (out_dir / "ricci_raw").exists() else set()
    ricci_done = {p.name[:-len(".FAITHFUL_ACTIVE_DONE.marker")] for p in (out_dir / "ricci_raw").glob("*.FAITHFUL_ACTIVE_DONE.marker")} if (out_dir / "ricci_raw").exists() else set()

    profiles, _ = discover_available_profiles(HUMANN_DIR)
    prof = set(profiles)
    print(f"Parseable MetaPhlAn profiles: {len(prof)}")
    print(f"External graphs:             {len(graphs)}")
    print(f"External H0 arrays:          {len(h0)}")
    print(f"Ricci CSV outputs:           {len(ricci)}")
    print(f"Ricci done markers:          {len(ricci_done)}")
    print(f"Profiles awaiting graph/H0:  {len(prof - (graphs & h0))}")
    print(f"Graphs awaiting Ricci:       {len(graphs - ricci_done)}")
    if prof - (graphs & h0):
        print("Next graph/H0 sample examples:", sorted(prof - (graphs & h0))[:10])
    if graphs - ricci_done:
        print("Next Ricci sample examples:", sorted(graphs - ricci_done)[:10])


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Strict frozen-IBDMDB external structural validation pipeline")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("preflight", help="Verify frozen training representation and H0/Ricci compatibility")
    p.add_argument("--graph-checks", type=int, default=3)
    p.add_argument("--h0-checks", type=int, default=5)
    p.set_defaults(func=cmd_preflight)

    p = sub.add_parser("build", help="Build external graphs + H0 from currently available MetaPhlAn profiles")
    p.add_argument(
        "--min-mapped-fraction",
        type=float,
        default=0.50,
        help="Fail a sample when < this fraction of species-level MetaPhlAn abundance maps to the frozen training species axis.",
    )
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--fail-fast", action="store_true")
    p.set_defaults(func=cmd_build)

    p = sub.add_parser("vectorize", help="Convert completed exact Ricci outputs to frozen training [B|K0] order")
    p.set_defaults(func=cmd_vectorize)

    p = sub.add_parser("status", help="Show current incremental processing counts")
    p.set_defaults(func=cmd_status)

    return ap


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
