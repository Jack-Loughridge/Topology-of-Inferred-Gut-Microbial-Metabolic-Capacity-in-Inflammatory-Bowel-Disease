#!/usr/bin/env python3
from __future__ import annotations

import os
import re
import math
import random
import itertools
import bz2
from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable
from collections import defaultdict

import numpy as np
import pandas as pd
import networkx as nx


class CONFIG:
    BASE_DIR = "/home/Jack/Real_Data/external_validation/serrano_gomez_ibd"

    DATA_DIR = "/home/Jack/Real_Data"
    REACTIONS_PATH = "/home/Jack/Real_Data/AGORA_reactions_canon.parquet"

    EC_TABLE = BASE_DIR + "/ge50_external_validation/humann_merged/ge50_ec_abundance_relab.tsv"
    OUT_DIR = BASE_DIR + "/ge50_external_validation/out_graphs_ec_033"

    # HUMAnN/MetaCyc reaction -> EC mapping, useful if AGORA reaction IDs overlap MetaCyc IDs.
    METACYC_RXN_EC_MAP = "/home/Jack/micromamba/envs/humann_py312/lib/python3.12/site-packages/humann/data/pathways/metacyc_reactions_level4ec_only.uniref.bz2"

    COL_REACTION_ID = "reaction_id"
    COL_INPUTS = "inputs"
    COL_OUTPUTS = "outputs"
    COL_CATALYSTS = "catalysts"

    RANDOM_SEED = 13
    TAU_THRESHOLD = 0.0001
    ROOTS_PER_INPUT = 50
    MAX_BETA_POOL = 2500

    EPSILON = 1e-8
    THETA_MIN = 0.0

    # For smoke test: use all columns currently present in merged EC table.
    # After 33 samples, this should be 33 sample columns.
    MAX_SAMPLES = 33

    CURRENCY = {
        "H2O","WATER","PROTON","H+","ATP","ADP","PI","PPI","CO2","CO(2)","NAD","NADH",
        "NADP","NADPH","FAD","FADH2","NAD(P)H","OXYGEN","O2","NH3","AMMONIA","HCO3-",
        "BICARBONATE","CO-A","COENZYME_A","COENZYME A","S-ADENOSYLMETHIONINE","SAM",
        "S-ADENOSYLHOMOCYSTEINE","SAH","UMP","UDP","UTP","AMP","CMP","CDP","CTP",
        "GMP","GDP","GTP"
    }


EC_RE = re.compile(r"(?<![A-Za-z0-9])(\d+\.\d+\.[0-9A-Za-z-]+\.[0-9A-Za-z-]+)(?![A-Za-z0-9])")


def _canon(x: str) -> str:
    return str(x).strip().upper().replace(" ", "_")


def _is_currency(x: str) -> bool:
    return _canon(x) in CONFIG.CURRENCY


def parse_listish(cell) -> List[str]:
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


@dataclass(frozen=True)
class Reaction:
    rid: str
    inputs: Tuple[str, ...]
    outputs: Tuple[str, ...]
    catalysts: Tuple[str, ...]
    ecs: Tuple[str, ...]


def extract_ecs_from_text(x) -> List[str]:
    if pd.isna(x):
        return []
    return sorted(set(EC_RE.findall(str(x))))


def load_metacyc_reaction_ec_map(path: str) -> Dict[str, List[str]]:
    rxn_to_ecs = defaultdict(list)

    if not os.path.exists(path):
        print(f"[WARN] MetaCyc reaction->EC map not found: {path}")
        return {}

    with bz2.open(path, "rt") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            rxn, ec_field = parts[0], parts[1]
            ecs = extract_ecs_from_text(ec_field)
            if ecs:
                rxn_to_ecs[rxn].extend(ecs)

    return {k: sorted(set(v)) for k, v in rxn_to_ecs.items()}


def load_ec_abundance(path: str, max_samples: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    feature_col = df.columns[0]

    # Keep only unstratified EC rows: e.g. 1.1.1.100, not 1.1.1.100|species.
    feature = df[feature_col].astype(str)
    mask = feature.str.match(r"^\d+\.\d+\.[0-9A-Za-z-]+\.[0-9A-Za-z-]+$") & ~feature.str.contains(r"\|", regex=True)
    df = df.loc[mask].copy()

    sample_cols = list(df.columns[1:])
    if max_samples is not None:
        sample_cols = sample_cols[:max_samples]

    df = df[[feature_col] + sample_cols]
    df = df.set_index(feature_col)
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    # rows = samples, columns = ECs
    abund = df.T
    abund.index = [str(x).replace("_genefamilies", "") for x in abund.index]
    abund.index.name = "sample_id"

    return abund


def load_reactions(path: str, rxn_to_ecs: Dict[str, List[str]]) -> List[Reaction]:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".parquet", ".pq"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_excel(path)

    need = [CONFIG.COL_REACTION_ID, CONFIG.COL_INPUTS, CONFIG.COL_OUTPUTS, CONFIG.COL_CATALYSTS]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"Missing reaction columns: {missing}")

    recs = []
    for _, row in df.iterrows():
        rid = str(row[CONFIG.COL_REACTION_ID]).strip()

        ins = tuple(
            _canon(x) for x in parse_listish(row[CONFIG.COL_INPUTS])
            if not _is_currency(x)
        )
        outs = tuple(
            _canon(x) for x in parse_listish(row[CONFIG.COL_OUTPUTS])
            if not _is_currency(x)
        )
        cats = tuple(str(x).strip() for x in parse_listish(row[CONFIG.COL_CATALYSTS]))

        ecs = set()
        ecs.update(extract_ecs_from_text(rid))
        ecs.update(extract_ecs_from_text(row[CONFIG.COL_CATALYSTS]))
        ecs.update(rxn_to_ecs.get(rid, []))

        if not ins or not outs:
            continue

        recs.append(
            Reaction(
                rid=rid,
                inputs=ins,
                outputs=outs,
                catalysts=cats,
                ecs=tuple(sorted(ecs)),
            )
        )

    return recs


def build_global_substrate_graph(reactions: List[Reaction]) -> nx.Graph:
    G = nx.Graph()
    for r in reactions:
        for x in r.inputs + r.outputs:
            if not _is_currency(x):
                G.add_node(_canon(x))

        for a, b in itertools.combinations(r.inputs, 2):
            G.add_edge(_canon(a), _canon(b))

        for a in r.inputs:
            for b in r.outputs:
                if not _is_currency(a) and not _is_currency(b):
                    G.add_edge(_canon(a), _canon(b))
    return G


def multi_input_gate(inputs: Tuple[str, ...], G0: nx.Graph, tau: float, roots_per_input: int, rng: random.Random) -> bool:
    if len(inputs) <= 1:
        return True

    nodes = list(G0.nodes)
    if not nodes:
        return False

    D = {a: [] for a in inputs}

    for Ai in inputs:
        trials = 0
        while len(D[Ai]) < roots_per_input and trials < roots_per_input * 4:
            trials += 1
            root = rng.choice(nodes)
            if not nx.has_path(G0, root, Ai):
                continue
            D[Ai].append(int(nx.shortest_path_length(G0, root, Ai)))

    if any(len(v) == 0 for v in D.values()):
        return False

    beta_pool = sorted(set(itertools.chain.from_iterable(D.values())))
    if not beta_pool:
        return False
    if len(beta_pool) > CONFIG.MAX_BETA_POOL:
        beta_pool = beta_pool[:CONFIG.MAX_BETA_POOL]

    def s_of_beta(beta: int) -> float:
        return max(min(abs(beta - d) for d in D[Ai]) for Ai in inputs)

    scores = [(beta, s_of_beta(beta)) for beta in beta_pool]
    scores.sort(key=lambda x: x[1])
    top3 = scores[:3] if len(scores) >= 3 else scores
    if not top3:
        return False

    avg_s = float(sum(s for _, s in top3)) / len(top3)
    return avg_s < tau


def accepted_edges_map(reactions: List[Reaction], G0: nx.Graph, rng: random.Random) -> Dict[Tuple[str, str], List[Reaction]]:
    edge2rxns = defaultdict(list)

    for r in reactions:
        # For external EC graphs, discard reactions with no EC mapping.
        if not r.ecs:
            continue

        if not multi_input_gate(r.inputs, G0, CONFIG.TAU_THRESHOLD, CONFIG.ROOTS_PER_INPUT, rng):
            continue

        for a in r.inputs:
            for b in r.outputs:
                A, B = _canon(a), _canon(b)
                if A == B or _is_currency(A) or _is_currency(B):
                    continue
                edge2rxns[(A, B)].append(r)

    return edge2rxns


def compute_supports(edge2rxns: Dict[Tuple[str, str], List[Reaction]], ec_abund: pd.DataFrame):
    sample_ids = [str(s) for s in ec_abund.index]
    available_ecs = set(ec_abund.columns)

    E = defaultdict(dict)
    edge_ec_rows = []

    for edge, rxns in edge2rxns.items():
        ecs = sorted(set(itertools.chain.from_iterable(r.ecs for r in rxns)))
        ecs_present = [ec for ec in ecs if ec in available_ecs]

        edge_ec_rows.append({
            "A": edge[0],
            "B": edge[1],
            "n_reactions": len(rxns),
            "n_ecs_mapped": len(ecs),
            "n_ecs_present": len(ecs_present),
            "ecs_present": ";".join(ecs_present[:50]),
        })

        if not ecs_present:
            for sid in sample_ids:
                E[edge][sid] = 0.0
            continue

        sums = ec_abund.loc[:, ecs_present].sum(axis=1)
        for sid, val in sums.items():
            E[edge][str(sid)] = float(val)

    for edge in edge2rxns.keys():
        for sid in sample_ids:
            E[edge].setdefault(sid, 0.0)

    return sample_ids, E, pd.DataFrame(edge_ec_rows)


def compute_baselines(E):
    return {edge: geometric_mean(sample_map.values()) for edge, sample_map in E.items()}


def build_and_save_graphs(ec_abund: pd.DataFrame, reactions: List[Reaction]) -> None:
    os.makedirs(CONFIG.OUT_DIR, exist_ok=True)
    rng = random.Random(CONFIG.RANDOM_SEED)

    G0 = build_global_substrate_graph(reactions)
    edge2rxns = accepted_edges_map(reactions, G0, rng)

    sample_ids, E, edge_ec_df = compute_supports(edge2rxns, ec_abund)
    Ehat = compute_baselines(E)

    edge_ec_df.to_csv(os.path.join(CONFIG.OUT_DIR, "edge_ec_mapping_diagnostics.csv"), index=False)

    rows = []
    for (A, B), base in Ehat.items():
        rows.append({"A": A, "B": B, "E_hat_geom": base})
    pd.DataFrame(rows).to_csv(os.path.join(CONFIG.OUT_DIR, "global_baselines.csv"), index=False)

    summary_rows = []

    for sid in sample_ids:
        Gs = nx.DiGraph(sample_id=sid)
        edge_rows = []

        for (A, B), rxns in edge2rxns.items():
            Es = E[(A, B)][sid]
            if Es < CONFIG.THETA_MIN:
                continue

            base = Ehat[(A, B)]
            if base <= 0:
                continue

            w = math.exp(-Es / (base + CONFIG.EPSILON))

            Gs.add_edge(
                A, B,
                weight=w,
                support=Es,
                baseline=base,
                n_reactions=len(rxns),
            )

            edge_rows.append({
                "A": A,
                "B": B,
                "support_E": Es,
                "baseline_Ehat": base,
                "weight": w,
                "n_reactions": len(rxns),
            })

        graph_base = os.path.join(CONFIG.OUT_DIR, f"{sid}")

        if Gs.number_of_edges() > 0:
            nx.write_graphml(Gs, graph_base + ".graphml")

            with open(graph_base + ".edgelist", "w", encoding="utf-8") as f:
                for u, v, d in Gs.edges(data=True):
                    f.write(f"{u}\t{v}\t{d.get('weight', 1.0):.10f}\n")

            pd.DataFrame(edge_rows).sort_values(["A", "B"]).to_csv(graph_base + "_edges.csv", index=False)

        else:
            with open(graph_base + "_EMPTY.txt", "w") as f:
                f.write("No edges passed thresholds for this sample.\n")

        summary_rows.append({
            "sample_id": sid,
            "n_nodes": Gs.number_of_nodes(),
            "n_edges": Gs.number_of_edges(),
            "total_support": float(sum(x["support_E"] for x in edge_rows)) if edge_rows else 0.0,
        })

    pd.DataFrame(summary_rows).to_csv(os.path.join(CONFIG.OUT_DIR, "graph_summary.csv"), index=False)

    print("[*] Graph summary:")
    print(pd.DataFrame(summary_rows).describe(include="all"))


def main():
    print("[*] Loading external EC abundance table...")
    ec_abund = load_ec_abundance(CONFIG.EC_TABLE, max_samples=CONFIG.MAX_SAMPLES)
    print("    EC abundance shape, samples x ECs:", ec_abund.shape)
    print("    First sample IDs:", list(ec_abund.index[:5]))

    print("[*] Loading MetaCyc reaction->EC map...")
    rxn_to_ecs = load_metacyc_reaction_ec_map(CONFIG.METACYC_RXN_EC_MAP)
    print("    MetaCyc reactions with EC mappings:", len(rxn_to_ecs))

    print("[*] Loading AGORA reactions...")
    reactions = load_reactions(CONFIG.REACTIONS_PATH, rxn_to_ecs)
    print("    Total reactions after currency filtering:", len(reactions))
    print("    Reactions with EC mapping:", sum(1 for r in reactions if r.ecs))

    print("[*] Building external EC-weighted graphs...")
    build_and_save_graphs(ec_abund, reactions)

    print("[*] Done. Output dir:", CONFIG.OUT_DIR)


if __name__ == "__main__":
    main()
