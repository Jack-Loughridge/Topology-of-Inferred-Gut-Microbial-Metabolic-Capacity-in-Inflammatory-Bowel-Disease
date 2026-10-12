#!/usr/bin/env python3
from __future__ import annotations

import os
import re
import math
import random
import itertools
from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import networkx as nx


class CONFIG:
    BASE_DIR = "/home/Jack/Real_Data/external_validation/serrano_gomez_ibd"
    HUMANN_DIR = BASE_DIR + "/ge50_external_validation/humann"
    MANIFEST = BASE_DIR + "/ge50_external_validation/manifests/PRJEB42155_ge50_local_fastq_pairs.tsv"

    REACTIONS_PATH = "/home/Jack/Real_Data/AGORA_reactions_canon.parquet"

    OUT_DIR = BASE_DIR + "/ge50_external_validation/out_graphs_species_proxy_033"
    SPECIES_PROXY_OUT = BASE_DIR + "/ge50_external_validation/humann_merged/ge50_species_proxy_from_genefamilies_033.tsv"

    MAX_SAMPLES = 33

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

    CURRENCY = {
        "H2O","WATER","PROTON","H+","ATP","ADP","PI","PPI","CO2","CO(2)","NAD","NADH",
        "NADP","NADPH","FAD","FADH2","NAD(P)H","OXYGEN","O2","NH3","AMMONIA","HCO3-",
        "BICARBONATE","CO-A","COENZYME_A","COENZYME A","S-ADENOSYLMETHIONINE","SAM",
        "S-ADENOSYLHOMOCYSTEINE","SAH","UMP","UDP","UTP","AMP","CMP","CDP","CTP",
        "GMP","GDP","GTP"
    }


@dataclass(frozen=True)
class Reaction:
    rid: str
    inputs: Tuple[str, ...]
    outputs: Tuple[str, ...]
    catalyst_keys: Tuple[str, ...]


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


def species_key_from_text(x: str) -> str | None:
    """
    Convert strings like:
      g__Bacteroides.s__Bacteroides_vulgatus
      Bacteroides_vulgatus_ATCC_8482
      Bacteroides vulgatus
    into:
      bacteroides_vulgatus
    """
    if x is None:
        return None

    s = str(x).strip()
    if not s or s.lower() in {"unclassified", "unknown"}:
        return None

    # HUMAnN / MetaPhlAn style
    m = re.search(r"s__([A-Za-z][A-Za-z0-9-]*)_([A-Za-z][A-Za-z0-9-]*)", s)
    if m:
        return f"{m.group(1).lower()}_{m.group(2).lower()}"

    # Sometimes species appears after s__ with spaces
    m = re.search(r"s__([A-Za-z][A-Za-z0-9-]*)[ _]([A-Za-z][A-Za-z0-9-]*)", s)
    if m:
        return f"{m.group(1).lower()}_{m.group(2).lower()}"

    cleaned = s
    cleaned = cleaned.replace("[", "").replace("]", "")
    cleaned = re.sub(r"^(g__|s__)", "", cleaned)
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", cleaned)
    toks = [t for t in cleaned.split("_") if t]

    # Remove common non-binomial tokens
    bad = {
        "strain", "subsp", "subspecies", "sp", "cf", "group",
        "bacterium", "uncultured", "unclassified", "unknown"
    }
    toks = [t for t in toks if t.lower() not in bad]

    if len(toks) >= 2:
        return f"{toks[0].lower()}_{toks[1].lower()}"

    return None


def geometric_mean(vals: Iterable[float]) -> float:
    arr = [float(v) for v in vals if float(v) > 0]
    if not arr:
        return 0.0
    return float(np.exp(np.mean(np.log(arr))))


def get_first_n_runs() -> List[str]:
    df = pd.read_csv(CONFIG.MANIFEST, sep="\t")
    run_col = df.columns[0]
    runs = [str(x) for x in df[run_col].head(CONFIG.MAX_SAMPLES)]
    return runs


def load_species_proxy_from_humann_genefamilies(runs: List[str]) -> pd.DataFrame:
    """
    Uses species-stratified HUMAnN genefamily rows as a species proxy.

    This is not a direct MetaPhlAn relative abundance table, but it lets us
    recover sample-specific species signals from the retained HUMAnN outputs.
    """
    rows = {}

    for run in runs:
        path = os.path.join(CONFIG.HUMANN_DIR, run, f"{run}_genefamilies.tsv")
        if not os.path.exists(path):
            raise FileNotFoundError(path)

        df = pd.read_csv(path, sep="\t", comment=None)
        feature_col = df.columns[0]
        value_col = df.columns[1]

        d = defaultdict(float)

        for feat, val in zip(df[feature_col].astype(str), pd.to_numeric(df[value_col], errors="coerce").fillna(0.0)):
            if "|" not in feat:
                continue

            taxon = feat.rsplit("|", 1)[-1]
            if "unclassified" in taxon.lower():
                continue

            key = species_key_from_text(taxon)
            if key is None:
                continue

            d[key] += float(val)

        total = sum(d.values())
        if total > 0:
            d = {k: v / total for k, v in d.items()}

        rows[run] = d
        print(f"    {run}: {len(d)} species-like keys, total before norm={total:.6g}")

    abund = pd.DataFrame.from_dict(rows, orient="index").fillna(0.0)
    abund.index.name = "sample_id"
    abund = abund.reindex(sorted(abund.columns), axis=1)

    os.makedirs(os.path.dirname(CONFIG.SPECIES_PROXY_OUT), exist_ok=True)
    abund.to_csv(CONFIG.SPECIES_PROXY_OUT, sep="\t")

    return abund


def build_global_graph_and_matched_reactions(species_keys: set[str]):
    print("[*] Reading AGORA reaction table...")
    ext = os.path.splitext(CONFIG.REACTIONS_PATH)[1].lower()
    if ext in (".parquet", ".pq"):
        df = pd.read_parquet(CONFIG.REACTIONS_PATH)
    else:
        df = pd.read_excel(CONFIG.REACTIONS_PATH)

    need = [CONFIG.COL_REACTION_ID, CONFIG.COL_INPUTS, CONFIG.COL_OUTPUTS, CONFIG.COL_CATALYSTS]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"Missing reaction columns: {missing}")

    print("    Reaction table shape:", df.shape)

    G0 = nx.Graph()
    matched_reactions: List[Reaction] = []

    n_usable = 0
    n_with_any_catalyst_key = 0
    n_with_present_catalyst_key = 0
    catalyst_key_counter = Counter()
    matched_catalyst_key_counter = Counter()

    for row in df.itertuples(index=False):
        rowd = row._asdict()

        rid = str(rowd[CONFIG.COL_REACTION_ID]).strip()

        ins = tuple(
            _canon(x) for x in parse_listish(rowd[CONFIG.COL_INPUTS])
            if not _is_currency(x)
        )
        outs = tuple(
            _canon(x) for x in parse_listish(rowd[CONFIG.COL_OUTPUTS])
            if not _is_currency(x)
        )

        if not ins or not outs:
            continue

        n_usable += 1

        for x in ins + outs:
            if not _is_currency(x):
                G0.add_node(_canon(x))

        for a, b in itertools.combinations(ins, 2):
            G0.add_edge(_canon(a), _canon(b))

        for a in ins:
            for b in outs:
                A, B = _canon(a), _canon(b)
                if A != B and not _is_currency(A) and not _is_currency(B):
                    G0.add_edge(A, B)

        cats = parse_listish(rowd[CONFIG.COL_CATALYSTS])
        cat_keys = sorted(set(k for k in (species_key_from_text(c) for c in cats) if k))

        if cat_keys:
            n_with_any_catalyst_key += 1
            catalyst_key_counter.update(cat_keys)

        present = tuple(sorted(k for k in cat_keys if k in species_keys))
        if present:
            n_with_present_catalyst_key += 1
            matched_catalyst_key_counter.update(present)
            matched_reactions.append(
                Reaction(
                    rid=rid,
                    inputs=ins,
                    outputs=outs,
                    catalyst_keys=present,
                )
            )

    os.makedirs(CONFIG.OUT_DIR, exist_ok=True)

    pd.DataFrame(
        [{"species_key": k, "n_reactions": v} for k, v in catalyst_key_counter.most_common()]
    ).to_csv(os.path.join(CONFIG.OUT_DIR, "agora_catalyst_species_key_counts.csv"), index=False)

    pd.DataFrame(
        [{"species_key": k, "n_matched_reactions": v} for k, v in matched_catalyst_key_counter.most_common()]
    ).to_csv(os.path.join(CONFIG.OUT_DIR, "matched_catalyst_species_key_counts.csv"), index=False)

    diag = {
        "n_usable_reactions": n_usable,
        "n_reactions_with_any_catalyst_species_key": n_with_any_catalyst_key,
        "n_reactions_with_present_external_species_key": n_with_present_catalyst_key,
        "n_matched_reaction_objects": len(matched_reactions),
        "n_global_nodes": G0.number_of_nodes(),
        "n_global_edges": G0.number_of_edges(),
        "n_external_species_keys": len(species_keys),
    }

    pd.DataFrame([diag]).to_csv(os.path.join(CONFIG.OUT_DIR, "reaction_species_match_diagnostics.csv"), index=False)

    print("[*] Reaction/species diagnostics:")
    for k, v in diag.items():
        print(f"    {k}: {v}")

    return G0, matched_reactions


def multi_input_gate(inputs: Tuple[str, ...], G0: nx.Graph, tau: float, roots_per_input: int, rng: random.Random) -> bool:
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


def accepted_edges_map(reactions: List[Reaction], G0: nx.Graph, rng: random.Random):
    edge2rxns = defaultdict(list)

    for i, r in enumerate(reactions, start=1):
        if i % 100000 == 0:
            print(f"    gated {i:,}/{len(reactions):,} matched reactions...")

        if not multi_input_gate(r.inputs, G0, CONFIG.TAU_THRESHOLD, CONFIG.ROOTS_PER_INPUT, rng):
            continue

        for a in r.inputs:
            for b in r.outputs:
                A, B = _canon(a), _canon(b)
                if A == B or _is_currency(A) or _is_currency(B):
                    continue
                edge2rxns[(A, B)].append(r)

    return edge2rxns


def compute_supports(edge2rxns, species_abund: pd.DataFrame):
    sample_ids = [str(s) for s in species_abund.index]
    available = set(species_abund.columns)

    E = defaultdict(dict)
    edge_rows = []

    for edge, rxns in edge2rxns.items():
        keys = sorted(set(itertools.chain.from_iterable(r.catalyst_keys for r in rxns)))
        keys_present = [k for k in keys if k in available]

        edge_rows.append({
            "A": edge[0],
            "B": edge[1],
            "n_reactions": len(rxns),
            "n_catalyst_species_keys": len(keys),
            "n_present_species_keys": len(keys_present),
            "present_species_keys": ";".join(keys_present[:80]),
        })

        if not keys_present:
            for sid in sample_ids:
                E[edge][sid] = 0.0
            continue

        sums = species_abund.loc[:, keys_present].sum(axis=1)
        for sid, val in sums.items():
            E[edge][str(sid)] = float(val)

    for edge in edge2rxns.keys():
        for sid in sample_ids:
            E[edge].setdefault(sid, 0.0)

    pd.DataFrame(edge_rows).to_csv(os.path.join(CONFIG.OUT_DIR, "edge_species_mapping_diagnostics.csv"), index=False)

    return sample_ids, E


def compute_baselines(E):
    return {edge: geometric_mean(sample_map.values()) for edge, sample_map in E.items()}


def build_and_save_graphs(species_abund: pd.DataFrame, G0: nx.Graph, matched_reactions: List[Reaction]):
    os.makedirs(CONFIG.OUT_DIR, exist_ok=True)
    rng = random.Random(CONFIG.RANDOM_SEED)

    print("[*] Applying original multi-input gate to matched reactions...")
    edge2rxns = accepted_edges_map(matched_reactions, G0, rng)
    print("    Accepted directed edges:", len(edge2rxns))

    sample_ids, E = compute_supports(edge2rxns, species_abund)
    Ehat = compute_baselines(E)

    pd.DataFrame(
        [{"A": A, "B": B, "E_hat_geom": base} for (A, B), base in Ehat.items()]
    ).to_csv(os.path.join(CONFIG.OUT_DIR, "global_baselines.csv"), index=False)

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

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(os.path.join(CONFIG.OUT_DIR, "graph_summary.csv"), index=False)

    print("[*] Graph summary:")
    print(summary.describe(include="all"))


def main():
    print("[*] Selecting first 33 manifest samples...")
    runs = get_first_n_runs()
    print("    First run:", runs[0])
    print("    Last run:", runs[-1])

    missing_done = [
        r for r in runs
        if not os.path.exists(os.path.join(CONFIG.HUMANN_DIR, r, f"{r}_humann.DONE"))
    ]
    if missing_done:
        raise RuntimeError(f"These first-33 samples do not have HUMAnN .DONE files: {missing_done[:10]}")

    print("[*] Building species proxy table from stratified HUMAnN genefamilies...")
    species_abund = load_species_proxy_from_humann_genefamilies(runs)
    print("    Species proxy table shape, samples x species_keys:", species_abund.shape)
    print("    Wrote:", CONFIG.SPECIES_PROXY_OUT)

    print("[*] Building global substrate scaffold and matching AGORA catalysts to external species keys...")
    G0, matched_reactions = build_global_graph_and_matched_reactions(set(species_abund.columns))

    if len(matched_reactions) == 0:
        raise RuntimeError(
            "No AGORA reactions matched the external species proxy. "
            "Check reaction_species_match_diagnostics.csv and catalyst naming."
        )

    print("[*] Building weighted directed graphs...")
    build_and_save_graphs(species_abund, G0, matched_reactions)

    print("[*] Done. Output dir:", CONFIG.OUT_DIR)


if __name__ == "__main__":
    main()
