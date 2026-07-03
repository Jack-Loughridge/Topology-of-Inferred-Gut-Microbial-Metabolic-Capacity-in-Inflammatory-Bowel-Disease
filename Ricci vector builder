#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
14_compute_ricci_faithful_pairwise_active.py

Faithful pairwise Ricci curvature builder for the original relative-weight
microbiome graphs, restricted to active directed edges.

Input:
    /home/Jack/Real_Data/out_graphs/*.graphml

Output:
    /home/Jack/Real_Data/Ricci_Summary_faithful_pairwise_active/<sample>_ricci.csv.gz

Active-edge convention:
    The graph builder encodes zero inferred microbial support as w = 1.
    Therefore, before any Ricci calculation, this script removes all directed
    edges with

        w >= 1 - active_tol

    and computes everything on the remaining active directed graph.

This means active edges are used for:
    - out-neighbour sets,
    - mu_x on {x} union active out-neighbours(x),
    - Monte Carlo random walks,
    - required reachability sources,
    - Ricci curvature output coordinates.

Faithful pairwise transport cost:
    d(u,v) = 0 if u=v
    d(u,v) = -log(epsilon_dist + q(u,v)) otherwise

where q(u,v) is estimated from Monte Carlo random walks starting at u, not only
from the edge source a.

For each active directed edge a -> b:
    - build mu_a on {a} union active out-neighbours(a)
    - build mu_b on {b} union active out-neighbours(b)
    - for every u in supp(mu_a), use Monte Carlo reachability from u
      to every v in supp(mu_b)
    - solve the same rho-form optimal transport LP
    - compute k_ab = 1 - W_ab / d_ab

Important:
    This is intentionally separate from both:
      - the old source-anchored Ricci_Summary, and
      - the first faithful_pairwise run that accidentally included w=1 edges.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import pickle
import random
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Any

import networkx as nx
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy import sparse


@dataclass
class Config:
    base_dir: str = "/home/Jack/Real_Data"
    graph_dir: str = "/home/Jack/Real_Data/out_graphs"
    output_dir: str = "/home/Jack/Real_Data/Ricci_Summary_faithful_pairwise_active"

    epsilon_dist: float = 1e-4
    c_single_out: float = 0.001
    beta: float = 1.4

    # "auto" tries edge_weight_attr first if supplied, then common GraphML keys.
    edge_weight_attr: str = "auto"
    active_tol: float = 1e-12

    n_paths: int = 1000
    max_steps: int = 10000
    seed: int = 13

    mu_prune_threshold: float = 1e-6

    # Practical LP guard. If a transport problem is too large, the script caps
    # supports to the largest-mass points and records that this happened.
    enable_support_cap: bool = True
    topk_support: int = 150
    max_lp_vars: int = 200000

    overwrite: bool = False
    max_samples: int | None = None
    max_edges_per_sample: int | None = None
    progress_every: int = 250
    save_reachability_cache: bool = False


def stable_seed(*parts, base_seed: int = 13) -> int:
    h = hashlib.sha256(str(base_seed).encode())

    for p in parts:
        h.update(b"||")
        h.update(str(p).encode())

    return int.from_bytes(h.digest()[:8], "little") % (2**32)


def sample_id_from_graph(path: Path) -> str:
    if path.name.endswith(".graphml"):
        return path.name[:-len(".graphml")]
    return path.stem


def _iter_edges_with_data(G_raw):
    """
    Works for DiGraph and MultiDiGraph returned by networkx.read_graphml.
    Yields (u, v, data).
    """
    if G_raw.is_multigraph():
        for u, v, _key, data in G_raw.edges(keys=True, data=True):
            yield u, v, data
    else:
        for u, v, data in G_raw.edges(data=True):
            yield u, v, data


def read_edge_weight(data: Dict[str, Any], cfg: Config) -> Tuple[float, str]:
    """
    Read edge weight robustly.

    Priority:
      1. cfg.edge_weight_attr, if not "auto" and present.
      2. "weight".
      3. "d1".
      4. "Weight".
      5. "D1".

    The script deliberately raises if no usable weight is found. It does not
    silently default to 1.0, because doing so can accidentally turn missing
    weights into inactive edges or corrupt transition probabilities.
    """
    candidate_keys: List[str] = []

    if cfg.edge_weight_attr and cfg.edge_weight_attr != "auto":
        candidate_keys.append(cfg.edge_weight_attr)

    candidate_keys.extend(["weight", "d1", "Weight", "D1"])

    seen = set()
    ordered_keys = []

    for key in candidate_keys:
        if key not in seen:
            seen.add(key)
            ordered_keys.append(key)

    for key in ordered_keys:
        if key in data:
            try:
                w = float(data[key])
            except Exception as e:
                raise ValueError(f"Could not convert edge attribute {key}={data[key]!r} to float") from e

            if not math.isfinite(w):
                raise ValueError(f"Non-finite edge weight {key}={data[key]!r}")

            return w, key

    raise KeyError(
        "No edge weight attribute found. "
        f"Available keys={list(data.keys())}; requested edge_weight_attr={cfg.edge_weight_attr!r}"
    )


def load_active_graph(path: Path, cfg: Config):
    """
    Load a GraphML file and return an active directed graph.

    Active edge:
        finite w < 1 - cfg.active_tol

    All inactive w=1 template edges are removed before Ricci computation.

    If duplicate directed edges are encountered, the lower weight is retained,
    because lower weight means stronger inferred support.
    """
    G_raw = nx.read_graphml(path)

    raw_nodes = [str(n) for n in G_raw.nodes()]
    raw_n_nodes = len(raw_nodes)
    raw_n_edges = G_raw.number_of_edges()

    G = nx.DiGraph()
    G.add_nodes_from(raw_nodes)

    n_weight_read = 0
    n_active = 0
    n_inactive = 0
    n_nonpositive = 0
    n_duplicate_pairs = 0

    weight_keys_used: Dict[str, int] = defaultdict(int)

    for u0, v0, data in _iter_edges_with_data(G_raw):
        u = str(u0)
        v = str(v0)

        w, key = read_edge_weight(data, cfg)
        weight_keys_used[key] += 1
        n_weight_read += 1

        if w <= 0.0:
            # The original formula should produce positive weights. Nonpositive
            # weights cannot be used in w^{-beta}; record and skip.
            n_nonpositive += 1
            continue

        if w < (1.0 - cfg.active_tol):
            if G.has_edge(u, v):
                n_duplicate_pairs += 1
                old_w = float(G[u][v]["weight"])
                if w < old_w:
                    G[u][v]["weight"] = float(w)
                    G[u][v]["original_weight"] = float(w)
                    G[u][v]["weight_key"] = key
            else:
                G.add_edge(
                    u,
                    v,
                    weight=float(w),
                    original_weight=float(w),
                    weight_key=key,
                )
                n_active += 1
        else:
            n_inactive += 1

    # Remove isolated nodes after active filtering. This does not change any
    # edge curvature, mu_x, or random walk, but makes graph diagnostics honest.
    isolated = [n for n in G.nodes() if G.in_degree(n) == 0 and G.out_degree(n) == 0]
    G.remove_nodes_from(isolated)

    active_stats = {
        "raw_n_nodes": int(raw_n_nodes),
        "raw_n_edges": int(raw_n_edges),
        "n_weight_read": int(n_weight_read),
        "active_n_nodes": int(G.number_of_nodes()),
        "active_n_edges": int(G.number_of_edges()),
        "n_active_edges_kept": int(G.number_of_edges()),
        "n_inactive_edges_removed": int(n_inactive),
        "n_nonpositive_edges_skipped": int(n_nonpositive),
        "n_duplicate_directed_pairs_seen": int(n_duplicate_pairs),
        "n_isolated_nodes_removed_after_active_filter": int(len(isolated)),
        "active_tol": float(cfg.active_tol),
        "weight_keys_used": dict(weight_keys_used),
    }

    return G, active_stats


def outgoing_probs(G: nx.DiGraph, x: str, cfg: Config, for_distance: bool) -> Dict[str, float]:
    """
    Transition probabilities over active out-neighbours only.

    Because G has already been active-filtered, successors(x) are exactly the
    active directed out-neighbours of x.
    """
    nbrs = list(G.successors(x))

    if not nbrs:
        return {}

    vals = []

    for y in nbrs:
        w = float(G[x][y]["weight"])
        w = max(w, 1e-12)
        vals.append(w ** (-cfg.beta))

    Z = float(sum(vals))

    if Z <= 0.0 or not math.isfinite(Z):
        probs = {y: 1.0 / len(nbrs) for y in nbrs}
    else:
        probs = {y: v / Z for y, v in zip(nbrs, vals)}

    # Same outdegree-one correction as original distance-stage code.
    if for_distance and len(nbrs) == 1:
        probs[nbrs[0]] = 1.0 - cfg.c_single_out

    return probs


def mu_dist(G: nx.DiGraph, x: str, cfg: Config) -> Dict[str, float]:
    """
    Probability measure mu_x on {x} union active out-neighbours(x).
    """
    nbrs = list(G.successors(x))
    k = len(nbrs)

    if k == 0:
        return {x: 1.0}

    p = outgoing_probs(G, x, cfg, for_distance=False)

    mu = {y: (k / (k + 1.0)) * p[y] for y in nbrs}
    mu[x] = 1.0 / (k + 1.0)

    return mu


def topk_cap(mu: Dict[str, float], k: int) -> Dict[str, float]:
    if len(mu) <= k:
        return mu

    items = sorted(mu.items(), key=lambda kv: kv[1], reverse=True)[:k]
    Z = float(sum(v for _, v in items))

    if Z <= 0.0 or not math.isfinite(Z):
        key = max(mu, key=mu.get)
        return {key: 1.0}

    return {kk: vv / Z for kk, vv in items}


def prune_and_normalise(
    mu: Dict[str, float],
    cfg: Config,
    centre: str | None = None,
) -> Dict[str, float]:
    """
    Remove support points with mass below cfg.mu_prune_threshold, then renormalise.

    The centre vertex is always retained. This matters for high-outdegree
    vertices, where mu_x(x)=1/(k+1) may be below the pruning threshold.
    """
    if not mu:
        if centre is None:
            raise ValueError("Cannot prune an empty measure without a centre")
        return {centre: 1.0}

    pruned = {k: v for k, v in mu.items() if v >= cfg.mu_prune_threshold}

    if centre is not None and centre in mu:
        pruned[centre] = mu[centre]

    if not pruned:
        key = centre if centre is not None and centre in mu else max(mu, key=mu.get)
        return {key: 1.0}

    Z = float(sum(pruned.values()))

    if Z <= 0.0 or not math.isfinite(Z):
        key = centre if centre is not None and centre in mu else max(mu, key=mu.get)
        return {key: 1.0}

    pruned = {k: v / Z for k, v in pruned.items()}

    return pruned


def maybe_cap_supports(mu_x: Dict[str, float], mu_y: Dict[str, float], cfg: Config):
    capped = False

    if not cfg.enable_support_cap:
        return mu_x, mu_y, capped

    n_vars = len(mu_x) * len(mu_y)

    if n_vars <= cfg.max_lp_vars:
        return mu_x, mu_y, capped

    # Cap both supports to top mass points. This is still pairwise faithful on
    # the retained supports, but it is an approximation. It is recorded.
    mu_x2 = topk_cap(mu_x, cfg.topk_support)
    mu_y2 = topk_cap(mu_y, cfg.topk_support)
    capped = True

    return mu_x2, mu_y2, capped


def get_mu(G: nx.DiGraph, x: str, cfg: Config, cache: Dict[str, Dict[str, float]]):
    if x not in cache:
        cache[x] = prune_and_normalise(mu_dist(G, x, cfg), cfg, centre=x)
    return cache[x]


def build_transition_cumulative(G: nx.DiGraph, cfg: Config):
    cumulative = {}

    for x in G.nodes():
        probs = outgoing_probs(G, x, cfg, for_distance=True)

        if not probs:
            continue

        acc = 0.0
        cum = []

        for y in sorted(probs.keys()):
            acc += probs[y]
            cum.append((y, acc))

        if cum:
            # Preserve original method: cumulative sampler always has an endpoint
            # at 1.0, while p_edge in the path-probability product can still be
            # 1 - c_single_out for outdegree-one vertices.
            cum[-1] = (cum[-1][0], 1.0)
            cumulative[x] = cum

    return cumulative


def choose_next(cum, rng: random.Random):
    r = rng.random()

    for y, c in cum:
        if r <= c:
            return y

    return cum[-1][0]


def required_reachability_sources(G: nx.DiGraph, cfg: Config):
    """
    To compute all active-edge curvatures faithfully, we need reachability from
    every node that appears in supp(mu_a) for some active edge source a.
    """
    mu_cache = {}
    required = set()

    for a, _ in G.edges():
        mu_a = get_mu(G, str(a), cfg, mu_cache)
        required.update(mu_a.keys())

    return sorted(required), mu_cache


def precompute_reachability(
    G: nx.DiGraph,
    graph_id: str,
    sources: List[str],
    cumulative,
    cfg: Config,
):
    """
    For every source u, simulate random walks from u and record every visited
    vertex before termination.

    q(u,v) = fraction of paths from u that visit v before:
        - dead end,
        - path probability falls below epsilon_dist,
        - max_steps is reached.

    This is the faithful pairwise reachability cache, now on the active graph.
    """
    cache: Dict[str, Dict[str, int]] = {}
    stats = []

    print(f"    precomputing faithful reachability for {len(sources)} sources", flush=True)

    for idx, src in enumerate(sources, 1):
        rng = random.Random(
            stable_seed("faithful_pairwise_reachability_active", graph_id, src, base_seed=cfg.seed)
        )

        counts: Dict[str, int] = defaultdict(int)
        capped_paths = 0
        total_steps = 0

        for _ in range(cfg.n_paths):
            cur = src
            p_prod = 1.0
            steps = 0

            # Starting point is reached with probability 1.
            visited = {cur}

            while True:
                cum = cumulative.get(cur)

                if not cum:
                    break

                nxt = choose_next(cum, rng)
                probs = outgoing_probs(G, cur, cfg, for_distance=True)
                p_edge = probs.get(nxt, 0.0)

                new_p = p_prod * p_edge

                if new_p < cfg.epsilon_dist:
                    break

                cur = nxt
                p_prod = new_p
                visited.add(cur)
                steps += 1

                if steps >= cfg.max_steps:
                    capped_paths += 1
                    break

            total_steps += steps

            for v in visited:
                counts[v] += 1

        cache[src] = dict(counts)

        stats.append(
            {
                "source": src,
                "unique_reached": int(len(counts)),
                "mean_steps": float(total_steps / cfg.n_paths),
                "capped_paths": int(capped_paths),
            }
        )

        if idx == 1 or idx % 250 == 0 or idx == len(sources):
            print(f"        reachability {idx}/{len(sources)} sources", flush=True)

    return cache, pd.DataFrame(stats)


def q_lookup(reach_cache: Dict[str, Dict[str, int]], u: str, v: str, cfg: Config) -> float:
    if u == v:
        return 1.0

    return reach_cache.get(u, {}).get(v, 0) / float(cfg.n_paths)


def cost_from_q(q: float, cfg: Config) -> float:
    # Kept identical to the previous faithful method.
    return float(-math.log(cfg.epsilon_dist + q))


def wasserstein_rho_sparse(mu_x: Dict[str, float], mu_y: Dict[str, float], C: np.ndarray):
    """
    Same rho-form LP as original code, but with sparse constraints.

    Variables rho_ij.

    Objective:
        sum_ij C_ij * rho_ij * mu_x_i

    Constraints:
        sum_j rho_ij = 1
        sum_i rho_ij * mu_x_i = mu_y_j
        0 <= rho_ij <= 1
    """
    xs = list(mu_x.keys())
    ys = list(mu_y.keys())

    n = len(xs)
    m = len(ys)

    mux = np.asarray([mu_x[x] for x in xs], dtype=float)
    muy = np.asarray([mu_y[y] for y in ys], dtype=float)

    c = (C * mux[:, None]).reshape(-1)

    rows = []
    cols = []
    data = []
    b = []

    # Row constraints: sum_j rho_ij = 1
    for i in range(n):
        r = len(b)
        for j in range(m):
            rows.append(r)
            cols.append(i * m + j)
            data.append(1.0)
        b.append(1.0)

    # Column constraints: sum_i rho_ij * mu_x_i = mu_y_j
    for j in range(m):
        r = len(b)
        for i in range(n):
            rows.append(r)
            cols.append(i * m + j)
            data.append(mux[i])
        b.append(muy[j])

    A_eq = sparse.coo_matrix(
        (data, (rows, cols)),
        shape=(len(b), n * m),
    ).tocsr()

    res = linprog(
        c,
        A_eq=A_eq,
        b_eq=np.asarray(b, dtype=float),
        bounds=(0.0, 1.0),
        method="highs",
    )

    if not res.success:
        return float("nan"), False, str(res.message)

    return float(res.fun), True, "ok"


def ricci_edge_faithful(
    G: nx.DiGraph,
    a: str,
    b: str,
    cfg: Config,
    mu_cache: Dict[str, Dict[str, float]],
    reach_cache: Dict[str, Dict[str, int]],
):
    mu_a = get_mu(G, a, cfg, mu_cache)
    mu_b = get_mu(G, b, cfg, mu_cache)

    mu_a, mu_b, capped = maybe_cap_supports(mu_a, mu_b, cfg)

    xs = list(mu_a.keys())
    ys = list(mu_b.keys())

    C = np.zeros((len(xs), len(ys)), dtype=float)

    for i, u in enumerate(xs):
        for j, v in enumerate(ys):
            if u == v:
                C[i, j] = 0.0
            else:
                q = q_lookup(reach_cache, u, v, cfg)
                C[i, j] = cost_from_q(q, cfg)

    W, ok, msg = wasserstein_rho_sparse(mu_a, mu_b, C)

    q_ab = q_lookup(reach_cache, a, b, cfg)
    d_ab = 0.0 if a == b else cost_from_q(q_ab, cfg)

    if (not ok) or (not np.isfinite(W)) or (not np.isfinite(d_ab)) or d_ab <= 0:
        k_ab = float("nan")
    else:
        k_ab = float(1.0 - W / d_ab)

    edge_weight = float(G[a][b]["weight"])

    return {
        "a": a,
        "b": b,
        "edge_weight": edge_weight,
        "d_ab": float(d_ab),
        "k_ab": float(k_ab),
        "W_ab": float(W),
        "q_ab": float(q_ab),
        "mu_a_size": int(len(mu_a)),
        "mu_b_size": int(len(mu_b)),
        "lp_vars": int(len(mu_a) * len(mu_b)),
        "support_capped": bool(capped),
        "lp_success": bool(ok),
        "lp_message": msg,
    }


def empty_ricci_dataframe() -> pd.DataFrame:
    cols = [
        "a",
        "b",
        "edge_weight",
        "d_ab",
        "k_ab",
        "W_ab",
        "q_ab",
        "mu_a_size",
        "mu_b_size",
        "lp_vars",
        "support_capped",
        "lp_success",
        "lp_message",
    ]
    return pd.DataFrame(columns=cols)


def process_graph(gpath: Path, cfg: Config):
    graph_id = sample_id_from_graph(gpath)

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"{graph_id}_ricci.csv.gz"
    done_marker = out_dir / f"{graph_id}.FAITHFUL_ACTIVE_DONE.marker"

    if out_path.exists() and out_path.stat().st_size > 0 and done_marker.exists() and not cfg.overwrite:
        print(f"Skipping {graph_id}: existing faithful active output found.")
        return {
            "sample_id": graph_id,
            "status": "skipped_exists",
            "out_file": str(out_path),
        }

    print(f"\n=== Faithful ACTIVE Ricci graph {graph_id} ===")
    t0 = time.time()

    G, active_stats = load_active_graph(gpath, cfg)

    print(
        f"    raw_nodes={active_stats['raw_n_nodes']} raw_edges={active_stats['raw_n_edges']} "
        f"active_nodes={G.number_of_nodes()} active_edges={G.number_of_edges()} "
        f"inactive_removed={active_stats['n_inactive_edges_removed']}",
        flush=True,
    )
    print(f"    weight_keys_used={active_stats['weight_keys_used']}", flush=True)

    all_edges = [(str(a), str(b)) for a, b in G.edges()]
    all_edges = sorted(all_edges, key=lambda x: (x[0], x[1]))

    if cfg.max_edges_per_sample is not None:
        all_edges = all_edges[: cfg.max_edges_per_sample]

    if len(all_edges) == 0:
        df = empty_ricci_dataframe()
        df.to_csv(out_path, index=False, compression="gzip")

        reach_stats = pd.DataFrame(columns=["source", "unique_reached", "mean_steps", "capped_paths"])
        reach_stats.to_csv(
            out_dir / f"{graph_id}_faithful_active_reachability_stats.csv",
            index=False,
        )

        diag = {
            "sample_id": graph_id,
            "status": "done_no_active_edges",
            **active_stats,
            "n_edges_written": 0,
            "n_required_reachability_sources": 0,
            "n_paths": int(cfg.n_paths),
            "n_nan_k_ab": 0,
            "min_k_ab": float("nan"),
            "median_k_ab": float("nan"),
            "max_k_ab": float("nan"),
            "median_d_ab": float("nan"),
            "median_W_ab": float("nan"),
            "n_support_capped_edges": 0,
            "mean_reachability_unique_reached": float("nan"),
            "total_capped_paths": 0,
            "seconds": float(time.time() - t0),
            "out_file": str(out_path),
        }

        done_marker.write_text(json.dumps(diag, indent=2))
        print(f"    done {graph_id}: no active edges after filtering", flush=True)
        return diag

    # Build mu cache partly while identifying required reachability sources.
    required_sources, mu_cache = required_reachability_sources(G, cfg)

    cumulative = build_transition_cumulative(G, cfg)

    reach_cache, reach_stats = precompute_reachability(
        G=G,
        graph_id=graph_id,
        sources=required_sources,
        cumulative=cumulative,
        cfg=cfg,
    )

    if cfg.save_reachability_cache:
        with gzip.open(out_dir / f"{graph_id}_faithful_active_reachability_cache.pkl.gz", "wb") as fh:
            pickle.dump(
                {
                    "counts": reach_cache,
                    "n_paths": cfg.n_paths,
                    "epsilon_dist": cfg.epsilon_dist,
                    "beta": cfg.beta,
                    "c_single_out": cfg.c_single_out,
                    "active_tol": cfg.active_tol,
                    "active_edge_convention": "w < 1 - active_tol",
                },
                fh,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    reach_stats.to_csv(
        out_dir / f"{graph_id}_faithful_active_reachability_stats.csv",
        index=False,
    )

    rows = []

    for idx, (a, b) in enumerate(all_edges, 1):
        r = ricci_edge_faithful(
            G=G,
            a=a,
            b=b,
            cfg=cfg,
            mu_cache=mu_cache,
            reach_cache=reach_cache,
        )
        rows.append(r)

        if idx == 1 or idx % cfg.progress_every == 0 or idx == len(all_edges):
            print(f"        ricci {idx}/{len(all_edges)} active edges", flush=True)

    df = pd.DataFrame(rows)

    # Keep core columns first for compatibility with old classifiers.
    first_cols = ["a", "b", "edge_weight", "d_ab", "k_ab", "W_ab"]
    other_cols = [c for c in df.columns if c not in first_cols]
    df = df[first_cols + other_cols]

    df.to_csv(out_path, index=False, compression="gzip")

    diag = {
        "sample_id": graph_id,
        "status": "done",
        **active_stats,
        "n_edges_written": int(len(df)),
        "n_required_reachability_sources": int(len(required_sources)),
        "n_paths": int(cfg.n_paths),
        "n_nan_k_ab": int(df["k_ab"].isna().sum()),
        "min_k_ab": float(df["k_ab"].min()) if len(df) else float("nan"),
        "median_k_ab": float(df["k_ab"].median()) if len(df) else float("nan"),
        "max_k_ab": float(df["k_ab"].max()) if len(df) else float("nan"),
        "median_d_ab": float(df["d_ab"].median()) if len(df) else float("nan"),
        "median_W_ab": float(df["W_ab"].median()) if len(df) else float("nan"),
        "n_support_capped_edges": int(df["support_capped"].sum()) if "support_capped" in df else 0,
        "mean_reachability_unique_reached": float(reach_stats["unique_reached"].mean()) if len(reach_stats) else float("nan"),
        "total_capped_paths": int(reach_stats["capped_paths"].sum()) if len(reach_stats) else 0,
        "seconds": float(time.time() - t0),
        "out_file": str(out_path),
    }

    done_marker.write_text(json.dumps(diag, indent=2))

    print(
        f"    done {graph_id}: active_edges={diag['n_edges_written']} "
        f"nan={diag['n_nan_k_ab']} median_k={diag['median_k_ab']:.4f} "
        f"time={diag['seconds']:.1f}s",
        flush=True,
    )

    return diag


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--base-dir", default=Config.base_dir)
    ap.add_argument("--graph-dir", default=Config.graph_dir)
    ap.add_argument("--output-dir", default=Config.output_dir)

    ap.add_argument("--n-paths", type=int, default=Config.n_paths)
    ap.add_argument("--max-steps", type=int, default=Config.max_steps)
    ap.add_argument("--seed", type=int, default=Config.seed)

    ap.add_argument("--beta", type=float, default=Config.beta)
    ap.add_argument("--epsilon-dist", type=float, default=Config.epsilon_dist)
    ap.add_argument("--c-single-out", type=float, default=Config.c_single_out)
    ap.add_argument("--mu-prune-threshold", type=float, default=Config.mu_prune_threshold)

    ap.add_argument("--edge-weight-attr", default=Config.edge_weight_attr)
    ap.add_argument("--active-tol", type=float, default=Config.active_tol)

    ap.add_argument("--topk-support", type=int, default=Config.topk_support)
    ap.add_argument("--max-lp-vars", type=int, default=Config.max_lp_vars)
    ap.add_argument("--disable-support-cap", action="store_true")

    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--max-edges-per-sample", type=int, default=None)
    ap.add_argument("--progress-every", type=int, default=Config.progress_every)

    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--save-reachability-cache", action="store_true")

    args = ap.parse_args()

    cfg = Config(
        base_dir=args.base_dir,
        graph_dir=args.graph_dir,
        output_dir=args.output_dir,
        epsilon_dist=args.epsilon_dist,
        c_single_out=args.c_single_out,
        beta=args.beta,
        edge_weight_attr=args.edge_weight_attr,
        active_tol=args.active_tol,
        n_paths=args.n_paths,
        max_steps=args.max_steps,
        seed=args.seed,
        mu_prune_threshold=args.mu_prune_threshold,
        enable_support_cap=not args.disable_support_cap,
        topk_support=args.topk_support,
        max_lp_vars=args.max_lp_vars,
        overwrite=args.overwrite,
        max_samples=args.max_samples,
        max_edges_per_sample=args.max_edges_per_sample,
        progress_every=args.progress_every,
        save_reachability_cache=args.save_reachability_cache,
    )

    graph_dir = Path(cfg.graph_dir).expanduser().resolve()
    out_dir = Path(cfg.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(graph_dir.glob("*.graphml"))

    if cfg.max_samples is not None:
        files = files[: cfg.max_samples]

    if not files:
        raise FileNotFoundError(f"No .graphml files found in {graph_dir}")

    (out_dir / "faithful_pairwise_active_ricci_config.json").write_text(
        json.dumps(asdict(cfg), indent=2)
    )

    print("[*] Faithful pairwise ACTIVE Ricci computation")
    print(f"    graph_dir: {graph_dir}")
    print(f"    output:    {out_dir}")
    print(f"    graphs:    {len(files)}")
    print(f"    n_paths:   {cfg.n_paths}")
    print(f"    beta:      {cfg.beta}")
    print(f"    epsilon:   {cfg.epsilon_dist}")
    print(f"    c_single:  {cfg.c_single_out}")
    print(f"    active:    w < 1 - {cfg.active_tol}")
    print(f"    weight attr: {cfg.edge_weight_attr}")
    print(f"    support cap enabled: {cfg.enable_support_cap}")

    diags = []

    for i, f in enumerate(files, 1):
        print(f"\n[{i}/{len(files)}] {sample_id_from_graph(f)}", flush=True)

        try:
            d = process_graph(f, cfg)
        except Exception as e:
            d = {
                "sample_id": sample_id_from_graph(f),
                "status": "failed",
                "error": repr(e),
            }
            print(f"    FAILED {d['sample_id']}: {e}", flush=True)

        diags.append(d)

        pd.DataFrame(diags).to_csv(
            out_dir / "faithful_pairwise_active_ricci_diagnostics.partial.csv",
            index=False,
        )

    pd.DataFrame(diags).to_csv(
        out_dir / "faithful_pairwise_active_ricci_diagnostics.csv",
        index=False,
    )

    print("\n[✓] Done")
    print(f"    diagnostics: {out_dir / 'faithful_pairwise_active_ricci_diagnostics.csv'}")


if __name__ == "__main__":
    main()
