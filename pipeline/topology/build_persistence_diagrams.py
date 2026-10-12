#!/usr/bin/env python3
"""
build_persistence_diagrams.py

Builds H0 and "H1-like" persistence summaries from directed substrate graphs.

- Input graphs:  ~/Real_Data/out_graphs/*.graphml
  Each graph is a directed weighted graph with:
    - Nodes: substrates (metabolites)
    - Edges: directed conversions, attribute 'weight' > 0

- H0 "persistence diagram":
    * Treat edges as undirected for connectivity.
    * Interpret edge *weight* directly as a distance / cost:
        - lower weight = biologically stronger / earlier connection
    * Run a Kruskal-style union–find over edges sorted by weight.
    * Each time two components merge, the younger one "dies" at that distance.
    * Output: N x 2 array: (birth, death) with birth ≡ 0.

- H1-like "cycle cloud":
    * Before cycle detection, remove any edge with weight > MAX_CYCLE_EDGE_WEIGHT.
      (We only keep reasonably strong / low-cost edges for cycles.)
    * Use Johnson's algorithm (networkx.simple_cycles) to find directed simple cycles
      on the filtered graph G_filt.
    * For each cycle C:
        - birth  = max edge weight on C (in G_filt)
        - length = number of nodes in C
        - depth  = basin metric via random walks to sinks on the FULL graph G:
            • For each node in C:
                ◦ Attempt N_PATHS_PER_NODE random directed walks to sinks.
                ◦ Each walk:
                    ▪ picks edges with probability ∝ weight
                    ▪ accumulates sum of squared weights along the path
                    ▪ terminates at a sink (node with no outgoing edges)
                      or on cycle / max length
            • depth = mean(sum of squared weights) over all successful
              sink-reaching walks from all cycle nodes.
            • cycles with no successful sink walks are discarded.
        - similarity_index:
            • First, count how many cycles each node participates in (in G_filt
              and within the MAX_CYCLES_PER_GRAPH cap).
            • For cycle C, similarity_index = (# of nodes in C with
              cycle_count > 1) / len(C).

- Outputs:
    * ~/Real_Data/out_pds/<sample>_H0.npy  (float64, shape [k, 2])
    * ~/Real_Data/out_pds/<sample>_H1.npy  (float64, shape [m, 4])
      Columns: [birth, depth, length, similarity_index]
    * ~/Real_Data/out_pds/pd_summary.csv:
        sample_id,label,n_H0_points,n_H1_points

Labels are optional, loaded from ~/Real_Data/sample_labels.csv if present
(with columns: sample_id,label).
"""

import os
import glob
import random
import logging
from typing import Dict, List, Tuple, Optional, Iterable, Set

import numpy as np
import pandas as pd
import networkx as nx


# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

GRAPH_DIR = "/home/Jack/Real_Data/out_graphs"
PD_OUTPUT_DIR = "/home/Jack/Real_Data/out_pds"
LABEL_CSV = "/home/Jack/Real_Data/sample_labels.csv"  # optional

# H0 (component) filtration
MIN_EDGE_WEIGHT = 1e-9  # safeguard if any weight <= 0

# Cycle detection / H1-like summary
MIN_CYCLE_LENGTH = 3
MAX_CYCLE_LENGTH = 50          # your previous value
MAX_CYCLES_PER_GRAPH = 10000    # your chosen cap
MAX_CYCLE_EDGE_WEIGHT = 0.13    # edges with weight > 0.1 excluded from cycles

# Depth metric via random walks to sinks
N_PATHS_PER_NODE = 10
MAX_WALK_LENGTH = 50
RNG_SEED = 42

# Logging
LOG_LEVEL = logging.INFO


# ---------------------------------------------------------------------
# UTIL: logging setup
# ---------------------------------------------------------------------

def setup_logging():
    logging.basicConfig(
        level=LOG_LEVEL,
        format="[%(asctime)s] %(levelname)s - %(message)s",
    )


# ---------------------------------------------------------------------
# UTIL: union-find for H0
# ---------------------------------------------------------------------

class UnionFind:
    def __init__(self, elements: Iterable):
        self.parent = {}
        self.rank = {}
        self.birth = {}

        for e in elements:
            self.parent[e] = e
            self.rank[e] = 0
            # All components born at t=0 in this filtration
            self.birth[e] = 0.0

    def find(self, x):
        # Path compression
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y) -> Tuple[Optional[object], Optional[float]]:
        """
        Union components containing x and y.

        Returns:
            (dead_rep, death_time_placeholder)
        where dead_rep is the representative that "dies" in the merge,
        and death_time_placeholder is None here (actual death time
        to be filled by caller based on filtration parameter).
        If x and y already in same component, returns (None, None).
        """
        rx = self.find(x)
        ry = self.find(ry := y)
        if rx == ry:
            return None, None

        bx = self.birth[rx]
        by = self.birth[ry]

        # The older component survives, younger one dies
        if bx <= by:
            alive, dead = rx, ry
        else:
            alive, dead = ry, rx

        # Union by rank
        if self.rank[alive] < self.rank[dead]:
            alive, dead = dead, alive

        self.parent[dead] = alive
        if self.rank[alive] == self.rank[dead]:
            self.rank[alive] += 1

        return dead, None


# ---------------------------------------------------------------------
# H0 persistence from a weighted directed graph
# ---------------------------------------------------------------------

def compute_H0_persistence(G: nx.DiGraph) -> np.ndarray:
    """
    Compute H0-like persistence from a weighted directed graph.

    - Treat edges as undirected for connectivity.
    - Use edge *weight* directly as the distance/cost:
        lower weight = earlier in the filtration.

    Returns:
        numpy array of shape [k, 2]: (birth=0, death=weight) for each merge.
    """
    if G.number_of_nodes() == 0:
        return np.zeros((0, 2), dtype=float)

    # Union-Find over nodes
    uf = UnionFind(G.nodes())

    # Undirected edge list with distances = weights
    edges = []
    for u, v, data in G.edges(data=True):
        w = float(data.get("weight", 1.0))
        if w <= 0:
            w = MIN_EDGE_WEIGHT
        dist = w  # distance = weight
        edges.append((u, v, dist))

    # Sort edges by distance ascending (stronger / lower-weight edges first)
    edges.sort(key=lambda e: e[2])

    intervals: List[Tuple[float, float]] = []

    for u, v, dist in edges:
        dead_rep, _ = uf.union(u, v)
        if dead_rep is not None:
            birth_time = uf.birth[dead_rep]  # always 0.0 here
            death_time = dist
            intervals.append((birth_time, death_time))

    if not intervals:
        return np.zeros((0, 2), dtype=float)

    return np.asarray(intervals, dtype=float)


# ---------------------------------------------------------------------
# Random-walk-based depth metric
# ---------------------------------------------------------------------

def build_adjacency(G: nx.DiGraph) -> Dict[object, List[Tuple[object, float]]]:
    """
    Build adjacency list with (neighbor, weight) for each node.
    """
    adj: Dict[object, List[Tuple[object, float]]] = {}
    for u, v, data in G.edges(data=True):
        w = float(data.get("weight", 1.0))
        if w <= 0:
            w = MIN_EDGE_WEIGHT
        adj.setdefault(u, []).append((v, w))
    # Ensure all nodes appear in adjacency (even sinks)
    for n in G.nodes():
        adj.setdefault(n, [])
    return adj


def is_sink(node, adjacency: Dict[object, List[Tuple[object, float]]]) -> bool:
    return len(adjacency.get(node, [])) == 0


def random_sink_path_score(
    start,
    adjacency: Dict[object, List[Tuple[object, float]]],
    max_len: int,
    rng: random.Random,
) -> Optional[float]:
    """
    Random directed walk starting at 'start' until:
        - reaches a sink (node with no outgoing edges): success
        - reaches max_len steps: failure (unless end is sink)
        - revisits a node (cycle): failure

    Edge selection is weighted ∝ edge weight.

    Returns:
        sum of squared weights along the path for successful sink-reaching
        walks, or None if no sink reached safely.
    """
    visited: Set[object] = {start}
    current = start
    score = 0.0

    for _ in range(max_len):
        nbrs = adjacency.get(current, [])
        if not nbrs:
            # already at sink: success
            return score

        # Weighted random choice by weight
        total_w = sum(w for _, w in nbrs)
        r = rng.random() * total_w
        acc = 0.0
        chosen_nbr = None
        chosen_w = None
        for nbr, w in nbrs:
            acc += w
            if r <= acc:
                chosen_nbr = nbr
                chosen_w = w
                break

        if chosen_nbr is None:
            return None  # safety fallback

        if chosen_nbr in visited:
            # hit a cycle; discard this path
            return None

        visited.add(chosen_nbr)
        score += chosen_w ** 2
        current = chosen_nbr

        if is_sink(current, adjacency):
            # reached sink successfully
            return score

    # exceeded max_len; consider failure unless final node is a sink
    if is_sink(current, adjacency):
        return score
    return None


def compute_cycle_depth(
    cycle: List[object],
    adjacency: Dict[object, List[Tuple[object, float]]],
    rng: random.Random,
) -> Optional[float]:
    """
    Compute depth metric for a single directed simple cycle.

    For each node in the cycle, run N_PATHS_PER_NODE random walks to sinks
    using the adjacency of the FULL graph.
    Gather all successful path scores = sum(weights^2), and return their mean.

    If no successful walks from any cycle node, return None.
    """
    all_scores: List[float] = []

    for node in cycle:
        for _ in range(N_PATHS_PER_NODE):
            s = random_sink_path_score(
                start=node,
                adjacency=adjacency,
                max_len=MAX_WALK_LENGTH,
                rng=rng,
            )
            if s is not None:
                all_scores.append(s)

    if not all_scores:
        return None

    return float(np.mean(all_scores))


# ---------------------------------------------------------------------
# H1-like: cycles → (birth, depth, length, similarity_index)
# ---------------------------------------------------------------------

def compute_H1_cycle_cloud(G: nx.DiGraph) -> np.ndarray:
    """
    Compute "H1-like" points from directed simple cycles.

    Steps:
        1. Create a filtered graph G_filt by removing any edge with
           weight > MAX_CYCLE_EDGE_WEIGHT.
        2. Run Johnson's algorithm (networkx.simple_cycles) on G_filt.
        3. Collect up to MAX_CYCLES_PER_GRAPH cycles that satisfy:
           MIN_CYCLE_LENGTH <= len(cycle) <= MAX_CYCLE_LENGTH.
        4. Compute per-node cycle participation counts (in G_filt).
        5. For each cycle C:
             birth  = max edge weight on C (in G_filt)
             length = |C|
             depth  = random-walk-based basin metric using adjacency of FULL G
             similarity_index =
                 (# of nodes in C that appear in >= 2 cycles)
                 / len(C)

    Returns:
        numpy array of shape [m, 4] with columns
            [birth, depth, length, similarity_index].
    """
    if G.number_of_nodes() == 0:
        return np.zeros((0, 4), dtype=float)

    # 1) Filter edges by weight for cycle detection
    G_filt = nx.DiGraph()
    G_filt.add_nodes_from(G.nodes(data=True))

    for u, v, data in G.edges(data=True):
        w = float(data.get("weight", 1.0))
        if w <= 0:
            w = MIN_EDGE_WEIGHT
        if w <= MAX_CYCLE_EDGE_WEIGHT:
            G_filt.add_edge(u, v, **data)

    if G_filt.number_of_edges() == 0:
        return np.zeros((0, 4), dtype=float)

    # Adjacency for depth is built on the FULL graph G (your requirement)
    adjacency = build_adjacency(G)
    rng = random.Random(RNG_SEED)

    # 2–3) Enumerate cycles with length and count constraints
    all_cycles: List[List[object]] = []
    n_cycles_seen = 0

    for cycle in nx.simple_cycles(G_filt):
        if len(cycle) < MIN_CYCLE_LENGTH:
            continue
        if len(cycle) > MAX_CYCLE_LENGTH:
            continue

        all_cycles.append(cycle)
        n_cycles_seen += 1

        if n_cycles_seen >= MAX_CYCLES_PER_GRAPH:
            logging.warning(
                "Reached MAX_CYCLES_PER_GRAPH=%d; stopping cycle search.",
                MAX_CYCLES_PER_GRAPH,
            )
            break

    if not all_cycles:
        return np.zeros((0, 4), dtype=float)

    # 4) Per-node cycle participation counts in G_filt
    node_cycle_count: Dict[object, int] = {}
    for cycle in all_cycles:
        for node in cycle:
            node_cycle_count[node] = node_cycle_count.get(node, 0) + 1

    # 5) Compute features for each cycle
    cycle_points: List[Tuple[float, float, float, float]] = []

    for cycle in all_cycles:
        # birth = max edge weight on cycle (on G_filt)
        edge_weights = []
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i + 1) % len(cycle)]
            data = G_filt.get_edge_data(u, v, default={})
            w = float(data.get("weight", 1.0))
            if w <= 0:
                w = MIN_EDGE_WEIGHT
            edge_weights.append(w)

        birth = max(edge_weights) if edge_weights else 0.0
        length = float(len(cycle))

        depth = compute_cycle_depth(cycle, adjacency, rng)
        if depth is None:
            # no successful sink-reaching walks; skip this cycle
            continue

        # similarity_index: fraction of nodes in this cycle that
        # belong to at least one *other* cycle
        shared_nodes = sum(
            1 for node in cycle if node_cycle_count.get(node, 0) > 1
        )
        similarity_index = shared_nodes / float(len(cycle))

        cycle_points.append((birth, depth, length, similarity_index))

    if not cycle_points:
        return np.zeros((0, 4), dtype=float)

    return np.asarray(cycle_points, dtype=float)


# ---------------------------------------------------------------------
# Label loading (optional)
# ---------------------------------------------------------------------

def load_labels(label_csv_path: str) -> Dict[str, str]:
    """
    Load sample_id → label from a CSV if it exists.

    Expects at least columns: 'sample_id', 'label'.
    """
    if not os.path.exists(label_csv_path):
        logging.warning("Label CSV not found at %s; using 'unknown' labels.", label_csv_path)
        return {}

    try:
        df = pd.read_csv(label_csv_path)
    except Exception as e:
        logging.warning("Failed to read label CSV (%s): %s", label_csv_path, e)
        return {}

    if "sample_id" not in df.columns or "label" not in df.columns:
        logging.warning(
            "Label CSV at %s missing required columns 'sample_id'/'label'; ignoring.",
            label_csv_path,
        )
        return {}

    mapping = dict(zip(df["sample_id"].astype(str), df["label"].astype(str)))
    logging.info("Loaded %d sample labels from %s", len(mapping), label_csv_path)
    return mapping


# ---------------------------------------------------------------------
# Per-graph processing
# ---------------------------------------------------------------------

def process_graph_file(
    graph_path: str,
    label_map: Dict[str, str],
    output_dir: str,
) -> Tuple[str, str, int, int]:
    """
    Load a graphml file, compute H0 and H1-like PDs, and save them.

    Returns:
        (sample_id, label, n_H0_points, n_H1_points)
    """
    sample_id = os.path.splitext(os.path.basename(graph_path))[0]

    logging.info("Processing sample %s from %s", sample_id, graph_path)

    # Read graph; force DiGraph to preserve direction
    G_raw = nx.read_graphml(graph_path)
    G = nx.DiGraph(G_raw)

    # H0 on the full graph (no edge filtering)
    pd_H0 = compute_H0_persistence(G)

    # H1-like on cycles defined in the filtered graph but depth on full G
    pd_H1 = compute_H1_cycle_cloud(G)

    # Ensure output dir exists
    os.makedirs(output_dir, exist_ok=True)

    # Save arrays
    h0_path = os.path.join(output_dir, f"{sample_id}_H0.npy")
    h1_path = os.path.join(output_dir, f"{sample_id}_H1.npy")

    np.save(h0_path, pd_H0)
    np.save(h1_path, pd_H1)

    label = label_map.get(sample_id, "unknown")

    logging.info(
        "Sample %s: label=%s, H0 points=%d, H1 points=%d",
        sample_id, label, pd_H0.shape[0], pd_H1.shape[0],
    )

    return sample_id, label, pd_H0.shape[0], pd_H1.shape[0]


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():
    setup_logging()

    if not os.path.isdir(GRAPH_DIR):
        raise RuntimeError(f"Graph directory not found: {GRAPH_DIR}")

    graph_files = sorted(
        glob.glob(os.path.join(GRAPH_DIR, "*.graphml"))
    )

    if not graph_files:
        logging.error("No .graphml files found in %s", GRAPH_DIR)
        return

    logging.info("Found %d graphs in %s", len(graph_files), GRAPH_DIR)

    # Load labels (optional)
    label_map = load_labels(LABEL_CSV)

    os.makedirs(PD_OUTPUT_DIR, exist_ok=True)

    summary_rows = []

    for gp in graph_files:
        sample_id, label, n_h0, n_h1 = process_graph_file(
            graph_path=gp,
            label_map=label_map,
            output_dir=PD_OUTPUT_DIR,
        )
        summary_rows.append(
            {
                "sample_id": sample_id,
                "label": label,
                "n_H0_points": n_h0,
                "n_H1_points": n_h1,
            }
        )

    # Save summary CSV
    summary_path = os.path.join(PD_OUTPUT_DIR, "pd_summary.csv")
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    logging.info("Wrote summary to %s", summary_path)


if __name__ == "__main__":
    main()

