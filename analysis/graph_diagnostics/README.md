# Graph diagnostics

This directory contains the canonical publication implementations for graph
diagnostics on the active directed metabolic graphs. All promoted generators
use the same safe GraphML parsing rule: child `data` elements remain intact
until their parent edge has been read, duplicate directed edges retain the
minimum weight, and active edges satisfy `w < 1 - active_tol`.

## Path-backbone Tables 1--5

`path_backbone_longrange_diagnostics.py` is the deterministic sampled-path
engine. It computes minimum-hop and minimax path summaries plus path-edge reuse
from 100 sources and up to 100 reachable targets per source using seed
`20260707`. Existing per-sample JSON results can be re-aggregated without
repeating the expensive path search:

```bash
python3 analysis/graph_diagnostics/path_backbone_longrange_diagnostics.py \
  --out-dir "$HOME/Real_Data/Path_Backbone_LongRange_Diagnostics" \
  --long-min-hop 4 --expected-samples 1317 --expected-participants 106 \
  --write-only

python3 analysis/graph_diagnostics/make_all_path_backbone_latex.py \
  --input-dir "$HOME/Real_Data/Path_Backbone_LongRange_Diagnostics"
```

`edge_concentration_weight_top001.py` generates the enhanced concentration and
median-weight table (Table 5) and now imports the adjacent canonical path
engine instead of an untracked file under `~/Real_Data`:

```bash
python3 analysis/graph_diagnostics/edge_concentration_weight_top001.py \
  --out-dir "$HOME/Real_Data/Path_Backbone_EdgeConcentration_WeightTop001" \
  --expected-samples 1317 --expected-participants 106 --write-only
```

The condition summaries and LaTeX tables report valid/total counts. Empty
active graphs retain a visible total count but do not enter moments for
undefined path or concentration metrics.

## Outdegree and edge-weight figures

The two canonical participant-balanced figure generators are
`outdegree_hist_active_participant_mean_cap6_excl0.py` and
`weight_density_curves_participant_average_clip099.py`. Both supersede local
snapshots whose streaming parsers cleared edge-data children prematurely.

Production commands:

```bash
python3 analysis/graph_diagnostics/outdegree_hist_active_participant_mean_cap6_excl0.py \
  --graph-dir "$HOME/Real_Data/out_graphs" \
  --metadata "$HOME/Real_Data/hmp2_metadata.csv" \
  --output-dir "$HOME/Real_Data/Outdegree_ParticipantAverage_Corrected_20260911" \
  --active-tol 1e-12 --sd-ddof 0 \
  --expected-samples 1317 --expected-participants 106

python3 analysis/graph_diagnostics/weight_density_curves_participant_average_clip099.py \
  --graph-dir "$HOME/Real_Data/out_graphs" \
  --metadata "$HOME/Real_Data/hmp2_metadata.csv" \
  --output-dir "$HOME/Real_Data/Weight_Density_ParticipantAverage_Corrected_20260911" \
  --active-tol 1e-12 --clip-max 0.99 --n-bin-edges 200 \
  --expected-samples 1317 --expected-participants 106
```

For both figures, an empty active graph contributes an all-zero sample curve or
histogram. This reproduces the historical participant hierarchy and is stated
in the generated run configuration rather than being an implicit exclusion.

## Directed clustering

`directed_clustering_participant_average.py` computes Fagiolo directed
clustering within each sample graph using only vertices incident to an active
edge. Active edges satisfy `w < 1 - 1e-12`; weighted clustering uses raw
support strength `s = 1 - w`, without global maximum normalization. Sample
values are then averaged within participant before the participant-level
condition summary is calculated. Population SD (`ddof=0`) is used.

The output records both total and valid observation counts. An active graph
with no retained edges has an undefined coefficient and is excluded from the
coefficient moments, while remaining visible in the total count and
`empty_active_graphs.csv`.

Production command:

```bash
python3 analysis/graph_diagnostics/directed_clustering_participant_average.py \
  --graph-dir "$HOME/Real_Data/out_graphs" \
  --metadata "$HOME/Real_Data/hmp2_metadata.csv" \
  --output-dir "$HOME/Real_Data/Directed_Clustering_ParticipantAverage_Corrected" \
  --active-tol 1e-12 \
  --sd-ddof 0 \
  --expected-samples 1317 \
  --expected-participants 106
```

## Sequential routing-attack removal

`sequential_routing_attack.py` is the recovered full generator for the
sequential-removal tables and plots. For each sample it deterministically
samples source and reachable target vertices, finds minimax paths, counts edge
use across the baseline paths, and then cumulatively removes the most-used
baseline edge that lies on each current path. Replacement paths are recomputed
after every removal. Results are calculated for each sample, averaged within
participant, and summarized by condition.

The publication run used the locked configuration at
`provenance/locked_run_configs/sequential_routing_attack_config.json`. The
implementation is self-contained: it no longer imports an untracked parser
from `~/Real_Data`.

Production command:

```bash
python3 analysis/graph_diagnostics/sequential_routing_attack.py \
  --graph-dir "$HOME/Real_Data/out_graphs" \
  --metadata "$HOME/Real_Data/hmp2_metadata.csv" \
  --output-dir "$HOME/Real_Data/Sequential_Routing_Attack_K8_Replay" \
  --n-source 10 --n-target 20 --k-max 8 --workers 4 \
  --active-tol 1e-12 --length-failure-value 25 \
  --seed 20260708 --sd-ddof 0 --no-resume
```

Eleven production sample graphs contained no active edge and were recorded as
failures by the sequential analysis. The remaining 1,306 sample graphs still
represented all 106 participants. This exclusion must be stated with the
sequential-removal results.

Run the focused tests with:

```bash
python3 -m unittest discover -s analysis/graph_diagnostics/tests -v
```

The same `unittest` suite is also discovered by the repository's pytest-based
CI environment.
