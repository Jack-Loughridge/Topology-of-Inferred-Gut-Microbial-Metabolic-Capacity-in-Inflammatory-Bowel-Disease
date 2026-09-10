# Graph diagnostics

This directory contains the canonical publication implementations for two
diagnostics on the active directed metabolic graphs.

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
