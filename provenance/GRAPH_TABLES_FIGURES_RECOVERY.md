# Recovery of Tables 1--5 and graph-distribution figures

## Scope

This record covers the generators for sampled-path Tables 1--5 and the active
outdegree and edge-weight distribution figures. It complements
`GRAPH_DIAGNOSTICS_CORRECTION.md`, which covers directed clustering and the
sequential routing attack.

## Audit finding

The recovered `path_backbone_longrange_diagnostics.py` parser is valid: it
retains edge `data` children until the parent edge is processed. Its completed
per-sample JSON results and aggregate tables were nonempty, and no samples were
listed as parser failures. The recovered edge-concentration generator used the
same valid parser but imported it from an untracked absolute home-directory
path. The promoted version imports the adjacent repository source.

Three historical/local figure sources used an unsafe streaming-parser pattern:
they handled an `edge` at its end event but cleared every other element at its
own end event. That clears each child `data` value before the parent edge is
processed and can turn all active-edge diagnostics into zeros. The archived
`Edge Concentration Diagnostics` file and the collected outdegree and
participant-average weight-density source snapshots therefore are not
canonical publication generators.

The promoted outdegree and weight-density scripts reuse the tested canonical
parser from `path_backbone_longrange_diagnostics.py`. Parser-focused unit tests
exercise nested GraphML edge data and reject the zero-output failure mode.

## Aggregation and empty-graph rules

- Tables 1--5: sample diagnostics are averaged within participant before
  condition summaries. Undefined path metrics from empty active graphs are
  excluded from metric moments, and tables display valid/total counts.
- Outdegree: per-sample histograms exclude zero-outdegree vertices. Empty
  active graphs contribute a zero histogram before participant averaging.
- Weight density: each nonempty sample curve integrates to one over
  `0 < w <= 0.99`; empty active graphs contribute a zero curve before
  participant averaging. Thus an aggregate curve can integrate to less than
  one, and the generated run configuration records those integrals.

These rules preserve the historical participant-balanced hierarchy while
making the 11 known empty active graphs explicit.

## Source audit identifiers

The inspected bundle was
`graph_tables_figures_audit_20260910T235633Z.tar.gz`. Its internal 65-file
manifest verified after normalizing the collection-machine absolute prefix.
Relevant recovered source SHA-256 values were:

- path-backbone generator: `7178c33bd17f1eea46624d6ac48bfca210f2967a6aec524df0abc711836943e9`;
- enhanced edge-concentration generator: `0f20fd0049ec9a3517ee4e5017e6339adde0a1d74a276b1352d383e3441a99ee`;
- all-path LaTeX assembler: `07528437a61ad1358414417ef018a8fb348a869db286c854d23ca5fd71ee69cb`.

## Result freeze

The cached path aggregation and both corrected figure reruns completed on
2026-09-11. Tables 1--5 reproduce every previously reviewed common numeric
metric exactly while adding valid/total counts: 1,306/1,317 sample graphs and
106/106 participants overall. The weight-density aggregate reproduces the
prior curve to floating-point precision (maximum absolute difference
`2.89e-15`).

The corrected outdegree generator produces nonzero distributions. Relative to
the earlier nonzero participant-level export, its largest condition/bin mean
change is 0.052686 vertices and its largest population-SD change is 0.143559
vertices; these differences are negligible for interpretation but the new
figure is the canonical manuscript copy. The archived parser-derived zero
summary remains noncanonical.

The exact comparison, input result-archive hash, aggregation policies, and
tracked-output hashes are recorded in
`provenance/graph_diagnostics/graph_tables_figures_result_audit_20260911.json`
and `graph_tables_figures_output_manifest_20260911.sha256`.

The result-freeze commit also normalises the combined Tables 1--4 writer to
emit exactly one final newline, avoiding a Git whitespace warning without
changing any LaTeX content or scientific value.
