# Graph-diagnostic result provenance

`directed_clustering_run_config_20260910.json` records the executed source
hash, input metadata hash, runtime versions, active-edge rule, aggregation
rule, and observed graph/participant counts for the corrected production
replay.

`directed_clustering_output_manifest_20260910.sha256` records every local
output from that replay. Only the non-identifying aggregate summary and LaTeX
table are tracked in Git. The sample-level, participant-level, empty-graph, and
failure files remain outside Git and belong in the controlled permanent
derived-results archive.

The four `*_run_config_20260911.json` files record the executed configurations
for the cached path aggregation, enhanced edge-concentration aggregation,
corrected outdegree replay, and corrected weight-density replay.

`graph_tables_figures_result_audit_20260911.json` records the numerical review
against the previously collected outputs. The accompanying SHA-256 manifest
covers every aggregate table, manuscript LaTeX file, figure, and run
configuration added by this result-freeze patch.

No sample identifiers, participant identifiers, per-sample feature values, or
individual predictions are included in the tracked result subset.
