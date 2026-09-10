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
