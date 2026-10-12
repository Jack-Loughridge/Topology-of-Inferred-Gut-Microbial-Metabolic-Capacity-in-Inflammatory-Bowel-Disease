# Graph-diagnostic source data

`sequential_condition_summary.csv` is the recovered aggregate source table for
the manuscript's four sequential routing-attack tables. It contains no sample
or participant identifiers. Its SHA-256 is
`5932016b519b60d1f9361b692bccc36bf94901742d4e03214221e717a478f57d`.

`directed_clustering_summary.csv` is the reviewed aggregate from the corrected
active-vertex production replay. It reports total and valid counts separately
and contains no sample or participant identifiers. Its SHA-256 is
`176b3f92fe719a6382b81d4c9b5bcbae751a4b3f8bfd3e47a720d80ca72e3bf1`.

The following additional public aggregate files support Tables 1--5 and the
two graph-distribution figure families:

- `path_backbone_condition_path_summary.csv`;
- `path_backbone_condition_edge_summary.csv`;
- `edge_concentration_weight_top001_condition_summary.csv`;
- `outdegree_hist_active_participant_mean_cap6_excl0_summary.csv`;
- `weight_density_curves_participant_average_clip099_curves.csv`.

They contain condition-level or curve-level aggregates only. Sample-level and
participant-level records remain in the controlled derived-results archive.
