# Graph-diagnostic manuscript tables

`sequential_attack_tables.tex` contains the four recovered routing-attack
tables and matches the corresponding manuscript tables. Its SHA-256 is
`1e43b235a36c06677791cffe7d766fd6c294f69d08c68302a0a1a00f27281966`.

`directed_clustering_table.tex` is the reviewed Table 6 replacement generated
by the corrected production replay. Its coefficients match the existing table
at three decimal places, while its valid/total column and caption disclose the
11 empty active graphs. Its SHA-256 is
`14ebd0cb33e9881f3486cccf4c9fe0cedb605606e622b9b5e2db767af4e6e5e3`.

`table_1_all_sampled_path_length_quantiles.tex` through
`table_4_all_sampled_mean_w2_quantiles.tex` are the canonical regenerated
Tables 1--4. `tables_1_to_4_path_backbone.tex` concatenates those same four
tables for convenient manuscript replacement.

`table_5_edge_concentration_weight_top001.tex` is the canonical regenerated
Table 5. All five tables report valid/total units. The stale historical file
named `all_sampled_path_backbone_tables_for_overleaf.tex` is intentionally not
tracked because it predates valid/total reporting and also contains an
auxiliary, non-numbered concentration table.
