# Manuscript-to-code map

## Purpose

This document maps each scientific claim, table, and figure to its canonical code, inputs, and generated outputs. Final manuscript numbering will be inserted after the revised results and figures are frozen.

| Manuscript role | Canonical code | Required input | Principal output | Runtime |
|---|---|---|---|---|
| AGORA2 reaction extraction and sanitisation | `pipeline/graph_construction/sanitize_agora_sbml.py`; `extract_agora_reactions.py` | AGORA2 SBML files | extracted reaction table | Heavy preprocessing |
| Canonical graph-input construction | `pipeline/graph_construction/canonicalize_graph_inputs.py` | audited species-abundance workbook and extracted AGORA2 reaction table | canonical species and reaction tables with provenance JSON | Moderate species / heavy streamed reactions |
| Sample metabolic graph construction | `pipeline/graph_construction/build_microbiome_graphs.py` | canonical reactions and species abundances | sample weighted directed graphs and baselines | Heavy CPU/storage |
| Tables 1--4, sampled-path quantiles | `analysis/graph_diagnostics/path_backbone_longrange_diagnostics.py`; `make_all_path_backbone_latex.py` | sample weighted directed graphs and metadata | `results/figure_source_data/graph_diagnostics/path_backbone_condition_path_summary.csv`; `path_backbone_condition_edge_summary.csv`; `results/manuscript_tables/graph_diagnostics/table_1_*` through `table_4_*`; combined `tables_1_to_4_path_backbone.tex` | Heavy path search; light cached re-aggregation |
| Table 5, path-edge concentration and weight | `analysis/graph_diagnostics/edge_concentration_weight_top001.py` | sample weighted directed graphs and metadata | `results/figure_source_data/graph_diagnostics/edge_concentration_weight_top001_condition_summary.csv`; `results/manuscript_tables/graph_diagnostics/table_5_edge_concentration_weight_top001.tex` | Heavy path search; light cached re-aggregation |
| Active-outdegree figure | `analysis/graph_diagnostics/outdegree_hist_active_participant_mean_cap6_excl0.py` | sample weighted directed graphs and metadata | `results/figure_source_data/graph_diagnostics/outdegree_hist_active_participant_mean_cap6_excl0_summary.csv`; matching PNG/PDF under `results/figures/graph_diagnostics/` | Moderate CPU |
| Active-edge weight-density figures | `analysis/graph_diagnostics/weight_density_curves_participant_average_clip099.py` | sample weighted directed graphs and metadata | `results/figure_source_data/graph_diagnostics/weight_density_curves_participant_average_clip099_curves.csv`; linear/log PNG/PDF under `results/figures/graph_diagnostics/` | Moderate CPU |
| Table 6, active-vertex directed clustering | `analysis/graph_diagnostics/directed_clustering_participant_average.py` | sample weighted directed graphs and metadata | `results/figure_source_data/graph_diagnostics/directed_clustering_summary.csv`; `results/manuscript_tables/graph_diagnostics/directed_clustering_table.tex` | Moderate CPU |
| Tables 53--56, sequential routing attack | `analysis/graph_diagnostics/sequential_routing_attack.py` | sample weighted directed graphs and metadata | `results/figure_source_data/graph_diagnostics/sequential_condition_summary.csv`; `results/manuscript_tables/graph_diagnostics/sequential_attack_tables.tex` | Heavy CPU |
| H0 persistence representation | `pipeline/topology/build_persistence_diagrams.py` | sample graphs | per-sample H0 diagrams | Heavy CPU |
| Faithful Ricci representation | `pipeline/topology/compute_ricci_faithful_pairwise_active.py` | active sample graphs | per-edge Ricci tables | Very heavy CPU |
| Standalone H0 IBD repeated analysis | `analysis/h0_alpha_pi/ibd_repeated_cv/h0_alpha_pi_repeated_cv.py` | H0 diagrams, metadata, locked IBD splits | repetition-level OOF metrics and interpretation curves | Heavy model fitting |
| Standalone H0 five-task table | `analysis/h0_alpha_pi/five_task_1x5/run_all_tasks.py` | H0 diagrams and shared task manifests | five-task H0 summary and OOF predictions | Heavy model fitting |
| IBD Ricci C sensitivity | `analysis/ricci/ibd_cpath/repeated_ricci_ibd_cpath.py` | faithful `[B | K0]` matrix and metadata | C-path performance and stability tables | Heavy CPU |
| Primary IBD Ricci feature-block ablation | `analysis/ricci/ibd_block_ablation/run_ibd_block_ablation.py` | faithful `[B | K0]` matrix, locked IBD split manifests, and completed combined `C=0.02` reference | `results/figure_source_data/ricci/ibd_feature_block_ablation/`; `results/manuscript_tables/ricci/ibd_feature_block_ablation.tex`; audit under `results/provenance/ricci/ibd_feature_block_ablation/` | Heavy CPU; light cached comparison |
| Five-task Ricci results | `analysis/ricci/five_task_repeated_cv/ricci_all_tasks.py` | faithful `[B | K0]` matrix and shared splits | repeated OOF performance and interpretation tables | Heavy CPU |
| Joint H0-Ricci results | `analysis/joint/repeated_cv/run_all_tasks.py`; `analysis/joint/core/` | H0 diagrams, Ricci matrix, metadata, shared splits | nested-CV performance and coefficient stability | Very heavy model fitting |
| Species-abundance comparators | `analysis/species/ibd_complete_case/species_benchmarks_repeated_cv.py`; `analysis/species/remaining_tasks/species_benchmarks_remaining_tasks.py` | canonical species table and shared splits | repeated comparator metrics and stability tables | Heavy CPU |
| Ricci manuscript interpretation | `analysis/interpretation/make_ricci_ibd_cpath_manuscript_outputs.py`; related interpretation scripts | completed model outputs and reaction annotations | manuscript-ready coefficient/process tables | Moderate |
| External profile acquisition | `external_validation/serrano_gomez/scripts/download_ge50_fastqs.sh`; profiling scripts | `PRJEB42155` manifest and FASTQs | verified taxonomic/functional profiles | Storage intensive |
| External v1 frozen structural validation | `external_validation/serrano_gomez/scripts/serrano_gomez_external_structural_frozen_ibdmdb_v1/external_structural_validation.py` | external profiles and frozen IBDMDB references | frozen-support graphs, H0, Ricci vectors | Very heavy CPU |
| Corrected strict-v1 external deployment | `external_validation/serrano_gomez/scripts/apply_frozen_species_benchmarks_external_ibd_last_occurrence_v1.py`; `apply_frozen_h0_alphapi_ensemble_external_ibd_v1_host_subject.py` | strict-v1 external features and frozen source models | `results/manuscript_tables/external_validation/species_benchmarks_strict_v1.tex`; strict rows in `h0_alphapi_strict_and_expanded.tex` | Moderate |
| External v2 expanded-community construction | `external_validation/serrano_gomez/scripts/build_expanded_external_v2.py` | completed external profiles and expanded-community prerequisites | v2 graphs, H0, full Ricci, and vectors | Very heavy CPU |
| Corrected expanded-v2 frozen deployment | `external_validation/serrano_gomez/scripts/apply_frozen_species_benchmarks_external_v2_last_occurrence_all184.py`; `apply_frozen_ricci_ibd_model_expanded_v2.py` | expanded-v2 features and frozen source models | `results/manuscript_tables/external_validation/species_benchmarks_expanded_v2.tex`; expanded rows in `h0_alphapi_strict_and_expanded.tex`; aggregate figure-source JSON | Moderate |

## Interpretation rule

The table above identifies canonical production code, not every historical exploration. Files under `archive/legacy_root/` must not be cited as the source of a manuscript result unless they are explicitly promoted, given a portable entry point, and mapped here before release.

## Fields to complete at result freeze

For every final manuscript item, add one row containing:

1. final figure or table number and caption shorthand;
2. exact input archive path and SHA-256;
3. exact command line or entry point;
4. locked configuration JSON;
5. exact output archive path and SHA-256;
6. Git commit and release tag;
7. expected runtime class;
8. whether the item can be regenerated in CI, from derived data, or only by the full pipeline.

No final table or figure should remain without an entry.
