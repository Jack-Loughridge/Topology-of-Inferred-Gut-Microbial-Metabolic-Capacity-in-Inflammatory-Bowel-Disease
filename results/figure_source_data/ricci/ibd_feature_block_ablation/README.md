# IBD Ricci feature-block ablation source data

These files support the prespecified comparison of the primary IBD-versus-
non-IBD Ricci classifier at `C=0.02` using:

- active-edge indicators only (`B only`);
- active-edge Ricci curvature only (`K0 only`);
- the original combined `[B|K0]` representation.

All three representations use the same 20 repetitions of the same five
participant-grouped folds, the same model seeds, fold-internal scaling,
class-weighted L1 logistic regression, and participant aggregation.

`ablation_performance_summary.csv` is the manuscript-facing summary.
`ablation_paired_deltas.csv` contains within-repetition differences between
representations. Its quantiles and proportions are descriptive measures of
split sensitivity, not p-values or confidence intervals from independent
cohorts. `ablation_repetition_metrics_long.csv` retains all 20 pooled
out-of-fold repetition estimates, including Brier score and log loss in
addition to the main reported metrics.

The representation-specific `*_repetition_pooled_oof_results.csv` files are
the compact inputs to the comparison. The `*_fold_results_100_models.csv`
files retain non-identifying fold-level diagnostics, including participant
overlap, selected-feature counts, iteration counts, and convergence status.
The `*_performance_summary.csv` files retain the complete production
summaries for each representation.

No sample identifiers, participant identifiers, individual predictions,
fitted models, or feature matrices are tracked here. The raw prediction-level
files were used for the audit recorded under
`results/provenance/ricci/ibd_feature_block_ablation/` and belong in the
permanent derived-results archive.
