# Current joint analysis: regularised v8

The canonical joint model for the final manuscript is **regularised v8**. Run it through `python analysis/joint/run.py` or the unchanged package under `regularised_v8/`. This replaces the earlier nested, five-task joint model as the current manuscript analysis.

The fixed experiment is IBD versus non-IBD, requested M=32, and the first five locked repetitions of five participant-grouped outer folds. There are 25 joint models and 25 matched Ricci controls. There is no inner resolution search. Alpha weights each death according to its source interval before original-death-axis Gaussian integration. The weighted-mean objective is

```
CE + ||beta_R||_1 / (0.02 * sum_w) + 0.001/2 * ||beta_H||_2^2
```

There is no H0 L1 penalty, alpha penalty, alpha smoothness penalty, or intercept penalty. The H0 ridge is a genuine model change from the earlier strictly unpenalised attempts. Local head and conditional-alpha checks use 1e-5. The old 100-refit stability protocol and 70/90-digit arithmetic are not part of v8. Every production fit in the supplied result export met the local targets. No finite-budget approximation contributed to that completed comparison.

## What is preserved

`regularised_v8/` contains all 29 files of the delivered source package, with all 28 payload files checked against its original `PACKAGE_MANIFEST.json`. The source is not rewritten to look more portable or to alter its objective after seeing performance. New wrappers, table generation and publication checks live outside that frozen package.

The older `core/` and `repeated_cv/` packages remain in place because they belong to the immutable 191-file production snapshot and reproduce historical analyses. Their internal README claims of being the final joint implementation are historical. They are **not** the entry points for the current joint results. See `CURRENT.json` and `docs/MANUSCRIPT_CODE_MAP.md`.

## Check without running any models

From the repository root:

```bash
python analysis/joint/verify_v8_release.py
python analysis/joint/regularised_v8/run_tests.py
python analysis/joint/make_v8_manuscript_tables.py \
  results/figure_source_data/joint/regularised_v8 \
  /tmp/joint_v8_tables
```

The first command uses only the standard library. The latter commands require the analysis dependencies. The generated LaTeX expects the manuscript's `\jointfinal{...}` bold-review macro.

## Reproduce the production analysis

This is the faithful production runner, not a claim that the raw-data reconstruction is self-contained. It intentionally requires the retained original VM input and split provenance. In the recorded environment:

```bash
python analysis/joint/run.py \
  --reference "$HOME/Real_Data/H0_Ricci_JointSparse_RepeatedCV_AllTasks" \
  --output "$HOME/Real_Data/H0_Ricci_IBD_Regularised_v8_REPRODUCTION"
```

Use a new output directory for a new reproduction. Never overwrite the original completed run. `--help` shows the actual optional arguments. The preflight validates the original matrix, diagrams, metadata, environment record, original-core parser identities, and the locked split manifests. It audits historical inner memberships but does not fit inner models.

A clean clone alone does not supply these observations, manifests, or original source-lock directory. They must be retained and made available under appropriate data access arrangements before a complete external reproduction can be claimed. The published aggregate archive does not include fitted states or individual predictions.

## Results and limits

The canonical aggregate CSVs are under `results/figure_source_data/joint/regularised_v8/`. They support modest participant AUC improvement in all five repetitions, mixed threshold improvements, and worse sample log loss in every repetition. Consensus results remain separate from mean repeated results. No formal test treats the five partitions as independent cohorts. There is no v8 subtype, external, unpenalised, finer-resolution or fixed-alpha control result.

The delivered source package is hash-verified. The review export records unchanged execution files but does not include the complete executed VM source manifest. That remaining provenance limitation is explicit in `provenance/joint_v8/REPRODUCIBILITY_BOUNDARIES.md`.
