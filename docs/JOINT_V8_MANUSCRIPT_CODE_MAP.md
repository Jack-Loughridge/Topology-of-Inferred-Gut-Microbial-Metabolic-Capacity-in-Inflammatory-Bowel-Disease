# Completed regularised joint comparison: manuscript to code

Review date: 11 October 2026. The current joint implementation is `analysis/joint/regularised_v8/`. The unchanged package is entered through `analysis/joint/run.py`. The former `core/` and `repeated_cv/` packages reproduce historical experiments only.

| Manuscript statement | Source in the frozen v8 package | Evidence or boundary |
|---|---|---|
| IBD only, fixed requested M=32, first five locked repetitions and 25 outer folds | `robust/config.py`, `robust/workflow.py::chosen_folds`, `run.py` | No inner models or resolution search. Historical partition audit is not extra fitting. |
| Positive finite deaths below 1-1e-12 and right-side source assignment | `audit/basis.py` | Multiplicity retained. Quantile boundaries merge at the 1e-8 minimum gap. Requested M is not a measured per-fold realised count. |
| Original-death-axis five-point quadrature and source alpha weighting | `audit/basis.py`, `robust/model.py::Problem` | Alpha is inside the sum over deaths, before integration. No output-bin alpha multiplication. |
| Training mean and population SD, H0 reference fixed at uniform alpha | `robust/model.py` and `robust/workflow.py` | Both H0 and Ricci are centred. Standard deviations at most 1e-12 become one. |
| Ricci C=.02 and H0 ridge .001 | `robust/config.py`, `robust/model.py` | Weighted-mean CE + L1(beta_R)/(.02 W) + .001 L2(beta_H)^2/2. No alpha, intercept or H0 L1 penalty. |
| Joint SLSQP, full Ricci gradient checks, bounded conditional rescues | `robust/model.py` | Full gradients include excluded working-set coordinates. Effective rescues per round are at most min(3, rescue_cycles), not the legacy field value 30. |
| Local convergence targets of 1e-5 | `robust/model.py` | Full head KKT and raw conditional alpha Frank-Wolfe bound at the same state. Not the earlier 100-refit stability test. |
| Declared finite-budget approximation option | `robust/model.py`, `robust/workflow.py` | All 50 observed fits converged. None used the approximation. No failed fold is omitted. |
| Saved prediction with extended accumulation | `robust/model.py`, `robust/workflow.py` | Float64 vs long-double score discrepancy is recorded. This is not 70/90-digit arithmetic and no invented discrepancy threshold is asserted. |
| First-argmax prediction and participant averaging | `robust/workflow.py`, vendored `metrics.py` | Class order nonIBD, IBD. Binary ties go to nonIBD. |
| Mean and sample SD across five pooled repetitions | `robust/workflow.py::aggregate` | Consensus averages probabilities before scoring and is not substituted for mean CV metrics. |
| Main participant table and five supplementary tables | `analysis/joint/make_v8_manuscript_tables.py` | Reads canonical aggregate CSVs. Uses the manuscript's `jointfinal` bold macro. |
| Completed output inventory and numerical checks | `results/figure_source_data/joint/regularised_v8/NUMERICAL_SUMMARY.csv` | 25 joint and 25 controls, all CONVERGED. Gradients are recorded production diagnostics, not independently reconstructed here. |

## Numerical result provenance

The original review archive hash is `060ff0950138dfa28a624595eb3532bce724248b5d9506f8a32a2cea8e90f4fe`. Its independently supplied sidecar and all 213 internal payload hashes were checked. The final published selection contains nine unchanged non-identifying aggregate/configuration files and separate checksums. Histories and individual observations are not needed to regenerate the manuscript tables.

The original delivered source ZIP hash is `34b85d1657f000954c54b1b2c84fbb66da258c3f57773ec3dbf9762ed0f570c9`. All 28 original payload files and their manifest are preserved. The complete executed VM manifest is absent from the curated result export. This is an outstanding provenance receipt, not evidence of a known source mismatch.

## Interpretation limits

The participant AUC gain is positive in all five repetitions, but threshold metrics improve in only three and sample log loss worsens in all five. Repetitions are not independent cohorts. No significance test, global optimum, fixed-alpha ablation, external joint transfer, subtype result or unpenalised joint benefit is inferred.
