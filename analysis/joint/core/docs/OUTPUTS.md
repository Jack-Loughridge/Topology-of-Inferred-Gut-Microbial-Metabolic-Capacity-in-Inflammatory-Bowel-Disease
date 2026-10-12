# Output definitions

## Preflight files

- `INPUT_AUDIT.json`: resolved paths, sample/participant counts, diagnosis checks and matrix dimensions.
- `INPUT_SAMPLE_AUDIT.csv`: aligned faithful Ricci samples and their raw H0 diagrams.
- `REFERENCE_OUTPUT_AUDIT.csv`: presence of the canonical five-task H0 and original-style Ricci reference outputs.

## Global files

- `ALL_FOLD_RESULTS.csv`: one held-out outer-fold row per task and fold.
- `ALL_SUMMARIES.csv`: mean and sample SD across outer folds.
- `ALL_TEST_PREDICTIONS.csv`: every sample's prediction from the single outer fold in which its participant was held out.
- `ALL_ALPHA_PROFILES.csv`: raw fold-specific alpha values and interval locations.
- `ALL_INNER_CONFIG_RESULTS.csv`: every inner-fold result for every `(M, lambda_H, lambda_R)` configuration.
- `ALL_SELECTED_CONFIGS.csv`: the configuration selected in each outer fold.
- `ALL_SPARSITY_DIAGNOSTICS.csv`: nonzero coefficient counts and alpha diagnostics.
- `RUN_CONFIG.json`: paths, matrix dimensions, labels, objective and complete run configuration.

## Confusion matrices

Each task has an aggregate confusion matrix over all held-out outer-fold predictions. CSV is the numerical source; PNG/PDF are manuscript figures; TEX is a LaTeX table.

## Alpha profiles

Because interval boundaries and selected interval counts are fold-specific, raw alpha profiles are retained. They are also interpolated to a task-specific common log-death grid before computing a mean and SD plot. Interpolation is used only for display, never for fitting or prediction.

## Coefficients

- `top_h0_coefficients.csv` reports the strongest effective H0 coefficients by class, together with interval boundaries, alpha and `alpha * beta`.
- `top_ricci_coefficients.csv` reports the strongest effective faithful Ricci coefficients by class and merges available reaction/process annotations by feature index.

For binary logistic regression, class-specific coefficient rows use the symmetric two-score representation. Their difference reproduces the fitted binary logit exactly.
