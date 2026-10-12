# Repeated IBD-vs-non-IBD Ricci C-path analysis

This package runs the corrected `[B | K0]` Ricci classifier for **IBD versus non-IBD** using:

- 20 repetitions of 5 participant-grouped folds;
- the same saved splits for every value of `C`;
- `C = 0.005, 0.01, 0.02, 0.05, 0.1`;
- fold-internal `StandardScaler`;
- L1 logistic regression with `solver="saga"` and `class_weight="balanced"`;
- reaction-level coefficient and held-out contribution distributions;
- process-level fold distributions and participant-balanced held-out contributions;
- saved fold scalers and coefficients for later locked external-cohort evaluation;
- one full-source model per `C` for later external use.

The primary performance distribution is the set of **20 pooled out-of-fold repetition estimates**, not the 100 overlapping fold scores treated as independent observations.

## Files

- `repeated_ricci_ibd_cpath.py`: main analysis.
- `validate_inputs.py`: quick structural validation before the long run.
- `launch_tmux.sh`: launches the full run in tmux with resume enabled.
- `smoke_test.py`: creates synthetic data and tests the full pipeline and resume logic.

## Expected input directory

By default:

```text
~/Real_Data/Ricci_Classifier_Faithful_Eps0001_n250_v3
```

It must contain:

```text
feature_matrix_B_K0.npz
matched_metadata.csv
edge_metadata.csv
```

`matched_metadata.csv` must contain `sample_id`, `participant_id`, and `cond`.

`edge_metadata.csv` must contain `edge`. A `process` column is strongly preferred. An external annotation CSV may instead be supplied with `ANNOTATION_CSV=/path/to/file.csv` when launching.

## Install on the VM

Copy or unzip this directory into `~/Real_Data`, then run:

```bash
cd ~/Real_Data/ricci_repeated_cv_ibd_cpath
python3 validate_inputs.py \
  --feature-dir ~/Real_Data/Ricci_Classifier_Faithful_Eps0001_n250_v3
```

The validator should report 1,317 rows and the expected participant counts for the current faithful feature matrix. It will also state whether process annotations are available.

## Launch the full run in tmux

```bash
cd ~/Real_Data/ricci_repeated_cv_ibd_cpath
bash launch_tmux.sh
```

The defaults are:

```text
session:     ricci_ibd_cpath
feature dir: ~/Real_Data/Ricci_Classifier_Faithful_Eps0001_n250_v3
output dir:  ~/Real_Data/Ricci_IBD_RepeatedCV_CPath
log:         ~/Real_Data/Ricci_IBD_RepeatedCV_CPath/full_run.log
```

Attach:

```bash
tmux attach -t ricci_ibd_cpath
```

Detach without stopping the run:

```text
Ctrl-b, then d
```

Follow the log outside tmux:

```bash
tail -f ~/Real_Data/Ricci_IBD_RepeatedCV_CPath/full_run.log
```

## Optional launch overrides

Use a different Python executable or output location:

```bash
PYTHON_BIN=/path/to/python \
OUTPUT_DIR=~/Real_Data/Ricci_IBD_RepeatedCV_CPath_v2 \
bash launch_tmux.sh
```

Provide a separate annotation table:

```bash
ANNOTATION_CSV=/path/to/edge_process_annotations.csv bash launch_tmux.sh
```

Increase the iteration cap only if convergence warnings occur:

```bash
MAX_ITER=20000 bash launch_tmux.sh
```

## Resume after interruption

The output path is deliberately stable. Each completed fold is saved immediately. Relaunching with the same output directory resumes completed folds:

```bash
cd ~/Real_Data/ricci_repeated_cv_ibd_cpath
bash launch_tmux.sh
```

Do not delete the `splits/` directory if the same participant partitions must later be reused by the species-abundance benchmarks.

## Main outputs

At the output root:

```text
all_C_performance_compact.csv
all_C_performance_summary.csv
all_C_repetition_pooled_oof_results.csv
paired_C_performance_comparisons.csv
all_C_full_source_model_inventory.csv
splits/participant_split_manifest.csv
splits/sample_split_manifest.csv
```

Within each `C_*` directory:

```text
fold_results_100_models.csv
repetition_pooled_oof_results.csv
repetition_mean_fold_results.csv
cross_validated_sample_predictions.csv
cross_validated_participant_predictions.csv
performance_summary.csv
reaction_coefficient_distributions.csv
reaction_summary_compact.csv
reaction_coefficient_distributions_selected_at_least_once.csv
top_stable_reactions.csv
top_stable_B_reactions.csv
top_stable_K0_reactions.csv
process_fold_distributions.csv
process_summary_distributions.csv
process_summary_compact.csv
process_summary_ranked.csv
fold_artifacts/*.npz
full_source_model/full_source_model.npz
```

### Reaction tables

The reaction distributions include:

- selection frequency across 100 fits;
- positive, negative, and zero counts;
- sign consistency conditional on selection;
- coefficient mean, SD, median, IQR, and 2.5–97.5% range including zeros;
- the same coefficient distribution conditional on selection;
- held-out participant-balanced absolute and signed contribution distributions;
- class-specific held-out contributions;
- stability flags at 50%, 70%, and 90% selection frequencies;
- the coefficient from the model trained on the full source cohort.

### Process tables

Each process is summarised separately for `B`, `K0`, and `Combined`. Metrics are first calculated within each fitted fold, then distributed across the 100 fold models. Held-out process contribution is never calculated on the training observations used to estimate that model.

## Later external validation

No Spanish data are required for this run. The saved `fold_artifacts/*.npz` files contain every source-fold scaler mean, scaler scale, coefficient vector, and intercept. The `full_source_model/*.npz` files contain the corresponding full-cohort models. These allow the curated Spanish feature matrix to be evaluated later without refitting or Spanish-derived scaling.
