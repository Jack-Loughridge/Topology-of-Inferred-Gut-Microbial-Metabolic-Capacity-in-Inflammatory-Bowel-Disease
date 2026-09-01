# Repeated species-abundance benchmarks for IBD classification

This repository evaluates three fixed species-abundance benchmark classifiers for **IBD (UC + CD) versus non-IBD** on the exact participant-grouped outer folds used by the repeated Ricci-curvature analysis.

## Models

1. **L1 logistic regression**
   - `penalty="l1"`
   - `solver="saga"`
   - `C=0.2`
   - `max_iter=10000`

2. **Random forest**
   - 1,000 trees
   - unrestricted depth
   - `min_samples_leaf=3`
   - `max_features="sqrt"`

3. **XGBoost**
   - 700 boosting rounds
   - maximum depth 3
   - learning rate 0.02
   - row subsampling 0.85
   - column subsampling 0.85
   - L2 regularization 2.0
   - L1 regularization 0.25

These hyperparameters are fixed manuscript baselines. They are not selected using outer-test observations.

## Core evaluation design

The program directly reads:

```text
~/Real_Data/Ricci_IBD_RepeatedCV_CPath/splits/sample_split_manifest.csv
```

It does **not** regenerate folds from a seed. The expected design is:

```text
20 repetitions × 5 participant-grouped folds
```

Every retained benchmark sample keeps the exact same repetition, fold, and train/test role as in the Ricci manifest. Ten samples with no usable canonical species-level profile are excluded explicitly; all 106 participants remain represented.

For each outer fold:

1. Load the exact training and test sample IDs from the shared manifest.
2. Apply a per-sample centered log-ratio transform using a pseudocount of `1e-6`.
3. Fit the variance filter on training observations only.
4. Fit the standard scaler on training observations only.
5. Construct fold-specific balanced sample weights from the training labels.
6. Fit the fixed benchmark model.
7. Evaluate the untouched outer-test observations.
8. Aggregate test probabilities by participant and evaluate again.

### XGBoost imbalance correction

The older binary XGBoost scripts applied both balanced sample weights and `scale_pos_weight=n_negative/n_positive`. That corrected the same class imbalance twice.

This implementation uses:

```text
balanced sample_weight
scale_pos_weight = 1.0
```

The correction is recorded in every XGBoost fold's `metrics.json`.


## Pre-specified complete-case rule

The canonical species table contains ten manifest samples whose 508 species-level features are all zero. Inspection of the broader taxonomic table showed that each is `UNKNOWN = 100%`, so these are not usable species-composition profiles. They are excluded before model fitting under the fixed rule:

```text
No nonzero species-level abundance after canonical filtering
```

The expected IBDMDB counts are:

```text
1,317 source-manifest samples
10 excluded all-zero canonical species profiles
1,307 complete-case benchmark samples
106 participants retained
```

The program asserts the expected count by default. It preserves the original fold assignments for every retained sample and writes the full cohort audit to:

```text
source_manifest_sample_master.csv
complete_case_sample_master.csv
sample_species_availability.csv
excluded_species_profiles.csv
complete_case_split_manifest.csv
complete_case_fold_counts.csv
participant_complete_case_counts.csv
```

Use `--expected-zero-profiles -1` only when deliberately applying the code to a different curated input and after inspecting the exclusions.

## Primary performance distribution

The 100 fold results per model are retained as descriptive diagnostics, but they are not treated as 100 independent estimates.

Within each repetition, the five held-out folds are pooled so every sample has exactly one out-of-fold prediction. The primary distribution therefore consists of:

```text
20 pooled out-of-fold estimates per model
```

This is calculated at both:

- sample level;
- participant level, after averaging a participant's held-out sample probabilities.

## Input safeguards

The code fails rather than silently altering the comparison when:

- the manifest is not exactly 20 × 5;
- a sample is not tested exactly once per repetition;
- participants overlap between train and test;
- a fold lacks a class;
- a manifest sample is missing from the species table;
- a manifest sample has conflicting labels or participant IDs;
- the label CSV disagrees with the manifest;
- the metadata CSV disagrees with the manifest;
- species features contain nonnumeric, infinite, or negative values;
- the observed all-zero-profile count differs from the pre-specified expectation;
- complete-case filtering would remove an entire participant;
- a retained sample is not tested exactly once per repetition after filtering.

Metadata conflicts outside the manifest sample set are ignored because they cannot enter this analysis.

## Files

```text
species_benchmarks_repeated_cv.py  Main repeated-CV analysis
validate_inputs.py                 Real-data preflight validation
self_test.py                       Synthetic end-to-end regression test
launch_tmux.sh                     Resource-limited TMUX launcher
requirements.txt                   Python dependencies
LICENSE                            MIT license
.gitignore                         Data/output exclusions
```

## Installation

Use the existing analysis environment when it already contains the dependencies, or create a new environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Self-test

Run before touching the real data:

```bash
OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 \
python3 self_test.py
```

The expected final line is:

```text
SELF-TEST PASSED: exact shared folds, explicit complete-case zero-profile exclusion, manifest-scoped validation, leakage-safe CLR pipeline, single XGBoost class weighting, convergence diagnostics, sample/participant OOF metrics, feature diagnostics, full-source models, plots, and resume behavior all succeeded.
```

The self-test verifies:

- exact use of a saved grouped-fold manifest;
- explicit exclusion of synthetic all-zero canonical profiles while retaining all participants;
- rejection when the expected all-zero-profile count is wrong;
- rejection of participant leakage;
- ignoring an unrelated metadata conflict;
- rejection of a conflict affecting a required sample;
- fold-internal preprocessing;
- all three model fits;
- `scale_pos_weight=1.0` for XGBoost;
- sample- and participant-level pooled OOF outputs;
- feature stability outputs;
- saved full-source model reload and prediction;
- plot creation;
- interruption/resume behavior.

## Real-data preflight

```bash
python3 validate_inputs.py \
  --species-file ~/Real_Data/Real_Species_Abundances_canon.xlsx \
  --species-sheet Sheet1 \
  --label-csv ~/Real_Data/sample_labels.csv \
  --metadata-csv ~/Real_Data/hmp2_metadata.csv \
  --split-manifest ~/Real_Data/Ricci_IBD_RepeatedCV_CPath/splits/sample_split_manifest.csv
```

The desired ending is:

```text
REPEATED SPECIES BENCHMARK PREFLIGHT: PASSED
```

## TMUX launch

The default launcher uses two CPU threads per model and a `nice` level of 5 so it can run alongside Ricci and GPU-based H0 jobs without trying to occupy the whole VM.

```bash
bash launch_tmux.sh
```

Attach:

```bash
tmux attach -t species_benchmarks_20x5
```

Detach without stopping:

```text
Ctrl-b, then d
```

Follow the log:

```bash
tail -f ~/Real_Data/Species_Benchmarks_RepeatedCV_IBD_CompleteCase/full_run.log
```

Check progress:

```bash
cat ~/Real_Data/Species_Benchmarks_RepeatedCV_IBD_CompleteCase/partial_progress/progress.json
```

As soon as one model completes all 100 folds, provisional performance and feature-stability tables are available under:

```text
partial_progress/completed_repetition_metrics.csv
partial_progress/completed_repetition_summary.csv
partial_progress/completed_feature_stability_<model>.csv
```

The next benchmark can continue running while those completed-model outputs are inspected.

Count completed folds:

```bash
find ~/Real_Data/Species_Benchmarks_RepeatedCV_IBD_CompleteCase/folds \
  -name FOLD_COMPLETE.json | wc -l
```

The final count is 300:

```text
3 models × 20 repetitions × 5 folds
```

### Resource overrides

Examples:

```bash
BENCHMARK_N_JOBS=1 bash launch_tmux.sh
```

```bash
BENCHMARK_N_JOBS=4 NICE_LEVEL=0 bash launch_tmux.sh
```

```bash
MODELS=logistic_l1,random_forest bash launch_tmux.sh
```

Avoid using unrestricted `n_jobs=-1` while other analyses are running.

## Resume behavior

Every fold has an atomic completion marker. Re-running the same command:

- reloads completed fold predictions and diagnostics;
- skips completed model fitting;
- resumes from the first incomplete fold;
- regenerates top-level summaries from the complete artifacts.

The run configuration stores hashes of all inputs and the method parameters. The program refuses to mix incompatible inputs or settings in one output directory.

## Main outputs

### Performance

```text
repetition_pooled_oof_metrics.csv
repetition_performance_summary.csv
all_outer_fold_metrics.csv
paired_model_differences_by_repetition.csv
paired_model_difference_summary.csv
consensus_prediction_metrics.csv
```

`repetition_performance_summary.csv` is the main manuscript-level summary. It reports mean, SD, median, IQR, and 2.5–97.5% empirical quantiles across the 20 pooled OOF repetitions.

Paired differences are calculated within the same repetition. For accuracy-like metrics, positive values mean model A performed better. For Brier score and log loss, the sign is reversed so positive still means model A performed better.

No p-values treating the 20 overlapping repetitions as independent cohorts are generated.

### Predictions and calibration

```text
all_outer_test_sample_predictions.csv.gz
all_outer_test_participant_predictions.csv.gz
consensus_predictions_by_sample.csv
consensus_predictions_by_participant.csv
consensus_calibration_tables.csv
```

### Model diagnostics

```text
model_fit_diagnostics.csv
partial_progress/completed_fold_metrics.csv
folds/<model>/repeat_XX/fold_Y/metrics.json
```

Diagnostics include:

- train and held-out metrics;
- train-test performance gaps;
- convergence warnings;
- logistic iteration count and nonzero coefficients;
- random-forest tree depths and leaf counts;
- XGBoost boosting rounds and features with positive gain;
- selected feature count after the training-only variance filter;
- class-weight ranges;
- fit and total runtimes.

### Repeated feature diagnostics

```text
feature_stability_logistic_l1.csv
feature_stability_random_forest.csv
feature_stability_xgboost.csv
top_100_features_<model>.csv
```

For every taxon, these contain:

- variance-filter selection frequency;
- model-use frequency;
- coefficient or importance distribution including zeros;
- distribution conditional on use;
- sign consistency for logistic regression.

Tree importances should be interpreted as predictive diagnostics rather than causal effects.

### Full-source models

After repeated CV, one model per benchmark is fitted on the complete-case source cohort and saved under:

```text
full_source_models/<model>/full_source_model.joblib
```

Each artifact contains:

- the fitted estimator;
- the exact training feature order;
- the CLR pseudocount;
- the training variance-filter mask;
- the training scaler mean and scale;
- task and class metadata.

These are intended for later locked external application. External data must be aligned to the saved training feature order and transformed using the saved preprocessing payload. External observations must not be used to refit the scaler or variance filter.

The apparent training metrics stored with these models are not generalization estimates.

## Direct command

The TMUX launcher is recommended, but the analysis can be run directly:

```bash
python3 species_benchmarks_repeated_cv.py \
  --species-file ~/Real_Data/Real_Species_Abundances_canon.xlsx \
  --species-sheet Sheet1 \
  --label-csv ~/Real_Data/sample_labels.csv \
  --metadata-csv ~/Real_Data/hmp2_metadata.csv \
  --split-manifest ~/Real_Data/Ricci_IBD_RepeatedCV_CPath/splits/sample_split_manifest.csv \
  --output-dir ~/Real_Data/Species_Benchmarks_RepeatedCV_IBD_CompleteCase \
  --models logistic_l1,random_forest,xgboost \
  --expected-repeats 20 \
  --expected-folds 5 \
  --zero-profile-policy exclude \
  --expected-zero-profiles 10 \
  --n-jobs 2
```

## Optional participant-balanced training sensitivity

The default preserves the manuscript benchmark training rule:

```text
--weighting-mode class_balanced
```

An optional sensitivity gives every participant equal total training weight within the class:

```text
--weighting-mode class_participant_balanced
```

Run that sensitivity in a different output directory. Do not mix it with the primary results.

## Data and GitHub

Do not commit raw IBDMDB-derived data or large output folders unless redistribution is explicitly permitted.

The included `.gitignore` excludes common local inputs, generated outputs, model files, caches, and logs. A public repository should contain the code and documentation, with instructions for authorized users to reconstruct the inputs.


## Self-test convergence note

The self-test uses a deliberately small synthetic dataset. SAGA convergence on such a fixture can vary slightly across scikit-learn and BLAS versions. This package therefore verifies that any logistic-regression `ConvergenceWarning` is captured and saved correctly rather than treating the warning itself as a package failure. The real analysis retains the manuscript setting `max_iter=10000` and records both `n_iter` and any convergence warning for every fold. Random-forest and XGBoost convergence-warning checks remain strict.
