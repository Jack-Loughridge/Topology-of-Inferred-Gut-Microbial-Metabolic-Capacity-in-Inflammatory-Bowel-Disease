# Repeated species-abundance benchmarks: remaining four IBD tasks

This repository runs the fixed species-abundance benchmark models for the four manuscript tasks that follow the completed **IBD vs non-IBD** run:

1. non-IBD vs UC vs CD;
2. non-IBD vs UC;
3. non-IBD vs CD;
4. UC vs CD (`CD_vs_UC` in the shared folder naming convention).

Each task uses **20 repetitions × 5 participant-grouped outer folds**. The repository does not generate or approximate split assignments. It reads the exact task-specific manifests created and audited by the joint H0+Ricci all-task preflight, so the species benchmarks and the topological classifier use the same held-out participants and samples before the documented species complete-case filtering.

## Fixed benchmark models

### L1 logistic regression

- `solver="saga"`
- `penalty="l1"`
- `C=0.2`
- `max_iter=10000`

For the three-class task, scikit-learn's multinomial SAGA implementation is used.

### Random forest

- 1,000 trees;
- unrestricted depth;
- `min_samples_leaf=3`;
- `max_features="sqrt"`.

### XGBoost

- 700 boosting rounds;
- depth 3;
- learning rate 0.02;
- row subsampling 0.85;
- column subsampling 0.85;
- L2 regularization 2.0;
- L1 regularization 0.25.

Balanced training-sample weights are used for every model. Binary XGBoost uses `scale_pos_weight=1.0`, avoiding the previous double correction for class imbalance. Multiclass XGBoost uses `multi:softprob` with balanced sample weights.

## Shared split manifests

The default split directory is:

```text
~/Real_Data/H0_Ricci_JointSparse_RepeatedCV_AllTasks/splits
```

It must contain:

```text
IBD_vs_nonIBD_split_manifest.csv
three_way_nonIBD_UC_CD_split_manifest.csv
nonIBD_vs_UC_split_manifest.csv
nonIBD_vs_CD_split_manifest.csv
CD_vs_UC_split_manifest.csv
```

These are written by:

```bash
cd ~/Real_Data/h0_ricci_joint_repeated_cv_all_tasks_v2
CORE_PY="$HOME/Real_Data/h0_ricci_joint_sparse/.venv/bin/python"
"$CORE_PY" validate_inputs.py
```

That preflight must pass before this benchmark package is launched.

## Complete-case species rule

The canonical species table contains ten source samples with an all-zero species-level vector. The broader taxonomic file showed `UNKNOWN = 100%` for each, so they cannot be treated as valid species profiles.

This program:

- computes the global zero-profile set from the union of all five shared task manifests;
- requires exactly ten such samples by default;
- excludes only the relevant intersection for each task;
- preserves every remaining sample's original task, repetition, fold and train/test role;
- rejects participant leakage, class loss or incomplete OOF coverage after filtering;
- records each task's exclusions explicitly.

## Fold-internal preprocessing

For every outer fold:

1. a centered log-ratio transform is applied per sample with pseudocount `1e-6`;
2. the variance filter is fitted on the outer-training samples only;
3. the standard scaler is fitted on the outer-training samples only;
4. balanced sample weights are calculated from the outer-training labels only;
5. the untouched outer-test samples are evaluated;
6. test probabilities are averaged within each held-out participant for participant-level evaluation.

The primary uncertainty distribution is the set of **20 pooled OOF repetition metrics**, not the 100 individual fold scores.

## Main files

```text
species_benchmarks_remaining_tasks.py  Main four-task repeated-CV runner
show_ibd_results.py                    Completion check and readable IBD result table
validate_inputs.py                     Real-data preflight wrapper
self_test.py                           Synthetic binary/multiclass end-to-end test
launch_tmux.sh                         Unattended resource-limited launcher
status.sh                              Progress checker
requirements.txt                       Python dependencies
CITATION.cff                           Citation metadata
LICENSE                                MIT license
.gitignore                             Data/output exclusions
```

## Reveal the completed IBD-vs-non-IBD results

The earlier IBD output is expected at:

```text
~/Real_Data/Species_Benchmarks_RepeatedCV_IBD_CompleteCase
```

Run:

```bash
cd ~/Real_Data/species_benchmarks_repeated_cv_remaining_tasks
python3 show_ibd_results.py
```

The script refuses to report final results unless:

- `RUN_COMPLETE.json` exists;
- 300 completed model-fold fits are recorded;
- 300 fold markers are present;
- all three models have 20 pooled repetition metrics at sample and participant level.

It prints the main accuracy, balanced accuracy, macro-F1, ROC-AUC and Brier summaries and writes:

```text
~/Real_Data/Species_Benchmarks_RepeatedCV_IBD_CompleteCase/ibd_primary_results_readout.csv
```

## Installation

Use the same environment as the completed benchmark when possible, or create a clean environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Self-test

```bash
OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 \
python3 self_test.py
```

Expected ending:

```text
SELF-TEST PASSED: exact shared task manifests, fixed four-task ordering, explicit global and task-level zero-profile exclusion, participant-leakage safeguards, train-only CLR/variance/scaling, binary and multiclass probability alignment and OOF metrics, single XGBoost class weighting, feature stability, full-source model persistence, final aggregation, and no-refit resume behavior all succeeded.
```

## Real-data preflight

```bash
python3 validate_inputs.py
```

Expected ending:

```text
REPEATED SPECIES BENCHMARK REMAINING-TASK PREFLIGHT: PASSED
```

The preflight prints the source and complete-case sample counts for each task. Do not launch if it fails.

## Launch all four tasks unattended

With other jobs running, use one thread per model:

```bash
N_JOBS=1 bash launch_tmux.sh
```

The fixed order is:

```text
non-IBD vs UC vs CD
non-IBD vs UC
non-IBD vs CD
UC vs CD
```

Attach:

```bash
tmux attach -t species_benchmarks_remaining_20x5
```

Detach without stopping the run:

```text
Ctrl-b, then d
```

Follow the log:

```bash
tail -f ~/Real_Data/logs/species_benchmarks_remaining_20x5.log
```

Check progress:

```bash
cd ~/Real_Data/species_benchmarks_repeated_cv_remaining_tasks
bash status.sh
```

The final outer-fit count is:

```text
4 tasks × 3 models × 20 repetitions × 5 folds = 1,200
```

The run is resumable at the completed fold level. Runtime thread count and plot/model-export choices are not part of the scientific fingerprint, so resource settings may be changed when resuming. Code, data, manifests, hyperparameters and random seeds are fingerprinted and cannot be mixed silently.

## Outputs

Default output root:

```text
~/Real_Data/Species_Benchmarks_RepeatedCV_RemainingTasks
```

Each task directory contains:

```text
RUN_COMPLETE.json
all_outer_fold_metrics.csv
all_outer_test_sample_predictions.csv.gz
all_outer_test_participant_predictions.csv.gz
repetition_pooled_oof_metrics.csv
repetition_performance_summary.csv
paired_model_differences_by_repetition.csv
paired_model_difference_summary.csv
consensus_predictions_by_sample.csv
consensus_predictions_by_participant.csv
consensus_prediction_metrics.csv
feature_stability_all_models.csv
feature_stability_<model>.csv
top_100_features_<model>.csv
complete_case_sample_master.csv
excluded_species_profiles.csv
complete_case_split_manifest.csv
complete_case_fold_counts.csv
plots/
full_source_models/
```

Combined four-task summaries are written under:

```text
~/Real_Data/Species_Benchmarks_RepeatedCV_RemainingTasks/aggregate
```

The main combined table is:

```text
repetition_performance_summary_all_tasks.csv
```

## Feature interpretation

For binary L1 logistic models, the saved coefficient refers to the second class in the locked class order:

- non-IBD vs UC: UC;
- non-IBD vs CD: CD;
- UC vs CD: UC, because the locked order is `(CD, UC)`.

For the three-class task, one coefficient vector is saved for each of `nonIBD`, `UC` and `CD`. Random-forest and XGBoost importance values are global rather than class-specific.

## GitHub

The full repository directory on the VM is:

```text
/home/Jack/Real_Data/species_benchmarks_repeated_cv_remaining_tasks
```

Upload the complete directory. Do not upload microbiome data, shared split outputs, fitted models or CV results unless the data-use terms explicitly permit redistribution.
