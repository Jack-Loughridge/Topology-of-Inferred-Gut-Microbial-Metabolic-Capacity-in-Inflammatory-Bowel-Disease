# Repeated Ricci classifiers: all five tasks

GitHub-ready repeated participant-grouped classification code for faithful Ricci `[B | K0]` features across the five locked IBDMDB tasks:

1. IBD vs non-IBD
2. non-IBD vs UC vs CD
3. non-IBD vs UC
4. non-IBD vs CD
5. UC vs CD

The default manuscript run uses **20 repetitions × 5 folds** and `C=0.02`. The code can also run a C-path sensitivity analysis.

## Model

For every outer fold:

1. select the locked train/test samples from the shared manifest;
2. fit `StandardScaler` on the outer-training `[B | K0]` matrix only;
3. train L1 logistic regression with SAGA and balanced class weights;
4. evaluate untouched held-out participants;
5. save the fold scaler, coefficients, intercept, predictions and held-out contribution summaries.

The three-way task uses multinomial probabilities; binary tasks use the positive-class logit coefficient.

## Expected inputs

By default:

```text
~/Real_Data/Ricci_Classifier_Faithful_Eps0001_n250_v3/
    feature_matrix_B_K0.npz
    matched_metadata.csv
    edge_metadata.csv

~/Real_Data/H0_Ricci_JointSparse_RepeatedCV_AllTasks/splits/
    IBD_vs_nonIBD_split_manifest.csv
    three_way_nonIBD_UC_CD_split_manifest.csv
    nonIBD_vs_UC_split_manifest.csv
    nonIBD_vs_CD_split_manifest.csv
    CD_vs_UC_split_manifest.csv
```

The package consumes these manifests exactly and never regenerates outer folds.

## Validate

```bash
python3 validate_inputs.py
```

Required ending:

```text
REPEATED RICCI ALL-TASK PREFLIGHT: PASSED
```

## Test

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
python3 self_test.py
```

## Run the manuscript model

```bash
N_JOBS=1 NICE_LEVEL=10 C_VALUES=0.02 bash launch_tmux.sh
```

Monitor:

```bash
bash status.sh
tail -f ~/Real_Data/logs/ricci_all_tasks_20x5.log
```

The default output is:

```text
~/Real_Data/Ricci_RepeatedCV_AllTasks
```

## Optional C path

Use comma-separated values:

```bash
C_VALUES="0.005,0.01,0.02,0.05,0.1" bash launch_tmux.sh
```

Use a new output directory when changing the C grid because the run lock treats the scientific configuration as immutable.

## Principal outputs

For each task and C:

```text
all_outer_fold_metrics.csv
all_oof_sample_predictions.csv
all_oof_participant_predictions.csv
repetition_pooled_oof_metrics.csv
repetition_performance_summary.csv
feature_coefficient_stability.csv.gz
top_stable_features.csv
process_fold_distributions.csv
process_summary.csv
consensus_sample_predictions.csv
consensus_participant_predictions.csv
consensus_confusion_sample.csv/.png
consensus_confusion_participant.csv/.png
folds/.../model_artifact.npz
full_source_model/full_source_model.npz
```

The all-task summary is:

```text
aggregate/repetition_performance_summary_all_tasks.csv
```

Use:

```bash
python3 show_results.py
```

## Statistical interpretation

The primary uncertainty distribution is the set of 20 pooled OOF repetition estimates. The 100 fold scores per task are retained descriptively and are not treated as 100 independent experiments.

Reaction-level stability includes coefficient distributions with zeros, selection frequencies, conditional coefficient distributions, sign consistency and held-out participant-balanced realised contributions. Process summaries are calculated fold-by-fold before being aggregated.

Input feature matrices, split outputs, trained artifacts and result directories are not included in the repository.
