# Standalone H0 Alpha-Pi: one locked 5-fold repetition across five IBD tasks

This repository completes the standalone H0 Alpha-Pi analysis using **one participant-grouped five-fold repetition for each of five classification tasks**. It replaces the abandoned plan to run 20 repeated partitions for H0, while retaining the corrected train-only adaptive-bin method and the same locked outer folds used by the repeated Ricci, species-benchmark, and joint H0+Ricci analyses.

## Final task order

1. `IBD_vs_nonIBD`
2. `three_way_nonIBD_UC_CD`
3. `nonIBD_vs_UC`
4. `nonIBD_vs_CD`
5. `CD_vs_UC` (reported as UC vs CD where appropriate)

The requested design is therefore **25 outer models**:

```text
5 tasks × 1 repetition × 5 outer folds = 25 outer folds
```

## How the interrupted IBD run is continued

The corrected IBD-vs-non-IBD package already completed fold 1 and had partial candidate checkpoints for fold 2. This repository deliberately reuses that work rather than restarting it.

`run_all_tasks.py`:

1. launches the existing corrected script at:
   `~/Real_Data/h0_alpha_pi_repeated_cv/h0_alpha_pi_repeated_cv.py`;
2. resumes its atomic candidate and fold artifacts under:
   `~/Real_Data/H0_AlphaPi_RepeatedCV_TrainOnlyBins_IBD`;
3. watches the five `repeat_01_fold_* / FOLD_COMPLETE.json` markers;
4. stops the old repeated-CV process immediately after repetition 1 reaches 5/5 folds;
5. verifies every held-out sample against the exact shared repetition-1 IBD manifest;
6. imports the five final folds into the all-task summary without using any incomplete later repetition.

The source fold models, candidate checkpoints, training histories, latent features and diagnostics remain in the original corrected IBD output directory. The consolidated all-task output records their exact source paths.

## Method for the other four tasks

For every outer fold:

1. the exact train/test sample and participant assignments are read from the shared H0+Ricci task manifest;
2. outer-training participants are divided into participant-grouped inner-train and validation subsets; a deterministic seed sequence is searched until both subsets contain every task class, avoiding scikit-learn-version-specific failures on grouped splits;
3. for each `q` in `{0.3, 0.4, 0.5, 0.6, 0.7}`, adaptive H0 bounds are built from **inner-training deaths only**;
4. the Alpha-Pi network is trained on inner training and the best epoch is selected on validation;
5. `q` is selected by validation balanced accuracy, macro F1, ROC-AUC, fewer realised intervals, and then larger `q`;
6. adaptive bounds are rebuilt from **all outer-training diagrams**;
7. a fresh model is trained on all outer-training samples for the selected number of epochs;
8. the untouched outer-test fold is evaluated once.

The primary prediction is always the cross-entropy/logit head. Nearest-centroid probabilities remain diagnostic.



### Version 1.1.1 self-test fixture correction

The synthetic two-fold fixture now assigns participants to outer folds explicitly within each class. This guarantees at least two outer-training participants per class, so the self-test evaluates the intended trainable path on every supported scikit-learn release. The production safeguard is unchanged: real folds with fewer than two outer-training participants in any class remain a hard error because an all-class inner train/validation split would be impossible.

### Inner-split portability safeguard

Scikit-learn releases can produce different `StratifiedGroupKFold` assignments for the same seed. The engine therefore tries up to 256 deterministic seeds, records the exact accepted seed and split membership, and stops with class-specific participant counts if no all-class participant-safe split exists. This changes only split construction robustness; it does not inspect validation outcomes or outer-test data.

### Preserved model defaults

```text
epochs maximum:        70
batch size:            32
learning rate:         4e-4
weight decay:          1e-4
quadrature points:     5
CE weight:             1000
within-class weight:   1
radius weight:         1
between-class weight:  200
between-class margin:  10
```

## Leakage and integrity safeguards

The code aborts if:

- a manifest sample lacks an H0 diagram;
- source labels or participant IDs disagree with the manifest;
- a participant crosses an outer or inner train/test boundary;
- a fold lacks a class;
- a repetition does not test every sample exactly once;
- completed predictions do not exactly match the locked held-out sample IDs;
- a completion marker is missing required artifacts;
- an existing task output has an incompatible configuration or input fingerprint.

Unrelated duplicate/conflicting rows outside a task manifest are ignored, while conflicts affecting required samples remain fatal.

## Outputs

The consolidated output directory is:

```text
~/Real_Data/H0_AlphaPi_1x5_AllTasks
```

Principal all-task tables:

```text
ALL_TASK_SUMMARY.csv
MANUSCRIPT_H0_1x5_TABLE.csv
RUN_COMPLETE.json
```

Each task folder includes:

```text
fold_results.csv
summary_mean_sd_across_folds.csv
pooled_oof_metrics.csv
all_oof_sample_predictions.csv
all_oof_participant_predictions.csv
all_candidate_validation_results.csv
aggregated_confusion_sample.csv/.png
aggregated_confusion_participant.csv/.png
classification_report_sample.txt
classification_report_participant.txt
per_class_recall_mean_sd_across_folds.csv
alpha_common_grid_mean_sd.csv
mean_alpha_across_folds.png
mean_alpha_normalized_across_folds.png
centered_logit_influence_common_grid_mean_sd.csv
mean_centered_logit_influence_across_folds.png
```

Binary tasks additionally receive:

```text
signed_logit_influence_common_grid_mean_sd.csv
mean_signed_logit_influence_across_folds.png
```

For newly fitted folds, `final_outer_model/` also contains:

- train-only final adaptive bounds;
- raw and normalized alpha arrays;
- classifier-head coefficients;
- class-center parameters;
- final model package;
- training history;
- held-out latent vectors and interval-level class-logit contributions;
- class-specific one-death influence curves.

## Installation on the VM

Extract the ZIP under `~/Real_Data`, then enter the repository:

```bash
cd ~/Real_Data/h0_alpha_pi_all_tasks_1x5
```

The existing Python environment that ran the corrected H0 package is normally sufficient. Requirements are listed in `requirements.txt`.

## Synthetic self-test

```bash
OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 \
python3 self_test.py
```

Expected ending:

```text
SELF-TEST PASSED: exact task manifests, binary and multiclass Alpha-Pi, train-only adaptive bounds, fresh outer-training refit, participant-safe inner/outer splits, candidate/fold resume, pooled sample and participant metrics, deterministic cross-version inner-split fallback, balanced synthetic outer-fold fixture, confusion tables, alpha curves, class-logit influence curves, and artifact persistence all succeeded.
```

## Real-data preflight

```bash
python3 validate_inputs.py
```

Required ending:

```text
STANDALONE H0 ALPHA-PI ALL-TASK 1x5 PREFLIGHT: PASSED
```

The preflight checks all five shared repetition-1 manifests and all required H0 diagrams. It verifies the exact SHA-256 of the corrected v2 IBD source being resumed, then proves row-for-row that repetition 1 of the original numeric-label IBD manifest is identical to the shared all-task IBD manifest in sample, participant, fold, role, label, and split seed before any resumed fitting is allowed.
It also constructs and audits all 20 generic inner train/validation splits for tasks 2–5 on the installed scikit-learn version, proving that both sides contain every class and remain participant-disjoint before the long run begins. Accepted inner seeds are written to `preflight_inner_split_audit.csv`.

## Launch unattended

The launcher uses one numerical thread and a low scheduling priority so it can coexist with Ricci, benchmark and joint-model runs.

```bash
NICE_LEVEL=10 bash launch_tmux.sh
```

Attach:

```bash
tmux attach -t h0_alpha_pi_all_tasks_1x5
```

Detach without stopping the analysis:

```text
Ctrl-b, then d
```

Follow the log:

```bash
tail -f ~/Real_Data/logs/h0_alpha_pi_all_tasks_1x5.log
```

Task-level status:

```bash
bash status.sh
```

Final completion is reached when status reports `25/25` and this file exists:

```text
~/Real_Data/H0_AlphaPi_1x5_AllTasks/RUN_COMPLETE.json
```

## Reveal completed results

```bash
python3 show_results.py
```

## Resume behaviour

Rerunning `bash launch_tmux.sh` after interruption resumes:

- existing corrected IBD candidate/fold artifacts;
- completed candidates for the remaining tasks;
- completed final outer folds;
- task-level aggregation.

No completed fold is refitted under a compatible configuration.

## GitHub repository

Upload the entire extracted directory:

```text
/home/Jack/Real_Data/h0_alpha_pi_all_tasks_1x5
```

Do not upload the H0 diagrams, source metadata, split-output directories, checkpoints or model results. The `.gitignore` excludes the common local data/output names, but `git status` should always be inspected before pushing.
