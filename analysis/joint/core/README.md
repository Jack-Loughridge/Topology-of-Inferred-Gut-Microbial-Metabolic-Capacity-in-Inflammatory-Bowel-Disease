# Joint WKPI + Ricci Sparse Classifier

GitHub-ready implementation of the final Alpha-Pi microbiome **joint H0 WKPI + faithful Ricci `[B | K0]` sparse classifier**.

This repository implements the agreed model directly. It is **not** the earlier late/logit-fusion model and it does not average predictions from separately fitted H0 and Ricci classifiers.

## Fixed methodology

For each task and each outer participant-grouped fold:

1. Raw H0 persistence diagrams are loaded for the samples in the fold.
2. H0 log-death intervals are fitted using **training participants only**.
3. The unweighted WKPI basis is computed with the established five-point Gauss-Legendre quadrature and log-Gaussian kernel.
4. A nonnegative learned alpha profile is constrained to have mean one.
5. The weighted WKPI vector is joined blockwise with the faithful Ricci feature vector `[B | K0]` without constructing repeated concatenated matrices.
6. A sparse logistic head is fitted with separate H0 and Ricci L1 penalties:

   \[
   \mathcal L_{CE}
   + \lambda_H\lVert\beta_H\rVert_1
   + \lambda_R\lVert\beta_R\rVert_1
   + \gamma\lVert\Delta\alpha\rVert_2^2.
   \]

7. Alpha and the L1 logistic head are alternated six times.
8. Hyperparameters are selected using three inner participant-grouped folds. The untouched outer participants are used once for final evaluation.

The logistic head is optimized directly in the original block units. The active-set proximal solver minimizes the SAGA-equivalent scaled objective

`weighted_mean_CE + lambda_H/(C*sum_w)*||beta_H||_1 + lambda_R/(C*sum_w)*||beta_R||_1`.

Only Ricci variables that currently violate the L1 KKT conditions enter the temporary working set. Every omitted variable is checked against the full gradient before a fit is accepted, so this is an exact optimization strategy rather than pre-filtering. Binary tasks retain the single-logit parameterization; the three-way task uses multinomial softmax.

### Default validation grid

- H0 interval count `M`: `96, 160, 224`
- H0 L1 multiplier `lambda_H`: `0.5, 1, 2`
- Ricci L1 multiplier `lambda_R`: `0.5, 1, 2`
- outer folds: `5`
- inner folds: `3`
- alternating alpha/L1 updates: `6`
- logistic `C`: `0.02`
- selection metric: ROC-AUC
- seed: `13`

## Five tasks

- `three_way_nonIBD_UC_CD`: non-IBD vs UC vs CD
- `IBD_vs_nonIBD`: CD + UC vs non-IBD
- `nonIBD_vs_UC`
- `nonIBD_vs_CD`
- `CD_vs_UC`

All splits are grouped by `participant_id` from the faithful Ricci metadata. A run stops immediately if participant overlap is detected.

## Canonical project inputs

```text
H0 reference outputs:
~/Real_Data/H0_AlphaPi_NestedQ_TrainOnlyBins

Original-style Ricci outputs and biological annotations:
~/Real_Data/Ricci_Classifier_OriginalStyle_AllTasks_NewVectors

Faithful Ricci matrix and matched metadata:
~/Real_Data/Ricci_Classifier_Faithful_Eps0001_n250_v3
```

The faithful Ricci directory must contain:

```text
feature_matrix_B_K0.npz
matched_metadata.csv
```

The H0 WKPI vectors cannot be reconstructed correctly from saved H0 predictions or aggregate alpha plots. The raw diagrams are therefore read from:

```text
~/Real_Data/out_pds/*_H0.npy
```

`--h0-pd-dir` can override that location. When omitted, the loader searches the H0 results manifests and the canonical sibling `out_pds` directory.

## Installation

Python 3.10 or later is recommended.

```bash
cd ~/Real_Data/h0_ricci_joint_sparse
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

For the existing VM environment, installing `requirements.txt` without a virtual environment is also sufficient:

```bash
pip install -r requirements.txt
```

## Validate inputs before the long run

```bash
cd ~/Real_Data/h0_ricci_joint_sparse
bash scripts/validate_inputs.sh
```

This checks sample alignment, participant IDs, diagnosis consistency, H0 diagram coverage, matrix dimensions, feature annotations, and the expected five-task reference outputs. It writes `INPUT_AUDIT.json`, `INPUT_SAMPLE_AUDIT.csv`, and `REFERENCE_OUTPUT_AUDIT.csv` without fitting a model.

## Benchmark one complete inner fold first

Before the manuscript run, execute all 27 configurations for the first three-way inner fold:

```bash
cd ~/Real_Data/h0_ricci_joint_sparse
bash scripts/benchmark_first_inner_fold.sh
```

This stops automatically after the benchmark and writes:

```text
BENCHMARK_FIRST_INNER_FOLD.csv
BENCHMARK_FIRST_INNER_FOLD.json
```

Every configuration records elapsed time, active Ricci count, optimizer iterations, objective and maximum full KKT violation. Do not launch all five tasks until this benchmark gives a credible runtime.

## Launch the full run in tmux

```bash
cd ~/Real_Data/h0_ricci_joint_sparse
bash scripts/launch_tmux.sh
```

The launcher creates a timestamped directory such as:

```text
~/Real_Data/H0_Ricci_JointSparse_ActiveSet_20260715_183000
```

Attach with:

```bash
tmux attach -t h0_ricci_joint_sparse
```

Detach with `Ctrl-b`, then `d`.

Monitor without attaching:

```bash
bash scripts/monitor_latest.sh
```

### Run one task first

This changes only the execution scope, not the task methodology:

```bash
TASKS=IBD_vs_nonIBD SESSION_NAME=h0_ricci_ibd bash scripts/launch_tmux.sh
```

## Direct command

```bash
python3 -u scripts/run_joint_sparse.py \
  --h0-results-dir ~/Real_Data/H0_AlphaPi_NestedQ_TrainOnlyBins \
  --h0-pd-dir ~/Real_Data/out_pds \
  --ricci-original-dir ~/Real_Data/Ricci_Classifier_OriginalStyle_AllTasks_NewVectors \
  --ricci-feature-dir ~/Real_Data/Ricci_Classifier_Faithful_Eps0001_n250_v3 \
  --out-dir ~/Real_Data/H0_Ricci_JointSparse_Nested_manual \
  --tasks all \
  --n-outer-splits 5 \
  --n-inner-splits 3 \
  --seed 13 \
  --n-jobs 3 \
  --selection-metric roc_auc \
  --interval-grid 96,160,224 \
  --lambda-h0-grid 0.5,1,2 \
  --lambda-ricci-grid 0.5,1,2 \
  --C 0.02 \
  --logistic-max-iter 8000 \
  --logistic-tol 1e-4 \
  --active-set-initial 512 \
  --active-set-batch 512 \
  --active-set-max-rounds 50 \
  --alternations 6 \
  --quad-points 5
```

## Main outputs

At run level:

```text
ALL_FOLD_RESULTS.csv
ALL_SUMMARIES.csv
ALL_TEST_PREDICTIONS.csv
ALL_ALPHA_PROFILES.csv
ALL_INNER_CONFIG_RESULTS.csv
ALL_SELECTED_CONFIGS.csv
ALL_SPARSITY_DIAGNOSTICS.csv
RICCI_FEATURE_METADATA_USED.csv
RUN_CONFIG.json
latex_joint_sparse_summary.tex
latex_selected_configurations.tex
```

For every task:

```text
fold_results.csv
test_predictions.csv
aggregated_confusion_matrix.csv/.png/.pdf/.tex
alpha_profiles_all_folds.csv
alpha_interpolated_common_grid.csv
mean_alpha_across_outer_folds.csv/.png/.pdf
latex_fold_results.tex
reference_comparison_original_style_ricci.csv
```

For every outer fold:

```text
selected_config.json
participant_group_split.csv
inner_config_results.csv
inner_config_summary.csv
checkpoints/inner_*/M_*/*.json/.npz
alpha_profile.csv
alternation_history.csv
sparsity_diagnostics.json
top_h0_coefficients.csv
top_ricci_coefficients.csv
test_confusion_matrix.csv/.png/.pdf/.tex
```

See [`docs/OUTPUTS.md`](docs/OUTPUTS.md) for definitions.

## Tests

```bash
pytest -q
```

The tests cover train-only interval fitting, H0 array loading, alpha constraints, participant leakage, class-ordered multiclass AUC, LaTeX safety, checkpoint round-trips, and binary/multinomial probability equivalence against converged scikit-learn SAGA fits. See `VERIFICATION.md` for the end-to-end audit.

## Repository layout

```text
src/h0_ricci_joint_sparse/
  cli.py          command-line entry point and input audit
  config.py       fixed run configuration
  cv.py           nested participant-grouped evaluation
  data.py         faithful Ricci and raw H0 loaders
  wkpi.py         train-only adaptive WKPI construction
  model.py        alternating alpha + KKT-safe block-L1 active-set solver
  metrics.py      task metrics
  outputs.py      diagnostics, plots, confusion matrices and LaTeX
scripts/
  benchmark_first_inner_fold.sh
  launch_tmux.sh
  monitor_latest.sh
  run_joint_sparse.py
  validate_inputs.sh
tests/
docs/
```

## Reproducibility notes

- The faithful Ricci matrix supplies the actual `[B | K0]` features.
- Original-style Ricci outputs are not substituted for the faithful matrix. They are used only to recover feature/process annotations and create reference comparisons.
- Every H0 interval set, WKPI standardizer, Ricci standardizer, alpha profile and classifier is fitted inside the applicable training split.
- No test sample or test participant contributes to interval boundaries, alpha learning, scaling, sparsity selection or hyperparameter selection.
- Ricci source data remain in CSR form, but each training/evaluation split is converted once to dense float64 because the observed matrix is approximately 71% nonzero.
- Three independent interval-count paths are warm-started from strong to weak penalties and may run concurrently. BLAS is restricted to one thread per worker.
- Each completed configuration and outer fold is checkpointed. Re-running with the same `OUT_DIR` resumes rather than discards completed work.
- The full grid remains computationally expensive. Reducing the grid or alternations changes the planned analysis and should be treated as a separate exploratory run.
