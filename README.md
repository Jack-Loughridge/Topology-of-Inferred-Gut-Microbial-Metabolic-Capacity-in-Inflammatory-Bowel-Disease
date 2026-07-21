# Repeated joint H0 Alpha-Pi + faithful Ricci classification

This repository is the publication-facing orchestration layer for the validated
`h0_ricci_joint_sparse` core classifier. It runs nested participant-grouped
cross-validation for five IBD classification tasks in a fixed order:

1. IBD vs non-IBD
2. non-IBD vs UC vs CD
3. non-IBD vs UC
4. non-IBD vs CD
5. UC vs CD (the historical core folder is `CD_vs_UC`)

Each task uses **20 repetitions × 5 outer folds**. All samples from a participant
remain in one fold. Hyperparameters are selected inside each outer-training set
using three participant-grouped inner folds.

## Final scientific configuration

The joint feature vector is the train-only H0 WKPI block concatenated with the
faithful Ricci presence/curvature block. The final inner grid is:

```text
H0 intervals M:       64, 96, 160
lambda_H0:            0.5, 1, 2, 4
lambda_Ricci:         0.5, 1, 2
global C:             0.02
inner folds:          3
selection metric:     ROC-AUC
```

There are 36 joint configurations per inner fold. No Ricci-only candidate is
included; Ricci-alone performance is available from the separate repeated Ricci
analysis.

### Resolution-normalised alpha smoothness

The core alpha loss originally used a fixed numerical coefficient for

```text
gamma * sum_j (alpha[j+1] - alpha[j])^2.
```

For samples of the same underlying smooth function, this finite-difference sum
shrinks approximately as `1/(M-1)`. This repository therefore installs an
audited, minimal core patch using

```text
gamma_eff(M) = gamma * (M - 1) / (160 - 1).
```

The previous M=160 objective is exactly unchanged. With base `gamma=0.01`, the
three effective coefficients are recorded in the preflight and selected-model
outputs.

### Audited compatibility with the current core

The current VM core is version 1.2.0 rather than the earlier v1.1.1 source for
which the first installer was written. The difference is a newer exact
orthant-polishing stage and associated solver diagnostics; it does not alter the
alpha-loss signature or its unique insertion point. The v2 installer accepts the
exact v1.2.0 source hash, preserves the orthant-polish code unchanged, and modifies
only the local `smoothness_gamma` used by `_alpha_loss_and_gradient`.

## Split locking

The existing repeated Ricci IBD-vs-non-IBD manifest is the source of the 20
split seeds. The program:

- reproduces the IBD folds exactly and refuses any mismatch;
- generates deterministic, task-specific manifests for the other four cohorts
  using the same repetition seeds and `StratifiedGroupKFold` design;
- saves every manifest before fitting;
- independently regenerates and verifies all 500 outer train/test splits using
  the core task loader;
- fingerprints the split manifests, raw H0 diagrams, Ricci feature matrix, sample metadata, core source, and orchestration source.

The generated manifests are reusable by later repeated H0, Ricci, or abundance
analyses for those same tasks.

## Required existing directories

Defaults assume:

```text
~/Real_Data/h0_ricci_joint_sparse
~/Real_Data/H0_AlphaPi_NestedQ_TrainOnlyBins
~/Real_Data/out_pds
~/Real_Data/Ricci_Classifier_OriginalStyle_AllTasks_NewVectors
~/Real_Data/Ricci_Classifier_Faithful_Eps0001_n250_v3
~/Real_Data/Ricci_IBD_RepeatedCV_CPath/splits/sample_split_manifest.csv
```

The core repository must contain one of the exact audited solver sources accepted by
the installer. The current VM source is core v1.2.0 with monotone FISTA, full KKT
certification and exact orthant-constrained L-BFGS-B polishing. Its unpatched
`model.py` SHA-256 is:

```text
b0eff25de9bf1400d5d1e3a5d3576ba1233ffdeda3420596cf24de65ed8e8e8e
```

The older post-FISTA v1.1.1 source remains supported for repository portability.
Unknown source hashes are rejected.

## Installation and preflight

Unzip this repository under `~/Real_Data`, then:

```bash
cd ~/Real_Data/h0_ricci_joint_repeated_cv_all_tasks_v2

OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 \
python3 self_test.py
```

Install the minimal resolution-normalisation patch into the existing core:

```bash
python3 install_core_patch.py \
  --joint-repo ~/Real_Data/h0_ricci_joint_sparse
```

The installer is deliberately strict. It recognises only exact allow-listed core
sources by SHA-256. For the current v1.2.0 orthant-polish source, it additionally
checks the deterministic patched hash
`fc6d0a8ec94172900c9ac25e431e5dfb601f0eddf0674c4f389c0993d6d0a230`. It
creates a timestamped backup, checks the helper and the alpha loss functionally,
runs the dedicated patch test, and then runs the full core pytest suite. Any
failure automatically restores the original source.

Run the real-data preflight with the **same validated core virtual environment** used for fitting:

```bash
~/Real_Data/h0_ricci_joint_sparse/.venv/bin/python validate_inputs.py
```

The expected ending is:

```text
REPEATED JOINT H0 + RICCI ALL-TASK PREFLIGHT: PASSED
```

The preflight loads the real inputs, audits all task cohorts, proves exact IBD
manifest reproduction, proves all 500 core-generated outer folds match their
locked manifests, verifies the alpha-smoothness helper numerically and records
source fingerprints.

## Launch and leave running

```bash
N_JOBS=1 bash launch_tmux.sh
```

`N_JOBS=1` is conservative while other analyses share the VM. Once those finish,
the same resumable output can be continued with `N_JOBS=2` or `N_JOBS=3` after
stopping and relaunching the tmux session. `N_JOBS` only parallelises the three
independent M paths; tasks and repetitions remain sequential so the requested
task order is preserved.

Monitor:

```bash
tmux attach -t h0_ricci_joint_all_tasks_20x5

tail -f ~/Real_Data/logs/h0_ricci_joint_all_tasks_20x5.log

cat ~/Real_Data/H0_Ricci_JointSparse_RepeatedCV_AllTasks/progress.json

cd ~/Real_Data/h0_ricci_joint_repeated_cv_all_tasks_v2
bash status.sh
```

The run is resumable at the core's inner-configuration and outer-fold
checkpoints. Completed task/repetition outputs are revalidated against their
manifest and exact core run configuration before being skipped. The run refuses to
resume if locked data, split files, the patched core, or scientific orchestration
source has changed.

## Output structure

```text
H0_Ricci_JointSparse_RepeatedCV_AllTasks/
├── orchestrator_config.json
├── splits/
│   ├── all_task_split_manifest.csv
│   ├── IBD_vs_nonIBD_split_manifest.csv
│   ├── three_way_nonIBD_UC_CD_split_manifest.csv
│   ├── nonIBD_vs_UC_split_manifest.csv
│   ├── nonIBD_vs_CD_split_manifest.csv
│   └── CD_vs_UC_split_manifest.csv
├── preflight/
├── runs/
│   └── <task>/repeat_01 ... repeat_20/
├── aggregate/
│   ├── repetition_metrics.csv
│   ├── repetition_performance_summary.csv
│   ├── all_test_predictions.csv
│   ├── participant_average_predictions.csv
│   ├── all_outer_fold_results.csv
│   ├── all_inner_config_results.csv
│   ├── all_selected_configs.csv
│   ├── selected_configuration_frequency.csv
│   └── coefficient_stability/<task>/
├── progress.json
└── RUN_COMPLETE.json
```

The primary uncertainty distribution is the 20 pooled out-of-fold estimates
per task, at both sample and participant level. The 100 fold scores per task are
retained as diagnostics rather than treated as independent experiments.

Metrics include accuracy, balanced accuracy, macro F1, ROC-AUC, log loss,
multiclass Brier score and confusion counts. Multiclass ROC-AUC is macro one-vs-
rest. Participant predictions average held-out class probabilities over all
samples from each participant.

Coefficient aggregation supports both binary and multiclass model states:

- Ricci selection frequency and signed coefficient distributions including
  implicit zeros;
- process/block coefficient distributions;
- interpolated alpha, H0 beta and alpha×beta curves across outer fits.

## Runtime expectations

This is a large analysis: 500 outer folds, each with 3 inner folds and 36 joint
configurations. It is designed to run unattended and resume safely, but it may
take several days on the current VM, especially while Ricci, H0 and abundance
runs are active. A numerical KKT failure is treated as fatal; the code does not
silently accept an unconverged fit. The final defaults increase the iteration
budget to 24,000 and active-set rounds to 100 because earlier 8,000-iteration
runs occasionally failed the full KKT check.

## GitHub contents

After extraction on the VM, the publication-facing repository is located at:

```text
~/Real_Data/h0_ricci_joint_repeated_cv_all_tasks_v2
```

Upload that **entire directory**, not only `run_all_tasks.py`. In particular:

- `run_all_tasks.py` is the direct command-line entry point;
- `src/joint_repeated_cv/` contains the orchestration, split-locking, verification, metrics and aggregation code;
- `install_core_patch.py` contains the audited resolution-normalisation patch installer;
- `tests/` and `self_test.py` provide unit and synthetic integration tests;
- `status.sh` reports tmux state, progress and the latest log lines;
- `.github/workflows/tests.yml` runs the portable test suite on GitHub Actions.

The validated `h0_ricci_joint_sparse` solver remains a separate repository/dependency.
Do not upload private or restricted microbiome input files, persistence diagrams,
model checkpoints or large result directories.

## Final manuscript launch

The included `launch_tmux.sh` runs all five tasks for repetition 1 only. This is the final computational plan for the joint model.

## Prepare a self-contained GitHub staging folder

The orchestration depends on the separate solver core at `~/Real_Data/h0_ricci_joint_sparse`. On the VM, run:

```bash
bash prepare_github_repo.sh
```

This creates:

```text
~/Real_Data/h0_ricci_joint_classifier_github
```

containing both `core/` and `orchestration/`, while excluding the virtual environment and generated outputs.
