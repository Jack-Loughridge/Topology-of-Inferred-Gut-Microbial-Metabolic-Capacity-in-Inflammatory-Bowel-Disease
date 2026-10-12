# Repeated H0 Alpha-Pi classifier (IBD vs non-IBD)

Package schema version: **2**.

This repository runs the leakage-corrected H0 Alpha-Pi classifier over the **same 20 × 5 participant-grouped outer folds** used by the repeated Ricci analysis.

The primary script is:

```text
h0_alpha_pi_repeated_cv.py
```

## Method implemented

For every outer fold, the code:

1. Reads the exact train/test assignment from the Ricci `sample_split_manifest.csv`; it does not regenerate outer folds.
2. Splits the outer-training participants once into grouped inner-train and validation subsets.
3. For each adaptive percentile step in `0.3, 0.4, 0.5, 0.6, 0.7`:
   - constructs adaptive H0 death-value boundaries from **inner-training diagrams only**;
   - trains the Alpha-Pi model on inner training;
   - selects the epoch using validation balanced accuracy, with prespecified tie-breaks;
   - evaluates the candidate on validation.
4. Selects the percentile step using validation balanced accuracy, macro F1 and logit ROC-AUC, followed by a **fewer-intervals** tie-break.
5. Rebuilds the selected adaptive boundaries from **all outer-training diagrams**.
6. Initializes a fresh model and trains on all outer-training samples for the selected number of epochs.
7. Evaluates once on the untouched outer-test participants.

The CE/logit head is the primary predictor. Nearest-centroid probabilities are diagnostic only.

## Why this is the corrected version

This implementation fixes the important weaknesses of the earlier five-fold script:

- no global precomputed adaptive intervals;
- no validation/test diagram determines its own boundaries;
- exact shared Ricci outer folds are loaded, not approximated from a seed;
- validation participants are reincorporated in a fresh final outer-training refit;
- exact performance ties favour fewer intervals rather than denser models;
- missing participant IDs are fatal rather than replaced with sample IDs;
- label and participant checks are restricted to samples in the locked outer-fold manifest, so unrelated conflicts elsewhere in the source metadata cannot block the analysis; conflicts for any manifest sample remain fatal;
- 20 pooled OOF repetition results are the main split-sensitivity distribution;
- sample-level and participant-level results are both saved;
- raw alpha is supplemented by a signed logit-influence curve;
- candidate and final-fold fits are resume-safe.

## Files

```text
h0_alpha_pi_repeated_cv.py   Main analysis
validate_inputs.py           Strict preflight validation
launch_tmux.sh               Conservative tmux launcher
self_test.py                 Fast functional test
requirements.txt             Python dependencies
```

## Required data

Default paths under `~/Real_Data`:

```text
out_pds/*_H0.npy
sample_labels.csv
hmp2_metadata.csv
Ricci_IBD_RepeatedCV_CPath/splits/sample_split_manifest.csv
```

The manifest must have the columns:

```text
repeat, fold, split_seed, role, sample_id, participant_id, label, label_name
```


### Manifest-scoped metadata validation

The IBDMDB metadata file can contain repeated external identifiers belonging to records that are not part of the H0/Ricci analysis. The validator therefore filters labels and participant metadata to the exact sample IDs in the locked split manifest **before** checking duplicates, missing values and conflicts. This is not a relaxation for analysed samples: every manifest sample must still have one unambiguous participant ID and a label consistent with the manifest.

The default output directory is:

```text
~/Real_Data/H0_AlphaPi_RepeatedCV_TrainOnlyBins_IBD
```

## Installation and launch

Unzip the package under `~/Real_Data`:

```bash
cd ~/Real_Data
unzip -o h0_alpha_pi_repeated_cv.zip
cd h0_alpha_pi_repeated_cv
```

Run the fast package self-test:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 self_test.py
```

Validate the real inputs before training:

```bash
python3 validate_inputs.py \
  --pd-dir ~/Real_Data/out_pds \
  --label-csv ~/Real_Data/sample_labels.csv \
  --metadata-csv ~/Real_Data/hmp2_metadata.csv \
  --split-manifest ~/Real_Data/Ricci_IBD_RepeatedCV_CPath/splits/sample_split_manifest.csv
```

Launch the full analysis:

```bash
bash launch_tmux.sh
```

Attach to the session:

```bash
tmux attach -t h0_alpha_pi_20x5
```

Detach without stopping it:

```text
Ctrl-b, then d
```

Follow the log:

```bash
tail -f ~/Real_Data/H0_AlphaPi_RepeatedCV_TrainOnlyBins_IBD/full_run.log
```

Check how many outer folds are complete:

```bash
find ~/Real_Data/H0_AlphaPi_RepeatedCV_TrainOnlyBins_IBD/folds \
  -name FOLD_COMPLETE.json | wc -l
```

The complete run should eventually report `100`.

Progress files are updated after every completed outer fold:

```text
~/Real_Data/H0_AlphaPi_RepeatedCV_TrainOnlyBins_IBD/partial_progress/progress.json
~/Real_Data/H0_AlphaPi_RepeatedCV_TrainOnlyBins_IBD/partial_progress/completed_outer_fold_results.csv
~/Real_Data/H0_AlphaPi_RepeatedCV_TrainOnlyBins_IBD/partial_progress/completed_repetition_sample_metrics.csv
~/Real_Data/H0_AlphaPi_RepeatedCV_TrainOnlyBins_IBD/partial_progress/completed_repetition_participant_metrics.csv
```

A repetition-level metric row becomes available as soon as all five folds of that repetition are complete.

## Concurrent running with Ricci

The launcher limits CPU numerical-library threads to one and lets PyTorch use CUDA when available. This is deliberate: Ricci can continue on CPU while Alpha-Pi runs primarily on the GPU.

Check resources before launch:

```bash
nproc
free -h
nvidia-smi
```

To use another GPU:

```bash
CUDA_VISIBLE_DEVICES=1 bash launch_tmux.sh
```

To force CPU execution:

```bash
EXTRA_ARGS='--device cpu' bash launch_tmux.sh
```

Avoid running unrestricted `n_jobs=-1` benchmark jobs concurrently with Ricci and H0.

## Resume behavior

Re-run the same launcher after an interruption:

```bash
bash launch_tmux.sh
```

Completed candidate fits and completed outer folds are loaded from disk. The program also compares the saved configuration and input fingerprints; it refuses to resume into an incompatible output directory.

To intentionally start a different configuration, use a new output directory:

```bash
OUTPUT_DIR=~/Real_Data/H0_AlphaPi_RepeatedCV_Alternative \
EXTRA_ARGS='--percentile-steps 0.4,0.5,0.6' \
bash launch_tmux.sh
```

## Main aggregate outputs

After all 100 folds finish:

```text
all_outer_fold_results.csv
all_candidate_validation_results.csv
all_oof_sample_predictions.csv
all_oof_participant_predictions.csv
repetition_pooled_oof_sample_metrics.csv
repetition_pooled_oof_participant_metrics.csv
performance_summary_across_repetitions.csv
selected_resolution_summary.csv
candidate_resolution_diagnostics.csv
all_final_model_interpretation_curves.csv
interpretation_curve_summary_across_100_models.csv
consensus_prediction_by_sample_across_repetitions.csv
consensus_prediction_by_participant_across_repetitions.csv
RUN_COMPLETE.json
```

The **primary performance rows** are the 20 rows in:

```text
repetition_pooled_oof_sample_metrics.csv
repetition_pooled_oof_participant_metrics.csv
```

The 100 fold scores are diagnostic and should not be treated as 100 independent estimates.

## Per-fold outputs

For each `repeat_XX_fold_Y`:

```text
split_membership.csv
candidate_summary.csv
selected_candidate.json
candidates/
  percentile_step_*/
    adaptive_bounds_inner_train.npy
    training_history.csv
    best_validation_predictions_sample.csv
    best_validation_predictions_participant.csv
    alpha_best.npy
    candidate_best_model.pt
    candidate_result.json
    CANDIDATE_COMPLETE.json
final_outer_model/
  adaptive_bounds_outer_train.npy
  interval_centers.npy
  alpha.npy
  logit_delta_weight.npy
  centers_parameter.npy
  training_history_outer_train.csv
  test_predictions_sample.csv
  test_predictions_participant.csv
  test_calibration_sample.csv
  test_calibration_participant.csv
  test_latent_features_and_contributions.npz
  interpretation_curves.csv
  final_outer_model.pt
  final_fold_result.json
FOLD_COMPLETE.json
```

## Interpretation curves

Two different quantities are saved and should not be conflated.

### Normalized alpha

`alpha_normalized_mean1_loggrid` describes which H0 death scales the learned representation emphasizes. Normalization reduces the arbitrary scaling trade-off between alpha and the final linear head.

### Signed single-death influence

`signed_single_death_logit_influence` is the direct change in the binary logit contrast caused by one H0 death at scale `d`:

```text
positive  -> pushes toward IBD
negative  -> pushes toward non-IBD
```

The code verifies the influence formula numerically for every completed final fold before writing `FOLD_COMPLETE.json`.

## Default model constants

The defaults preserve the most recent Alpha-Pi formulation:

```text
maximum candidate epochs = 70
batch size               = 32
learning rate            = 4e-4
weight decay             = 1e-4
quadrature points        = 5
lambda CE                = 1000
lambda within            = 1
lambda radius            = 1
lambda between           = 200
between-center margin    = 10
alpha                     = softplus(raw_alpha) + 1e-8
```

Gradient clipping is disabled by default to preserve the original optimization. Gradient norms and finite-value checks are still recorded.

## Reproducibility note

The code requests deterministic PyTorch algorithms by default and fixes outer-independent model and inner-split seeds. Exact bitwise equality can still depend on hardware, CUDA, PyTorch and BLAS versions; these are written to `run_config.json`.
