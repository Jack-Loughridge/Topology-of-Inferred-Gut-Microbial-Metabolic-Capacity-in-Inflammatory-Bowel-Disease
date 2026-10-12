# Primary IBD Ricci feature-block ablation

This package runs the two prespecified ablations needed to determine whether
IBD-versus-non-IBD discrimination comes from reaction-edge presence, local
Ricci geometry, or their combination:

1. `B only`: active-edge indicators only;
2. `K0 only`: active-edge Ricci curvature only, with zero for absent edges.

The completed `[B | K0]` model is not refitted. Its saved repetition results
and split manifests are used as the comparison reference.

## Scientific lock

Except for the feature block, the ablations match the completed primary model:

- IBD (UC + CD) versus non-IBD;
- `C=0.02` only;
- 20 repetitions x 5 participant-grouped folds;
- byte-identical saved split manifests from `Ricci_IBD_RepeatedCV_CPath`;
- `StandardScaler` fitted on each outer-training fold only;
- L1 `LogisticRegression(solver="saga", class_weight="balanced")`;
- identical model seeds, `max_iter=10000`, and `tol=1e-4`;
- arithmetic mean of held-out sample probabilities within participant;
- the same sample- and participant-level metrics and repeated-OOF summaries.

The driver imports the hash-locked production engine rather than copying its
classifier implementation. It refuses to run if that engine's SHA-256 differs
from the source frozen in the publication repository.

## Expected repository location

Place this directory at:

```text
analysis/ricci/ibd_block_ablation/
```

The default engine path then resolves to:

```text
analysis/ricci/ibd_cpath/repeated_ricci_ibd_cpath.py
```

## Preflight

From the repository root on the Azure VM:

```bash
python3 analysis/ricci/ibd_block_ablation/run_ibd_block_ablation.py \
  --validate-only
```

The required ending is:

```text
RICCI IBD FEATURE-BLOCK ABLATION PREFLIGHT: PASSED
```

## Run

Directly:

```bash
OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 \
python3 analysis/ricci/ibd_block_ablation/run_ibd_block_ablation.py
```

Or in a resumable tmux session:

```bash
bash analysis/ricci/ibd_block_ablation/launch_tmux.sh
```

Defaults:

```text
feature input:       ~/Real_Data/Ricci_Classifier_Faithful_Eps0001_n250_v3
combined reference:  ~/Real_Data/Ricci_IBD_RepeatedCV_CPath
ablation output:     ~/Real_Data/Ricci_IBD_FeatureBlock_Ablation_C002
log:                 ~/Real_Data/logs/ricci_ibd_feature_block_ablation.log
```

Run only one block, if resuming operationally:

```bash
python3 analysis/ricci/ibd_block_ablation/run_ibd_block_ablation.py --blocks B
python3 analysis/ricci/ibd_block_ablation/run_ibd_block_ablation.py --blocks K0
```

The top-level comparison is regenerated from every completed block found in the
output directory, so sequential block runs are supported.

## Principal outputs

```text
Ricci_IBD_FeatureBlock_Ablation_C002/
  B_only/
  K0_only/
  ablation_repetition_metrics_long.csv
  ablation_performance_summary.csv
  ablation_paired_deltas.csv
  ricci_feature_block_ablation.tex
  ABLATION_RUN_COMPLETE.json
  ablation_output_manifest.json
```

Each block directory retains the full canonical C-path output layout, including
fold predictions, fold artifacts, coefficient stability, process summaries,
and a block-specific provenance record.

`ablation_paired_deltas.csv` reports three prespecified contrasts on the same 20
repetitions:

- `K0 only minus B only`;
- `[B|K0] minus B only`;
- `[B|K0] minus K0 only`.

The reported quantiles and proportions of positive deltas describe split
sensitivity. They are not p-values or confidence intervals from independent
cohorts.

## Acceptance checks

The run fails rather than silently continuing if:

- the production engine hash has changed;
- the combined reference configuration differs;
- a frozen manifest has missing folds, participant leakage, class loss, or
  incomplete test coverage;
- copied split files are not byte-identical to the combined reference;
- B is not binary or K0 is empty/non-finite;
- an output directory contains an incompatible scientific lock;
- any fold records a convergence warning or reaches `max_iter`;
- a B-only result selects a K0 coefficient or vice versa;
- paired repetition counts or cohort sizes disagree.

## Publication interpretation

Do not select a different C separately for either block. The primary question
is performance at the already prespecified `C=0.02` under identical splits and
preprocessing. The ablation should be described as a feature-block analysis,
not as two newly tuned competitor models.

