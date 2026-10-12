#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-$HOME/Real_Data/H0_Ricci_JointSparse_InputAudit_${STAMP}}"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"
python3 -u scripts/run_joint_sparse.py \
  --h0-results-dir "${H0_RESULTS_DIR:-$HOME/Real_Data/H0_AlphaPi_NestedQ_TrainOnlyBins}" \
  --h0-pd-dir "${H0_PD_DIR:-$HOME/Real_Data/out_pds}" \
  --ricci-original-dir "${RICCI_ORIGINAL_DIR:-$HOME/Real_Data/Ricci_Classifier_OriginalStyle_AllTasks_NewVectors}" \
  --ricci-feature-dir "${RICCI_FEATURE_DIR:-$HOME/Real_Data/Ricci_Classifier_Faithful_Eps0001_n250_v3}" \
  --out-dir "$OUT_DIR" \
  --validate-only
