#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION_NAME="${SESSION_NAME:-h0_ricci_joint_sparse}"
STAMP="$(date +%Y%m%d_%H%M%S)"

H0_RESULTS_DIR="${H0_RESULTS_DIR:-$HOME/Real_Data/H0_AlphaPi_NestedQ_TrainOnlyBins}"
H0_PD_DIR="${H0_PD_DIR:-$HOME/Real_Data/out_pds}"
RICCI_ORIGINAL_DIR="${RICCI_ORIGINAL_DIR:-$HOME/Real_Data/Ricci_Classifier_OriginalStyle_AllTasks_NewVectors}"
RICCI_FEATURE_DIR="${RICCI_FEATURE_DIR:-$HOME/Real_Data/Ricci_Classifier_Faithful_Eps0001_n250_v3}"
OUT_DIR="${OUT_DIR:-$HOME/Real_Data/H0_Ricci_JointSparse_ActiveSet_${STAMP}}"
TASKS="${TASKS:-all}"
N_JOBS="${N_JOBS:-3}"

mkdir -p "$OUT_DIR"
RUN_LOG="$OUT_DIR/run.log"
COMMAND_FILE="$OUT_DIR/run_command.sh"

cat > "$COMMAND_FILE" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "$REPO_ROOT"
source "$REPO_ROOT/.venv/bin/activate"
export PYTHONPATH="$REPO_ROOT/src:\${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
/usr/bin/time -v python3 -u scripts/run_joint_sparse.py \\
  --h0-results-dir "$H0_RESULTS_DIR" \\
  --h0-pd-dir "$H0_PD_DIR" \\
  --ricci-original-dir "$RICCI_ORIGINAL_DIR" \\
  --ricci-feature-dir "$RICCI_FEATURE_DIR" \\
  --out-dir "$OUT_DIR" \\
  --tasks "$TASKS" \\
  --n-outer-splits 5 \\
  --n-inner-splits 3 \\
  --seed 13 \\
  --n-jobs "$N_JOBS" \\
  --selection-metric roc_auc \\
  --interval-grid 96,160,224 \\
  --lambda-h0-grid 0.5,1,2 \\
  --lambda-ricci-grid 0.5,1,2 \\
  --C 0.02 \\
  --logistic-max-iter 8000 \\
  --logistic-tol 1e-4 \\
  --active-set-initial 512 \\
  --active-set-batch 512 \\
  --active-set-max-rounds 50 \\
  --alternations 6 \\
  --quad-points 5 \\
  2>&1 | tee -a "$RUN_LOG"
EOF
chmod +x "$COMMAND_FILE"

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "[ERROR] tmux session '$SESSION_NAME' already exists." >&2
  echo "Attach with: tmux attach -t $SESSION_NAME" >&2
  exit 1
fi

tmux new-session -d -s "$SESSION_NAME" "bash '$COMMAND_FILE'"

cat <<EOF
[STARTED] tmux session: $SESSION_NAME
[OUTPUT]  $OUT_DIR
[LOG]     $RUN_LOG
[WORKERS] $N_JOBS parallel M paths; one BLAS thread per worker
[RESUME]  Re-run with OUT_DIR="$OUT_DIR" to continue from checkpoints.

Attach:
  tmux attach -t $SESSION_NAME

Detach:
  Ctrl-b then d

Monitor without attaching:
  tail -f "$RUN_LOG"
EOF
