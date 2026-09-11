#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME="${SESSION_NAME:-species_benchmarks_remaining_20x5}"
REAL_DATA_DIR="${REAL_DATA_DIR:-$HOME/Real_Data}"
REPO_DIR="${REPO_DIR:-$REAL_DATA_DIR/species_benchmarks_repeated_cv_remaining_tasks}"
OUTPUT_DIR="${OUTPUT_DIR:-$REAL_DATA_DIR/Species_Benchmarks_RepeatedCV_RemainingTasks}"
SPLIT_DIR="${SPLIT_DIR:-$REAL_DATA_DIR/H0_Ricci_JointSparse_RepeatedCV_AllTasks/splits}"
LOG_DIR="${LOG_DIR:-$REAL_DATA_DIR/logs}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/species_benchmarks_remaining_20x5.log}"
N_JOBS="${N_JOBS:-1}"
NICE_LEVEL="${NICE_LEVEL:-5}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

mkdir -p "$LOG_DIR"
cd "$REPO_DIR"

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "TMUX session already exists: $SESSION_NAME"
  echo "Attach: tmux attach -t $SESSION_NAME"
  exit 1
fi

for required in \
  "$SPLIT_DIR/IBD_vs_nonIBD_split_manifest.csv" \
  "$SPLIT_DIR/three_way_nonIBD_UC_CD_split_manifest.csv" \
  "$SPLIT_DIR/nonIBD_vs_UC_split_manifest.csv" \
  "$SPLIT_DIR/nonIBD_vs_CD_split_manifest.csv" \
  "$SPLIT_DIR/CD_vs_UC_split_manifest.csv"; do
  if [[ ! -f "$required" ]]; then
    echo "Missing shared split manifest: $required" >&2
    echo "Run the H0+Ricci all-task validate_inputs.py first." >&2
    exit 1
  fi
done

COMMAND=$(cat <<EOF
set -o pipefail
cd "$REPO_DIR"
export OMP_NUM_THREADS="$N_JOBS"
export MKL_NUM_THREADS="$N_JOBS"
export OPENBLAS_NUM_THREADS="$N_JOBS"
export NUMEXPR_NUM_THREADS="$N_JOBS"
nice -n "$NICE_LEVEL" "$PYTHON_BIN" species_benchmarks_remaining_tasks.py \\
  --split-dir "$SPLIT_DIR" \\
  --output-dir "$OUTPUT_DIR" \\
  --n-jobs "$N_JOBS" 2>&1 | tee -a "$LOG_FILE"
EOF
)

tmux new-session -d -s "$SESSION_NAME" "bash -lc $(printf '%q' "$COMMAND")"

echo "Started tmux session: $SESSION_NAME"
echo "Attach: tmux attach -t $SESSION_NAME"
echo "Log:    tail -f $LOG_FILE"
echo "Output: $OUTPUT_DIR"
