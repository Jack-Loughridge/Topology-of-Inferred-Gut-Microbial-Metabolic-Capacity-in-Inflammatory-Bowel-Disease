#!/usr/bin/env bash
set -euo pipefail
SESSION_NAME="${SESSION_NAME:-ricci_all_tasks_20x5}"
REAL_DATA_DIR="${REAL_DATA_DIR:-$HOME/Real_Data}"
REPO_DIR="${REPO_DIR:-$REAL_DATA_DIR/ricci_all_tasks_20x5}"
FEATURE_DIR="${FEATURE_DIR:-$REAL_DATA_DIR/Ricci_Classifier_Faithful_Eps0001_n250_v3}"
SPLIT_DIR="${SPLIT_DIR:-$REAL_DATA_DIR/H0_Ricci_JointSparse_RepeatedCV_AllTasks/splits}"
OUTPUT_DIR="${OUTPUT_DIR:-$REAL_DATA_DIR/Ricci_RepeatedCV_AllTasks}"
LOG_DIR="${LOG_DIR:-$REAL_DATA_DIR/logs}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/ricci_all_tasks_20x5.log}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
N_JOBS="${N_JOBS:-1}"
NICE_LEVEL="${NICE_LEVEL:-5}"
C_VALUES="${C_VALUES:-0.02}"
command -v tmux >/dev/null 2>&1 || { echo "[error] tmux not found" >&2; exit 1; }
[[ -f "$REPO_DIR/ricci_all_tasks.py" ]] || { echo "[error] Missing $REPO_DIR/ricci_all_tasks.py" >&2; exit 1; }
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then echo "[error] Session exists: $SESSION_NAME" >&2; exit 1; fi
mkdir -p "$LOG_DIR"
RUNNER="$LOG_DIR/${SESSION_NAME}_runner.sh"
{
  printf '#!/usr/bin/env bash\nset -euo pipefail\ncd %q\n' "$REPO_DIR"
  printf 'export OMP_NUM_THREADS=%q\nexport MKL_NUM_THREADS=%q\nexport OPENBLAS_NUM_THREADS=%q\nexport NUMEXPR_NUM_THREADS=%q\n' "$N_JOBS" "$N_JOBS" "$N_JOBS" "$N_JOBS"
  printf 'exec > >(tee -a %q) 2>&1\n' "$LOG_FILE"
  printf 'nice -n %q %q %q' "$NICE_LEVEL" "$PYTHON_BIN" "$REPO_DIR/ricci_all_tasks.py"
  printf ' %q %q' --feature-dir "$FEATURE_DIR"
  printf ' %q %q' --split-dir "$SPLIT_DIR"
  printf ' %q %q' --output-dir "$OUTPUT_DIR"
  printf ' %q %q' --c-values "$C_VALUES"
  printf ' %q %q' --n-jobs "$N_JOBS"
  printf '\n'
} > "$RUNNER"
chmod 700 "$RUNNER"; bash -n "$RUNNER"
tmux new-session -d -s "$SESSION_NAME" "bash $(printf '%q' "$RUNNER")"
sleep 2
if ! tmux has-session -t "$SESSION_NAME" 2>/dev/null; then tail -n 100 "$LOG_FILE" >&2 || true; exit 1; fi
echo "Started: $SESSION_NAME"; echo "Log: tail -f $LOG_FILE"; echo "Output: $OUTPUT_DIR"
