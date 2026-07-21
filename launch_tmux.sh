#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REAL_DATA_DIR="${REAL_DATA_DIR:-$HOME/Real_Data}"
SESSION_NAME="${SESSION_NAME:-h0_ricci_joint_all_tasks_1x5}"
OUTPUT_DIR="${OUTPUT_DIR:-$REAL_DATA_DIR/H0_Ricci_JointSparse_RepeatedCV_AllTasks}"
LOG_DIR="${LOG_DIR:-$REAL_DATA_DIR/logs}"
N_JOBS="${N_JOBS:-1}"
NICE_LEVEL="${NICE_LEVEL:-5}"
CORE_PYTHON="${CORE_PYTHON:-$REAL_DATA_DIR/h0_ricci_joint_sparse/.venv/bin/python}"

if ! command -v tmux >/dev/null 2>&1; then
  echo "[ERROR] tmux is not installed or not on PATH." >&2
  exit 1
fi

if [[ ! -x "$CORE_PYTHON" ]]; then
  echo "[ERROR] Core virtual-environment Python is missing or not executable:" >&2
  echo "  $CORE_PYTHON" >&2
  exit 1
fi

mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/h0_ricci_joint_all_tasks_1x5.log"
COMMAND_FILE="$LOG_DIR/h0_ricci_joint_all_tasks_1x5_command.sh"

cat > "$COMMAND_FILE" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "$REPO_ROOT"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
nice -n "$NICE_LEVEL" "$CORE_PYTHON" -u run_all_tasks.py \\
  --output-dir "$OUTPUT_DIR" \\
  --n-jobs "$N_JOBS" \\
  2>&1 | tee -a "$LOG_FILE"
EOF
chmod +x "$COMMAND_FILE"

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "[ERROR] tmux session '$SESSION_NAME' already exists." >&2
  echo "Attach with: tmux attach -t $SESSION_NAME" >&2
  exit 1
fi

tmux new-session -d -s "$SESSION_NAME" "bash '$COMMAND_FILE'"
sleep 2
if ! tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "[ERROR] The tmux session exited immediately. Inspect:" >&2
  echo "  tail -100 '$LOG_FILE'" >&2
  exit 1
fi

cat <<EOF
Started tmux session: $SESSION_NAME
Task order: IBD vs non-IBD -> 3-way -> non-IBD vs UC -> non-IBD vs CD -> UC vs CD
Requested range: repetition 1 only (25 outer folds total)
Output: $OUTPUT_DIR
Log:    $LOG_FILE
Python:  $CORE_PYTHON
Workers: $N_JOBS interval-count path(s), one BLAS thread each

Attach:
  tmux attach -t $SESSION_NAME

Monitor:
  tail -f "$LOG_FILE"

Progress:
  cat "$OUTPUT_DIR/progress.json"
  column -s, -t < "$OUTPUT_DIR/progress_by_task.csv" | less -S
EOF
