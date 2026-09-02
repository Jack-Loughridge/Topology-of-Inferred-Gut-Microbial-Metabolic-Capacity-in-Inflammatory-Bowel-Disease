#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME="${SESSION_NAME:-species_benchmarks_all_tasks_20x5}"
REAL_DATA_DIR="${REAL_DATA_DIR:-$HOME/Real_Data}"
REPO_DIR="${REPO_DIR:-$REAL_DATA_DIR/species_benchmarks_all_tasks_20x5}"
OUTPUT_DIR="${OUTPUT_DIR:-$REAL_DATA_DIR/Species_Benchmarks_RepeatedCV_AllTasks}"
SPLIT_DIR="${SPLIT_DIR:-$REAL_DATA_DIR/H0_Ricci_JointSparse_RepeatedCV_AllTasks/splits}"
LOG_DIR="${LOG_DIR:-$REAL_DATA_DIR/logs}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/species_benchmarks_all_tasks_20x5.log}"
N_JOBS="${N_JOBS:-1}"
NICE_LEVEL="${NICE_LEVEL:-5}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

command -v tmux >/dev/null 2>&1 || { echo "[error] tmux not found" >&2; exit 1; }
[[ -f "$REPO_DIR/species_benchmarks_all_tasks.py" ]] || { echo "[error] Missing main script in $REPO_DIR" >&2; exit 1; }
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "[error] TMUX session already exists: $SESSION_NAME" >&2
  exit 1
fi
for task in IBD_vs_nonIBD three_way_nonIBD_UC_CD nonIBD_vs_UC nonIBD_vs_CD CD_vs_UC; do
  [[ -f "$SPLIT_DIR/${task}_split_manifest.csv" ]] || { echo "[error] Missing $SPLIT_DIR/${task}_split_manifest.csv" >&2; exit 1; }
done
mkdir -p "$LOG_DIR"
RUNNER_FILE="$LOG_DIR/${SESSION_NAME}_runner.sh"
{
  printf '#!/usr/bin/env bash\nset -euo pipefail\n'
  printf 'cd %q\n' "$REPO_DIR"
  printf 'export OMP_NUM_THREADS=%q\nexport MKL_NUM_THREADS=%q\nexport OPENBLAS_NUM_THREADS=%q\nexport NUMEXPR_NUM_THREADS=%q\n' "$N_JOBS" "$N_JOBS" "$N_JOBS" "$N_JOBS"
  printf 'exec > >(tee -a %q) 2>&1\n' "$LOG_FILE"
  printf 'echo %q\n' '=================================================================='
  printf 'echo %q "$(date -u +%%Y-%%m-%%dT%%H:%%M:%%SZ)"\n' 'ALL-TASK SPECIES BENCHMARKS — started UTC:'
  printf 'nice -n %q %q %q' "$NICE_LEVEL" "$PYTHON_BIN" "$REPO_DIR/species_benchmarks_all_tasks.py"
  printf ' %q %q' --split-dir "$SPLIT_DIR"
  printf ' %q %q' --output-dir "$OUTPUT_DIR"
  printf ' %q %q' --n-jobs "$N_JOBS"
  printf '\n'
  printf 'echo %q "$(date -u +%%Y-%%m-%%dT%%H:%%M:%%SZ)"\n' 'Completed UTC:'
} > "$RUNNER_FILE"
chmod 700 "$RUNNER_FILE"
bash -n "$RUNNER_FILE"
tmux new-session -d -s "$SESSION_NAME" "bash $(printf '%q' "$RUNNER_FILE")"
sleep 2
if ! tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "[error] Session exited immediately; last log lines:" >&2
  tail -n 80 "$LOG_FILE" >&2 || true
  exit 1
fi
echo "Started tmux session: $SESSION_NAME"
echo "Attach: tmux attach -t $SESSION_NAME"
echo "Log:    tail -f $LOG_FILE"
echo "Output: $OUTPUT_DIR"
