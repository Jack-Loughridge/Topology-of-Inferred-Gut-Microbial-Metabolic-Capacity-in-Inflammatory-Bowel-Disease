#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/Real_Data/h0_alpha_pi_all_tasks_1x5}"
LOG_DIR="${LOG_DIR:-$HOME/Real_Data/logs}"
SESSION="${SESSION:-h0_alpha_pi_all_tasks_1x5}"
RUNNER="$LOG_DIR/${SESSION}_runner.sh"
LOG_FILE="$LOG_DIR/${SESSION}.log"
NICE_LEVEL="${NICE_LEVEL:-10}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

mkdir -p "$LOG_DIR"

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "Session already exists: $SESSION"
    echo "Attach: tmux attach -t $SESSION"
    exit 0
fi

if pgrep -af '[h]0_alpha_pi_repeated_cv.py' >/dev/null 2>&1; then
    echo "ERROR: an existing standalone H0 repeated-CV process is already running."
    pgrep -af '[h]0_alpha_pi_repeated_cv.py' || true
    exit 1
fi

cat > "$RUNNER" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "$REPO_DIR"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
exec > >(tee -a "$LOG_FILE") 2>&1

echo "================================================================"
echo "H0 ALPHA-PI: ONE LOCKED REPETITION x FIVE FOLDS, ALL FIVE TASKS"
echo "Started UTC: \$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "================================================================"

nice -n "$NICE_LEVEL" "$PYTHON_BIN" -u run_all_tasks.py

echo "================================================================"
echo "Finished UTC: \$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "================================================================"
EOF
chmod 700 "$RUNNER"
bash -n "$RUNNER"

tmux new-session -d -s "$SESSION" "bash '$RUNNER'"
sleep 5
if ! tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "ERROR: tmux session exited during startup. Recent log:"
    tail -n 120 "$LOG_FILE" 2>/dev/null || true
    exit 1
fi

echo "Started tmux session: $SESSION"
echo "Attach: tmux attach -t $SESSION"
echo "Log:    tail -f $LOG_FILE"
echo "Status: bash $REPO_DIR/status.sh"
echo "Output: $HOME/Real_Data/H0_AlphaPi_1x5_AllTasks"
