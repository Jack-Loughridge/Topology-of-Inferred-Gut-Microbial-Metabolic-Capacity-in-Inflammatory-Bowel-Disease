#!/usr/bin/env bash
set -euo pipefail

REAL_DATA_DIR="${REAL_DATA_DIR:-$HOME/Real_Data}"
SESSION_NAME="${SESSION_NAME:-h0_ricci_joint_all_tasks_20x5}"
OUTPUT_DIR="${OUTPUT_DIR:-$REAL_DATA_DIR/H0_Ricci_JointSparse_RepeatedCV_AllTasks}"
LOG_FILE="${LOG_FILE:-$REAL_DATA_DIR/logs/h0_ricci_joint_all_tasks_20x5.log}"

if command -v tmux >/dev/null 2>&1 && tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "tmux: RUNNING ($SESSION_NAME)"
else
  echo "tmux: not running ($SESSION_NAME)"
fi

echo "output: $OUTPUT_DIR"
if [[ -f "$OUTPUT_DIR/progress.json" ]]; then
  echo
  cat "$OUTPUT_DIR/progress.json"
else
  echo "progress.json: not created yet"
fi

echo
FOLD_COUNT=0
if [[ -d "$OUTPUT_DIR/runs" ]]; then
  FOLD_COUNT="$(find "$OUTPUT_DIR/runs" -path '*/fold_*/COMPLETE.json' -type f | wc -l | tr -d ' ')"
fi
echo "completed outer-fold markers: $FOLD_COUNT / 500"

if [[ -f "$LOG_FILE" ]]; then
  echo
  echo "last 25 log lines:"
  tail -25 "$LOG_FILE"
else
  echo "log: not created yet ($LOG_FILE)"
fi
