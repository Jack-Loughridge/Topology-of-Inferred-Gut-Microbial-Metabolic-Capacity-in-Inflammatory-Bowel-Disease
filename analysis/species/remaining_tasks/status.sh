#!/usr/bin/env bash
set -euo pipefail

REAL_DATA_DIR="${REAL_DATA_DIR:-$HOME/Real_Data}"
OUTPUT_DIR="${OUTPUT_DIR:-$REAL_DATA_DIR/Species_Benchmarks_RepeatedCV_RemainingTasks}"
SESSION_NAME="${SESSION_NAME:-species_benchmarks_remaining_20x5}"
LOG_FILE="${LOG_FILE:-$REAL_DATA_DIR/logs/species_benchmarks_remaining_20x5.log}"

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "TMUX: running ($SESSION_NAME)"
else
  echo "TMUX: not running"
fi

if [[ -f "$OUTPUT_DIR/progress.json" ]]; then
  cat "$OUTPUT_DIR/progress.json"
else
  count=0
  if [[ -d "$OUTPUT_DIR" ]]; then
    count=$(find "$OUTPUT_DIR" -name FOLD_COMPLETE.json | wc -l)
  fi
  echo "Completed fold markers: $count / 1200"
fi

for task in three_way_nonIBD_UC_CD nonIBD_vs_UC nonIBD_vs_CD CD_vs_UC; do
  if [[ -f "$OUTPUT_DIR/$task/RUN_COMPLETE.json" ]]; then
    echo "$task: COMPLETE"
  else
    echo "$task: pending/running"
  fi
done

echo "Log: $LOG_FILE"
