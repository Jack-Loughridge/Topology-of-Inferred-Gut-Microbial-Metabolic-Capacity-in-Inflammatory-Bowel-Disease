#!/usr/bin/env bash
set -euo pipefail
REAL_DATA_DIR="${REAL_DATA_DIR:-$HOME/Real_Data}"
OUTPUT_DIR="${OUTPUT_DIR:-$REAL_DATA_DIR/Species_Benchmarks_RepeatedCV_AllTasks}"
SESSION_NAME="${SESSION_NAME:-species_benchmarks_all_tasks_20x5}"
LOG_FILE="${LOG_FILE:-$REAL_DATA_DIR/logs/species_benchmarks_all_tasks_20x5.log}"
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then echo "TMUX: running ($SESSION_NAME)"; else echo "TMUX: not running"; fi
if [[ -f "$OUTPUT_DIR/progress.json" ]]; then cat "$OUTPUT_DIR/progress.json"; else
  count=0; [[ -d "$OUTPUT_DIR" ]] && count=$(find "$OUTPUT_DIR" -name FOLD_COMPLETE.json | wc -l)
  echo "Completed model-fold markers: $count / 1500"
fi
for task in IBD_vs_nonIBD three_way_nonIBD_UC_CD nonIBD_vs_UC nonIBD_vs_CD CD_vs_UC; do
  [[ -f "$OUTPUT_DIR/$task/RUN_COMPLETE.json" ]] && state=COMPLETE || state=pending/running
  echo "$task: $state"
done
echo "Log: $LOG_FILE"
