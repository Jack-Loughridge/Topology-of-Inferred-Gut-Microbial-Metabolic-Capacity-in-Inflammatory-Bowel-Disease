#!/usr/bin/env bash
set -euo pipefail
REAL_DATA_DIR="${REAL_DATA_DIR:-$HOME/Real_Data}"
OUTPUT_DIR="${OUTPUT_DIR:-$REAL_DATA_DIR/Ricci_RepeatedCV_AllTasks}"
SESSION_NAME="${SESSION_NAME:-ricci_all_tasks_20x5}"
LOG_FILE="${LOG_FILE:-$REAL_DATA_DIR/logs/ricci_all_tasks_20x5.log}"
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then echo "TMUX: running ($SESSION_NAME)"; else echo "TMUX: not running"; fi
[[ -f "$OUTPUT_DIR/progress.json" ]] && cat "$OUTPUT_DIR/progress.json" || echo "No progress.json yet"
echo "Log: $LOG_FILE"
