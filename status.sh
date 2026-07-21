#!/usr/bin/env bash
set -euo pipefail

BASE="${REAL_DATA_DIR:-$HOME/Real_Data}"
OUTPUT="$BASE/H0_AlphaPi_1x5_AllTasks"
IBD_SOURCE="$BASE/H0_AlphaPi_RepeatedCV_TrainOnlyBins_IBD"
LOG="$BASE/logs/h0_alpha_pi_all_tasks_1x5.log"
SESSION="h0_alpha_pi_all_tasks_1x5"

if tmux has-session -t "$SESSION" 2>/dev/null; then
    session_status="running"
else
    session_status="not running"
fi

echo "Session: $session_status"
echo

ibd=0
for fold in 1 2 3 4 5; do
    [[ -f "$IBD_SOURCE/folds/repeat_01_fold_${fold}/FOLD_COMPLETE.json" ]] && ibd=$((ibd+1))
done
printf '%-30s %s/5\n' "IBD_vs_nonIBD" "$ibd"

total="$ibd"
for task in three_way_nonIBD_UC_CD nonIBD_vs_UC nonIBD_vs_CD CD_vs_UC; do
    count=0
    for fold in 1 2 3 4 5; do
        [[ -f "$OUTPUT/$task/folds/repeat_01_fold_${fold}/FOLD_COMPLETE.json" ]] && count=$((count+1))
    done
    printf '%-30s %s/5\n' "$task" "$count"
    total=$((total+count))
done

echo
echo "Completed outer folds: $total/25"
[[ -f "$OUTPUT/RUN_COMPLETE.json" ]] && echo "RUN_COMPLETE.json: present" || echo "RUN_COMPLETE.json: absent"

if [[ -f "$LOG" ]]; then
    echo
    echo "Recent log:"
    tail -n 30 "$LOG"
fi
