#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BASE="$HOME/Real_Data"
SESSION="joint_ibd_regularised_v8"
mkdir -p "$BASE/logs"
if pgrep -af '[j]oint_.*(v4|v5|v6|v7)/(run.py|run_fixed.py)' >/dev/null; then
    echo 'An earlier model runner is active. Stop it before launching v8.' >&2
    exit 2
fi
if pgrep -af '[j]oint_ibd_regularised_v8/run.py' >/dev/null; then
    echo 'A v8 runner is already active. No duplicate launched.'
    exit 0
fi
if command -v tmux >/dev/null 2>&1 && tmux has-session -t "$SESSION" 2>/dev/null; then
    echo 'The v8 tmux session already exists. Check its status rather than launching twice.'
    exit 0
fi
if command -v tmux >/dev/null 2>&1; then
    printf -v CMD 'bash %q' "$HERE/run_with_log.sh"
    tmux new-session -d -s "$SESSION" "$CMD"
    echo "Started $SESSION. Check $BASE/logs/joint_ibd_regularised_v8.log"
else
    nohup bash "$HERE/run_with_log.sh" >/dev/null 2>&1 < /dev/null &
    echo "Started PID $!. Check $BASE/logs/joint_ibd_regularised_v8.log"
fi
