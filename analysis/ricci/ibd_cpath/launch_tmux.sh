#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SESSION="${SESSION:-ricci_ibd_cpath}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
FEATURE_DIR="${FEATURE_DIR:-$HOME/Real_Data/Ricci_Classifier_Faithful_Eps0001_n250_v3}"
OUTPUT_DIR="${OUTPUT_DIR:-$HOME/Real_Data/Ricci_IBD_RepeatedCV_CPath}"
ANNOTATION_CSV="${ANNOTATION_CSV:-}"
MAX_ITER="${MAX_ITER:-10000}"
TOL="${TOL:-1e-4}"
N_JOBS="${N_JOBS:-1}"

mkdir -p "$OUTPUT_DIR"
LOG_FILE="$OUTPUT_DIR/full_run.log"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "A tmux session named '$SESSION' already exists."
  echo "Attach with: tmux attach -t $SESSION"
  exit 1
fi

CMD=(
  "$PYTHON_BIN" "$SCRIPT_DIR/repeated_ricci_ibd_cpath.py"
  --feature-dir "$FEATURE_DIR"
  --output-dir "$OUTPUT_DIR"
  --c-values 0.005 0.01 0.02 0.05 0.1
  --n-repeats 20
  --n-splits 5
  --split-seed 13
  --model-seed 20260717
  --max-iter "$MAX_ITER"
  --tol "$TOL"
  --n-jobs "$N_JOBS"
  --resume
)

if [[ -n "$ANNOTATION_CSV" ]]; then
  CMD+=(--edge-annotation-csv "$ANNOTATION_CSV")
fi

printf -v CMD_QUOTED '%q ' "${CMD[@]}"
printf -v WORKDIR_QUOTED '%q' "$SCRIPT_DIR"
printf -v LOG_QUOTED '%q' "$LOG_FILE"

TMUX_COMMAND="cd $WORKDIR_QUOTED && $CMD_QUOTED 2>&1 | tee -a $LOG_QUOTED"
tmux new-session -d -s "$SESSION" "$TMUX_COMMAND"

echo "Started tmux session: $SESSION"
echo "Output directory: $OUTPUT_DIR"
echo "Log: $LOG_FILE"
echo
echo "Attach: tmux attach -t $SESSION"
echo "Follow log: tail -f '$LOG_FILE'"
echo "Detach from tmux: Ctrl-b, then d"
