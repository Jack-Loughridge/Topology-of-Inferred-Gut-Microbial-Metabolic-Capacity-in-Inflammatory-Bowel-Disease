#!/usr/bin/env bash
set -euo pipefail

REAL_DATA_DIR="${REAL_DATA_DIR:-$HOME/Real_Data}"
GE50="$REAL_DATA_DIR/external_validation/serrano_gomez_ibd/ge50_external_validation"
OUT_DIR="${OUT_DIR:-$GE50/structural_frozen_ibdmdb}"
GRAPH_DIR="$OUT_DIR/graphs"
RICCI_OUT="$OUT_DIR/ricci_raw"
LOG_DIR="$OUT_DIR/logs"
SESSION_NAME="${SESSION_NAME:-ge50_external_ricci_frozen_ibdmdb}"

mkdir -p "$RICCI_OUT" "$LOG_DIR"

RICCI_SOURCE="${RICCI_SOURCE:-}"
if [[ -z "$RICCI_SOURCE" ]]; then
  RICCI_SOURCE="$(find "$REAL_DATA_DIR" -type f -name '14_compute_ricci_faithful_pairwise_active.py' -print -quit)"
fi
if [[ -z "$RICCI_SOURCE" || ! -f "$RICCI_SOURCE" ]]; then
  echo "[ERROR] Could not find 14_compute_ricci_faithful_pairwise_active.py under $REAL_DATA_DIR" >&2
  exit 1
fi

if [[ ! -f "$OUT_DIR/provenance/PREFLIGHT_PASSED.json" ]]; then
  echo "[ERROR] Strict external preflight has not passed. Run external_structural_validation.py preflight first." >&2
  exit 1
fi

LOG="$LOG_DIR/ricci_n250.log"
CMD="$LOG_DIR/ricci_n250_command.sh"

cat > "$CMD" <<EOF
#!/usr/bin/env bash
set -euo pipefail
python3 -u "$RICCI_SOURCE" \\
  --base-dir "$REAL_DATA_DIR" \\
  --graph-dir "$GRAPH_DIR" \\
  --output-dir "$RICCI_OUT" \\
  --n-paths 250 \\
  --max-steps 10000 \\
  --seed 13 \\
  --beta 1.4 \\
  --epsilon-dist 0.0001 \\
  --c-single-out 0.001 \\
  --mu-prune-threshold 0.000001 \\
  --edge-weight-attr weight \\
  --active-tol 0.000000000001 \\
  --topk-support 150 \\
  --max-lp-vars 200000 \\
  --progress-every 250 \\
  2>&1 | tee -a "$LOG"
EOF
chmod +x "$CMD"

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "[ERROR] tmux session '$SESSION_NAME' already exists." >&2
  echo "Attach: tmux attach -t $SESSION_NAME" >&2
  exit 1
fi

tmux new-session -d -s "$SESSION_NAME" "bash '$CMD'"

echo "[STARTED] exact faithful Ricci n=250"
echo "Session: $SESSION_NAME"
echo "Log:     $LOG"
echo "Output:  $RICCI_OUT"
echo "Source:  $RICCI_SOURCE"
echo
echo "Attach:  tmux attach -t $SESSION_NAME"
echo "Monitor: tail -f '$LOG'"
