#!/usr/bin/env bash
set -euo pipefail

# Launch the repeated H0 Alpha-Pi analysis in a detached tmux session.
# Environment variables may override every path shown below.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REAL_DATA_DIR="${REAL_DATA_DIR:-$HOME/Real_Data}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SESSION_NAME="${SESSION_NAME:-h0_alpha_pi_20x5}"
PD_DIR="${PD_DIR:-$REAL_DATA_DIR/out_pds}"
LABEL_CSV="${LABEL_CSV:-$REAL_DATA_DIR/sample_labels.csv}"
METADATA_CSV="${METADATA_CSV:-$REAL_DATA_DIR/hmp2_metadata.csv}"
SPLIT_MANIFEST="${SPLIT_MANIFEST:-$REAL_DATA_DIR/Ricci_IBD_RepeatedCV_CPath/splits/sample_split_manifest.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-$REAL_DATA_DIR/H0_AlphaPi_RepeatedCV_TrainOnlyBins_IBD}"
LOG_FILE="${LOG_FILE:-$OUTPUT_DIR/full_run.log}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

mkdir -p "$OUTPUT_DIR"

for required in "$SCRIPT_DIR/h0_alpha_pi_repeated_cv.py" "$SCRIPT_DIR/validate_inputs.py" "$SPLIT_MANIFEST"; do
  if [[ ! -e "$required" ]]; then
    echo "[error] Required path not found: $required" >&2
    exit 1
  fi
done

if ! command -v tmux >/dev/null 2>&1; then
  echo "[error] tmux is not installed or is not on PATH." >&2
  exit 1
fi

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "[error] tmux session '$SESSION_NAME' already exists."
  echo "Attach with: tmux attach -t $SESSION_NAME"
  exit 1
fi

# Keep CPU-side numerical libraries conservative while Ricci and benchmark jobs
# may be running concurrently. Alpha-Pi will use CUDA automatically when the
# active PyTorch installation provides it.
RUNNER="$OUTPUT_DIR/run_in_tmux.sh"
cat > "$RUNNER" <<EOF
#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"

{
  echo "================================================================================"
  echo "Repeated H0 Alpha-Pi launch"
  date --iso-8601=seconds || date
  echo "Host: \$(hostname)"
  echo "Python: \$($PYTHON_BIN --version 2>&1)"
  echo "nproc: \$(nproc 2>/dev/null || echo unavailable)"
  free -h 2>/dev/null || true
  nvidia-smi 2>/dev/null || true
  echo "================================================================================"

  "$PYTHON_BIN" "$SCRIPT_DIR/validate_inputs.py" \
    --pd-dir "$PD_DIR" \
    --label-csv "$LABEL_CSV" \
    --metadata-csv "$METADATA_CSV" \
    --split-manifest "$SPLIT_MANIFEST"

  "$PYTHON_BIN" "$SCRIPT_DIR/h0_alpha_pi_repeated_cv.py" \
    --pd-dir "$PD_DIR" \
    --label-csv "$LABEL_CSV" \
    --metadata-csv "$METADATA_CSV" \
    --split-manifest "$SPLIT_MANIFEST" \
    --output-dir "$OUTPUT_DIR" \
    $EXTRA_ARGS

  status=\$?
  echo "================================================================================"
  echo "Finished with status \$status"
  date --iso-8601=seconds || date
  echo "================================================================================"
  exit \$status
} 2>&1 | tee -a "$LOG_FILE"
EOF
chmod +x "$RUNNER"

tmux new-session -d -s "$SESSION_NAME" "bash '$RUNNER'"

echo "Started tmux session: $SESSION_NAME"
echo "Attach:   tmux attach -t $SESSION_NAME"
echo "Detach:   Ctrl-b, then d"
echo "Log:      tail -f '$LOG_FILE'"
echo "Output:   $OUTPUT_DIR"
