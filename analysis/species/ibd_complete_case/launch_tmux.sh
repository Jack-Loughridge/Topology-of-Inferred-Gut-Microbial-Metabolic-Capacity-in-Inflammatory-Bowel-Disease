#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REAL_DATA_DIR="${REAL_DATA_DIR:-$HOME/Real_Data}"
SESSION_NAME="${SESSION_NAME:-species_benchmarks_20x5}"
OUTPUT_DIR="${OUTPUT_DIR:-$REAL_DATA_DIR/Species_Benchmarks_RepeatedCV_IBD_CompleteCase}"
SPECIES_FILE="${SPECIES_FILE:-$REAL_DATA_DIR/Real_Species_Abundances_canon.xlsx}"
SPECIES_SHEET="${SPECIES_SHEET:-Sheet1}"
LABEL_CSV="${LABEL_CSV:-$REAL_DATA_DIR/sample_labels.csv}"
METADATA_CSV="${METADATA_CSV:-$REAL_DATA_DIR/hmp2_metadata.csv}"
SPLIT_MANIFEST="${SPLIT_MANIFEST:-$REAL_DATA_DIR/Ricci_IBD_RepeatedCV_CPath/splits/sample_split_manifest.csv}"
BENCHMARK_N_JOBS="${BENCHMARK_N_JOBS:-2}"
NICE_LEVEL="${NICE_LEVEL:-5}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MODELS="${MODELS:-logistic_l1,random_forest,xgboost}"
ZERO_PROFILE_POLICY="${ZERO_PROFILE_POLICY:-exclude}"
EXPECTED_ZERO_PROFILES="${EXPECTED_ZERO_PROFILES:-10}"

if ! command -v tmux >/dev/null 2>&1; then
  echo "[error] tmux is not installed or not on PATH." >&2
  exit 1
fi

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "[error] tmux session '$SESSION_NAME' already exists." >&2
  echo "Attach with: tmux attach -t $SESSION_NAME" >&2
  exit 1
fi

LOG_DIR="${LOG_DIR:-$REAL_DATA_DIR/logs}"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/species_benchmarks_20x5.log"

printf -v CMD '%q ' \
  "$PYTHON_BIN" "$SCRIPT_DIR/species_benchmarks_repeated_cv.py" \
  --species-file "$SPECIES_FILE" \
  --species-sheet "$SPECIES_SHEET" \
  --label-csv "$LABEL_CSV" \
  --metadata-csv "$METADATA_CSV" \
  --split-manifest "$SPLIT_MANIFEST" \
  --output-dir "$OUTPUT_DIR" \
  --models "$MODELS" \
  --expected-repeats 20 \
  --expected-folds 5 \
  --zero-profile-policy "$ZERO_PROFILE_POLICY" \
  --expected-zero-profiles "$EXPECTED_ZERO_PROFILES" \
  --n-jobs "$BENCHMARK_N_JOBS"

RUNNER=$(cat <<EOF
set -euo pipefail
cd $(printf '%q' "$SCRIPT_DIR")
export OMP_NUM_THREADS=$(printf '%q' "$BENCHMARK_N_JOBS")
export MKL_NUM_THREADS=$(printf '%q' "$BENCHMARK_N_JOBS")
export OPENBLAS_NUM_THREADS=$(printf '%q' "$BENCHMARK_N_JOBS")
export NUMEXPR_NUM_THREADS=$(printf '%q' "$BENCHMARK_N_JOBS")
{
  echo '=================================================================='
  echo 'SPECIES BENCHMARK 20x5 RUN'
  echo 'Started UTC:' "\$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo 'Host:' "\$(hostname)"
  echo 'Working directory:' "\$(pwd)"
  echo 'nproc:' "\$(nproc)"
  echo 'Threads per model:' $(printf '%q' "$BENCHMARK_N_JOBS")
  echo 'nice level:' $(printf '%q' "$NICE_LEVEL")
  echo 'Models:' $(printf '%q' "$MODELS")
  echo 'Zero-profile policy:' $(printf '%q' "$ZERO_PROFILE_POLICY")
  echo 'Expected zero profiles:' $(printf '%q' "$EXPECTED_ZERO_PROFILES")
  echo 'Memory:'
  free -h || true
  echo 'Python:'
  $(printf '%q' "$PYTHON_BIN") --version
  echo '=================================================================='
  nice -n $(printf '%q' "$NICE_LEVEL") $CMD
  echo 'Completed UTC:' "\$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} 2>&1 | tee -a $(printf '%q' "$LOG_FILE")
EOF
)

tmux new-session -d -s "$SESSION_NAME" "bash -lc $(printf '%q' "$RUNNER")"

echo "Started tmux session: $SESSION_NAME"
echo "Attach: tmux attach -t $SESSION_NAME"
echo "Log:    tail -f $LOG_FILE"
echo "Output: $OUTPUT_DIR"
