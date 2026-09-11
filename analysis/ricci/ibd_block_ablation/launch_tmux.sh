#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SESSION_NAME="${SESSION_NAME:-ricci_ibd_block_ablation}"
NICE_LEVEL="${NICE_LEVEL:-10}"
LOG_DIR="${LOG_DIR:-${HOME}/Real_Data/logs}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/ricci_ibd_feature_block_ablation.log}"

mkdir -p "${LOG_DIR}"

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION_NAME}" >&2
  exit 1
fi

COMMAND=$(printf \
  'cd %q && export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONHASHSEED=0 && exec nice -n %q %q %q 2>&1 | tee %q' \
  "${SCRIPT_DIR}" \
  "${NICE_LEVEL}" \
  "${PYTHON_BIN}" \
  "${SCRIPT_DIR}/run_ibd_block_ablation.py" \
  "${LOG_FILE}")

tmux new-session -d -s "${SESSION_NAME}" "bash -lc ${COMMAND@Q}"
echo "Started tmux session: ${SESSION_NAME}"
echo "Log: ${LOG_FILE}"
echo "Attach: tmux attach -t ${SESSION_NAME}"

