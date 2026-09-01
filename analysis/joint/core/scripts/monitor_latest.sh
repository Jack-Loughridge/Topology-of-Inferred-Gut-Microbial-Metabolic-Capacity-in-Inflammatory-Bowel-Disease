#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ge 1 ]]; then
  OUT_DIR="$1"
else
  OUT_DIR="$(
    find "$HOME/Real_Data" -maxdepth 1 -mindepth 1 -type d \
      \( -name 'H0_Ricci_JointSparse_ActiveSet_*' -o -name 'H0_Ricci_ActiveSet_Benchmark_*' -o -name 'H0_Ricci_JointSparse_Nested_*' \) \
      -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-
  )"
fi

if [[ -z "${OUT_DIR:-}" || ! -d "$OUT_DIR" ]]; then
  echo "No joint sparse output directory found." >&2
  exit 1
fi

echo "[OUTPUT] $OUT_DIR"
if [[ -f "$OUT_DIR/BENCHMARK_FIRST_INNER_FOLD.json" ]]; then
  cat "$OUT_DIR/BENCHMARK_FIRST_INNER_FOLD.json"
  echo
fi
if [[ -f "$OUT_DIR/ALL_SUMMARIES.csv" ]]; then
  column -s, -t < "$OUT_DIR/ALL_SUMMARIES.csv" 2>/dev/null || cat "$OUT_DIR/ALL_SUMMARIES.csv"
elif [[ -f "$OUT_DIR/run.log" ]]; then
  tail -n 100 "$OUT_DIR/run.log"
else
  echo "No run.log or summary file exists yet."
fi
