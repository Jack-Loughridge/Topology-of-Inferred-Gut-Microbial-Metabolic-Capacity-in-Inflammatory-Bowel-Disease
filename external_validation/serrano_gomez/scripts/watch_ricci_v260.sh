#!/usr/bin/env bash
set -u
REAL="$HOME/Real_Data"
GE50="$REAL/external_validation/serrano_gomez_ibd/ge50_external_validation"
OUT="$GE50/structural_frozen_ibdmdb"
GRAPH_DIR="$OUT/graphs"
RICCI_OUT="$OUT/ricci_raw"
LOG_DIR="$OUT/logs"
ESV="$GE50/scripts/serrano_gomez_external_structural_frozen_ibdmdb_v1/external_structural_validation.py"
RICCI="$REAL/14_compute_ricci_faithful_pairwise_active.py"
mkdir -p "$RICCI_OUT" "$LOG_DIR"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
while true; do
  graphs=$(find "$GRAPH_DIR" -maxdepth 1 -type f -name '*.graphml' 2>/dev/null | wc -l)
  done_n=$(find "$RICCI_OUT" -maxdepth 1 -type f -name '*.FAITHFUL_ACTIVE_DONE.marker' 2>/dev/null | wc -l)
  echo
  echo "============================================================"
  echo "[RICCI WATCH] $(date) graphs=$graphs done=$done_n"
  echo "============================================================"
  if [ "$graphs" -gt "$done_n" ]; then
    nice -n 10 python3 -u "$RICCI" \
      --base-dir "$REAL" \
      --graph-dir "$GRAPH_DIR" \
      --output-dir "$RICCI_OUT" \
      --n-paths 250 \
      --max-steps 10000 \
      --seed 13 \
      --beta 1.4 \
      --epsilon-dist 0.0001 \
      --c-single-out 0.001 \
      --mu-prune-threshold 0.000001 \
      --edge-weight-attr weight \
      --active-tol 0.000000000001 \
      --topk-support 150 \
      --max-lp-vars 200000 \
      --progress-every 250
    rc=$?
    echo "[RICCI WATCH] Ricci pass exit=$rc"
    done_n=$(find "$RICCI_OUT" -maxdepth 1 -type f -name '*.FAITHFUL_ACTIVE_DONE.marker' 2>/dev/null | wc -l)
    if [ "$done_n" -gt 0 ]; then
      python3 -u "$ESV" --out-dir "$OUT" vectorize || echo "[RICCI WATCH] vectorize returned nonzero; will retry next pass"
    fi
  else
    echo "[RICCI WATCH] no pending graphs"
  fi
  echo "[RICCI WATCH] sleeping 300s"
  sleep 300
done
