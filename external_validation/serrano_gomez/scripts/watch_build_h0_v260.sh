#!/usr/bin/env bash
set -u
GE50="$HOME/Real_Data/external_validation/serrano_gomez_ibd/ge50_external_validation"
SCRIPT="$GE50/scripts/stream_build_h0_v260.py"
LOG="$GE50/structural_frozen_ibdmdb/logs/build_h0_v260_stream.log"
mkdir -p "$(dirname "$LOG")"
while true; do
  echo
  echo "============================================================"
  echo "[BUILD/H0 WATCH] $(date)"
  echo "============================================================"
  nice -n 10 python3 -u "$SCRIPT"
  rc=$?
  echo "[BUILD/H0 WATCH] pass exit=$rc; sleeping 300s"
  sleep 300
done
