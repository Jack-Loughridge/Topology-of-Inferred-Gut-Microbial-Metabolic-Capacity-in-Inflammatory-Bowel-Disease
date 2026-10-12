#!/usr/bin/env bash
set -u

GE50="$HOME/Real_Data/external_validation/serrano_gomez_ibd/ge50_external_validation"
RUNNER="$GE50/scripts/run_metaphlan2_v260_incremental.sh"

echo "============================================================"
echo "GE50 MetaPhlAn2 watcher"
echo "Started: $(date)"
echo "Rescan interval: 5 minutes"
echo "============================================================"

while true; do
    echo
    echo "[WATCH] Scan started: $(date)"

    bash "$RUNNER"

    n_inputs=$(find "$GE50/clean_fastq_for_humann" \
        -name '*_clean_combined.fastq.gz' | wc -l)

    n_done=$(find "$GE50/metaphlan2_v260_profiles" \
        -name '*.COMPLETE' | wc -l)

    echo "[WATCH] Inputs currently present: $n_inputs"
    echo "[WATCH] MetaPhlAn2 complete:       $n_done"
    echo "[WATCH] Sleeping 300 seconds..."

    sleep 300
done
