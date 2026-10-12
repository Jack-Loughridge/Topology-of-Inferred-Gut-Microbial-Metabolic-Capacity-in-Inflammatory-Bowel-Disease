#!/usr/bin/env bash
set -u
export PATH="$HOME/micromamba/envs/humann_py312/bin:$PATH"

GE50="$HOME/Real_Data/external_validation/serrano_gomez_ibd/ge50_external_validation"
MPA="$HOME/Real_Data/tools/MetaPhlAn2/metaphlan2.py"
DB="$HOME/Real_Data/tools/MetaPhlAn2/db_v20/mpa_v20_m200"

INPUT="$GE50/clean_fastq_for_humann"
OUT="$GE50/metaphlan2_v260_profiles"

mkdir -p "$OUT"

echo "============================================================"
echo "GE50 MetaPhlAn2 v2.6.0"
echo "Start: $(date)"
echo "============================================================"

TOTAL=0
DONE=0
SKIP=0
FAIL=0

for fq in "$INPUT"/*_clean_combined.fastq.gz; do
    [ -e "$fq" ] || continue

    sample="$(basename "$fq" _clean_combined.fastq.gz)"

    profile="$OUT/${sample}_profile.tsv"
    bowtie="$OUT/${sample}.bowtie2.bz2"
    marker="$OUT/${sample}.COMPLETE"

    TOTAL=$((TOTAL+1))

    echo
    echo "============================================================"
    echo "[*] $sample"
    echo "============================================================"

    if [ -s "$profile" ] && [ -f "$marker" ]; then
        echo "[SKIP] complete"
        SKIP=$((SKIP+1))
        continue
    fi

    rm -f "$profile" "$marker"

    if zcat "$fq" | \
       nice -n 10 python3 "$MPA" \
          --input_type fastq \
          --bowtie2db "$DB" \
          --mpa_pkl "${DB}.pkl" \
          --bowtie2out "$bowtie" \
          --nproc 1 \
          > "$profile"
    then
        touch "$marker"
        DONE=$((DONE+1))
        echo "[OK] $sample"
    else
        FAIL=$((FAIL+1))
        echo "[FAILED] $sample"
        rm -f "$profile"
    fi
done

echo
echo "============================================================"
echo "FINISHED: $(date)"
echo "Seen:     $TOTAL"
echo "New OK:   $DONE"
echo "Skipped:  $SKIP"
echo "Failed:   $FAIL"
echo "============================================================"
