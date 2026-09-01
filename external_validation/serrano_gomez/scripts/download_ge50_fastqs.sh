#!/usr/bin/env bash
set -euo pipefail

BASE="/home/Jack/Real_Data/external_validation/serrano_gomez_ibd/ge50_external_validation"
MANIFEST="$BASE/manifests/PRJEB42155_ge50_fastq_manifest.tsv"
URLS="$BASE/manifests/PRJEB42155_ge50_fastq_urls.txt"
FASTQ_DIR="$BASE/fastq"
LOG_DIR="$BASE/logs"

mkdir -p "$FASTQ_DIR" "$LOG_DIR"

echo "[*] Downloading FASTQs"
echo "[*] FASTQ_DIR: $FASTQ_DIR"
echo "[*] URLS: $URLS"

if command -v aria2c >/dev/null 2>&1; then
  echo "[*] Using aria2c"
  aria2c \
    --continue=true \
    --max-concurrent-downloads=4 \
    --split=4 \
    --min-split-size=5M \
    --dir="$FASTQ_DIR" \
    --input-file="$URLS" \
    2>&1 | tee "$LOG_DIR/download_aria2c.log"
else
  echo "[*] aria2c not found; using wget sequentially"
  while IFS= read -r url; do
    [[ -z "$url" ]] && continue
    wget -c -P "$FASTQ_DIR" "$url"
  done < "$URLS" 2>&1 | tee "$LOG_DIR/download_wget.log"
fi

echo "[*] Download complete"
echo "[*] Files:"
find "$FASTQ_DIR" -maxdepth 1 -type f -name "*.fastq.gz" | wc -l
