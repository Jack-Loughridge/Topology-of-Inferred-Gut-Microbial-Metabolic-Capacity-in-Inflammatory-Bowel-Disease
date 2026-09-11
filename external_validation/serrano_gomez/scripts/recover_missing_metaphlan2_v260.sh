#!/usr/bin/env bash
set -euo pipefail

GE50="$HOME/Real_Data/external_validation/serrano_gomez_ibd/ge50_external_validation"
MANIFEST="$GE50/manifests/PRJEB42155_ge50_run_manifest.tsv"
FASTQ="$GE50/fastq"
CLEAN="$GE50/clean_fastq_for_humann"
OUT="$GE50/metaphlan2_v260_profiles"
RUNNER="$GE50/scripts/run_metaphlan2_v260_incremental.sh"
LOG="$GE50/logs/recover_missing_metaphlan2_v260.log"

mkdir -p "$FASTQ" "$CLEAN" "$OUT" "$GE50/logs"

exec > >(tee -a "$LOG") 2>&1

echo "============================================================"
echo "GE50 MetaPhlAn2 v2.6.0 missing-profile recovery"
echo "Started: $(date)"
echo "============================================================"

python3 - "$MANIFEST" "$OUT" > "$GE50/manifests/metaphlan2_v260_missing.tsv" <<'PY'
import sys
from pathlib import Path
import pandas as pd

manifest = Path(sys.argv[1])
outdir = Path(sys.argv[2])

df = pd.read_csv(manifest, sep="\t", dtype=str)

required = [
    "run_accession",
    "R1_url", "R2_url",
    "R1_md5", "R2_md5",
    "R1_bytes", "R2_bytes",
]
missing_cols = [c for c in required if c not in df.columns]
if missing_cols:
    raise SystemExit(f"Manifest missing columns: {missing_cols}")

rows = []

for _, r in df.iterrows():
    run = r["run_accession"]
    profile = outdir / f"{run}_profile.tsv"
    marker = outdir / f"{run}.COMPLETE"

    if profile.is_file() and profile.stat().st_size > 0 and marker.is_file():
        continue

    rows.append(r[required].to_dict())

pd.DataFrame(rows, columns=required).to_csv(
    sys.stdout, sep="\t", index=False
)

print(
    f"[INFO] Missing valid MetaPhlAn profiles: {len(rows)}",
    file=sys.stderr
)
PY

MISSING="$GE50/manifests/metaphlan2_v260_missing.tsv"

N_MISSING=$(( $(wc -l < "$MISSING") - 1 ))

echo "[INFO] Profiles requiring recovery: $N_MISSING"

if [[ "$N_MISSING" -eq 0 ]]; then
    echo "[✓] Nothing to do."
    exit 0
fi

download_file () {
    local url="$1"
    local dest="$2"

    rm -f "${dest}.part"

    echo "[DOWNLOAD] $url"

    # Try HTTPS version of ENA FTP URL first.
    https_url="${url/ftp:\/\/ftp.sra.ebi.ac.uk/https:\/\/ftp.sra.ebi.ac.uk}"

    if wget \
        --tries=5 \
        --timeout=60 \
        --continue \
        -O "${dest}.part" \
        "$https_url"
    then
        mv "${dest}.part" "$dest"
        return 0
    fi

    echo "[WARN] HTTPS download failed; trying original manifest URL"
    rm -f "${dest}.part"

    if wget \
        --tries=5 \
        --timeout=60 \
        --continue \
        -O "${dest}.part" \
        "$url"
    then
        mv "${dest}.part" "$dest"
        return 0
    fi

    rm -f "${dest}.part"
    return 1
}

verify_file () {
    local file="$1"
    local expected_md5="$2"
    local expected_bytes="$3"

    if [[ ! -s "$file" ]]; then
        echo "[ERROR] Missing/empty file: $file"
        return 1
    fi

    actual_bytes=$(stat -c '%s' "$file")

    if [[ -n "$expected_bytes" && "$expected_bytes" != "nan" ]]; then
        if [[ "$actual_bytes" != "$expected_bytes" ]]; then
            echo "[ERROR] Size mismatch:"
            echo "        file=$file"
            echo "        expected=$expected_bytes"
            echo "        actual=$actual_bytes"
            return 1
        fi
    fi

    actual_md5=$(md5sum "$file" | awk '{print $1}')

    if [[ -n "$expected_md5" && "$expected_md5" != "nan" ]]; then
        if [[ "$actual_md5" != "$expected_md5" ]]; then
            echo "[ERROR] MD5 mismatch:"
            echo "        file=$file"
            echo "        expected=$expected_md5"
            echo "        actual=$actual_md5"
            return 1
        fi
    fi

    echo "[VERIFY OK] $(basename "$file")"
}

SUCCESS=0
FAILED=0
INDEX=0

tail -n +2 "$MISSING" |
while IFS=$'\t' read -r RUN R1_URL R2_URL R1_MD5 R2_MD5 R1_BYTES R2_BYTES
do
    INDEX=$((INDEX + 1))

    echo
    echo "################################################################"
    echo "[$INDEX/$N_MISSING] $RUN"
    echo "################################################################"

    PROFILE="$OUT/${RUN}_profile.tsv"
    MARKER="$OUT/${RUN}.COMPLETE"

    if [[ -s "$PROFILE" && -f "$MARKER" ]]; then
        echo "[SKIP] Already complete"
        continue
    fi

    R1="$FASTQ/${RUN}_1.fastq.gz"
    R2="$FASTQ/${RUN}_2.fastq.gz"
    COMBINED="$CLEAN/${RUN}_clean_combined.fastq.gz"

    rm -f "$PROFILE" "$MARKER" "$COMBINED"

    sample_ok=1

    # ---------------------------------------------------------
    # R1
    # ---------------------------------------------------------
    if [[ -s "$R1" ]]; then
        echo "[INFO] Existing R1 found; verifying"
        if ! verify_file "$R1" "$R1_MD5" "$R1_BYTES"; then
            echo "[WARN] Existing R1 invalid; redownloading"
            rm -f "$R1"
        fi
    fi

    if [[ ! -s "$R1" ]]; then
        if ! download_file "$R1_URL" "$R1"; then
            echo "[ERROR] R1 download failed"
            sample_ok=0
        elif ! verify_file "$R1" "$R1_MD5" "$R1_BYTES"; then
            echo "[ERROR] R1 verification failed"
            sample_ok=0
        fi
    fi

    # ---------------------------------------------------------
    # R2
    # ---------------------------------------------------------
    if [[ "$sample_ok" -eq 1 ]]; then
        if [[ -s "$R2" ]]; then
            echo "[INFO] Existing R2 found; verifying"
            if ! verify_file "$R2" "$R2_MD5" "$R2_BYTES"; then
                echo "[WARN] Existing R2 invalid; redownloading"
                rm -f "$R2"
            fi
        fi

        if [[ ! -s "$R2" ]]; then
            if ! download_file "$R2_URL" "$R2"; then
                echo "[ERROR] R2 download failed"
                sample_ok=0
            elif ! verify_file "$R2" "$R2_MD5" "$R2_BYTES"; then
                echo "[ERROR] R2 verification failed"
                sample_ok=0
            fi
        fi
    fi

    # ---------------------------------------------------------
    # Build EXACT same input used by SKIP_KNEADDATA=1 pipeline
    # ---------------------------------------------------------
    if [[ "$sample_ok" -eq 1 ]]; then
        echo "[BUILD] Concatenating R1 + R2"
        cat "$R1" "$R2" > "$COMBINED"

        if [[ ! -s "$COMBINED" ]]; then
            echo "[ERROR] Combined FASTQ is empty"
            sample_ok=0
        fi
    fi

    # ---------------------------------------------------------
    # Run existing canonical MetaPhlAn2 v2.6 runner
    # ---------------------------------------------------------
    if [[ "$sample_ok" -eq 1 ]]; then
        echo "[METAPHLAN] Running canonical existing runner"

        if ! bash "$RUNNER"; then
            echo "[ERROR] MetaPhlAn runner returned non-zero"
            sample_ok=0
        fi
    fi

    # ---------------------------------------------------------
    # Validate output
    # ---------------------------------------------------------
    if [[ "$sample_ok" -eq 1 ]]; then
        if [[ -s "$PROFILE" && -f "$MARKER" ]]; then
            echo "[SUCCESS] $RUN"
        else
            echo "[ERROR] Missing valid profile/COMPLETE marker"
            sample_ok=0
        fi
    fi

    # Combined file is always disposable.
    rm -f "$COMBINED"

    if [[ "$sample_ok" -eq 1 ]]; then
        # These raw files were downloaded/recovered solely for this step.
        # Delete after confirmed successful MetaPhlAn completion.
        rm -f "$R1" "$R2"
        echo "[CLEANUP] Removed temporary R1/R2"
    else
        echo "[FAILED] $RUN"
        echo "[INFO] Retaining any downloaded R1/R2 for diagnosis/retry."
    fi

    echo "[DISK]"
    df -h "$GE50" | tail -1

done

echo
echo "============================================================"
echo "RECOVERY PASS FINISHED: $(date)"
echo "============================================================"

N_PROFILE=$(find "$OUT" -maxdepth 1 -name '*_profile.tsv' -size +0c | wc -l)
N_COMPLETE=$(find "$OUT" -maxdepth 1 -name '*.COMPLETE' | wc -l)

echo "Non-empty profiles: $N_PROFILE"
echo "COMPLETE markers:   $N_COMPLETE"
echo "Expected cohort:    184"

if [[ "$N_PROFILE" -eq 184 && "$N_COMPLETE" -eq 184 ]]; then
    echo "[✓] ALL 184 METAPHLAN2 v2.6 PROFILES COMPLETE"
else
    echo "[!] Cohort not yet complete."
fi
