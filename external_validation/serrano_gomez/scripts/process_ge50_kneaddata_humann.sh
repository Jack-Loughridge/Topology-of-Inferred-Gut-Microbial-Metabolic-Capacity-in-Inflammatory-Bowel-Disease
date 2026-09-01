#!/usr/bin/env bash
set -euo pipefail

BASE="/home/Jack/Real_Data/external_validation/serrano_gomez_ibd/ge50_external_validation"
RUN_MANIFEST="$BASE/manifests/PRJEB42155_ge50_run_manifest.tsv"
FASTQ_DIR="$BASE/fastq"
KNEAD_OUT="$BASE/kneaddata"
CLEAN_FASTQ_DIR="$BASE/clean_fastq_for_humann"
HUMANN_OUT="$BASE/humann"
MERGED_OUT="$BASE/humann_merged"
LOG_DIR="$BASE/logs"

THREADS="${THREADS:-1}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"   # 0 means all
SKIP_KNEADDATA="${SKIP_KNEADDATA:-0}"

mkdir -p "$KNEAD_OUT" "$CLEAN_FASTQ_DIR" "$HUMANN_OUT" "$MERGED_OUT" "$LOG_DIR"

echo "[*] Processing GE50 cohort"
echo "[*] THREADS=$THREADS"
echo "[*] MAX_SAMPLES=$MAX_SAMPLES"
echo "[*] SKIP_KNEADDATA=$SKIP_KNEADDATA"

if ! command -v humann >/dev/null 2>&1; then
  echo "[ERROR] humann not found on PATH. Activate the HUMAnN environment first."
  exit 1
fi

if [[ "$SKIP_KNEADDATA" != "1" ]]; then
  if ! command -v kneaddata >/dev/null 2>&1; then
    echo "[ERROR] kneaddata not found on PATH. Activate the KneadData/HUMAnN environment first."
    echo "        Or rerun with SKIP_KNEADDATA=1 to concatenate raw paired FASTQs directly for HUMAnN."
    exit 1
  fi

  if [[ -z "${KNEADDATA_DB:-}" ]]; then
    echo "[ERROR] KNEADDATA_DB is not set."
    echo "        Example:"
    echo "        export KNEADDATA_DB=/path/to/kneaddata_human_genome_db"
    exit 1
  fi
fi

python3 - <<'PY3'
import pandas as pd
from pathlib import Path

base = Path("/home/Jack/Real_Data/external_validation/serrano_gomez_ibd/ge50_external_validation")
run_manifest = base / "manifests" / "PRJEB42155_ge50_run_manifest.tsv"
fastq_dir = base / "fastq"

df = pd.read_csv(run_manifest, sep="\t", dtype=str)

rows = []
for _, r in df.iterrows():
    run = r["run_accession"]
    r1 = Path(str(r["R1_url"]).split("/")[-1])
    r2 = Path(str(r["R2_url"]).split("/")[-1])
    rows.append({
        "run_accession": run,
        "r1": str(fastq_dir / r1),
        "r2": str(fastq_dir / r2),
    })

out = pd.DataFrame(rows)
out.to_csv(base / "manifests" / "PRJEB42155_ge50_local_fastq_pairs.tsv", sep="\t", index=False)
print("Written local pairs:", len(out))
PY3

PAIR_FILE="$BASE/manifests/PRJEB42155_ge50_local_fastq_pairs.tsv"

N=0

tail -n +2 "$PAIR_FILE" | while IFS=$'\t' read -r RUN R1 R2; do
  N=$((N + 1))

  if [[ "$MAX_SAMPLES" != "0" && "$N" -gt "$MAX_SAMPLES" ]]; then
    echo "[*] Reached MAX_SAMPLES=$MAX_SAMPLES"
    break
  fi

  echo ""
  echo "============================================================"
  echo "[*] Sample $N: $RUN"
  echo "============================================================"

  CLEAN_COMBINED="$CLEAN_FASTQ_DIR/${RUN}_clean_combined.fastq.gz"
  HUMANN_DONE="$HUMANN_OUT/${RUN}/${RUN}_humann.DONE"

  if [[ -f "$HUMANN_DONE" ]]; then
    echo "[*] HUMAnN already done for $RUN; skipping"
    continue
  fi

  if [[ ! -f "$R1" || ! -f "$R2" ]]; then
    echo "[ERROR] Missing FASTQs for $RUN"
    echo "R1=$R1"
    echo "R2=$R2"
    exit 1
  fi

  if [[ "$SKIP_KNEADDATA" == "1" ]]; then
    echo "[*] SKIP_KNEADDATA=1; concatenating raw paired FASTQs for HUMAnN"
    if [[ ! -f "$CLEAN_COMBINED" ]]; then
      cat "$R1" "$R2" > "$CLEAN_COMBINED"
    fi
  else
    KOUT="$KNEAD_OUT/$RUN"
    mkdir -p "$KOUT"

    echo "[*] Running KneadData for $RUN"
    kneaddata \
      --input1 "$R1" \
      --input2 "$R2" \
      --reference-db "$KNEADDATA_DB" \
      --output "$KOUT" \
      --output-prefix "$RUN" \
      --threads "$THREADS" \
      --remove-intermediate-output \
      2>&1 | tee "$LOG_DIR/${RUN}_kneaddata.log"

    echo "[*] Finding KneadData paired output files"

    CLEAN_R1="$(find "$KOUT" -type f \( -name "*paired*1*.fastq" -o -name "*paired*1*.fastq.gz" \) | head -n 1 || true)"
    CLEAN_R2="$(find "$KOUT" -type f \( -name "*paired*2*.fastq" -o -name "*paired*2*.fastq.gz" \) | head -n 1 || true)"

    if [[ -z "$CLEAN_R1" || -z "$CLEAN_R2" ]]; then
      echo "[ERROR] Could not find KneadData paired outputs for $RUN in $KOUT"
      find "$KOUT" -type f | sed 's#^#  #'
      exit 1
    fi

    echo "[*] Combining KneadData paired reads for HUMAnN"
    if [[ "$CLEAN_R1" == *.gz && "$CLEAN_R2" == *.gz ]]; then
      zcat "$CLEAN_R1" "$CLEAN_R2" | gzip -c > "$CLEAN_COMBINED"
    else
      cat "$CLEAN_R1" "$CLEAN_R2" | gzip -c > "$CLEAN_COMBINED"
    fi
  fi

  mkdir -p "$HUMANN_OUT/$RUN"

  echo "[*] Running HUMAnN for $RUN"
  humann \
    --input "$CLEAN_COMBINED" \
    --output "$HUMANN_OUT/$RUN" \
    --threads "$THREADS" \
    --search-mode uniref90 \
    --metaphlan-options "--bowtie2db /home/Jack/Real_Data/humann_databases/metaphlan_vJun23 -x mpa_vJun23_CHOCOPhlAnSGB_202307" \
    --output-basename "$RUN" \
    2>&1 | tee "$LOG_DIR/${RUN}_humann.log"

  rm -rf "$HUMANN_OUT/$RUN/${RUN}_humann_temp"
  rm -f "$CLEAN_COMBINED"

  touch "$HUMANN_DONE"
done

echo ""
echo "============================================================"
echo "[*] Joining HUMAnN outputs"
echo "============================================================"

if command -v humann_join_tables >/dev/null 2>&1; then
  JOIN_IN="$MERGED_OUT/table_inputs"
  rm -rf "$JOIN_IN"
  mkdir -p "$JOIN_IN"

  find "$HUMANN_OUT" -mindepth 2 -maxdepth 2 -type f \( -name "*_genefamilies.tsv" -o -name "*_pathabundance.tsv" \) | while read -r f; do
    ln -sf "$(readlink -f "$f")" "$JOIN_IN/$(basename "$f")"
  done

  humann_join_tables \
    --input "$JOIN_IN" \
    --file_name genefamilies \
    --output "$MERGED_OUT/ge50_genefamilies_joined.tsv"

  humann_join_tables \
    --input "$JOIN_IN" \
    --file_name pathabundance \
    --output "$MERGED_OUT/ge50_pathabundance_joined.tsv"

  if command -v humann_regroup_table >/dev/null 2>&1; then
    humann_regroup_table \
      --input "$MERGED_OUT/ge50_genefamilies_joined.tsv" \
      --groups uniref90_rxn \
      --output "$MERGED_OUT/ge50_rxn_abundance_joined.tsv"

    python3 - <<'PY_EC'
from pathlib import Path
import bz2
import pandas as pd
from collections import defaultdict

rxn_table = Path("/home/Jack/Real_Data/external_validation/serrano_gomez_ibd/ge50_external_validation/humann_merged/ge50_rxn_abundance_joined.tsv")
mapping = Path("/home/Jack/micromamba/envs/humann_py312/lib/python3.12/site-packages/humann/data/pathways/metacyc_reactions_level4ec_only.uniref.bz2")
out_raw = Path("/home/Jack/Real_Data/external_validation/serrano_gomez_ibd/ge50_external_validation/humann_merged/ge50_ec_abundance_joined.tsv")
out_relab = Path("/home/Jack/Real_Data/external_validation/serrano_gomez_ibd/ge50_external_validation/humann_merged/ge50_ec_abundance_relab.tsv")

rxn_to_ecs = defaultdict(list)

with bz2.open(mapping, "rt") as f:
    for line in f:
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 2:
            continue
        rxn, ec_field = parts[0], parts[1]
        ecs = [x.strip() for x in ec_field.replace(",", " ").split() if x.strip()]
        for ec in ecs:
            if "." in ec:
                rxn_to_ecs[rxn].append(ec)

df = pd.read_csv(rxn_table, sep="\t")
feature_col = df.columns[0]
sample_cols = list(df.columns[1:])

acc = defaultdict(lambda: [0.0] * len(sample_cols))

for _, row in df.iterrows():
    feature = str(row[feature_col])
    if feature.startswith(("UNMAPPED", "UNGROUPED")):
        continue

    if "|" in feature:
        rxn, strat = feature.split("|", 1)
        suffix = "|" + strat
    else:
        rxn, suffix = feature, ""

    ecs = rxn_to_ecs.get(rxn, [])
    if not ecs:
        continue

    vals = [float(row[c]) for c in sample_cols]
    for ec in ecs:
        key = ec + suffix
        cur = acc[key]
        for i, v in enumerate(vals):
            cur[i] += v

out = pd.DataFrame([[k] + v for k, v in sorted(acc.items())], columns=[feature_col] + sample_cols)
out.to_csv(out_raw, sep="\t", index=False)

rel = out.copy()
col_sums = rel[sample_cols].sum(axis=0).replace(0, pd.NA)
rel[sample_cols] = rel[sample_cols].div(col_sums, axis=1).fillna(0)
rel.to_csv(out_relab, sep="\t", index=False)

print("Wrote EC raw:", out_raw, out.shape)
print("Wrote EC relab:", out_relab, rel.shape)
PY_EC
  fi

  if command -v humann_renorm_table >/dev/null 2>&1; then

    humann_renorm_table \
      --input "$MERGED_OUT/ge50_pathabundance_joined.tsv" \
      --output "$MERGED_OUT/ge50_pathabundance_relab.tsv" \
      --units relab || true
  fi
else
  echo "[WARN] humann_join_tables not found; skipping merge."
fi

echo "[✓] Processing complete"
echo "Outputs:"
echo "  HUMAnN: $HUMANN_OUT"
echo "  merged: $MERGED_OUT"
