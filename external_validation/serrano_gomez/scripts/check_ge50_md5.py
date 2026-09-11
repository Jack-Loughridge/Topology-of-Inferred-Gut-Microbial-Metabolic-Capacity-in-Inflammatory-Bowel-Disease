#!/usr/bin/env python3
from pathlib import Path
import hashlib
import pandas as pd
import sys

BASE = Path("/home/Jack/Real_Data/external_validation/serrano_gomez_ibd/ge50_external_validation")
MANIFEST = BASE / "manifests" / "PRJEB42155_ge50_fastq_manifest.tsv"
FASTQ_DIR = BASE / "fastq"

df = pd.read_csv(MANIFEST, sep="\t", dtype=str)

bad = []
missing = []

for _, r in df.iterrows():
    url = str(r["fastq_url"])
    expected = str(r.get("md5", "")).strip()
    fn = url.rstrip("/").split("/")[-1]
    path = FASTQ_DIR / fn

    if not path.exists():
        missing.append(fn)
        continue

    if expected and expected.lower() != "nan":
        h = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        got = h.hexdigest()
        if got != expected:
            bad.append((fn, expected, got))

print("Expected files:", len(df))
print("Missing:", len(missing))
print("Bad md5:", len(bad))

if missing:
    print("\nMissing examples:")
    for x in missing[:20]:
        print(x)

if bad:
    print("\nBad md5 examples:")
    for x in bad[:20]:
        print(x)

if missing or bad:
    sys.exit(1)

print("[✓] All downloaded FASTQs passed existence/MD5 checks")
