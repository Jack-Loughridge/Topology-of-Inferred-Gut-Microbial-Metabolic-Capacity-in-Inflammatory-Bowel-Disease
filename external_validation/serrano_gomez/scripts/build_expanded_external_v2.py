#!/usr/bin/env python3
"""Build the Serrano-Gomez expanded-community external structural replication.

This is deliberately separate from the frozen-support v1 external validation.
All expanded-edge support baselines are estimated, without labels, from the
locked 184-sample external cohort.  The graph family is stored losslessly as a
common directed scaffold plus a sparse sample-by-edge support matrix; active
GraphML files are written for the faithful active-edge Ricci program.

Stages are restartable and fail closed:
  prepare        curate profiles and map them conservatively to AGORA catalysts
  scaffold       build/gate the cohort AGORA reaction scaffold and supports
  graphs-h0      write active GraphML graphs and exact full-scaffold H0 diagrams
  ricci-pilot    run faithful Ricci on low/median/high-coverage pilot samples
  ricci-full     run faithful Ricci on all 184 active graphs
  vectorize      create full-expanded and frozen-edge B/K0 matrices
  predict        exploratory frozen C=0.02 prediction on the expanded projection
  status         report checkpoint state without changing outputs

The exploratory prediction is not labelled as strict external validation,
because its graph weights use external-cohort normalization.
"""

from __future__ import annotations

import argparse
import ast
import csv
import gzip
import hashlib
import itertools
import json
import math
import os
import random
import re
import shutil
import sqlite3
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import sparse


SCRIPT_VERSION = "2.0.0"
EXPECTED_SAMPLES = 184
EXPECTED_PAIRED_V1 = 90
GRAPH_SEED = 13
TAU_THRESHOLD = 1e-4
ROOTS_PER_INPUT = 50
MAX_BETA_POOL = 2500
WEIGHT_EPSILON = 1e-8
ACTIVE_TOL = 1e-12

RICCI_EPSILON_DIST = 1e-4
RICCI_C_SINGLE_OUT = 0.001
RICCI_BETA = 1.4
RICCI_N_PATHS = 1000
RICCI_MAX_STEPS = 10000
RICCI_MU_PRUNE = 1e-6
RICCI_TOPK_SUPPORT = 150
RICCI_MAX_LP_VARS = 200000

EXPECTED_AGORA_SHA256 = "ec906cd5f08448991b3958112fd4367f17141e0bbbe7d0333c8382a5c36af755"
EXPECTED_AUDIT_SCRIPT_SHA256 = "933ceaa7dfe3330b515d081fd823afa17772f0eed507d42e1c8a63cbf8f55182"
EXPECTED_STRUCTURAL_SOURCE_SHA256 = "d1f9758ce24a235915c4ae6be6bb9e57fc274a9da48eab6aaf47b528dea6f863"
EXPECTED_RICCI_SOURCE_SHA256 = "d1076a7a6a4e0bda74db8b997d1118120fa573dfb93126c0d1b2f4438c0154e8"
EXPECTED_FROZEN_EDGE_SHA256 = "a3cb8e9b0f19677dbffdd5b8d6ca3119b14a06c6643d0781347872e6687936a3"
EXPECTED_FROZEN_MODEL_SHA256 = "e3152d4006266962c63efdaf11f11cb2ca88ca6cdad1b728183b41e53304638d"

CURRENCY = {
    "H2O", "WATER", "PROTON", "H+", "ATP", "ADP", "PI", "PPI", "CO2", "CO(2)",
    "NAD", "NADH", "NADP", "NADPH", "FAD", "FADH2", "NAD(P)H", "OXYGEN", "O2",
    "NH3", "AMMONIA", "HCO3-", "BICARBONATE", "CO-A", "COENZYME_A", "COENZYME A",
    "S-ADENOSYLMETHIONINE", "SAM", "S-ADENOSYLHOMOCYSTEINE", "SAH", "UMP", "UDP",
    "UTP", "AMP", "CMP", "CDP", "CTP", "GMP", "GDP", "GTP",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def die(message: str) -> None:
    raise RuntimeError(message)


def sha256(path: Path, chunk: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def stable_seed(*parts: object, base_seed: int = GRAPH_SEED) -> int:
    h = hashlib.sha256(str(base_seed).encode())
    for part in parts:
        h.update(b"||")
        h.update(str(part).encode())
    return int.from_bytes(h.digest()[:8], "little") % (2**32)


def atomic_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, path)


def require_file(path: Path, label: str, expected_sha: Optional[str] = None) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        die(f"Missing or empty {label}: {path}")
    if expected_sha is not None:
        found = sha256(path)
        if found != expected_sha:
            die(f"SHA-256 mismatch for {label}: expected {expected_sha}, found {found}: {path}")


def require_free_gib(path: Path, minimum: float, stage: str) -> float:
    free = shutil.disk_usage(path).free / (1024**3)
    if free < minimum:
        die(f"{stage} requires at least {minimum:.1f} GiB free; found {free:.2f} GiB")
    return free


def marker(path: Path, stage: str, details: Mapping[str, object]) -> None:
    payload = {
        "status": "complete",
        "stage": stage,
        "completed_utc": utc_now(),
        "script_version": SCRIPT_VERSION,
        "script_sha256": sha256(Path(__file__).resolve()),
        **dict(details),
    }
    atomic_json(path, payload)


def canon_metabolite(x: object) -> str:
    return str(x).strip().upper().replace(" ", "_")


def is_currency(x: object) -> bool:
    return canon_metabolite(x) in CURRENCY


def parse_listish(cell: object) -> List[str]:
    if cell is None or (isinstance(cell, float) and math.isnan(cell)):
        return []
    if isinstance(cell, (list, tuple, set, np.ndarray)):
        return [str(x).strip() for x in cell if str(x).strip()]
    s = str(cell).strip()
    if not s:
        return []
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, (list, tuple, set)):
            return [str(x).strip() for x in parsed if str(x).strip()]
        if isinstance(parsed, str):
            s = parsed
    except Exception:
        pass
    for sep in (";", "|"):
        s = s.replace(sep, ",")
    return [t.strip().strip("'\"") for t in s.split(",") if t.strip().strip("'\"")]


def canonical_metabolite_tuple(cell: object) -> Tuple[str, ...]:
    vals = [canon_metabolite(x) for x in parse_listish(cell) if not is_currency(x)]
    return tuple(dict.fromkeys(v for v in vals if v))


def species_terminal_name(x: object) -> str:
    s = str(x).strip()
    if "|" in s:
        species_tokens = [p for p in s.split("|") if p.startswith("s__")]
        s = species_tokens[-1] if species_tokens else s.split("|")[-1]
    if s.startswith("s__"):
        s = s[3:]
    return s.strip()


def species_key(x: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", species_terminal_name(x).lower())


def read_table_auto(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t", low_memory=False)
    return pd.read_csv(path, low_memory=False)


def parse_metaphlan_species(path: Path) -> Tuple[pd.Series, dict]:
    attempts: List[pd.DataFrame] = []
    for kwargs in (
        {"sep": "\t", "comment": "#", "low_memory": False},
        {"sep": "\t", "comment": "#", "header": None, "low_memory": False},
    ):
        try:
            df = pd.read_csv(path, **kwargs)
            if len(df.columns) >= 2 and len(df):
                attempts.append(df)
        except Exception:
            continue
    if not attempts:
        die(f"Could not parse MetaPhlAn profile: {path}")

    known_tax = {"clade_name", "taxonomy", "taxon", "clade", "#clade_name", "name"}
    known_abund = {"relative_abundance", "abundance", "relative abundance", "rel_abundance"}
    best: Optional[Tuple[pd.Series, dict]] = None
    for df in attempts:
        lower = {str(c).strip().lower(): c for c in df.columns}
        tax_col = next((lower[x] for x in known_tax if x in lower), None)
        abund_col = next((lower[x] for x in known_abund if x in lower), None)
        if tax_col is None:
            tax_col = df.columns[0]
        if abund_col is None:
            numeric = []
            for c in df.columns:
                if c == tax_col:
                    continue
                vals = pd.to_numeric(df[c], errors="coerce")
                if float(vals.notna().mean()) > 0.80:
                    numeric.append(c)
            if not numeric:
                continue
            abund_col = numeric[-1]

        rows: List[Tuple[str, float]] = []
        for tax0, val0 in zip(df[tax_col], pd.to_numeric(df[abund_col], errors="coerce")):
            if pd.isna(val0):
                continue
            val = float(val0)
            tax = str(tax0).strip()
            if not math.isfinite(val) or val < 0:
                continue
            is_species = ("|s__" in tax and "|t__" not in tax) or tax.startswith("s__")
            if is_species:
                name = species_terminal_name(tax)
                if name:
                    rows.append((name, val))
        if not rows:
            continue
        s = pd.DataFrame(rows, columns=["species", "abundance"]).groupby("species")["abundance"].sum()
        total = float(s.sum())
        scale = 1.0
        if 0 < total <= 1.5 and float(s.max()) <= 1.0 + 1e-9:
            s *= 100.0
            total = float(s.sum())
            scale = 100.0
        if total <= 0 or total > 110 or float(s.max()) > 100.0 + 1e-6:
            continue
        candidate = (s.astype(float).sort_index(), {
            "n_species_rows": int(len(s)),
            "species_abundance_sum_percent": total,
            "species_abundance_max_percent": float(s.max()),
            "unit_scale_factor_applied": scale,
            "tax_col": str(tax_col),
            "abundance_col": str(abund_col),
        })
        if best is None or len(candidate[0]) > len(best[0]):
            best = candidate
    if best is None:
        die(f"No valid species-level MetaPhlAn rows: {path}")
    return best


def unique_key_maps(axis: Sequence[str]) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    buckets: Dict[str, List[str]] = defaultdict(list)
    for x in axis:
        k = species_key(x)
        if k:
            buckets[k].append(str(x))
    unique = {k: v[0] for k, v in buckets.items() if len(v) == 1}
    ambiguous = {k: sorted(v) for k, v in buckets.items() if len(v) > 1}
    return unique, ambiguous


def map_species_profile(ext: pd.Series, catalyst_axis: Sequence[str]) -> Tuple[pd.Series, dict, pd.DataFrame]:
    out = pd.Series(0.0, index=list(catalyst_axis), dtype=float)
    unique, ambiguous = unique_key_maps(catalyst_axis)
    exact = {str(x): str(x) for x in catalyst_axis}
    lower: Dict[str, List[str]] = defaultdict(list)
    for x in catalyst_axis:
        lower[str(x).lower()].append(str(x))
    rows = []
    mapped = unmatched = ambiguous_total = 0.0
    for raw0, val0 in ext.items():
        raw = str(raw0)
        terminal = species_terminal_name(raw)
        val = float(val0)
        target = None
        method = ""
        if raw in exact:
            target, method = exact[raw], "exact"
        elif terminal in exact:
            target, method = exact[terminal], "terminal_exact"
        elif len(lower.get(terminal.lower(), [])) == 1:
            target, method = lower[terminal.lower()][0], "case_insensitive_exact"
        else:
            key = species_key(terminal)
            if key in unique:
                target, method = unique[key], "unique_normalized"
            elif key in ambiguous:
                ambiguous_total += val
                rows.append({"external_species": raw, "abundance_percent": val, "agora_catalyst": "",
                             "status": "ambiguous", "method": "", "candidates": " | ".join(ambiguous[key])})
                continue
        if target is None:
            unmatched += val
            rows.append({"external_species": raw, "abundance_percent": val, "agora_catalyst": "",
                         "status": "unmatched", "method": "", "candidates": ""})
        else:
            out.loc[target] += val
            mapped += val
            rows.append({"external_species": raw, "abundance_percent": val, "agora_catalyst": target,
                         "status": "mapped", "method": method, "candidates": ""})
    total = float(ext.sum())
    meta = {
        "raw_species_total_percent": total,
        "agora_mapped_abundance_percent": mapped,
        "agora_coverage_fraction": mapped / total if total > 0 else float("nan"),
        "agora_unmatched_abundance_percent": unmatched,
        "agora_ambiguous_abundance_percent": ambiguous_total,
        "detected_species": int(len(ext)),
        "agora_catalysts_nonzero": int((out > 0).sum()),
    }
    return out, meta, pd.DataFrame(rows)


class UnionFind:
    def __init__(self, n: int):
        self.parent = np.arange(n, dtype=np.int64)
        self.rank = np.zeros(n, dtype=np.int8)
        self.components = n

    def find(self, x: int) -> int:
        p = int(self.parent[x])
        while p != int(self.parent[p]):
            self.parent[p] = self.parent[int(self.parent[p])]
            p = int(self.parent[p])
        while x != p:
            nxt = int(self.parent[x])
            self.parent[x] = p
            x = nxt
        return p

    def union(self, a: int, b: int) -> bool:
        a, b = self.find(a), self.find(b)
        if a == b:
            return False
        if self.rank[a] < self.rank[b]:
            a, b = b, a
        self.parent[b] = a
        if self.rank[a] == self.rank[b]:
            self.rank[a] += 1
        self.components -= 1
        return True


def multi_input_gate(inputs: Tuple[str, ...], g0: nx.Graph, rng: random.Random) -> bool:
    if len(inputs) <= 1:
        return True
    nodes = list(g0.nodes)
    if not nodes:
        return False
    dists: Dict[str, List[int]] = {x: [] for x in inputs}
    for x in inputs:
        if x not in g0:
            return False
        trials = 0
        while len(dists[x]) < ROOTS_PER_INPUT and trials < ROOTS_PER_INPUT * 4:
            trials += 1
            root = rng.choice(nodes)
            try:
                dists[x].append(int(nx.shortest_path_length(g0, root, x)))
            except nx.NetworkXNoPath:
                continue
    if any(not v for v in dists.values()):
        return False
    pool = sorted(set(itertools.chain.from_iterable(dists.values())))[:MAX_BETA_POOL]
    scores = []
    for beta in pool:
        scores.append(max(min(abs(beta - d) for d in dists[x]) for x in inputs))
    top = sorted(scores)[:3]
    return bool(top) and (float(sum(top)) / len(top) < TAU_THRESHOLD)


class Paths:
    def __init__(self, base: Path, ext: Path, out: Path):
        self.base = base
        self.ext = ext
        self.out = out
        self.audit = ext / "expanded_v2_preflight_audit"
        self.audit_json = self.audit / "V2_EXPANDED_PREFLIGHT_AUDIT.json"
        self.audit_marker = self.audit / "AUDIT_COMPLETE.json"
        self.profile_inventory = self.audit / "profile_inventory.csv"
        self.coverage_preflight = self.audit / "preliminary_taxonomic_coverage.csv"
        self.agora = base / "AGORA_reactions_canon.parquet"
        self.metadata = ext / "manifests" / "PRJEB42155_ge50_clean_metadata_full.tsv"
        self.structural_source = ext / "scripts" / "serrano_gomez_external_structural_frozen_ibdmdb_v1" / "external_structural_validation.py"
        self.ricci_source = base / "14_compute_ricci_faithful_pairwise_active.py"
        self.frozen_edges = ext / "structural_frozen_ibdmdb" / "ricci_features_frozen_training_order" / "edge_metadata.csv"
        self.frozen_v1_meta = ext / "structural_frozen_ibdmdb" / "ricci_features_frozen_training_order" / "matched_metadata.csv"
        self.adaptive_intervals = base / "out_pds" / "adaptive_intervals.npy"
        self.frozen_model = base / "Ricci_IBD_RepeatedCV_CPath" / "C_0p02" / "full_source_model" / "full_source_model.npz"
        self.curated = out / "01_curated"
        self.scaffold = out / "02_scaffold_support"
        self.graphs = out / "03_active_graphs"
        self.h0 = out / "04_h0"
        self.ricci_pilot = out / "05_ricci_pilot"
        self.ricci = out / "06_ricci_full"
        self.features = out / "07_features"
        self.predictions = out / "08_exploratory_prediction"
        self.logs = out / "logs"


def validate_foundation(p: Paths) -> dict:
    require_file(p.audit_json, "v2 preflight audit")
    require_file(p.audit_marker, "v2 preflight marker")
    audit = json.loads(p.audit_json.read_text())
    done = json.loads(p.audit_marker.read_text())
    if not audit.get("ready_for_pipeline_construction") or audit.get("blockers"):
        die("Preflight audit does not authorize pipeline construction")
    if int(done.get("selected_profiles", -1)) != EXPECTED_SAMPLES:
        die("Preflight marker does not lock 184 selected profiles")
    require_file(p.profile_inventory, "profile inventory")
    require_file(p.agora, "AGORA reaction parquet", EXPECTED_AGORA_SHA256)
    require_file(p.metadata, "external metadata")
    require_file(p.structural_source, "frozen structural source", EXPECTED_STRUCTURAL_SOURCE_SHA256)
    require_file(p.ricci_source, "faithful Ricci source", EXPECTED_RICCI_SOURCE_SHA256)
    require_file(p.frozen_edges, "frozen edge metadata", EXPECTED_FROZEN_EDGE_SHA256)
    require_file(p.adaptive_intervals, "frozen H0 intervals")
    return audit


def scan_catalyst_axis(agora: Path) -> Tuple[List[str], pd.DataFrame]:
    pf = pq.ParquetFile(agora)
    raw_values: set[str] = set()
    for rg in range(pf.num_row_groups):
        arr = pf.read_row_group(rg, columns=["catalysts"]).column("catalysts").to_pylist()
        raw_values.update(str(x) for x in arr if x is not None and str(x).strip())
    rows = []
    tokens = set()
    for raw in sorted(raw_values):
        parsed = parse_listish(raw)
        for token in parsed:
            if token:
                tokens.add(token)
                rows.append({"raw_catalyst_value": raw, "agora_catalyst": token})
    axis = sorted(tokens)
    if len(axis) < 1000:
        die(f"AGORA catalyst axis unexpectedly small: {len(axis)}")
    return axis, pd.DataFrame(rows)


def match_metadata(metadata: Path, sample_ids: Sequence[str]) -> Tuple[pd.DataFrame, str]:
    df = read_table_auto(metadata)
    wanted = set(map(str, sample_ids))
    candidates = []
    for c in df.columns:
        series = df[c].dropna().astype(str)
        vals = set(series)
        overlap = len(wanted & vals)
        if overlap:
            selected_series = series[series.isin(wanted)]
            duplicates = int(selected_series.duplicated().sum())
            candidates.append((overlap, duplicates, str(c), c))
    if not candidates:
        die("No metadata column overlaps selected profile IDs")
    candidates.sort(key=lambda x: (-x[0], x[1], x[2]))
    overlap, _, _, col = candidates[0]
    if overlap != EXPECTED_SAMPLES:
        die(f"Best metadata ID column {col!r} overlaps only {overlap}/184 samples")
    out = df[df[col].astype(str).isin(wanted)].copy()
    out.insert(0, "sample_id_v2", out[col].astype(str))
    if out["sample_id_v2"].duplicated().any():
        die(f"Metadata ID column {col!r} is not unique for selected samples")
    out = out.set_index("sample_id_v2").loc[list(sample_ids)].reset_index()
    return out, str(col)


def stage_prepare(p: Paths, overwrite: bool = False) -> None:
    done = p.curated / "PREPARE_COMPLETE.json"
    if done.exists() and not overwrite:
        print(f"prepare: already complete ({done})")
        return
    validate_foundation(p)
    require_free_gib(p.ext, 20.0, "prepare")
    p.curated.mkdir(parents=True, exist_ok=True)

    inv = pd.read_csv(p.profile_inventory)
    selected = inv[inv["parse_status"].astype(str).eq("selected")].copy()
    selected["sample_id"] = selected["sample_id"].astype(str)
    selected = selected.sort_values("sample_id").reset_index(drop=True)
    if len(selected) != EXPECTED_SAMPLES or selected["sample_id"].duplicated().any():
        die(f"Profile inventory must contain 184 unique selected rows; found {len(selected)}")
    for row in selected.itertuples(index=False):
        pp = Path(row.profile_path)
        require_file(pp, f"profile {row.sample_id}", str(row.profile_sha256))

    print("[prepare] Scanning full AGORA catalyst axis...")
    catalyst_axis, raw_map = scan_catalyst_axis(p.agora)
    (p.curated / "agora_catalyst_axis.txt").write_text("\n".join(catalyst_axis) + "\n")
    raw_map.to_csv(p.curated / "catalyst_raw_value_map.csv", index=False)

    abundance_rows = []
    mapping_frames = []
    coverage_rows = []
    profile_rows = []
    print(f"[prepare] Parsing and mapping {len(selected)} profiles without labels...")
    for i, row in enumerate(selected.itertuples(index=False), 1):
        sid = str(row.sample_id)
        s, parse_meta = parse_metaphlan_species(Path(row.profile_path))
        mapped, map_meta, audit = map_species_profile(s, catalyst_axis)
        abundance_rows.append(mapped.rename(sid))
        audit.insert(0, "sample_id", sid)
        mapping_frames.append(audit)
        coverage_rows.append({"sample_id": sid, **parse_meta, **map_meta})
        profile_rows.append({
            "sample_id": sid, "profile_path": str(row.profile_path),
            "profile_sha256": str(row.profile_sha256), **parse_meta,
        })
        if i == 1 or i % 25 == 0 or i == len(selected):
            print(f"    profiles {i}/{len(selected)}")

    abundance = pd.DataFrame(abundance_rows).fillna(0.0)
    abundance.index = [str(x.name) for x in abundance_rows]
    abundance.index.name = "sample_id"
    abundance = abundance.loc[selected["sample_id"], catalyst_axis]
    if not np.isfinite(abundance.to_numpy()).all() or (abundance.to_numpy() < 0).any():
        die("Mapped abundance matrix contains invalid values")
    abundance.reset_index().to_parquet(p.curated / "agora_abundance_percent.parquet", index=False)
    pd.DataFrame(profile_rows).to_csv(p.curated / "profile_manifest.csv", index=False)
    coverage = pd.DataFrame(coverage_rows)
    coverage.to_csv(p.curated / "taxonomic_coverage.csv", index=False)
    pd.concat(mapping_frames, ignore_index=True).to_csv(
        p.curated / "species_mapping_long.csv.gz", index=False, compression="gzip"
    )

    # Labels enter only after the label-blind abundance artefacts above exist.
    matched_meta, id_col = match_metadata(p.metadata, selected["sample_id"].tolist())
    matched_meta.to_csv(p.curated / "matched_external_metadata_184.tsv.gz", sep="\t", index=False, compression="gzip")
    v1 = pd.read_csv(p.frozen_v1_meta, usecols=["sample_id"])
    paired = selected[selected["sample_id"].isin(set(v1["sample_id"].astype(str)))][["sample_id"]]
    if len(paired) != EXPECTED_PAIRED_V1:
        die(f"Expected 90 paired v1 samples, found {len(paired)}")
    paired.to_csv(p.curated / "paired_v1_sample_ids.csv", index=False)

    qs = coverage.set_index("sample_id")["agora_coverage_fraction"].sort_values()
    pilot_ids = [str(qs.index[0]), str(qs.index[len(qs)//2]), str(qs.index[-1])]
    pd.DataFrame({"sample_id": pilot_ids, "selection": ["minimum_coverage", "median_coverage", "maximum_coverage"]}).to_csv(
        p.curated / "ricci_pilot_samples.csv", index=False
    )

    provenance = {
        "methodological_role": "expanded-community structural replication; not strict frozen external validation",
        "label_firewall": "profiles and AGORA mapping were completed before metadata was merged",
        "n_samples": int(len(abundance)), "n_agora_catalysts": int(len(catalyst_axis)),
        "metadata_id_column": id_col, "pilot_sample_ids": pilot_ids,
        "coverage_min": float(coverage["agora_coverage_fraction"].min()),
        "coverage_median": float(coverage["agora_coverage_fraction"].median()),
        "coverage_max": float(coverage["agora_coverage_fraction"].max()),
        "agora_sha256": sha256(p.agora), "profile_inventory_sha256": sha256(p.profile_inventory),
    }
    atomic_json(p.curated / "CURATION_PROVENANCE.json", provenance)
    marker(done, "prepare", provenance)
    print(json.dumps(provenance, indent=2))


def sqlite_reaction_db(path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(path)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    con.execute("PRAGMA temp_store=FILE")
    con.execute("""
        CREATE TABLE IF NOT EXISTS reaction_signatures (
            inputs_key TEXT NOT NULL,
            outputs_key TEXT NOT NULL,
            catalyst TEXT NOT NULL,
            row_count INTEGER NOT NULL,
            PRIMARY KEY (inputs_key, outputs_key, catalyst)
        )
    """)
    return con


def stage_scaffold(p: Paths, overwrite: bool = False) -> None:
    done = p.scaffold / "SCAFFOLD_COMPLETE.json"
    if done.exists() and not overwrite:
        print(f"scaffold: already complete ({done})")
        return
    validate_foundation(p)
    require_file(p.curated / "PREPARE_COMPLETE.json", "prepare marker")
    require_free_gib(p.ext, 15.0, "scaffold")
    p.scaffold.mkdir(parents=True, exist_ok=True)
    db_path = p.scaffold / "community_reaction_signatures.sqlite"
    if overwrite and db_path.exists():
        db_path.unlink()
    con = sqlite_reaction_db(db_path)

    raw_map = pd.read_csv(p.curated / "catalyst_raw_value_map.csv", dtype=str)
    abundance = pd.read_parquet(p.curated / "agora_abundance_percent.parquet").set_index("sample_id")
    observed = set(abundance.columns[(abundance > 0).any(axis=0)])
    selected_raw = set(raw_map.loc[raw_map["agora_catalyst"].isin(observed), "raw_catalyst_value"].astype(str))
    raw_to_tokens = raw_map.groupby("raw_catalyst_value")["agora_catalyst"].apply(list).to_dict()
    if not selected_raw:
        die("No AGORA catalyst raw values correspond to mapped external species")

    g0 = nx.Graph()
    pf = pq.ParquetFile(p.agora)
    total_selected_rows = 0
    print(f"[scaffold] Streaming {pf.num_row_groups} AGORA parquet row groups...")
    insert_sql = """
        INSERT INTO reaction_signatures(inputs_key, outputs_key, catalyst, row_count)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(inputs_key, outputs_key, catalyst)
        DO UPDATE SET row_count = row_count + excluded.row_count
    """
    for rg in range(pf.num_row_groups):
        df = pf.read_row_group(rg, columns=["inputs", "outputs", "catalysts"]).to_pandas()
        df["catalysts"] = df["catalysts"].astype(str)
        df = df[df["catalysts"].isin(selected_raw)]
        records = []
        for row in df.itertuples(index=False):
            ins = canonical_metabolite_tuple(row.inputs)
            outs = canonical_metabolite_tuple(row.outputs)
            if not ins or not outs:
                continue
            cats = [x for x in raw_to_tokens.get(str(row.catalysts), []) if x in observed]
            if not cats:
                continue
            ik, ok = "\x1f".join(ins), "\x1f".join(outs)
            for cat in cats:
                records.append((ik, ok, cat))
            g0.add_nodes_from(ins)
            g0.add_nodes_from(outs)
            g0.add_edges_from(itertools.combinations(ins, 2))
            g0.add_edges_from((a, b) for a in ins for b in outs)
        if records:
            counts = Counter(records)
            con.executemany(insert_sql, [(a, b, c, n) for (a, b, c), n in counts.items()])
            con.commit()
            total_selected_rows += int(sum(counts.values()))
        print(f"    row group {rg+1}/{pf.num_row_groups}: retained rows cumulative={total_selected_rows:,}")
    if total_selected_rows == 0:
        die("No usable AGORA rows retained for the external mapped community")

    # Identical input predicates are gated once, in first SQLite row order.
    # This removes strain/model duplicates from the stochastic decision while
    # preserving their multiplicity in n_reaction_rows.
    rng = random.Random(GRAPH_SEED)
    gate_cache: Dict[str, bool] = {}
    edge_cats: Dict[Tuple[str, str], set[str]] = defaultdict(set)
    edge_rows: Counter[Tuple[str, str]] = Counter()
    edge_signatures: Counter[Tuple[str, str]] = Counter()
    n_signatures = n_accepted_signatures = 0
    print("[scaffold] Applying the production multi-input gate...")
    cur = con.execute("SELECT inputs_key, outputs_key, catalyst, row_count FROM reaction_signatures ORDER BY rowid")
    for ik, ok, cat, row_count in cur:
        n_signatures += 1
        ins = tuple(str(ik).split("\x1f"))
        outs = tuple(str(ok).split("\x1f"))
        if ik not in gate_cache:
            gate_cache[ik] = multi_input_gate(ins, g0, rng)
        if not gate_cache[ik]:
            continue
        n_accepted_signatures += 1
        for a in ins:
            for b in outs:
                if a == b or is_currency(a) or is_currency(b):
                    continue
                e = (a, b)
                edge_cats[e].add(str(cat))
                edge_rows[e] += int(row_count)
                edge_signatures[e] += 1
        if n_signatures % 100000 == 0:
            print(f"    signatures {n_signatures:,}; accepted edges {len(edge_cats):,}")
    con.close()
    if not edge_cats:
        die("The accepted expanded edge scaffold is empty")

    edges = sorted(edge_cats)
    edge_index = {e: i for i, e in enumerate(edges)}
    cat_index = {c: i for i, c in enumerate(abundance.columns)}
    incidence_rows, incidence_cols = [], []
    long_rows = []
    meta_rows = []
    for e in edges:
        j = edge_index[e]
        cats = sorted(edge_cats[e])
        for cat in cats:
            incidence_rows.append(cat_index[cat])
            incidence_cols.append(j)
            long_rows.append({"edge_index": j, "source": e[0], "target": e[1], "agora_catalyst": cat})
        meta_rows.append({
            "edge_index": j, "source": e[0], "target": e[1],
            "n_distinct_catalysts": len(cats), "n_reaction_rows": int(edge_rows[e]),
            "n_unique_reaction_signatures": int(edge_signatures[e]),
        })
    incidence = sparse.csr_matrix(
        (np.ones(len(incidence_rows), dtype=np.float64), (incidence_rows, incidence_cols)),
        shape=(len(abundance.columns), len(edges)),
    )
    abundance_sparse = sparse.csr_matrix(abundance.to_numpy(dtype=np.float64))
    supports = (abundance_sparse @ incidence).tocsr()
    supports.eliminate_zeros()
    csc = supports.tocsc()
    baselines = np.zeros(len(edges), dtype=np.float64)
    positive_counts = np.zeros(len(edges), dtype=np.int32)
    for j in range(len(edges)):
        vals = csc.data[csc.indptr[j]:csc.indptr[j+1]]
        vals = vals[vals > 0]
        if len(vals):
            baselines[j] = float(np.exp(np.mean(np.log(vals))))
            positive_counts[j] = len(vals)
    if not np.all(np.isfinite(baselines)) or np.any(baselines <= 0):
        bad = np.flatnonzero((~np.isfinite(baselines)) | (baselines <= 0))[:10]
        die(f"Every cohort catalyst edge must have a positive finite baseline; bad indices={bad.tolist()}")

    edge_meta = pd.DataFrame(meta_rows)
    edge_meta["positive_sample_count"] = positive_counts
    edge_meta["baseline_positive_support_geomean"] = baselines
    edge_meta.to_parquet(p.scaffold / "expanded_edge_metadata.parquet", index=False)
    pd.DataFrame(long_rows).to_parquet(p.scaffold / "edge_catalyst_mapping.parquet", index=False)
    sparse.save_npz(p.scaffold / "support_matrix_samples_by_edges.npz", supports, compressed=True)
    np.save(p.scaffold / "edge_baselines.npy", baselines)
    pd.DataFrame({"sample_index": np.arange(len(abundance)), "sample_id": abundance.index}).to_csv(
        p.scaffold / "sample_order.csv", index=False
    )
    nx.write_graphml(g0, p.scaffold / "gate_substrate_graph.graphml.gz")

    provenance = {
        "n_samples": int(supports.shape[0]), "n_edges": int(supports.shape[1]),
        "n_gate_nodes": int(g0.number_of_nodes()), "n_gate_edges": int(g0.number_of_edges()),
        "n_observed_agora_catalysts": int(len(observed)),
        "retained_agora_reaction_rows": int(total_selected_rows),
        "unique_reaction_signatures": int(n_signatures),
        "accepted_reaction_signatures": int(n_accepted_signatures),
        "unique_input_predicates": int(len(gate_cache)),
        "accepted_input_predicates": int(sum(gate_cache.values())),
        "graph_seed": GRAPH_SEED, "tau_threshold": TAU_THRESHOLD,
        "roots_per_input": ROOTS_PER_INPUT, "weight_epsilon": WEIGHT_EPSILON,
        "baseline_rule": "geometric mean of positive external-cohort raw supports for every expanded edge",
        "reaction_scope": "reactions catalysed by at least one conservatively mapped species observed in the locked external cohort",
        "duplicate_policy": "identical canonical input predicates gated once; reaction-row multiplicity retained for audit only; edge support uses distinct catalyst species",
        "storage": "common scaffold + sparse supports; exact full graphs are reconstructable; active GraphML written in next stage",
    }
    atomic_json(p.scaffold / "SCAFFOLD_PROVENANCE.json", provenance)
    marker(done, "scaffold", provenance)
    print(json.dumps(provenance, indent=2))


def h0_from_scaffold(
    nodes: Sequence[str], directed_edges: Sequence[Tuple[str, str]], weights: np.ndarray,
) -> Tuple[np.ndarray, int, int]:
    node_index = {n: i for i, n in enumerate(nodes)}
    best: Dict[Tuple[int, int], float] = {}
    for (a, b), w0 in zip(directed_edges, weights):
        u, v = node_index[a], node_index[b]
        if u > v:
            u, v = v, u
        w = float(w0)
        if not math.isfinite(w):
            continue
        if w <= 0:
            w = 1e-9
        if (u, v) not in best or w < best[(u, v)]:
            best[(u, v)] = w
    ordered = sorted(((w, u, v) for (u, v), w in best.items()), key=lambda x: (x[0], x[1], x[2]))
    uf = UnionFind(len(nodes))
    deaths = []
    for w, u, v in ordered:
        if uf.union(u, v):
            deaths.append(w)
    arr = np.column_stack([np.zeros(len(deaths), dtype=float), np.asarray(deaths, dtype=float)]) if deaths else np.zeros((0, 2))
    return arr, len(best), int(uf.components)


def stage_graphs_h0(p: Paths, overwrite: bool = False) -> None:
    done = p.h0 / "GRAPHS_H0_COMPLETE.json"
    if done.exists() and not overwrite:
        print(f"graphs-h0: already complete ({done})")
        return
    validate_foundation(p)
    require_file(p.scaffold / "SCAFFOLD_COMPLETE.json", "scaffold marker")
    require_free_gib(p.ext, 10.0, "graphs-h0")
    p.graphs.mkdir(parents=True, exist_ok=True)
    p.h0.mkdir(parents=True, exist_ok=True)
    meta = pd.read_parquet(p.scaffold / "expanded_edge_metadata.parquet").sort_values("edge_index")
    samples = pd.read_csv(p.scaffold / "sample_order.csv").sort_values("sample_index")
    supports = sparse.load_npz(p.scaffold / "support_matrix_samples_by_edges.npz").tocsr()
    baselines = np.load(p.scaffold / "edge_baselines.npy")
    edges = list(zip(meta["source"].astype(str), meta["target"].astype(str)))
    nodes = sorted(set(meta["source"].astype(str)) | set(meta["target"].astype(str)))
    intervals = np.asarray(np.load(p.adaptive_intervals), dtype=float).ravel()
    if len(intervals) == 0 or not np.isfinite(intervals).all() or np.any(np.diff(intervals) < 0):
        die("Frozen adaptive interval coordinate array must be finite and nondecreasing")
    ecdf = np.zeros((len(samples), len(intervals)), dtype=np.float64)
    h0_summary = []
    graph_summary = []

    print(f"[graphs-h0] Building {len(samples)} active GraphML files and exact full-scaffold H0 diagrams...")
    for i, srow in enumerate(samples.itertuples(index=False), 1):
        sid = str(srow.sample_id)
        row = supports.getrow(int(srow.sample_index))
        active_idx = row.indices
        active_support = row.data.astype(float)
        active_weights = np.exp(-active_support / (baselines[active_idx] + WEIGHT_EPSILON))
        if not np.isfinite(active_weights).all() or np.any(active_weights <= 0) or np.any(active_weights >= 1):
            die(f"Invalid active weights for {sid}")
        g = nx.DiGraph(
            sample_id=sid,
            representation="expanded_external_v2_active_projection",
            full_scaffold_edges=int(len(edges)),
            weight_formula="exp(-support/(positive_external_geomean+1e-8))",
        )
        for j, sup, w in zip(active_idx, active_support, active_weights):
            r = meta.iloc[int(j)]
            g.add_edge(str(r.source), str(r.target), weight=float(w), support=float(sup),
                       baseline=float(baselines[int(j)]), edge_index=int(j),
                       n_reactions=int(r.n_reaction_rows))
        graph_path = p.graphs / f"{sid}.graphml"
        nx.write_graphml(g, graph_path)

        full_weights = np.ones(len(edges), dtype=np.float64)
        full_weights[active_idx] = active_weights
        diagram, n_undirected, n_components = h0_from_scaffold(nodes, edges, full_weights)
        np.save(p.h0 / f"{sid}_H0.npy", diagram)
        deaths = diagram[:, 1] if len(diagram) else np.zeros(0)
        ecdf[i-1, :] = np.searchsorted(np.sort(deaths), intervals, side="right") / max(1, len(deaths))
        h0_summary.append({
            "sample_id": sid, "n_nodes_full_scaffold": len(nodes),
            "n_directed_edges_full_scaffold": len(edges), "n_undirected_edges": n_undirected,
            "n_H0_points": len(diagram), "n_components_final": n_components,
            "n_deaths_lt_1": int(np.sum(deaths < 1.0 - ACTIVE_TOL)),
            "n_deaths_eq_1": int(np.sum(np.isclose(deaths, 1.0, atol=ACTIVE_TOL, rtol=0))),
        })
        graph_summary.append({
            "sample_id": sid, "active_nodes": g.number_of_nodes(), "active_edges": g.number_of_edges(),
            "weight_min": float(active_weights.min()) if len(active_weights) else np.nan,
            "weight_median": float(np.median(active_weights)) if len(active_weights) else np.nan,
            "weight_max": float(active_weights.max()) if len(active_weights) else np.nan,
            "support_sum": float(active_support.sum()),
        })
        if i == 1 or i % 20 == 0 or i == len(samples):
            print(f"    samples {i}/{len(samples)}")

    np.save(p.h0 / "frozen_h0_boundary_ecdf.npy", ecdf)
    np.save(p.h0 / "frozen_h0_boundaries.npy", intervals)
    pd.DataFrame(h0_summary).to_csv(p.h0 / "h0_summary.csv", index=False)
    pd.DataFrame(graph_summary).to_csv(p.graphs / "active_graph_summary.csv", index=False)
    provenance = {
        "n_samples": int(len(samples)), "full_scaffold_nodes": int(len(nodes)),
        "full_scaffold_directed_edges": int(len(edges)),
        "stored_graph_semantics": "active edges only; inactive scaffold edges have exactly weight 1 and are reconstructed from shared scaffold",
        "h0_semantics": "underlying undirected full scaffold; reciprocal edges use minimum weight; union-find deaths in increasing weight",
        "h0_frozen_projection": "empirical CDF of full-scaffold H0 death values evaluated at each frozen adaptive coordinate",
        "frozen_h0_coordinates": int(len(intervals)), "frozen_h0_intervals_sha256": sha256(p.adaptive_intervals),
    }
    atomic_json(p.h0 / "GRAPHS_H0_PROVENANCE.json", provenance)
    marker(done, "graphs-h0", provenance)
    print(json.dumps(provenance, indent=2))


def ricci_command(p: Paths, graph_dir: Path, out_dir: Path, overwrite: bool) -> List[str]:
    cmd = [
        sys.executable, str(p.ricci_source), "--base-dir", str(p.base),
        "--graph-dir", str(graph_dir), "--output-dir", str(out_dir),
        "--n-paths", str(RICCI_N_PATHS), "--max-steps", str(RICCI_MAX_STEPS),
        "--seed", str(GRAPH_SEED), "--beta", str(RICCI_BETA),
        "--epsilon-dist", str(RICCI_EPSILON_DIST), "--c-single-out", str(RICCI_C_SINGLE_OUT),
        "--mu-prune-threshold", str(RICCI_MU_PRUNE), "--edge-weight-attr", "weight",
        "--active-tol", str(ACTIVE_TOL), "--topk-support", str(RICCI_TOPK_SUPPORT),
        "--max-lp-vars", str(RICCI_MAX_LP_VARS), "--progress-every", "250",
    ]
    if overwrite:
        cmd.append("--overwrite")
    return cmd


def audit_ricci_outputs(graph_dir: Path, out_dir: Path, expected: int) -> dict:
    graphs = sorted(graph_dir.glob("*.graphml"))
    markers = sorted(out_dir.glob("*.FAITHFUL_ACTIVE_DONE.marker"))
    files = sorted(out_dir.glob("*_ricci.csv.gz"))
    if len(graphs) != expected or len(markers) != expected or len(files) != expected:
        die(f"Ricci completion mismatch: graphs={len(graphs)}, markers={len(markers)}, files={len(files)}, expected={expected}")
    n_rows = n_nan = n_capped = 0
    for f in files:
        df = pd.read_csv(f)
        n_rows += len(df)
        if "k_ab" not in df:
            die(f"Missing k_ab in {f}")
        n_nan += int(pd.to_numeric(df["k_ab"], errors="coerce").isna().sum())
        if "support_capped" in df:
            n_capped += int(df["support_capped"].astype(str).str.lower().isin({"true", "1"}).sum())
    return {"graphs": len(graphs), "ricci_files": len(files), "ricci_rows": n_rows,
            "nonfinite_k_ab": n_nan, "support_capped_edges": n_capped}


def stage_ricci_pilot(p: Paths, overwrite: bool = False) -> None:
    validate_foundation(p)
    require_file(p.h0 / "GRAPHS_H0_COMPLETE.json", "graphs-H0 marker")
    p.ricci_pilot.mkdir(parents=True, exist_ok=True)
    graph_dir = p.ricci_pilot / "graphs"
    graph_dir.mkdir(parents=True, exist_ok=True)
    pilot = pd.read_csv(p.curated / "ricci_pilot_samples.csv")
    if len(pilot) != 3:
        die("Ricci pilot manifest must contain exactly three samples")
    for sid in pilot["sample_id"].astype(str):
        source = p.graphs / f"{sid}.graphml"
        require_file(source, f"pilot graph {sid}")
        link = graph_dir / source.name
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(source)
    out_dir = p.ricci_pilot / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = ricci_command(p, graph_dir, out_dir, overwrite)
    atomic_json(p.ricci_pilot / "RICCI_PILOT_COMMAND.json", {"argv": cmd, "created_utc": utc_now()})
    subprocess.run(cmd, check=True)
    report = audit_ricci_outputs(graph_dir, out_dir, 3)
    atomic_json(p.ricci_pilot / "RICCI_PILOT_AUDIT.json", report)
    marker(p.ricci_pilot / "RICCI_PILOT_COMPLETE.json", "ricci-pilot", report)
    print(json.dumps(report, indent=2))


def stage_ricci_full(p: Paths, overwrite: bool = False) -> None:
    validate_foundation(p)
    require_file(p.ricci_pilot / "RICCI_PILOT_COMPLETE.json", "Ricci pilot completion marker")
    require_free_gib(p.ext, 7.0, "ricci-full")
    p.ricci.mkdir(parents=True, exist_ok=True)
    cmd = ricci_command(p, p.graphs, p.ricci, overwrite)
    atomic_json(p.ricci / "RICCI_FULL_COMMAND.json", {"argv": cmd, "created_utc": utc_now()})
    subprocess.run(cmd, check=True)
    report = audit_ricci_outputs(p.graphs, p.ricci, EXPECTED_SAMPLES)
    report.update({
        "epsilon_dist": RICCI_EPSILON_DIST, "c_single_out": RICCI_C_SINGLE_OUT,
        "beta": RICCI_BETA, "n_paths": RICCI_N_PATHS, "max_steps": RICCI_MAX_STEPS,
        "seed": GRAPH_SEED, "active_rule": f"weight < 1 - {ACTIVE_TOL}",
        "ricci_source_sha256": sha256(p.ricci_source),
    })
    atomic_json(p.ricci / "RICCI_FULL_AUDIT.json", report)
    marker(p.ricci / "RICCI_FULL_COMPLETE.json", "ricci-full", report)
    print(json.dumps(report, indent=2))


def parse_frozen_edge(value: object) -> Tuple[str, str]:
    s = str(value).strip()
    try:
        obj = ast.literal_eval(s)
        if isinstance(obj, (tuple, list)) and len(obj) == 2:
            return str(obj[0]), str(obj[1])
    except Exception:
        pass
    for sep in (" -> ", "->", "\t", "||"):
        if sep in s:
            a, b = s.split(sep, 1)
            return a.strip().strip("'\"()[]"), b.strip().strip("'\"()[]")
    m = re.match(r"^\(?\s*['\"]?(.*?)['\"]?\s*,\s*['\"]?(.*?)['\"]?\s*\)?$", s)
    if m:
        return m.group(1), m.group(2)
    die(f"Cannot parse frozen edge coordinate: {value!r}")


def write_feature_set(
    out_dir: Path, sample_ids: Sequence[str], edges: Sequence[Tuple[str, str]],
    b: sparse.csr_matrix, k0: sparse.csr_matrix, provenance: Mapping[str, object],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if b.shape != k0.shape or b.shape != (len(sample_ids), len(edges)):
        die(f"Feature matrix shape mismatch in {out_dir}")
    if np.setdiff1d(b.data, np.array([1.0])).size:
        die("B matrix is not binary sparse-one encoded")
    if not np.isfinite(k0.data).all():
        die("K0 matrix contains nonfinite stored values")
    if (k0 - k0.multiply(b)).nnz:
        die("K0 is nonzero where B=0")
    x = sparse.hstack([b.astype(np.float64), k0.astype(np.float64)], format="csr")
    sparse.save_npz(out_dir / "B_matrix.npz", b, compressed=True)
    sparse.save_npz(out_dir / "K0_matrix.npz", k0, compressed=True)
    sparse.save_npz(out_dir / "feature_matrix_B_K0.npz", x, compressed=True)
    pd.DataFrame({"sample_index": np.arange(len(sample_ids)), "sample_id": sample_ids}).to_csv(
        out_dir / "sample_order.csv", index=False
    )
    pd.DataFrame({"edge_index": np.arange(len(edges)), "source": [x[0] for x in edges],
                  "target": [x[1] for x in edges]}).to_csv(out_dir / "edge_metadata.csv", index=False)
    with (out_dir / "feature_names.txt").open("w") as fh:
        for prefix in ("B", "K0"):
            for a, bb in edges:
                fh.write(f"{prefix}::{a}->{bb}\n")
    prov = {**dict(provenance), "n_samples": len(sample_ids), "n_edges": len(edges),
            "matrix_shape": list(x.shape), "layout": "[all B coordinates, all K0 coordinates]"}
    atomic_json(out_dir / "FEATURE_MATRIX_PROVENANCE.json", prov)


def stage_vectorize(p: Paths, overwrite: bool = False) -> None:
    done = p.features / "VECTORIZATION_COMPLETE.json"
    if done.exists() and not overwrite:
        print(f"vectorize: already complete ({done})")
        return
    validate_foundation(p)
    require_file(p.ricci / "RICCI_FULL_COMPLETE.json", "full Ricci marker")
    p.features.mkdir(parents=True, exist_ok=True)
    samples = pd.read_csv(p.scaffold / "sample_order.csv").sort_values("sample_index")
    sample_ids = samples["sample_id"].astype(str).tolist()
    support = sparse.load_npz(p.scaffold / "support_matrix_samples_by_edges.npz").tocsr()
    edge_meta = pd.read_parquet(p.scaffold / "expanded_edge_metadata.parquet").sort_values("edge_index")
    expanded_edges = list(zip(edge_meta["source"].astype(str), edge_meta["target"].astype(str)))
    expanded_index = {e: i for i, e in enumerate(expanded_edges)}
    b = support.copy().astype(np.float64)
    b.data[:] = 1.0
    k_rows, k_cols, k_vals = [], [], []
    n_nonfinite = n_missing = n_extra = 0
    per_sample = []
    for i, sid in enumerate(sample_ids):
        f = p.ricci / f"{sid}_ricci.csv.gz"
        require_file(f, f"Ricci output {sid}")
        df = pd.read_csv(f)
        if df.duplicated(["a", "b"]).any():
            die(f"Duplicate Ricci coordinates for {sid}")
        ricci_map = {(str(r.a), str(r.b)): r.k_ab for r in df.itertuples(index=False)}
        active_cols = set(b.getrow(i).indices.tolist())
        ricci_cols = set()
        for e, val0 in ricci_map.items():
            j = expanded_index.get(e)
            if j is None:
                n_extra += 1
                continue
            ricci_cols.add(j)
            val = pd.to_numeric(pd.Series([val0]), errors="coerce").iloc[0]
            if pd.isna(val) or not math.isfinite(float(val)):
                n_nonfinite += 1
                continue
            if float(val) != 0.0:
                k_rows.append(i); k_cols.append(j); k_vals.append(float(val))
        missing = active_cols - ricci_cols
        if missing:
            n_missing += len(missing)
        extra_active = ricci_cols - active_cols
        if extra_active:
            die(f"Ricci output contains {len(extra_active)} inactive edges for {sid}")
        per_sample.append({"sample_id": sid, "active_edges": len(active_cols), "ricci_rows": len(df),
                           "missing_ricci_rows": len(missing),
                           "nonfinite_k_ab": int(pd.to_numeric(df["k_ab"], errors="coerce").isna().sum())})
    if n_missing or n_extra:
        die(f"Ricci/scaffold coordinate mismatch: missing={n_missing}, extra={n_extra}")
    k0 = sparse.csr_matrix((k_vals, (k_rows, k_cols)), shape=b.shape, dtype=np.float64)
    common_prov = {
        "representation": "expanded-community external-v2 faithful active Ricci",
        "nonfinite_policy": "B remains 1; nonfinite active k_ab is stored as K0=0 and counted",
        "nonfinite_k_ab_total": int(n_nonfinite), "ricci_source_sha256": sha256(p.ricci_source),
        "external_normalization": True,
    }
    write_feature_set(p.features / "full_expanded", sample_ids, expanded_edges, b, k0, common_prov)

    frozen_df = pd.read_csv(p.frozen_edges).sort_values("edge_index")
    frozen_edges = [parse_frozen_edge(x) for x in frozen_df["edge"]]
    if len(frozen_edges) != len(set(frozen_edges)):
        die("Frozen edge axis contains duplicate parsed coordinates")
    frozen_index = {e: j for j, e in enumerate(frozen_edges)}
    fb_rows, fb_cols, fk_rows, fk_cols, fk_vals = [], [], [], [], []
    k0_csr = k0.tocsr()
    for i in range(len(sample_ids)):
        for j in b.getrow(i).indices:
            fj = frozen_index.get(expanded_edges[int(j)])
            if fj is not None:
                fb_rows.append(i); fb_cols.append(fj)
        row = k0_csr.getrow(i)
        for j, val in zip(row.indices, row.data):
            fj = frozen_index.get(expanded_edges[int(j)])
            if fj is not None:
                fk_rows.append(i); fk_cols.append(fj); fk_vals.append(float(val))
    fb = sparse.csr_matrix((np.ones(len(fb_rows)), (fb_rows, fb_cols)), shape=(len(sample_ids), len(frozen_edges)))
    fk = sparse.csr_matrix((fk_vals, (fk_rows, fk_cols)), shape=fb.shape)
    frozen_prov = {
        **common_prov,
        "representation": "frozen IBDMDB edge projection of external-normalized expanded graphs",
        "warning": "exploratory robustness representation; not the frozen-support strict external validation",
        "frozen_edge_metadata_sha256": sha256(p.frozen_edges),
    }
    write_feature_set(p.features / "frozen_edge_projection", sample_ids, frozen_edges, fb, fk, frozen_prov)
    pd.DataFrame(per_sample).to_csv(p.features / "vectorization_audit.csv", index=False)
    report = {
        "n_samples": len(sample_ids), "expanded_edges": len(expanded_edges),
        "frozen_edges": len(frozen_edges), "nonfinite_k_ab_zero_filled": n_nonfinite,
        "expanded_active_B": int(b.nnz), "frozen_projection_active_B": int(fb.nnz),
    }
    marker(done, "vectorize", report)
    print(json.dumps(report, indent=2))


def binary_label(value: object) -> int:
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, (int, np.integer)) and int(value) in (0, 1):
        return int(value)
    if isinstance(value, (float, np.floating)) and math.isfinite(float(value)) and float(value) in (0.0, 1.0):
        return int(value)
    s = re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())
    if s in {"1", "ibd", "case", "cd", "uc", "crohns", "crohnsdisease", "ulcerativecolitis"}:
        return 1
    if s in {"0", "nonibd", "nonibdcontrol", "healthy", "healthycontrol", "control"}:
        return 0
    die(f"Unsupported IBD binary label: {value!r}")


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def metrics_binary(y: np.ndarray, prob: np.ndarray) -> dict:
    from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                                 brier_score_loss, confusion_matrix, f1_score, log_loss, roc_auc_score)
    pred = (prob >= 0.5).astype(int)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    return {
        "n": int(len(y)), "class_counts": {"nonIBD": int((y == 0).sum()), "IBD": int((y == 1).sum())},
        "threshold": 0.5, "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "roc_auc": float(roc_auc_score(y, prob)), "average_precision_ibd": float(average_precision_score(y, prob)),
        "brier": float(brier_score_loss(y, prob)), "log_loss": float(log_loss(y, prob, labels=[0, 1])),
        "recall_nonIBD": float(cm[0, 0] / cm[0].sum()), "recall_IBD": float(cm[1, 1] / cm[1].sum()),
        "confusion_matrix": cm.tolist(),
    }


def stage_predict(p: Paths, overwrite: bool = False) -> None:
    done = p.predictions / "EXPLORATORY_PREDICTION_COMPLETE.json"
    if done.exists() and not overwrite:
        print(f"predict: already complete ({done})")
        return
    validate_foundation(p)
    require_file(p.features / "VECTORIZATION_COMPLETE.json", "vectorization marker")
    require_file(p.frozen_model, "frozen Ricci C=0.02 model", EXPECTED_FROZEN_MODEL_SHA256)
    p.predictions.mkdir(parents=True, exist_ok=True)
    x = sparse.load_npz(p.features / "frozen_edge_projection" / "feature_matrix_B_K0.npz").toarray().astype(np.float64)
    order = pd.read_csv(p.features / "frozen_edge_projection" / "sample_order.csv")
    with np.load(p.frozen_model, allow_pickle=False) as z:
        coef = np.asarray(z["coef"], dtype=np.float64).reshape(-1)
        intercept = float(np.asarray(z["intercept"]).reshape(-1)[0])
        mean = np.asarray(z["scaler_mean"], dtype=np.float64).reshape(-1)
        scale = np.asarray(z["scaler_scale"], dtype=np.float64).reshape(-1)
    if x.shape[1] != len(coef) or len(mean) != len(coef) or len(scale) != len(coef):
        die(f"Frozen projection/model feature mismatch: X={x.shape}, coef={coef.shape}")
    if np.any(scale <= 0) or not np.isfinite(scale).all():
        die("Frozen model scaler is invalid")
    score = ((x - mean) / scale) @ coef + intercept
    prob = sigmoid(score)
    pred = pd.DataFrame({
        "sample_id": order["sample_id"].astype(str), "decision_score": score,
        "probability_IBD": prob, "predicted_label_0p5": np.where(prob >= 0.5, "IBD", "nonIBD"),
    })
    # Probabilities are created before labels are read.
    pred.to_csv(p.predictions / "label_blind_sample_probabilities.csv", index=False)
    meta = pd.read_csv(p.curated / "matched_external_metadata_184.tsv.gz", sep="\t")
    label_col = next((c for c in ["external_label_ibd_binary", "PRJEB42155_ge50_clean_metadata_full__external_label_ibd_binary"] if c in meta), None)
    if label_col is None:
        die("Cannot locate binary IBD label column in matched external metadata")
    eval_df = pred.merge(meta, left_on="sample_id", right_on="sample_id_v2", how="left", validate="one_to_one")
    eval_df["y_binary"] = eval_df[label_col].map(binary_label)
    sample_metrics = metrics_binary(eval_df["y_binary"].to_numpy(int), eval_df["probability_IBD"].to_numpy(float))
    participant_col = next((c for c in ["participant_id", "host_subject_id", "patient_id"] if c in eval_df), None)
    participant_metrics = None
    if participant_col is not None:
        ids = eval_df[participant_col].astype(str)
        missing = eval_df[participant_col].isna() | ids.str.strip().str.lower().isin({"", "nan", "none", "missing"})
        eval_df["participant_id_v2"] = ids
        eval_df.loc[missing, "participant_id_v2"] = "sample::" + eval_df.loc[missing, "sample_id"].astype(str)
        consistency = eval_df.groupby("participant_id_v2")["y_binary"].nunique()
        if (consistency > 1).any():
            die("Participant groups contain inconsistent labels")
        part = eval_df.groupby("participant_id_v2", as_index=False).agg(
            probability_IBD=("probability_IBD", "mean"), y_binary=("y_binary", "first"), n_samples=("sample_id", "size")
        )
        part["predicted_label_0p5"] = np.where(part["probability_IBD"] >= 0.5, "IBD", "nonIBD")
        part.to_csv(p.predictions / "participant_predictions_probability_averaged.csv", index=False)
        participant_metrics = metrics_binary(part["y_binary"].to_numpy(int), part["probability_IBD"].to_numpy(float))
    eval_df.to_csv(p.predictions / "sample_predictions_with_labels.csv", index=False)
    report = {
        "interpretation": "exploratory frozen-model transfer to external-normalized expanded-graph frozen coordinates; not strict external validation",
        "sample": sample_metrics, "participant_probability_averaged": participant_metrics,
        "model_sha256": sha256(p.frozen_model), "n_features": int(x.shape[1]),
    }
    atomic_json(p.predictions / "exploratory_external_metrics.json", report)
    marker(done, "predict", {"sample_predictions": len(pred), "model_sha256": sha256(p.frozen_model)})
    print(json.dumps(report, indent=2))


def stage_status(p: Paths) -> None:
    checks = [
        ("prepare", p.curated / "PREPARE_COMPLETE.json"),
        ("scaffold", p.scaffold / "SCAFFOLD_COMPLETE.json"),
        ("graphs-h0", p.h0 / "GRAPHS_H0_COMPLETE.json"),
        ("ricci-pilot", p.ricci_pilot / "RICCI_PILOT_COMPLETE.json"),
        ("ricci-full", p.ricci / "RICCI_FULL_COMPLETE.json"),
        ("vectorize", p.features / "VECTORIZATION_COMPLETE.json"),
        ("predict", p.predictions / "EXPLORATORY_PREDICTION_COMPLETE.json"),
    ]
    rows = []
    for name, path in checks:
        rows.append({"stage": name, "complete": path.exists(), "marker": str(path)})
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"disk_free_GiB={shutil.disk_usage(p.ext).free/(1024**3):.2f}")
    if p.graphs.exists():
        print(f"active_graphml={len(list(p.graphs.glob('*.graphml')))}")
    if p.ricci.exists():
        print(f"ricci_markers={len(list(p.ricci.glob('*.FAITHFUL_ACTIVE_DONE.marker')))}")


def main() -> None:
    home = Path.home()
    default_base = home / "Real_Data"
    default_ext = default_base / "external_validation" / "serrano_gomez_ibd" / "ge50_external_validation"
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("stage", choices=["prepare", "scaffold", "graphs-h0", "ricci-pilot", "ricci-full", "vectorize", "predict", "status"])
    ap.add_argument("--base", type=Path, default=default_base)
    ap.add_argument("--external-root", type=Path, default=default_ext)
    ap.add_argument("--output-root", type=Path, default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    base = args.base.expanduser().resolve()
    ext = args.external_root.expanduser().resolve()
    out = (args.output_root or (ext / "expanded_community_structural_replication_v2")).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    paths = Paths(base, ext, out)
    paths.logs.mkdir(parents=True, exist_ok=True)
    print("=" * 108)
    print("EXPANDED EXTERNAL V2 — AGORA COMMUNITY STRUCTURAL REPLICATION")
    print("=" * 108)
    print(f"stage: {args.stage}\nexternal root: {ext}\noutput root: {out}")
    dispatch = {
        "prepare": stage_prepare, "scaffold": stage_scaffold, "graphs-h0": stage_graphs_h0,
        "ricci-pilot": stage_ricci_pilot, "ricci-full": stage_ricci_full,
        "vectorize": stage_vectorize, "predict": stage_predict,
    }
    if args.stage == "status":
        stage_status(paths)
    else:
        dispatch[args.stage](paths, args.overwrite)


if __name__ == "__main__":
    main()
