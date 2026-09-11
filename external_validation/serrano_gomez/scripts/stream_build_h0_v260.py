#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

REAL = Path.home() / "Real_Data"
GE50 = REAL / "external_validation" / "serrano_gomez_ibd" / "ge50_external_validation"
PROFILE_DIR = GE50 / "metaphlan2_v260_profiles"
OUT = GE50 / "structural_frozen_ibdmdb"
ESV_PATH = GE50 / "scripts" / "serrano_gomez_external_structural_frozen_ibdmdb_v1" / "external_structural_validation.py"
MAP_PATH = REAL / "ibdmdb_raw_to_canonical_species_mapping_candidate.csv"
RAW_TRAIN = REAL / "Real_Species_Abundances.xlsx"
CAN_TRAIN = REAL / "Real_Species_Abundances_canon.xlsx"
MIN_MAPPED_FRACTION = 0.50
TOL = 1e-10


def load_esv():
    spec = importlib.util.spec_from_file_location("external_structural_validation", ESV_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {ESV_PATH}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def is_species_taxon(tax: str) -> bool:
    return (("|s__" in tax and "|t__" not in tax) or tax.startswith("s__"))


def terminal_taxon_name(tax: str) -> str:
    part = str(tax).split("|")[-1]
    if "__" in part:
        return part.split("__", 1)[1]
    return part


def read_full_metaphlan_profile(path: Path):
    vals = defaultdict(float)
    with path.open("rt", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\n\r")
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            tax = parts[0].strip()
            numeric = []
            for x in parts[1:]:
                try:
                    numeric.append(float(x))
                except Exception:
                    pass
            if not numeric:
                continue
            val = numeric[-1]
            if not np.isfinite(val) or val < 0:
                raise RuntimeError(f"Invalid abundance {val!r} in {path}: {tax}")
            vals[tax] += float(val)

    if not vals:
        raise RuntimeError(f"No taxonomic abundance rows parsed from {path}")

    species_vals = [v for k, v in vals.items() if is_species_taxon(k)]
    if not species_vals:
        raise RuntimeError(f"No species-level rows parsed from {path}")

    species_total = float(sum(species_vals))
    species_max = float(max(species_vals))
    scale = 1.0
    if 0 < species_total <= 1.5 and species_max <= 1.0 + 1e-9:
        vals = defaultdict(float, {k: 100.0 * v for k, v in vals.items()})
        species_total *= 100.0
        species_max *= 100.0
        scale = 100.0

    if species_total <= 0 or species_total > 110.0:
        raise RuntimeError(f"Species-level abundance sum={species_total:.6f}% is invalid for {path}")

    return dict(vals), {
        "raw_species_total_percent": species_total,
        "species_abundance_max_percent": species_max,
        "unit_scale_factor_applied": scale,
    }


def load_mapping():
    if not MAP_PATH.exists():
        raise RuntimeError(f"Missing validated mapping candidate: {MAP_PATH}")
    m = pd.read_csv(MAP_PATH)
    need = {"raw_column", "canonical_target", "mapping_type"}
    if not need.issubset(m.columns):
        raise RuntimeError(f"Mapping file lacks columns {sorted(need)}: {MAP_PATH}")
    m = m[list(need)].dropna(subset=["raw_column", "canonical_target"]).copy()
    m["raw_column"] = m["raw_column"].astype(str)
    m["canonical_target"] = m["canonical_target"].astype(str)
    m["mapping_type"] = m["mapping_type"].astype(str)
    m = m.drop_duplicates(["raw_column", "canonical_target"]).reset_index(drop=True)
    return m


def audit_mapping_for_graph(esv, scaffold: pd.DataFrame, training_cols, mapping: pd.DataFrame):
    # Only catalyst coordinates in the accepted frozen scaffold can affect graph weights.
    graph_catalysts = set()
    for x in scaffold["catalysts_json"]:
        graph_catalysts.update(json.loads(x))
    graph_targets = sorted(set(training_cols) & graph_catalysts)

    mapped_targets = set(mapping["canonical_target"])
    missing_targets = sorted(set(graph_targets) - mapped_targets)

    raw = pd.read_excel(RAW_TRAIN)
    can = pd.read_excel(CAN_TRAIN)
    if len(raw) != len(can):
        raise RuntimeError("Training raw/canonical row counts differ")
    if not np.array_equal(raw.iloc[:, 0].astype(str).to_numpy(), can.iloc[:, 0].astype(str).to_numpy()):
        raise RuntimeError("Training raw/canonical rows are not positionally aligned")

    missing_raw_sources = sorted(set(mapping["raw_column"]) - set(raw.columns.astype(str)))
    if missing_raw_sources:
        raise RuntimeError("Mapping source columns missing from raw training table: " + ", ".join(missing_raw_sources[:20]))

    audit = []
    bad = []
    for target in graph_targets:
        srcs = mapping.loc[mapping["canonical_target"].eq(target), "raw_column"].drop_duplicates().tolist()
        recon = np.zeros(len(raw), dtype=float)
        for src in srcs:
            recon += pd.to_numeric(raw[src], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        truth = pd.to_numeric(can[target], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        d = np.abs(recon - truth)
        row = {
            "canonical_target": target,
            "n_sources": len(srcs),
            "max_abs_diff": float(d.max()),
            "mean_abs_diff": float(d.mean()),
        }
        audit.append(row)
        if d.max() >= TOL:
            bad.append(row)

    audit_path = OUT / "audits" / "canonical_mapping_graph_relevant_audit.csv"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(audit).to_csv(audit_path, index=False)
    if bad:
        msg = "; ".join(f"{x['canonical_target']} max={x['max_abs_diff']:.6g}" for x in bad[:20])
        raise RuntimeError("Raw->canonical mapping is not exact for graph-relevant catalysts: " + msg)

    # A graph-relevant coordinate with no recovered source has just been
    # reconstructed as the zero vector above. Reaching this point therefore
    # proves that its frozen canonical training coordinate is also identically
    # zero across all training rows.
    zero_source_targets = missing_targets
    if zero_source_targets:
        print(
            "[audit] allowing source-less graph catalysts proven all-zero in training: "
            + ", ".join(zero_source_targets),
            flush=True,
        )

    # Freeze mapping identity so outputs cannot silently mix mappings.
    map_sha = esv.sha256_file(MAP_PATH)
    sig_path = OUT / "provenance" / "METAPHLAN2_V260_CANONICAL_MAPPING.json"
    sig = {
        "mapping_path": str(MAP_PATH),
        "mapping_sha256": map_sha,
        "training_raw_path": str(RAW_TRAIN),
        "training_raw_sha256": esv.sha256_file(RAW_TRAIN),
        "training_canonical_path": str(CAN_TRAIN),
        "training_canonical_sha256": esv.sha256_file(CAN_TRAIN),
        "n_graph_relevant_training_catalysts": len(graph_targets),
        "graph_relevant_zero_source_training_catalysts": zero_source_targets,
        "graph_relevant_mapping_validation": (
            "exact_across_all_1360_rows_or_zero_source_coordinate_proven_all_zero"
        ),
        "tolerance": TOL,
        "profile_source": "MetaPhlAn2 v2.6.0 / mpa_v20_m200",
        "no_external_renormalisation": True,
    }
    if sig_path.exists():
        old = json.loads(sig_path.read_text())
        for key in ["mapping_sha256", "training_raw_sha256", "training_canonical_sha256"]:
            if old.get(key) != sig.get(key):
                raise RuntimeError(f"Frozen MetaPhlAn2 canonicalization provenance changed for {key}; use a new output directory")
    else:
        esv.json_dump(sig_path, sig)
        # Human-readable frozen copy.
        mapping.to_csv(OUT / "provenance" / "ibdmdb_raw_to_canonical_species_mapping_frozen.csv", index=False)

    return map_sha, graph_targets, zero_source_targets


def canonicalize_profile(
    path: Path, training_cols, mapping: pd.DataFrame, zero_source_targets
):
    vals, meta0 = read_full_metaphlan_profile(path)
    out = pd.Series(0.0, index=list(training_cols), dtype=float)

    source_to_targets = defaultdict(list)
    for _, r in mapping.iterrows():
        source_to_targets[str(r["raw_column"])].append((str(r["canonical_target"]), str(r["mapping_type"])))

    mapped_species_sources = set()
    audit_rows = []

    for source, targets in source_to_targets.items():
        if source not in vals:
            continue
        val = float(vals[source])
        for target, method in targets:
            if target not in out.index:
                raise RuntimeError(f"Mapping target absent from frozen training axis: {target}")
            out.loc[target] += val
            audit_rows.append({
                "raw_taxon": source,
                "abundance_percent": val,
                "canonical_target": target,
                "mapping_type": method,
                "rank": source.split("|")[-1].split("__", 1)[0] if "__" in source.split("|")[-1] else "",
            })
        if is_species_taxon(source):
            mapped_species_sources.add(source)

    # Only for graph-relevant coordinates that were proven identically zero
    # throughout IBDMDB training, permit exact terminal-species identity when
    # that old-MetaPhlAn species appears externally. No synonym, edit-distance,
    # SGB or other fuzzy mapping is performed.
    allowed_zero_source = set(zero_source_targets)
    for tax, val in vals.items():
        if not is_species_taxon(tax) or tax in mapped_species_sources:
            continue
        terminal = terminal_taxon_name(tax)
        if terminal not in allowed_zero_source:
            continue
        out.loc[terminal] += float(val)
        mapped_species_sources.add(tax)
        audit_rows.append({
            "raw_taxon": tax,
            "abundance_percent": float(val),
            "canonical_target": terminal,
            "mapping_type": "exact_terminal_all_zero_training_fallback",
            "rank": "s",
        })

    species_total = float(meta0["raw_species_total_percent"])
    mapped_species_total = float(sum(vals[s] for s in mapped_species_sources))
    mapped_fraction = mapped_species_total / species_total if species_total > 0 else 0.0

    # Add explicit species-level unmatched rows to the audit.
    for tax, val in vals.items():
        if is_species_taxon(tax) and tax not in mapped_species_sources:
            audit_rows.append({
                "raw_taxon": tax,
                "abundance_percent": float(val),
                "canonical_target": "",
                "mapping_type": "unmatched_species",
                "rank": "s",
            })

    meta = {
        **meta0,
        "mapped_abundance_total_percent": mapped_species_total,
        "mapped_fraction_of_species_abundance": mapped_fraction,
        "canonical_feature_sum_percent": float(out.sum()),
        "n_external_species": int(sum(1 for k in vals if is_species_taxon(k))),
        "n_training_species_nonzero": int((out > 0).sum()),
        "no_external_renormalisation": True,
        "canonicalization_source": str(MAP_PATH),
    }
    return out, meta, pd.DataFrame(audit_rows)


def main():
    esv = load_esv()
    if not PROFILE_DIR.exists():
        raise RuntimeError(f"MetaPhlAn2 profile directory not found: {PROFILE_DIR}")

    _, scaffold, baselines = esv.require_preflight(OUT)
    training_cols, training_stats = esv.load_training_species_axis(CAN_TRAIN)
    mapping = load_mapping()
    map_sha, _, zero_source_targets = audit_mapping_for_graph(
        esv, scaffold, training_cols, mapping
    )

    graphs_dir = OUT / "graphs"
    edges_dir = OUT / "edges"
    h0_dir = OUT / "h0"
    species_dir = OUT / "species_projection"
    marker_dir = OUT / "sample_markers"
    ricci_dir = OUT / "ricci_raw"
    for d in [graphs_dir, edges_dir, h0_dir, species_dir, marker_dir, ricci_dir, OUT / "audits"]:
        d.mkdir(parents=True, exist_ok=True)

    complete = sorted(PROFILE_DIR.glob("*.COMPLETE"))
    print(f"[scan] MetaPhlAn2 COMPLETE markers: {len(complete)}", flush=True)

    built = skipped = failed = 0
    status = []

    for cm in complete:
        sid = cm.name[:-len(".COMPLETE")]
        profile = PROFILE_DIR / f"{sid}_profile.tsv"
        if not profile.exists() or profile.stat().st_size == 0:
            print(f"[FAILED] {sid}: COMPLETE marker exists but profile is missing/empty", flush=True)
            failed += 1
            continue

        profile_sha = esv.sha256_file(profile)
        marker_path = marker_dir / f"{sid}.json"
        graph_path = graphs_dir / f"{sid}.graphml"
        h0_path = h0_dir / f"{sid}_H0.npy"
        edges_path = edges_dir / f"{sid}_edges.csv.gz"
        proj_path = species_dir / f"{sid}_species_projection.csv.gz"

        old = json.loads(marker_path.read_text()) if marker_path.exists() else None
        unchanged = (
            old is not None
            and old.get("profile_sha256") == profile_sha
            and old.get("canonical_mapping_sha256") == map_sha
            and graph_path.exists() and h0_path.exists() and edges_path.exists() and proj_path.exists()
        )
        if unchanged:
            skipped += 1
            continue

        try:
            abundance, mmap, audit_df = canonicalize_profile(
                profile, training_cols, mapping, zero_source_targets
            )
            mapped_frac = float(mmap["mapped_fraction_of_species_abundance"])
            if mapped_frac < MIN_MAPPED_FRACTION:
                raise RuntimeError(
                    f"Mapped only {mapped_frac:.3%} of species-level abundance, below required {MIN_MAPPED_FRACTION:.3%}"
                )

            supports = esv.compute_external_supports(scaffold, abundance)
            G, edge_df = esv.build_external_graph(sid, scaffold, baselines, supports)
            if G.number_of_edges() != len(scaffold):
                raise RuntimeError(f"Graph edges={G.number_of_edges()} but frozen scaffold={len(scaffold)}")
            active_edges = sum(float(d["weight"]) < 1.0 - esv.ACTIVE_TOL for _, _, d in G.edges(data=True))
            if active_edges <= 0:
                raise RuntimeError("External graph has zero active edges")

            pd_h0 = esv.h0_pd_from_digraph(G)
            nontrivial_h0 = esv.filtered_h0_deaths(pd_h0)

            tmp_graph = graph_path.with_suffix(".graphml.tmp")
            nx.write_graphml(G, tmp_graph)
            tmp_graph.replace(graph_path)
            edge_df.to_csv(edges_path, index=False, compression="gzip")
            audit_df.to_csv(proj_path, index=False, compression="gzip")
            np.save(h0_path, pd_h0, allow_pickle=False)

            # Any rebuilt graph invalidates any previously computed Ricci for this sample.
            # This is intentionally unconditional on rebuild so stale results can never survive.
            esv.invalidate_ricci_for_sample(OUT, sid)

            marker = {
                "sample_id": sid,
                "status": "built_metaphlan2_v260",
                "profile_path": str(profile),
                "profile_sha256": profile_sha,
                "canonical_mapping_sha256": map_sha,
                **mmap,
                "training_abundance_row_sum_median": training_stats["row_sum_median"],
                "n_template_edges": int(G.number_of_edges()),
                "n_active_edges": int(active_edges),
                "n_h0_finite_merges": int(len(pd_h0)),
                "n_h0_nontrivial_deaths_0_lt_d_lt_1": int(len(nontrivial_h0)),
                "graph_path": str(graph_path),
                "h0_path": str(h0_path),
                "profile_generation": "MetaPhlAn2 v2.6.0 / mpa_v20_m200",
                "no_external_cohort_baseline": True,
                "no_external_compositional_renormalisation": True,
                "denominator_source": str(esv.FROZEN_BASELINES_PATH),
            }
            esv.json_dump(marker_path, marker)
            status.append(marker)
            built += 1
            print(
                f"[built] {sid}: mapped={mapped_frac:.1%}, active_edges={active_edges:,}, H0_nontrivial={len(nontrivial_h0):,}",
                flush=True,
            )
        except Exception as e:
            failed += 1
            status.append({"sample_id": sid, "status": "FAILED", "profile_path": str(profile), "error": f"{type(e).__name__}: {e}"})
            print(f"[FAILED] {sid}: {type(e).__name__}: {e}", flush=True)

    if status:
        pd.DataFrame(status).to_csv(OUT / "audits" / "stream_build_status_v260.csv", index=False)

    complete_sids = sorted(p.name[:-len("_H0.npy")] for p in h0_dir.glob("*_H0.npy"))
    try:
        esv.metadata_for_samples(complete_sids).to_csv(OUT / "external_sample_metadata.csv", index=False)
    except Exception as e:
        print(f"[warning] metadata refresh failed: {type(e).__name__}: {e}", flush=True)

    print(f"[summary] built={built} skipped={skipped} failed={failed} graphs={len(list(graphs_dir.glob('*.graphml')))} h0={len(list(h0_dir.glob('*_H0.npy')))}", flush=True)


if __name__ == "__main__":
    main()
