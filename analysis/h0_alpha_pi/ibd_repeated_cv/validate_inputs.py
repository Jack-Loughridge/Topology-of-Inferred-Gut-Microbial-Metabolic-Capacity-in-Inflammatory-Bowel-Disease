#!/usr/bin/env python3
"""Preflight validation for the repeated H0 Alpha-Pi analysis."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from h0_alpha_pi_repeated_cv import (
    discover_pd_paths,
    load_and_validate_manifest,
    load_h0_bundle,
    make_input_fingerprint,
)


def main() -> None:
    home = Path.home() / "Real_Data"
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--pd-dir", type=Path, default=home / "out_pds")
    parser.add_argument("--label-csv", type=Path, default=home / "sample_labels.csv")
    parser.add_argument("--metadata-csv", type=Path, default=home / "hmp2_metadata.csv")
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=home / "Ricci_IBD_RepeatedCV_CPath" / "splits" / "sample_split_manifest.csv",
    )
    parser.add_argument("--expected-repeats", type=int, default=20)
    parser.add_argument("--expected-splits", type=int, default=5)
    parser.add_argument("--require-exact-pd-set", action="store_true")
    args = parser.parse_args()

    manifest, sample_meta, n_repeats, n_splits = load_and_validate_manifest(args.split_manifest)
    if n_repeats != args.expected_repeats:
        raise ValueError(f"Expected {args.expected_repeats} repeats, manifest has {n_repeats}")
    if n_splits != args.expected_splits:
        raise ValueError(f"Expected {args.expected_splits} folds, manifest has {n_splits}")

    bundle = load_h0_bundle(
        args.pd_dir,
        args.label_csv,
        args.metadata_csv,
        sample_meta,
        args.require_exact_pd_set,
    )
    fingerprint = make_input_fingerprint(
        args.split_manifest,
        args.label_csv,
        args.metadata_csv,
        bundle.pd_paths,
    )

    deaths = np.asarray([len(x) for x in bundle.deaths_list], dtype=int)
    participant_meta = bundle.metadata[["participant_id", "label"]].drop_duplicates("participant_id")
    sample_counts = bundle.metadata["label"].value_counts().sort_index()
    participant_counts = participant_meta["label"].value_counts().sort_index()

    print("=" * 100)
    print("REPEATED H0 ALPHA-PI PREFLIGHT: PASSED")
    print("=" * 100)
    print(f"Split manifest        : {args.split_manifest}")
    print(f"Outer design          : {n_repeats} repeats x {n_splits} folds")
    print(f"H0 directory          : {args.pd_dir}")
    print(f"Samples               : {len(bundle.sample_ids)}")
    print(f"Participants          : {bundle.metadata['participant_id'].nunique()}")
    print(f"Sample non-IBD / IBD  : {sample_counts.get(0, 0)} / {sample_counts.get(1, 0)}")
    print(f"Person non-IBD / IBD  : {participant_counts.get(0, 0)} / {participant_counts.get(1, 0)}")
    print(f"H0 deaths per sample  : min={deaths.min()}, median={np.median(deaths):.1f}, max={deaths.max()}")
    print(f"CUDA available        : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device           : {torch.cuda.get_device_name(0)}")
    print("Input fingerprints:")
    for key, value in fingerprint.items():
        print(f"  {key}: {value}")
    print("\nValidated safeguards:")
    print("  - every manifest sample has exactly one participant and binary label")
    print("  - no participant is assigned both labels")
    print("  - every repetition tests every sample exactly once")
    print("  - no sample or participant overlaps train and test within a fold")
    print("  - every manifest sample has a label row, metadata row and H0 diagram")
    print("  - label/participant validation is restricted to the locked manifest sample set")
    print("  - label and participant metadata agree with the shared Ricci manifest")
    print("=" * 100)


if __name__ == "__main__":
    main()
