#!/usr/bin/env python3
"""Preflight validation for the repeated species-abundance benchmarks."""

from __future__ import annotations

import argparse
from pathlib import Path

from species_benchmarks_repeated_cv import (
    config_from_args,
    load_and_verify_inputs,
    package_versions,
    parse_args as parse_main_args,
)


def parse_args() -> argparse.Namespace:
    base = Path.home() / "Real_Data"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--species-file",
        type=Path,
        default=base / "Real_Species_Abundances_canon.xlsx",
    )
    parser.add_argument("--species-sheet", default="Sheet1")
    parser.add_argument("--label-csv", type=Path, default=base / "sample_labels.csv")
    parser.add_argument("--metadata-csv", type=Path, default=base / "hmp2_metadata.csv")
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=base
        / "Ricci_IBD_RepeatedCV_CPath"
        / "splits"
        / "sample_split_manifest.csv",
    )
    parser.add_argument("--expected-repeats", type=int, default=20)
    parser.add_argument("--expected-folds", type=int, default=5)
    parser.add_argument(
        "--zero-profile-policy", choices=["exclude", "error"], default="exclude"
    )
    parser.add_argument("--expected-zero-profiles", type=int, default=10)
    parser.add_argument("--require-xgboost", action="store_true", default=True)
    parser.add_argument("--no-require-xgboost", dest="require_xgboost", action="store_false")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    main_argv = [
        "--species-file",
        str(args.species_file),
        "--species-sheet",
        str(args.species_sheet),
        "--label-csv",
        str(args.label_csv),
        "--metadata-csv",
        str(args.metadata_csv),
        "--split-manifest",
        str(args.split_manifest),
        "--expected-repeats",
        str(args.expected_repeats),
        "--expected-folds",
        str(args.expected_folds),
        "--zero-profile-policy",
        str(args.zero_profile_policy),
        "--expected-zero-profiles",
        str(args.expected_zero_profiles),
    ]
    config = config_from_args(parse_main_args(main_argv))
    versions = package_versions()
    if args.require_xgboost and versions.get("xgboost") is None:
        raise RuntimeError(
            "XGBoost is not importable. Install requirements.txt before launching all benchmarks."
        )
    bundle = load_and_verify_inputs(config)
    master = bundle["sample_master"]

    print("=" * 100)
    print("SPECIES BENCHMARK PREFLIGHT")
    print("=" * 100)
    print("Species file:", config.species_file)
    print("Split manifest:", config.split_manifest)
    print("Source manifest samples:", len(bundle["source_sample_master"]))
    print("Complete-case samples:", len(master))
    print("Excluded all-zero species profiles:", len(bundle["excluded_samples"]))
    if len(bundle["excluded_samples"]):
        print("Excluded sample IDs:")
        print(bundle["excluded_samples"][["sample_id", "participant_id", "label_name", "exclusion_reason"]].to_string(index=False))
    print("Participants retained:", master["participant_id"].nunique())
    print("Species features:", len(bundle["feature_names"]))
    print("Sample class counts:")
    print(master["label"].map({0: "non-IBD", 1: "IBD"}).value_counts().to_string())
    print("Participant class counts:")
    print(
        master.drop_duplicates("participant_id")["label"]
        .map({0: "non-IBD", 1: "IBD"})
        .value_counts()
        .to_string()
    )
    print("Species diagnostics:")
    for key, value in bundle["species_diagnostics"].items():
        print(f"  {key}: {value}")
    print("Software:")
    for key, value in versions.items():
        print(f"  {key}: {value}")
    print("=" * 100)
    print("REPEATED SPECIES BENCHMARK PREFLIGHT: PASSED")
    print("=" * 100)


if __name__ == "__main__":
    main()
