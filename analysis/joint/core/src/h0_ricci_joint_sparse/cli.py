from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .config import RunConfig
from .cv import run_nested_cv
from .data import load_all_inputs


def _int_tuple(value: str) -> tuple[int, ...]:
    parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not parsed:
        raise argparse.ArgumentTypeError("Expected at least one comma-separated integer.")
    return parsed


def _float_tuple(value: str) -> tuple[float, ...]:
    parsed = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not parsed:
        raise argparse.ArgumentTypeError("Expected at least one comma-separated number.")
    if any(item <= 0 for item in parsed):
        raise argparse.ArgumentTypeError("All block penalties must be positive.")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    home = Path.home()
    stamp = time.strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(
        description=(
            "Leak-safe joint train-only WKPI + faithful Ricci classifier with "
            "warm-started KKT-safe block-L1 active sets."
        )
    )
    parser.add_argument(
        "--h0-results-dir", type=Path,
        default=home / "Real_Data" / "H0_AlphaPi_NestedQ_TrainOnlyBins",
        help="Existing train-only H0 nested-run directory used as the canonical H0 reference.",
    )
    parser.add_argument(
        "--h0-pd-dir", type=Path, default=None,
        help="Raw H0 NPY directory. Auto-resolves to ~/Real_Data/out_pds when omitted.",
    )
    parser.add_argument(
        "--ricci-original-dir", type=Path,
        default=home / "Real_Data" / "Ricci_Classifier_OriginalStyle_AllTasks_NewVectors",
        help="Original-style Ricci outputs used for biological feature annotations and comparisons.",
    )
    parser.add_argument(
        "--ricci-feature-dir", type=Path,
        default=home / "Real_Data" / "Ricci_Classifier_Faithful_Eps0001_n250_v3",
        help="Faithful feature_matrix_B_K0.npz and matched_metadata.csv directory.",
    )
    parser.add_argument(
        "--out-dir", type=Path,
        default=home / "Real_Data" / f"H0_Ricci_JointSparse_Nested_{stamp}",
    )
    parser.add_argument("--tasks", default="all", help="all or comma-separated task folders.")
    parser.add_argument("--n-outer-splits", type=int, default=5)
    parser.add_argument("--n-inner-splits", type=int, default=3)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--n-jobs", type=int, default=3,
        help="Parallel independent M paths. Each worker is restricted to one BLAS thread.",
    )
    parser.add_argument(
        "--selection-metric", choices=("accuracy", "balanced_accuracy", "macro_f1", "roc_auc"),
        default="roc_auc",
    )
    parser.add_argument("--interval-grid", type=_int_tuple, default=(96, 160, 224))
    parser.add_argument("--lambda-h0-grid", type=_float_tuple, default=(0.5, 1.0, 2.0))
    parser.add_argument("--lambda-ricci-grid", type=_float_tuple, default=(0.5, 1.0, 2.0))
    parser.add_argument("--C", dest="c_value", type=float, default=0.02)
    parser.add_argument(
        "--logistic-max-iter", type=int, default=8000,
        help="Total proximal iterations available to each logistic fit across working-set rounds.",
    )
    parser.add_argument(
        "--logistic-tol", type=float, default=1e-4,
        help="Required full KKT tolerance for every fitted logistic head.",
    )
    parser.add_argument("--alternations", type=int, default=6)
    parser.add_argument("--alpha-smoothness-gamma", type=float, default=0.01)
    parser.add_argument("--alpha-optimizer-max-iter", type=int, default=100)
    parser.add_argument("--quad-points", type=int, default=5)
    parser.add_argument("--common-alpha-grid-size", type=int, default=300)
    parser.add_argument("--top-coefficients", type=int, default=100)
    parser.add_argument("--active-set-initial", type=int, default=512)
    parser.add_argument("--active-set-batch", type=int, default=512)
    parser.add_argument("--active-set-max-rounds", type=int, default=50)
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Ignore existing configuration/fold checkpoints in --out-dir.",
    )
    parser.add_argument(
        "--benchmark-first-inner-fold", action="store_true",
        help=(
            "Run exactly one complete 27-configuration inner fold, write timing/KKT "
            "results, and stop before outer-model fitting."
        ),
    )
    parser.add_argument(
        "--validate-only", action="store_true",
        help="Load and audit all inputs, then stop before model fitting.",
    )
    return parser


def _config_from_args(args: argparse.Namespace) -> RunConfig:
    return RunConfig(
        h0_results_dir=args.h0_results_dir.expanduser().resolve(),
        h0_pd_dir=args.h0_pd_dir.expanduser().resolve() if args.h0_pd_dir else None,
        ricci_original_dir=args.ricci_original_dir.expanduser().resolve(),
        ricci_feature_dir=args.ricci_feature_dir.expanduser().resolve(),
        out_dir=args.out_dir.expanduser().resolve(),
        tasks=args.tasks,
        n_outer_splits=args.n_outer_splits,
        n_inner_splits=args.n_inner_splits,
        seed=args.seed,
        n_jobs=max(1, args.n_jobs),
        selection_metric=args.selection_metric,
        interval_grid=tuple(args.interval_grid),
        lambda_h0_grid=tuple(args.lambda_h0_grid),
        lambda_ricci_grid=tuple(args.lambda_ricci_grid),
        c_value=args.c_value,
        logistic_max_iter=args.logistic_max_iter,
        logistic_tolerance=args.logistic_tol,
        n_alternations=args.alternations,
        alpha_smoothness_gamma=args.alpha_smoothness_gamma,
        alpha_optimizer_max_iter=args.alpha_optimizer_max_iter,
        quad_points=args.quad_points,
        common_alpha_grid_size=args.common_alpha_grid_size,
        top_coefficients=args.top_coefficients,
        active_set_initial=max(0, args.active_set_initial),
        active_set_batch=max(1, args.active_set_batch),
        active_set_max_rounds=max(1, args.active_set_max_rounds),
        resume=not args.no_resume,
        benchmark_first_inner_fold=bool(args.benchmark_first_inner_fold),
    )


def _reference_output_audit(config: RunConfig) -> pd.DataFrame:
    task_pairs = (
        ("non-IBD vs UC vs CD", "3way", "three_way_nonIBD_UC_CD"),
        ("IBD vs non-IBD", "IBD_vs_nonIBD", "IBD_vs_nonIBD"),
        ("non-IBD vs UC", "nonIBD_vs_UC", "nonIBD_vs_UC"),
        ("non-IBD vs CD", "nonIBD_vs_CD", "nonIBD_vs_CD"),
        ("CD vs UC", "CD_vs_UC", "CD_vs_UC"),
    )
    rows: list[dict] = []
    for task, h0_folder, ricci_folder in task_pairs:
        h0_root = config.h0_results_dir / h0_folder
        ricci_root = config.ricci_original_dir / ricci_folder
        rows.append({
            "task": task,
            "h0_task_directory": str(h0_root),
            "h0_task_directory_exists": h0_root.is_dir(),
            "h0_fold_results_selected_q_exists": (h0_root / "fold_results_selected_q.csv").exists(),
            "h0_summary_selected_q_exists": (h0_root / "summary_selected_q.csv").exists(),
            "h0_selected_prediction_files": len(list(h0_root.glob("fold_*/selected_q/test_predictions.csv"))),
            "ricci_task_directory": str(ricci_root),
            "ricci_task_directory_exists": ricci_root.is_dir(),
            "ricci_fold_results_exists": (ricci_root / "fold_results.csv").exists(),
            "ricci_confusion_matrix_exists": (ricci_root / "aggregated_confusion_matrix.csv").exists(),
            "ricci_selected_coefficient_files": len(list(ricci_root.glob("*coefficient*.csv"))),
        })
    return pd.DataFrame(rows)


def _write_preflight(inputs, config: RunConfig) -> None:
    config.out_dir.mkdir(parents=True, exist_ok=True)
    counts = pd.DataFrame({
        "sample_id": inputs.sample_ids,
        "participant_id": inputs.participant_ids,
        "diagnosis": inputs.labels3,
        "h0_death_count": [len(values) for values in inputs.h0_deaths],
        "h0_min_death": [float(np.min(values)) for values in inputs.h0_deaths],
        "h0_max_death": [float(np.max(values)) for values in inputs.h0_deaths],
        "h0_path": inputs.h0_paths,
    })
    counts.to_csv(config.out_dir / "INPUT_SAMPLE_AUDIT.csv", index=False)
    reference_audit = _reference_output_audit(config)
    reference_audit.to_csv(config.out_dir / "REFERENCE_OUTPUT_AUDIT.csv", index=False)
    payload = {
        "sample_count": int(len(inputs.sample_ids)),
        "participant_count": int(len(set(inputs.participant_ids))),
        "label_counts": pd.Series(inputs.labels3).value_counts().to_dict(),
        "ricci_shape": list(inputs.ricci.shape),
        "ricci_nnz": int(inputs.ricci.nnz),
        "ricci_density": float(inputs.ricci.nnz / np.prod(inputs.ricci.shape)),
        "solver_storage": "CSR source; each train/evaluation split converted once to dense float64 and scaled train-only",
        "resolved_h0_pd_dir": str(inputs.h0_pd_dir),
        "h0_results_dir": str(config.h0_results_dir),
        "ricci_original_dir": str(config.ricci_original_dir),
        "ricci_feature_dir": str(config.ricci_feature_dir),
        "source_columns": inputs.source_columns,
        "participant_label_conflicts": int(
            inputs.metadata.groupby("_participant_id")["_label3"].nunique().gt(1).sum()
        ),
        "h0_reference_tasks_present": int(reference_audit["h0_task_directory_exists"].sum()),
        "ricci_reference_tasks_present": int(reference_audit["ricci_task_directory_exists"].sum()),
    }
    (config.out_dir / "INPUT_AUDIT.json").write_text(json.dumps(payload, indent=2) + "\n")
    if payload["participant_label_conflicts"]:
        raise ValueError(
            f"Found {payload['participant_label_conflicts']} participants with multiple diagnosis labels. "
            "Resolve this before grouped classification."
        )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = _config_from_args(args)
    config.out_dir.mkdir(parents=True, exist_ok=True)

    print("[START] Joint train-only WKPI + faithful Ricci block-L1 active-set classifier", flush=True)
    print(f"[INFO] H0 results:       {config.h0_results_dir}", flush=True)
    print(f"[INFO] H0 raw PDs:       {config.h0_pd_dir or 'auto'}", flush=True)
    print(f"[INFO] Ricci reference:  {config.ricci_original_dir}", flush=True)
    print(f"[INFO] Ricci features:   {config.ricci_feature_dir}", flush=True)
    print(f"[INFO] Output:           {config.out_dir}", flush=True)
    print(f"[INFO] Parallel M paths: {config.n_jobs}", flush=True)
    print(f"[INFO] Resume:           {config.resume}", flush=True)

    try:
        inputs = load_all_inputs(
            h0_results_dir=config.h0_results_dir,
            ricci_original_dir=config.ricci_original_dir,
            ricci_feature_dir=config.ricci_feature_dir,
            h0_pd_dir=config.h0_pd_dir,
        )
        _write_preflight(inputs, config)
        print(
            f"[READY] {len(inputs.sample_ids)} samples, {len(set(inputs.participant_ids))} participants, "
            f"Ricci shape={inputs.ricci.shape}, density={inputs.ricci.nnz / np.prod(inputs.ricci.shape):.3%}, "
            f"H0 diagrams={len(inputs.h0_deaths)}",
            flush=True,
        )
        if args.validate_only:
            print(f"[DONE] Validation-only audit written to {config.out_dir}", flush=True)
            return 0
        results = run_nested_cv(inputs, config)
        if bool(results.get("benchmark_only", False)):
            print("\n[DONE] First-inner-fold benchmark complete", flush=True)
            print(f"[OUTPUT] {config.out_dir}", flush=True)
            return 0
        print("\n[DONE] Joint sparse run complete", flush=True)
        summaries = results["summaries"]
        if isinstance(summaries, pd.DataFrame):
            print(summaries.to_string(index=False), flush=True)
        print(f"[OUTPUT] {config.out_dir}", flush=True)
        return 0
    except Exception as exc:
        print(f"\n[ERROR] {type(exc).__name__}: {exc}\n", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
