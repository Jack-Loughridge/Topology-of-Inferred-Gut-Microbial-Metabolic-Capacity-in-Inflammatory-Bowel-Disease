#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

MODEL_ORDER = ["logistic_l1", "random_forest", "xgboost"]
MODEL_LABELS = {
    "logistic_l1": "L1 logistic",
    "random_forest": "Random forest",
    "xgboost": "XGBoost",
}


def fmt(mean: object, sd: object) -> str:
    try:
        return f"{float(mean):.4f} ± {float(sd):.4f}"
    except Exception:
        return "NA"


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and print the completed IBD-vs-non-IBD benchmark results.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.home() / "Real_Data" / "Species_Benchmarks_RepeatedCV_IBD_CompleteCase",
    )
    args = parser.parse_args()
    out = args.output_dir.expanduser().resolve()
    marker = out / "RUN_COMPLETE.json"
    print("=" * 112)
    print("IBD VS NON-IBD SPECIES BENCHMARK RESULT CHECK")
    print("=" * 112)
    print("Output:", out)
    if not marker.exists():
        partial = out / "partial_progress" / "progress.json"
        print("RUN_COMPLETE.json: MISSING")
        if partial.exists():
            print("Partial progress:")
            print(partial.read_text())
        raise SystemExit("The run is not confirmed complete. Do not treat partial outputs as final.")
    completion = json.loads(marker.read_text())
    print("RUN_COMPLETE.json: PRESENT")
    print("Completed folds recorded:", completion.get("completed_folds"))
    fold_markers = len(list((out / "folds").glob("*/repeat_*/fold_*/FOLD_COMPLETE.json")))
    print("Fold completion markers found:", fold_markers)
    if int(completion.get("completed_folds", -1)) != 300 or fold_markers != 300:
        raise SystemExit("Completion markers do not confirm all 300 model-fold fits.")

    summary_path = out / "repetition_performance_summary.csv"
    rep_path = out / "repetition_pooled_oof_metrics.csv"
    paired_path = out / "paired_model_difference_summary.csv"
    diagnostics_path = out / "model_fit_diagnostics.csv"
    for path in (summary_path, rep_path, paired_path, diagnostics_path):
        if not path.exists():
            raise SystemExit(f"Required final output is missing: {path}")

    summary = pd.read_csv(summary_path)
    repetition = pd.read_csv(rep_path)
    if len(repetition) != 3 * 2 * 20:
        raise SystemExit(f"Expected 120 pooled repetition rows; found {len(repetition)}")
    counts = repetition.groupby(["model", "evaluation_level"])["repeat"].nunique()
    if not counts.eq(20).all():
        raise SystemExit(f"Not every model/level has 20 repetitions:\n{counts}")

    rows = []
    for level in ("sample", "participant"):
        for model in MODEL_ORDER:
            row = summary[summary["model"].eq(model) & summary["evaluation_level"].eq(level)]
            if len(row) != 1:
                raise SystemExit(f"Missing/duplicate summary row for {model}, {level}")
            row = row.iloc[0]
            rows.append({
                "Level": level,
                "Model": MODEL_LABELS[model],
                "Accuracy": fmt(row.get("accuracy_mean"), row.get("accuracy_sd")),
                "Balanced accuracy": fmt(row.get("balanced_accuracy_mean"), row.get("balanced_accuracy_sd")),
                "Macro F1": fmt(row.get("macro_f1_mean"), row.get("macro_f1_sd")),
                "ROC-AUC": fmt(row.get("roc_auc_mean"), row.get("roc_auc_sd")),
                "Brier": fmt(row.get("brier_mean"), row.get("brier_sd")),
            })
    compact = pd.DataFrame(rows)
    print("\nPrimary distribution: 20 pooled out-of-fold repetition estimates")
    print(compact.to_string(index=False))

    compact_path = out / "ibd_primary_results_readout.csv"
    compact.to_csv(compact_path, index=False)

    paired = pd.read_csv(paired_path)
    paired = paired[
        paired["evaluation_level"].eq("sample")
        & paired["metric"].isin(["balanced_accuracy", "macro_f1", "roc_auc"])
    ].copy()
    if len(paired):
        paired["comparison"] = paired["model_a"].map(MODEL_LABELS) + " minus " + paired["model_b"].map(MODEL_LABELS)
        display = paired[[
            "comparison", "metric", "mean_delta_a_better_positive", "sd_delta",
            "q025_delta", "q975_delta", "fraction_a_better"
        ]]
        print("\nPaired within-repetition sample-level differences (positive favours first model):")
        print(display.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    diagnostics = pd.read_csv(diagnostics_path)
    def boolean_count(series: pd.Series) -> int:
        return int(series.astype(str).str.strip().str.lower().isin({"1", "true", "yes"}).sum())

    warning_count = boolean_count(diagnostics["convergence_warning"])
    limit_count = boolean_count(diagnostics["convergence_at_iteration_limit"])
    print("\nDiagnostics:")
    print("  Folds with recorded convergence warnings:", warning_count)
    print("  Logistic fits at iteration limit:", limit_count)
    print("\nCompact result table written to:", compact_path)
    print("Plots directory:", out / "plots")
    print("Top-feature tables:")
    for model in MODEL_ORDER:
        print(" ", out / f"top_100_features_{model}.csv")
    print("=" * 112)


if __name__ == "__main__":
    main()
