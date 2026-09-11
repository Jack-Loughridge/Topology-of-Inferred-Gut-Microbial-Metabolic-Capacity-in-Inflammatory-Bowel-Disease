#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path.home() / "Real_Data" / "H0_Ricci_JointSparse_RepeatedCV_AllTasks"
AGG = BASE / "aggregate"
OUT = Path.home() / "Real_Data" / "H0_Ricci_JointSparse_Manuscript_Outputs"

TASK_ORDER = {
    "IBD_vs_nonIBD": 1,
    "three_way_nonIBD_UC_CD": 2,
    "nonIBD_vs_UC": 3,
    "nonIBD_vs_CD": 4,
    "CD_vs_UC": 5,
}

TASK_DISPLAY = {
    "IBD_vs_nonIBD": "IBD vs non-IBD",
    "three_way_nonIBD_UC_CD": "non-IBD vs UC vs CD",
    "nonIBD_vs_UC": "non-IBD vs UC",
    "nonIBD_vs_CD": "non-IBD vs CD",
    "CD_vs_UC": "UC vs CD",
}

CLASS_ORDER = {
    "IBD_vs_nonIBD": ["nonIBD", "IBD"],
    "three_way_nonIBD_UC_CD": ["nonIBD", "UC", "CD"],
    "nonIBD_vs_UC": ["nonIBD", "UC"],
    "nonIBD_vs_CD": ["nonIBD", "CD"],
    "CD_vs_UC": ["CD", "UC"],
}

MODEL_NAME = "H0 + Ricci"

def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return path

def fmt_mean_sd(mean, sd, digits=3):
    mean = pd.to_numeric(pd.Series([mean]), errors="coerce").iloc[0]
    sd = pd.to_numeric(pd.Series([sd]), errors="coerce").iloc[0]
    if pd.isna(mean):
        return "--"
    if pd.isna(sd):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} ± {sd:.{digits}f}"

def fmt_pct(x, digits=1):
    x = pd.to_numeric(pd.Series([x]), errors="coerce").iloc[0]
    if pd.isna(x):
        return "--"
    return f"{100*x:.{digits}f}"

def safe_task_name(task_folder: str) -> str:
    return TASK_DISPLAY.get(task_folder, task_folder)

def ordered_classes(task_folder: str, observed: list[str]) -> list[str]:
    preferred = CLASS_ORDER.get(task_folder, [])
    obs = list(dict.fromkeys(observed))
    ordered = [c for c in preferred if c in obs]
    ordered += [c for c in obs if c not in ordered]
    return ordered

def write_latex_table(df: pd.DataFrame, path: Path, caption: str, label: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    latex = df.to_latex(
        index=False,
        escape=True,
        na_rep="--",
        column_format="l" * len(df.columns),
        longtable=False,
    )
    latex = latex.replace("\\toprule", "\\toprule\n")
    latex = latex.replace("\\midrule", "\\midrule\n")
    latex = latex.replace("\\bottomrule", "\\bottomrule\n")
    latex = latex.replace("\\begin{table}", "\\begin{table}[htbp]")
    latex = latex.replace("\\begin{tabular}", f"\\caption{{{caption}}}\n\\label{{{label}}}\n\\begin{{tabular}}")
    path.write_text(latex, encoding="utf-8")

def build_performance_tables():
    summary = pd.read_csv(require(AGG / "repetition_performance_summary.csv"))

    out_rows = []
    for level in ["sample", "participant"]:
        sub = summary.loc[summary["level"].eq(level)].copy()
        if sub.empty:
            continue
        sub["task_order"] = sub["task_folder"].map(TASK_ORDER)
        sub = sub.sort_values(["task_order"], kind="mergesort")

        formatted_rows = []
        for _, row in sub.iterrows():
            formatted_rows.append({
                "Task": safe_task_name(row["task_folder"]),
                "Level": level,
                "Completed repetitions": int(row["n_repetitions"]),
                "Accuracy": fmt_mean_sd(row.get("accuracy_mean"), row.get("accuracy_sd")),
                "Balanced accuracy": fmt_mean_sd(row.get("balanced_accuracy_mean"), row.get("balanced_accuracy_sd")),
                "Macro F1": fmt_mean_sd(row.get("macro_f1_mean"), row.get("macro_f1_sd")),
                "ROC-AUC": fmt_mean_sd(row.get("roc_auc_mean"), row.get("roc_auc_sd")),
            })
        out = pd.DataFrame(formatted_rows)
        out_rows.append(out)

        stem = f"joint_h0_ricci_performance_{level}"
        out.to_csv(OUT / f"{stem}.csv", index=False)
        write_latex_table(
            out,
            OUT / f"{stem}.tex",
            caption=f"{MODEL_NAME} performance by task ({level}-level pooled out-of-fold summaries).",
            label=f"tab:{stem}",
        )

    combined = pd.concat(out_rows, ignore_index=True) if out_rows else pd.DataFrame()
    if not combined.empty:
        combined.to_csv(OUT / "joint_h0_ricci_performance_all_levels.csv", index=False)
        write_latex_table(
            combined,
            OUT / "joint_h0_ricci_performance_all_levels.tex",
            caption=f"{MODEL_NAME} performance across all tasks.",
            label="tab:joint_h0_ricci_performance_all_levels",
        )

def consensus_confusion_from_predictions(frame: pd.DataFrame, id_column: str, level_name: str):
    prob_cols = [c for c in frame.columns if c.startswith("probability_standard_")]
    if not prob_cols:
        raise ValueError("No probability_standard_* columns found.")

    outputs = []

    for task_folder, task_df in frame.groupby("task_folder", sort=False):
        task_name = safe_task_name(task_folder)
        classes_observed = [c.replace("probability_standard_", "") for c in prob_cols if c in task_df.columns]
        classes = ordered_classes(task_folder, classes_observed)

        keep_prob_cols = [f"probability_standard_{c}" for c in classes]
        group_cols = ["task_order", "task_folder", "task", id_column, "true_label_standard"]
        agg = task_df[group_cols + keep_prob_cols].groupby(group_cols, as_index=False).mean()

        probs = agg[keep_prob_cols].to_numpy(float)
        pred_idx = np.argmax(probs, axis=1)
        agg["predicted_label_standard"] = [classes[i] for i in pred_idx]

        rows = []
        for true_class in classes:
            sub = agg.loc[agg["true_label_standard"].eq(true_class)].copy()
            row = {
                "Task": task_name,
                "Level": level_name,
                "Model": MODEL_NAME,
                "True class": true_class,
            }
            for pred_class in classes:
                count = int((sub["predicted_label_standard"] == pred_class).sum())
                row[f"Predicted {pred_class}"] = count
            total = int(len(sub))
            correct = int((sub["predicted_label_standard"] == true_class).sum())
            row["Total"] = total
            row["Correct"] = correct
            row["Row accuracy (%)"] = fmt_pct(correct / total if total else np.nan)
            rows.append(row)

        out = pd.DataFrame(rows)
        outputs.append((task_folder, out))

        csv_path = OUT / "confusion_tables" / level_name / f"{task_folder}_confusion.csv"
        tex_path = OUT / "confusion_tables" / level_name / f"table_{task_folder}_confusion.tex"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(csv_path, index=False)
        write_latex_table(
            out,
            tex_path,
            caption=f"Consensus confusion table for {MODEL_NAME}: {task_name} ({level_name} level).",
            label=f"tab:{task_folder}_{level_name}_confusion",
        )

def build_confusion_tables():
    sample_pred = pd.read_csv(require(AGG / "all_test_predictions.csv"))
    participant_pred = pd.read_csv(require(AGG / "participant_average_predictions.csv"))

    if "task_order" not in sample_pred.columns:
        sample_pred["task_order"] = sample_pred["task_folder"].map(TASK_ORDER)
    if "task_order" not in participant_pred.columns:
        participant_pred["task_order"] = participant_pred["task_folder"].map(TASK_ORDER)

    consensus_confusion_from_predictions(sample_pred, "sample_id", "sample")
    consensus_confusion_from_predictions(participant_pred, "participant_id", "participant")

def build_selected_config_tables():
    path = AGG / "selected_configuration_frequency.csv"
    if not path.exists():
        return

    df = pd.read_csv(path)
    if df.empty:
        return

    task_col = "run_task_folder" if "run_task_folder" in df.columns else "task_folder"
    report_col = "report_task" if "report_task" in df.columns else "task"
    interval_col = "selected_intervals" if "selected_intervals" in df.columns else ("n_intervals" if "n_intervals" in df.columns else None)

    rows = []
    for _, row in df.iterrows():
        out = {
            "Task": safe_task_name(row[task_col]),
            "Completed task-fold fits": int(row["outer_fold_count"]),
            "Within-task frequency (%)": fmt_pct(row["outer_fold_frequency_within_task"]),
            "lambda_H0": row["lambda_h0"] if "lambda_h0" in row else np.nan,
            "lambda_Ricci": row["lambda_ricci"] if "lambda_ricci" in row else np.nan,
        }
        if interval_col is not None:
            out["Selected intervals"] = row[interval_col]
        rows.append(out)

    out_df = pd.DataFrame(rows)
    sort_cols = [c for c in ["Task", "Completed task-fold fits"] if c in out_df.columns]
    ascending = [True, False][:len(sort_cols)]
    if sort_cols:
        out_df = out_df.sort_values(sort_cols, ascending=ascending, kind="mergesort")

    out_df.to_csv(OUT / "joint_h0_ricci_selected_configuration_frequency.csv", index=False)
    write_latex_table(
        out_df,
        OUT / "joint_h0_ricci_selected_configuration_frequency.tex",
        caption=f"Selected hyperparameter configurations for {MODEL_NAME}.",
        label="tab:joint_h0_ricci_selected_configurations",
    )

def write_run_summary():
    progress_path = BASE / "progress.json"
    summary_path = OUT / "run_summary.json"

    payload = {}
    if progress_path.exists():
        payload = json.loads(progress_path.read_text(encoding="utf-8"))

    payload["source_output_dir"] = str(BASE)
    payload["aggregate_dir"] = str(AGG)
    payload["manuscript_output_dir"] = str(OUT)

    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    build_performance_tables()
    build_confusion_tables()
    build_selected_config_tables()
    write_run_summary()

    print("H0 + RICCI MANUSCRIPT OUTPUTS COMPLETE")
    print(f"Output directory: {OUT}")
    print(f"Performance tables: {OUT / 'joint_h0_ricci_performance_all_levels.csv'}")
    print(f"Confusion tables: {OUT / 'confusion_tables'}")
    print(f"Selected configuration table: {OUT / 'joint_h0_ricci_selected_configuration_frequency.csv'}")

if __name__ == "__main__":
    main()
