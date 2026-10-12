#!/usr/bin/env python3
"""Create manuscript-ready species benchmark tables and confusion matrices.

Inputs are the completed IBD-vs-nonIBD benchmark output and the completed
remaining-four-task output. Performance uncertainty is summarized across the
20 pooled out-of-fold repetitions. Confusion matrices use consensus OOF
probabilities averaged across repetitions, so each sample contributes once.
"""

from __future__ import annotations

from pathlib import Path
import math
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

IBD_ROOT = Path.home() / "Real_Data" / "Species_Benchmarks_RepeatedCV_IBD_CompleteCase"
REMAINING_ROOT = Path.home() / "Real_Data" / "Species_Benchmarks_RepeatedCV_RemainingTasks"
OUT_ROOT = Path.home() / "Real_Data" / "Species_Benchmarks_Manuscript_Outputs"

TASKS = [
    ("IBD_vs_nonIBD", "IBD vs non-IBD", IBD_ROOT, ["non-IBD", "IBD"]),
    (
        "three_way_nonIBD_UC_CD",
        "non-IBD vs UC vs CD",
        REMAINING_ROOT / "three_way_nonIBD_UC_CD",
        ["non-IBD", "UC", "CD"],
    ),
    ("nonIBD_vs_UC", "non-IBD vs UC", REMAINING_ROOT / "nonIBD_vs_UC", ["non-IBD", "UC"]),
    ("nonIBD_vs_CD", "non-IBD vs CD", REMAINING_ROOT / "nonIBD_vs_CD", ["non-IBD", "CD"]),
    ("CD_vs_UC", "CD vs UC", REMAINING_ROOT / "CD_vs_UC", ["CD", "UC"]),
]

MODEL_LABELS = {
    "logistic_l1": "L1 logistic regression",
    "random_forest": "Random forest",
    "xgboost": "XGBoost",
}
PERFORMANCE_ORDER = ["random_forest", "xgboost", "logistic_l1"]
CONFUSION_ORDER = ["logistic_l1", "random_forest", "xgboost"]
METRICS = [
    ("accuracy", "Accuracy"),
    ("balanced_accuracy", "Balanced acc."),
    ("macro_f1", "Macro F1"),
    ("roc_auc", "ROC-AUC"),
]


def die(message: str) -> None:
    raise RuntimeError(message)


def normalize_label(value: object) -> str:
    text = str(value).strip()
    return {"nonIBD": "non-IBD", "non-IBD": "non-IBD"}.get(text, text)


def latex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(char, char) for char in text)


def mean_sd(row: pd.Series, metric: str) -> str:
    mean = float(row[f"{metric}_mean"])
    sd = float(row[f"{metric}_sd"])
    if not (math.isfinite(mean) and math.isfinite(sd)):
        return "--"
    return f"{mean:.3f} $\\pm$ {sd:.3f}"


def verify_inputs() -> None:
    missing: list[str] = []
    for _, _, root, _ in TASKS:
        for path in (
            root / "RUN_COMPLETE.json",
            root / "repetition_performance_summary.csv",
            root / "consensus_predictions_by_sample.csv",
        ):
            if not path.exists():
                missing.append(str(path))
    if missing:
        die("Missing required completed-run files:\n  " + "\n  ".join(missing))


def read_performance(folder: str, task_name: str, root: Path) -> pd.DataFrame:
    path = root / "repetition_performance_summary.csv"
    frame = pd.read_csv(path)
    required = {"model", "evaluation_level", "n_repetitions"}
    required.update(f"{metric}_{suffix}" for metric, _ in METRICS for suffix in ("mean", "sd"))
    missing = required.difference(frame.columns)
    if missing:
        die(f"{path} is missing columns: {sorted(missing)}")
    frame = frame[frame["evaluation_level"].eq("sample")].copy()
    if set(frame["model"]) != set(MODEL_LABELS):
        die(f"Unexpected model set in {path}: {sorted(frame['model'].unique())}")
    if not (pd.to_numeric(frame["n_repetitions"], errors="coerce") == 20).all():
        die(f"{path} does not contain 20 pooled OOF repetitions for every model")
    frame.insert(0, "task_folder", folder)
    frame.insert(1, "task", task_name)
    return frame


def read_consensus(root: Path) -> pd.DataFrame:
    path = root / "consensus_predictions_by_sample.csv"
    frame = pd.read_csv(path)
    required = {"model", "sample_id", "true_label", "pred_label"}
    missing = required.difference(frame.columns)
    if missing:
        die(f"{path} is missing columns: {sorted(missing)}")
    frame = frame.copy()
    frame["true_label"] = frame["true_label"].map(normalize_label)
    frame["pred_label"] = frame["pred_label"].map(normalize_label)
    if frame.duplicated(["model", "sample_id"]).any():
        die(f"Duplicate model/sample consensus rows in {path}")
    if "n_repetitions" in frame.columns:
        reps = pd.to_numeric(frame["n_repetitions"], errors="coerce")
        if reps.isna().any() or not (reps == 20).all():
            die(f"{path} does not contain exactly 20 predictions per sample")
    return frame


def write_performance_outputs() -> None:
    task_order = {folder: index for index, (folder, _, _, _) in enumerate(TASKS)}
    model_order = {model: index for index, model in enumerate(PERFORMANCE_ORDER)}
    combined = pd.concat(
        [read_performance(folder, name, root) for folder, name, root, _ in TASKS],
        ignore_index=True,
    )
    combined["task_order"] = combined["task_folder"].map(task_order)
    combined["model_order"] = combined["model"].map(model_order)
    combined = combined.sort_values(["task_order", "model_order"], kind="mergesort")

    numeric_columns = ["task_folder", "task", "model", "model_label", "n_repetitions"]
    numeric_columns += [f"{metric}_{suffix}" for metric, _ in METRICS for suffix in ("mean", "sd")]
    combined[numeric_columns].to_csv(OUT_ROOT / "benchmark_performance_all_tasks_numeric.csv", index=False)

    formatted_rows = []
    for _, row in combined.iterrows():
        formatted = {"Task": row["task"], "Model": MODEL_LABELS[row["model"]]}
        for metric, label in METRICS:
            formatted[label] = f"{row[f'{metric}_mean']:.3f} ± {row[f'{metric}_sd']:.3f}"
        formatted_rows.append(formatted)
    pd.DataFrame(formatted_rows).to_csv(
        OUT_ROOT / "benchmark_performance_all_tasks_formatted.csv", index=False
    )

    all_rows = []
    for _, row in combined.iterrows():
        values = [latex_escape(row["task"]), latex_escape(MODEL_LABELS[row["model"]])]
        values += [mean_sd(row, metric) for metric, _ in METRICS]
        all_rows.append(" & ".join(values) + r" \\")

    all_tex = "\n".join(
        [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\small",
            r"\caption{Species-abundance benchmark performance across all five classification tasks. Values are mean $\pm$ SD across 20 pooled out-of-fold repetitions; each repetition pools five participant-grouped outer folds.}",
            r"\label{tab:species_benchmark_all_tasks}",
            r"\begin{tabular}{llcccc}",
            r"\toprule",
            r"Task & Model & Accuracy & Balanced acc. & Macro F1 & ROC-AUC \\",
            r"\midrule",
            *all_rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    (OUT_ROOT / "table_benchmark_performance_all_tasks.tex").write_text(all_tex, encoding="utf-8")

    by_task = OUT_ROOT / "performance_by_task"
    by_task.mkdir(exist_ok=True)
    for folder, task_name, _, _ in TASKS:
        subset = combined[combined["task_folder"].eq(folder)]
        rows = []
        for _, row in subset.iterrows():
            values = [latex_escape(MODEL_LABELS[row["model"]])]
            values += [mean_sd(row, metric) for metric, _ in METRICS]
            rows.append(" & ".join(values) + r" \\")
        tex = "\n".join(
            [
                r"\begin{table}[htbp]",
                r"\centering",
                r"\small",
                f"\\caption{{Species-abundance benchmark performance for {latex_escape(task_name)}. Values are mean $\\pm$ SD across 20 pooled out-of-fold repetitions; each repetition pools five participant-grouped outer folds.}}",
                f"\\label{{tab:species_benchmark_{folder}}}",
                r"\begin{tabular}{lcccc}",
                r"\toprule",
                r"Model & Accuracy & Balanced acc. & Macro F1 & ROC-AUC \\",
                r"\midrule",
                *rows,
                r"\bottomrule",
                r"\end{tabular}",
                r"\end{table}",
                "",
            ]
        )
        (by_task / f"table_{folder}_performance.tex").write_text(tex, encoding="utf-8")


def confusion_matrix(frame: pd.DataFrame, model: str, classes: list[str]) -> np.ndarray:
    subset = frame[frame["model"].eq(model)]
    if subset.empty:
        die(f"No consensus predictions for model {model}")
    unexpected = (set(subset["true_label"]) | set(subset["pred_label"])) - set(classes)
    if unexpected:
        die(f"Unexpected class labels for {model}: {sorted(unexpected)}")
    matrix = np.zeros((len(classes), len(classes)), dtype=int)
    indices = {label: index for index, label in enumerate(classes)}
    for true_label, pred_label in zip(subset["true_label"], subset["pred_label"]):
        matrix[indices[true_label], indices[pred_label]] += 1
    return matrix


def plot_confusion(matrix: np.ndarray, classes: list[str], title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 4.7))
    image = ax.imshow(matrix, cmap="Blues", interpolation="nearest")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(np.arange(len(classes)), labels=classes)
    ax.set_yticks(np.arange(len(classes)), labels=classes)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)
    threshold = matrix.max() / 2 if matrix.size else 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j,
                i,
                str(int(matrix[i, j])),
                ha="center",
                va="center",
                color="white" if matrix[i, j] > threshold else "black",
                fontsize=11,
            )
    fig.tight_layout()
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def write_confusion_outputs() -> None:
    table_dir = OUT_ROOT / "confusion_tables"
    image_dir = OUT_ROOT / "confusion_matrices_png"
    table_dir.mkdir(exist_ok=True)
    image_dir.mkdir(exist_ok=True)

    for folder, task_name, root, classes in TASKS:
        frame = read_consensus(root)
        matrices: dict[str, np.ndarray] = {}
        long_rows = []
        for model in CONFUSION_ORDER:
            matrix = confusion_matrix(frame, model, classes)
            matrices[model] = matrix
            for row_index, true_label in enumerate(classes):
                row = {"Model": MODEL_LABELS[model], "True class": true_label}
                for col_index, predicted_label in enumerate(classes):
                    row[f"Predicted {predicted_label}"] = int(matrix[row_index, col_index])
                long_rows.append(row)
            plot_confusion(
                matrix,
                classes,
                f"{task_name} — {MODEL_LABELS[model]}",
                image_dir / f"{folder}_{model}.png",
            )
        pd.DataFrame(long_rows).to_csv(table_dir / f"{folder}_confusion.csv", index=False)

        header = ["Model", "True class"] + [f"Predicted {label}" for label in classes]
        rows = []
        for model in CONFUSION_ORDER:
            matrix = matrices[model]
            for row_index, true_label in enumerate(classes):
                values = [latex_escape(MODEL_LABELS[model]), latex_escape(true_label)]
                values += [str(int(value)) for value in matrix[row_index, :]]
                rows.append(" & ".join(values) + r" \\")
        tex = "\n".join(
            [
                r"\begin{table}[htbp]",
                r"\centering",
                r"\small",
                f"\\caption{{Consensus held-out sample confusion matrices for the species-abundance benchmarks on {latex_escape(task_name)}. Each sample's out-of-fold probabilities are averaged across 20 repetitions, so every sample contributes once. Rows denote true labels and columns denote predicted labels.}}",
                f"\\label{{tab:species_benchmark_confusion_{folder}}}",
                f"\\begin{{tabular}}{{ll{'r' * len(classes)}}}",
                r"\toprule",
                " & ".join(latex_escape(item) for item in header) + r" \\",
                r"\midrule",
                *rows,
                r"\bottomrule",
                r"\end{tabular}",
                r"\end{table}",
                "",
            ]
        )
        (table_dir / f"table_{folder}_confusion.tex").write_text(tex, encoding="utf-8")


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    verify_inputs()
    write_performance_outputs()
    write_confusion_outputs()
    print("=" * 100)
    print("SPECIES BENCHMARK MANUSCRIPT OUTPUTS COMPLETE")
    print(f"Output directory: {OUT_ROOT}")
    print(f"Combined performance table: {OUT_ROOT / 'table_benchmark_performance_all_tasks.tex'}")
    print(f"Task-specific performance tables: {OUT_ROOT / 'performance_by_task'}")
    print(f"Confusion tables: {OUT_ROOT / 'confusion_tables'}")
    print(f"Confusion PNGs: {OUT_ROOT / 'confusion_matrices_png'}")
    print("=" * 100)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
