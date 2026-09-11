from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _escape_latex_text(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in text)


def latex_escape(value: object) -> str:
    """Escape text while preserving explicit ``$...$`` LaTeX math spans."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "--"
    text = str(value)
    pieces: list[str] = []
    cursor = 0
    while cursor < len(text):
        math_start = text.find("$", cursor)
        if math_start < 0:
            pieces.append(_escape_latex_text(text[cursor:]))
            break
        math_end = text.find("$", math_start + 1)
        if math_end < 0:
            pieces.append(_escape_latex_text(text[cursor:]))
            break
        pieces.append(_escape_latex_text(text[cursor:math_start]))
        pieces.append(text[math_start : math_end + 1])
        cursor = math_end + 1
    return "".join(pieces)


def format_mean_sd(mean: float, sd: float, digits: int = 3) -> str:
    if not (math.isfinite(float(mean)) and math.isfinite(float(sd))):
        return "--"
    return f"{float(mean):.{digits}f} $\\pm$ {float(sd):.{digits}f}"


def dataframe_to_latex_table(
    dataframe: pd.DataFrame,
    path: Path,
    caption: str,
    label: str,
    align: str | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if dataframe.empty:
        path.write_text("% Empty table\n")
        return
    align = align or ("l" + "r" * (len(dataframe.columns) - 1))
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\resizebox{\textwidth}{!}{%",
        rf"\begin{{tabular}}{{{align}}}",
        r"\toprule",
        " & ".join(latex_escape(column) for column in dataframe.columns) + r" \\",
        r"\midrule",
    ]
    for _, row in dataframe.iterrows():
        lines.append(" & ".join(latex_escape(row[column]) for column in dataframe.columns) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}"])
    path.write_text("\n".join(lines) + "\n")


def save_confusion_artifacts(
    matrix: np.ndarray,
    classes: tuple[str, ...],
    out_prefix: Path,
    title: str,
) -> None:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    dataframe = pd.DataFrame(matrix, index=classes, columns=classes)
    dataframe.index.name = "true_label"
    dataframe.to_csv(out_prefix.with_suffix(".csv"))

    figure, axis = plt.subplots(figsize=(5.5, 4.8))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axis.set(
        xticks=np.arange(len(classes)),
        yticks=np.arange(len(classes)),
        xticklabels=classes,
        yticklabels=classes,
        ylabel="True label",
        xlabel="Predicted label",
        title=title,
    )
    threshold = matrix.max() / 2.0 if matrix.size else 0.0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                f"{int(matrix[row, column])}",
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "black",
            )
    figure.tight_layout()
    figure.savefig(out_prefix.with_suffix(".png"), dpi=220, bbox_inches="tight")
    figure.savefig(out_prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)

    tex = dataframe.reset_index()
    tex.columns = ["True class"] + [f"Predicted {label}" for label in classes]
    dataframe_to_latex_table(
        tex,
        out_prefix.with_suffix(".tex"),
        caption=title,
        label=f"tab:{out_prefix.parent.name}_{out_prefix.name}".replace("-", "_"),
    )


def save_alpha_plot(alpha_profiles: pd.DataFrame, out_prefix: Path, title: str) -> None:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    if alpha_profiles.empty:
        return
    grouped = alpha_profiles.groupby("common_log_center", sort=True)["alpha_interpolated"]
    summary = grouped.agg(["mean", "std", "count"]).reset_index()
    summary["std"] = summary["std"].fillna(0.0)
    summary.to_csv(out_prefix.with_suffix(".csv"), index=False)

    x = summary["common_log_center"].to_numpy(float)
    mean = summary["mean"].to_numpy(float)
    std = summary["std"].to_numpy(float)
    figure, axis = plt.subplots(figsize=(7.5, 4.6))
    axis.plot(x, mean, linewidth=2)
    axis.fill_between(x, mean - std, mean + std, alpha=0.2)
    axis.axhline(1.0, linestyle="--", linewidth=1)
    axis.set_xlabel("log H0 death-value interval centre")
    axis.set_ylabel("Learned alpha")
    axis.set_title(title)
    figure.tight_layout()
    figure.savefig(out_prefix.with_suffix(".png"), dpi=220, bbox_inches="tight")
    figure.savefig(out_prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def write_global_latex_tables(
    summaries: pd.DataFrame,
    fold_results: pd.DataFrame,
    selected_configs: pd.DataFrame,
    out_dir: Path,
) -> None:
    rows = []
    for _, row in summaries.iterrows():
        rows.append({
            "Task": row["task"],
            "Accuracy": format_mean_sd(row["accuracy_mean"], row["accuracy_sd"]),
            "Balanced acc.": format_mean_sd(row["balanced_accuracy_mean"], row["balanced_accuracy_sd"]),
            "Macro F1": format_mean_sd(row["macro_f1_mean"], row["macro_f1_sd"]),
            "ROC-AUC": format_mean_sd(row["roc_auc_mean"], row["roc_auc_sd"]),
            "Intervals": format_mean_sd(row["selected_intervals_mean"], row["selected_intervals_sd"], digits=1),
            "$\\lambda_H$": format_mean_sd(row["lambda_h0_mean"], row["lambda_h0_sd"]),
            "$\\lambda_R$": format_mean_sd(row["lambda_ricci_mean"], row["lambda_ricci_sd"]),
        })
    n_outer_folds = int(summaries["n_folds"].max()) if not summaries.empty else 0
    dataframe_to_latex_table(
        pd.DataFrame(rows),
        out_dir / "latex_joint_sparse_summary.tex",
        caption=(
            "Participant-grouped nested joint WKPI--Ricci sparse classifier performance. "
            "H0 intervals, alpha, scaling and both block penalties are selected using outer-training participants only. "
            f"Values are mean $\\pm$ SD across {n_outer_folds} outer folds."
        ),
        label="tab:joint_wkpi_ricci_sparse_summary",
    )

    for task_folder, task_rows in fold_results.groupby("task_folder", sort=False):
        table = task_rows[[
            "fold", "train_samples", "test_samples", "train_participants", "test_participants",
            "accuracy", "balanced_accuracy", "macro_f1", "roc_auc",
            "selected_intervals", "lambda_h0", "lambda_ricci",
        ]].copy()
        table.columns = [
            "Fold", "Train", "Test", "Train participants", "Test participants",
            "Accuracy", "Balanced acc.", "Macro F1", "ROC-AUC",
            "Intervals", "$\\lambda_H$", "$\\lambda_R$",
        ]
        for column in ["Accuracy", "Balanced acc.", "Macro F1", "ROC-AUC", "$\\lambda_H$", "$\\lambda_R$"]:
            table[column] = table[column].map(lambda value: f"{float(value):.3f}")
        dataframe_to_latex_table(
            table,
            out_dir / task_folder / "latex_fold_results.tex",
            caption=f"Outer-fold performance for {task_rows.iloc[0]['task']} using the joint WKPI--Ricci sparse classifier.",
            label=f"tab:joint_{task_folder}_folds",
        )

    selected = selected_configs[[
        "task", "fold", "selected_intervals", "lambda_h0", "lambda_ricci", "inner_selection_score"
    ]].copy()
    selected.columns = ["Task", "Fold", "Intervals", "$\\lambda_H$", "$\\lambda_R$", "Inner ROC-AUC"]
    selected["Inner ROC-AUC"] = selected["Inner ROC-AUC"].map(lambda value: f"{float(value):.3f}")
    dataframe_to_latex_table(
        selected,
        out_dir / "latex_selected_configurations.tex",
        caption="Fold-specific hyperparameters selected entirely within the outer-training participants.",
        label="tab:joint_wkpi_ricci_selected_configs",
    )


def write_manifest(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
