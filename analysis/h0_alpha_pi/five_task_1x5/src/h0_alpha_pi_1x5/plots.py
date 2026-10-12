from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_confusion(matrix: np.ndarray, class_names: tuple[str, ...], title: str, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(matrix, interpolation="nearest")
    figure.colorbar(image, ax=axis)
    axis.set_title(title)
    ticks = np.arange(len(class_names))
    axis.set_xticks(ticks, class_names, rotation=45, ha="right")
    axis.set_yticks(ticks, class_names)
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    threshold = matrix.max() / 2.0 if matrix.size and matrix.max() else 0.0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                str(int(matrix[row, column])),
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "black",
            )
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def plot_training(history: pd.DataFrame, title: str, path: Path) -> None:
    if history.empty:
        return
    figure, axis = plt.subplots(figsize=(10, 5))
    if "train_balanced_accuracy" in history:
        axis.plot(history["epoch"], history["train_balanced_accuracy"], label="train balanced accuracy")
    if "validation_balanced_accuracy" in history:
        axis.plot(history["epoch"], history["validation_balanced_accuracy"], label="validation balanced accuracy")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Balanced accuracy")
    axis.set_title(title)
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def plot_curve(x: np.ndarray, mean: np.ndarray, sd: np.ndarray | None, ylabel: str, title: str, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.plot(x, mean, linewidth=1.8, label="mean")
    if sd is not None:
        axis.fill_between(x, mean - sd, mean + sd, alpha=0.25, label="±1 SD across folds")
    axis.set_xscale("log")
    axis.set_xlabel("H0 death value")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def plot_multiclass_curves(
    x: np.ndarray,
    means: dict[str, np.ndarray],
    sds: dict[str, np.ndarray],
    ylabel: str,
    title: str,
    path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(10, 5))
    for label, mean in means.items():
        axis.plot(x, mean, linewidth=1.5, label=label)
        sd = sds[label]
        axis.fill_between(x, mean - sd, mean + sd, alpha=0.15)
    axis.set_xscale("log")
    axis.set_xlabel("H0 death value")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)
