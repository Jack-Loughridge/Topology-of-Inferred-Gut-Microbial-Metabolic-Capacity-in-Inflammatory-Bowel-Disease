#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from h0_alpha_pi_1x5.tasks import TASKS


def main() -> None:
    output = Path.home() / "Real_Data" / "H0_AlphaPi_1x5_AllTasks"
    combined = output / "MANUSCRIPT_H0_1x5_TABLE.csv"
    if combined.exists():
        frame = pd.read_csv(combined)
    else:
        pieces = []
        for order, task in enumerate(TASKS, start=1):
            path = output / task.folder / "summary_mean_sd_across_folds.csv"
            if path.exists():
                piece = pd.read_csv(path)
                piece.insert(0, "task_order", order)
                pieces.append(piece)
        if not pieces:
            raise SystemExit(f"No completed task summaries found under {output}")
        frame = pd.concat(pieces, ignore_index=True).sort_values("task_order")
    columns = [
        "task",
        "n_samples",
        "n_participants",
        "sample_accuracy_mean",
        "sample_accuracy_sd",
        "sample_balanced_accuracy_mean",
        "sample_balanced_accuracy_sd",
        "sample_macro_f1_mean",
        "sample_macro_f1_sd",
        "sample_roc_auc_mean",
        "sample_roc_auc_sd",
        "selected_q_step_mean",
        "selected_q_step_sd",
        "selected_intervals_mean",
        "selected_intervals_sd",
    ]
    available = [column for column in columns if column in frame.columns]
    print("=" * 120)
    print("STANDALONE H0 ALPHA-PI: ONE LOCKED 5-FOLD REPETITION")
    print("=" * 120)
    print(frame[available].to_string(index=False))
    print("=" * 120)
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
