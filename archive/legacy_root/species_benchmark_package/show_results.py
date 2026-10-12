#!/usr/bin/env python3
from pathlib import Path
import pandas as pd

root = Path.home() / "Real_Data" / "Species_Benchmarks_RepeatedCV_AllTasks"
path = root / "aggregate" / "repetition_performance_summary_all_tasks.csv"
if not path.exists():
    raise SystemExit(f"Missing {path}; the all-task run has not completed aggregation.")
frame = pd.read_csv(path)
metrics = [c for c in ["accuracy_mean", "balanced_accuracy_mean", "macro_f1_mean", "roc_auc_mean"] if c in frame]
cols = [c for c in ["task", "evaluation_level", "model_label", *metrics] if c in frame]
print(frame[cols].to_string(index=False))
