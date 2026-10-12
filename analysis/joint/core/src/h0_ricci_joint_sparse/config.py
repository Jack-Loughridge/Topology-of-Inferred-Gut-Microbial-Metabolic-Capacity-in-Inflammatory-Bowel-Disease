from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class RunConfig:
    h0_results_dir: Path
    ricci_original_dir: Path
    ricci_feature_dir: Path
    out_dir: Path
    h0_pd_dir: Path | None = None
    tasks: str = "all"
    n_outer_splits: int = 5
    n_inner_splits: int = 3
    seed: int = 13
    n_jobs: int = 3
    selection_metric: str = "roc_auc"
    interval_grid: tuple[int, ...] = (96, 160, 224)
    lambda_h0_grid: tuple[float, ...] = (0.5, 1.0, 2.0)
    lambda_ricci_grid: tuple[float, ...] = (0.5, 1.0, 2.0)
    c_value: float = 0.02
    logistic_max_iter: int = 8000
    logistic_tolerance: float = 1e-4
    n_alternations: int = 6
    alpha_smoothness_gamma: float = 0.01
    alpha_optimizer_max_iter: int = 100
    quad_points: int = 5
    common_alpha_grid_size: int = 300
    top_coefficients: int = 100
    active_set_initial: int = 512
    active_set_batch: int = 512
    active_set_max_rounds: int = 50
    resume: bool = True
    benchmark_first_inner_fold: bool = False

    def serialisable(self) -> dict[str, Any]:
        output = asdict(self)
        for key, value in list(output.items()):
            if isinstance(value, Path):
                output[key] = str(value)
            elif isinstance(value, tuple):
                output[key] = list(value)
        return output
