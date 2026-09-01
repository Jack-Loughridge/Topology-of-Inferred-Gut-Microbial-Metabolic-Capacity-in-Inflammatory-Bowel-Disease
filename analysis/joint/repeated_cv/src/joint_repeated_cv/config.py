from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .tasks import TASK_ORDER


SCHEMA_VERSION = "h0-ricci-joint-all-tasks-repeated-cv-v2"
CODE_VERSION = "2.0.1"


@dataclass(frozen=True)
class RunConfig:
    joint_repo: str
    ibd_split_manifest: str
    h0_results_dir: str
    h0_pd_dir: str
    ricci_original_dir: str
    ricci_feature_dir: str
    output_dir: str
    expected_repeats: int = 20
    expected_folds: int = 5
    n_inner_splits: int = 3
    n_jobs: int = 1
    selection_metric: str = "roc_auc"
    interval_grid: tuple[int, ...] = (64, 96, 160)
    lambda_h0_grid: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0)
    lambda_ricci_grid: tuple[float, ...] = (0.5, 1.0, 2.0)
    c_value: float = 0.02
    alpha_smoothness_gamma: float = 0.01
    alpha_smoothness_reference_intervals: int = 160
    logistic_max_iter: int = 24000
    logistic_tolerance: float = 1e-4
    active_set_initial: int = 512
    active_set_batch: int = 512
    active_set_max_rounds: int = 100
    alternations: int = 6
    quad_points: int = 5
    task_order: tuple[str, ...] = TASK_ORDER
    start_task_index: int = 1
    end_task_index: int = 5
    start_repeat: int = 1
    end_repeat: int = 20

    def scientific_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        for key, value in list(payload.items()):
            if isinstance(value, tuple):
                payload[key] = list(value)
        for key in ("n_jobs", "start_task_index", "end_task_index", "start_repeat", "end_repeat"):
            payload.pop(key, None)
        payload.update({"schema_version": SCHEMA_VERSION, "code_version": CODE_VERSION})
        return payload


def _int_tuple(value: str) -> tuple[int, ...]:
    parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not parsed or any(item <= 0 for item in parsed):
        raise argparse.ArgumentTypeError("Expected positive comma-separated integers.")
    if len(set(parsed)) != len(parsed):
        raise argparse.ArgumentTypeError("Grid values must be unique.")
    return parsed


def _float_tuple(value: str) -> tuple[float, ...]:
    parsed = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not parsed or any(item <= 0 for item in parsed):
        raise argparse.ArgumentTypeError("Expected positive comma-separated numbers.")
    if len(set(parsed)) != len(parsed):
        raise argparse.ArgumentTypeError("Grid values must be unique.")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    base = Path.home() / "Real_Data"
    parser = argparse.ArgumentParser(
        description=(
            "Run the validated joint H0+Ricci sparse classifier for five IBD tasks using "
            "20 repetitions of participant-grouped five-fold nested cross-validation."
        )
    )
    parser.add_argument("--joint-repo", type=Path, default=base / "h0_ricci_joint_sparse")
    parser.add_argument(
        "--ibd-split-manifest",
        type=Path,
        default=base / "Ricci_IBD_RepeatedCV_CPath" / "splits" / "sample_split_manifest.csv",
    )
    parser.add_argument(
        "--h0-results-dir", type=Path, default=base / "H0_AlphaPi_NestedQ_TrainOnlyBins"
    )
    parser.add_argument("--h0-pd-dir", type=Path, default=base / "out_pds")
    parser.add_argument(
        "--ricci-original-dir",
        type=Path,
        default=base / "Ricci_Classifier_OriginalStyle_AllTasks_NewVectors",
    )
    parser.add_argument(
        "--ricci-feature-dir",
        type=Path,
        default=base / "Ricci_Classifier_Faithful_Eps0001_n250_v3",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=base / "H0_Ricci_JointSparse_RepeatedCV_AllTasks",
    )
    parser.add_argument("--expected-repeats", type=int, default=20)
    parser.add_argument("--expected-folds", type=int, default=5)
    parser.add_argument("--n-inner-splits", type=int, default=3)
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Parallel interval-count paths inside the core. Tasks and repetitions remain sequential.",
    )
    parser.add_argument(
        "--selection-metric",
        choices=("accuracy", "balanced_accuracy", "macro_f1", "roc_auc"),
        default="roc_auc",
    )
    parser.add_argument("--interval-grid", type=_int_tuple, default=(64, 96, 160))
    parser.add_argument("--lambda-h0-grid", type=_float_tuple, default=(0.5, 1.0, 2.0, 4.0))
    parser.add_argument("--lambda-ricci-grid", type=_float_tuple, default=(0.5, 1.0, 2.0))
    parser.add_argument("--C", dest="c_value", type=float, default=0.02)
    parser.add_argument("--alpha-smoothness-gamma", type=float, default=0.01)
    parser.add_argument("--alpha-smoothness-reference-intervals", type=int, default=160)
    parser.add_argument("--logistic-max-iter", type=int, default=24000)
    parser.add_argument("--logistic-tol", type=float, default=1e-4)
    parser.add_argument("--active-set-initial", type=int, default=512)
    parser.add_argument("--active-set-batch", type=int, default=512)
    parser.add_argument("--active-set-max-rounds", type=int, default=100)
    parser.add_argument("--alternations", type=int, default=6)
    parser.add_argument("--quad-points", type=int, default=5)
    parser.add_argument("--start-task-index", type=int, default=1)
    parser.add_argument("--end-task-index", type=int, default=5)
    parser.add_argument("--start-repeat", type=int, default=1)
    parser.add_argument("--end-repeat", type=int, default=20)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--skip-core-tests", action="store_true")
    parser.add_argument("--overwrite-incompatible-output", action="store_true")
    return parser


def config_from_args(args: argparse.Namespace) -> RunConfig:
    return RunConfig(
        joint_repo=str(args.joint_repo.expanduser().resolve()),
        ibd_split_manifest=str(args.ibd_split_manifest.expanduser().resolve()),
        h0_results_dir=str(args.h0_results_dir.expanduser().resolve()),
        h0_pd_dir=str(args.h0_pd_dir.expanduser().resolve()),
        ricci_original_dir=str(args.ricci_original_dir.expanduser().resolve()),
        ricci_feature_dir=str(args.ricci_feature_dir.expanduser().resolve()),
        output_dir=str(args.output_dir.expanduser().resolve()),
        expected_repeats=int(args.expected_repeats),
        expected_folds=int(args.expected_folds),
        n_inner_splits=int(args.n_inner_splits),
        n_jobs=max(1, int(args.n_jobs)),
        selection_metric=str(args.selection_metric),
        interval_grid=tuple(int(value) for value in args.interval_grid),
        lambda_h0_grid=tuple(float(value) for value in args.lambda_h0_grid),
        lambda_ricci_grid=tuple(float(value) for value in args.lambda_ricci_grid),
        c_value=float(args.c_value),
        alpha_smoothness_gamma=float(args.alpha_smoothness_gamma),
        alpha_smoothness_reference_intervals=int(args.alpha_smoothness_reference_intervals),
        logistic_max_iter=int(args.logistic_max_iter),
        logistic_tolerance=float(args.logistic_tol),
        active_set_initial=int(args.active_set_initial),
        active_set_batch=int(args.active_set_batch),
        active_set_max_rounds=int(args.active_set_max_rounds),
        alternations=int(args.alternations),
        quad_points=int(args.quad_points),
        task_order=TASK_ORDER,
        start_task_index=int(args.start_task_index),
        end_task_index=int(args.end_task_index),
        start_repeat=int(args.start_repeat),
        end_repeat=int(args.end_repeat),
    )
