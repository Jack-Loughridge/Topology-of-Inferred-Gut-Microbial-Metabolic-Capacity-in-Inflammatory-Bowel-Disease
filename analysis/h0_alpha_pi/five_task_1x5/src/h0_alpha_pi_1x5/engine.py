from __future__ import annotations

import copy
import json
import math
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn import __version__ as sklearn_version
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold

from .data import TaskBundle, fold_indices, load_task_bundle, load_task_manifest
from .metrics import metrics_from_prediction_frame, participant_predictions, probability_metrics
from .model import (
    AlphaPiModel,
    build_bounds,
    class_weighted_ce,
    full_loss,
    make_loader,
    sigma_from_bounds,
)
from .plots import plot_confusion, plot_curve, plot_multiclass_curves, plot_training
from .tasks import TaskSpec
from .util import (
    atomic_csv,
    atomic_json,
    atomic_npy,
    atomic_npz,
    atomic_torch,
    canonical_frame_sha256,
    describe,
    now_iso,
    set_all_seeds,
    sha256_file,
)


@dataclass(frozen=True)
class EngineConfig:
    pd_dir: Path
    label_csv: Path
    metadata_csv: Path
    split_manifest: Path
    output_dir: Path
    task: TaskSpec
    repeat: int = 1
    expected_folds: int = 5
    percentile_steps: tuple[float, ...] = (0.3, 0.4, 0.5, 0.6, 0.7)
    inner_splits: int = 5
    inner_split_attempts: int = 256
    epochs: int = 70
    batch_size: int = 32
    learning_rate: float = 4e-4
    weight_decay: float = 1e-4
    quad_points: int = 5
    point_chunk: int = 256
    lambda_within: float = 1.0
    lambda_radius: float = 1.0
    lambda_between: float = 200.0
    lambda_ce: float = 1000.0
    margin_between: float = 10.0
    max_grad_norm: float = 0.0
    min_interval_width: float = 1e-8
    model_seed: int = 1337
    inner_split_seed: int = 7331
    num_workers: int = 0
    print_every: int = 10
    common_grid_size: int = 300
    deterministic: bool = True
    device: str = "cpu"
    resume: bool = True
    make_plots: bool = True


def _candidate_seed(config: EngineConfig, fold: int) -> int:
    return int(config.model_seed + config.repeat * 100_003 + fold * 1_009 + 17)


def _final_seed(config: EngineConfig, fold: int) -> int:
    return int(config.model_seed + config.repeat * 100_003 + fold * 1_009 + 53)


def _percentile_tag(value: float) -> str:
    return f"percentile_step_{value:.6g}".replace(".", "p").replace("-", "m")


def _safe(value: Any) -> float:
    try:
        output = float(value)
    except Exception:
        return float("-inf")
    return output if math.isfinite(output) else float("-inf")


def _run_config(config: EngineConfig, manifest: pd.DataFrame, bundle: TaskBundle) -> dict[str, Any]:
    manifest_columns = ["task_folder", "repeat", "fold", "split_seed", "role", "sample_id", "participant_id", "label"]
    return {
        "schema_version": 2,
        "created_utc": now_iso(),
        "task_folder": config.task.folder,
        "task": config.task.report_name,
        "class_order": list(config.task.class_order),
        "repeat": config.repeat,
        "expected_folds": config.expected_folds,
        "paths": {
            "pd_dir": str(config.pd_dir.resolve()),
            "label_csv": str(config.label_csv.resolve()),
            "metadata_csv": str(config.metadata_csv.resolve()),
            "split_manifest": str(config.split_manifest.resolve()),
            "output_dir": str(config.output_dir.resolve()),
        },
        "selection": {
            "percentile_steps": list(config.percentile_steps),
            "inner_splits": config.inner_splits,
            "inner_split_attempts": config.inner_split_attempts,
            "inner_split_policy": "first all-class StratifiedGroupKFold split across a deterministic seed sequence",
            "epoch_order": ["validation balanced accuracy", "validation macro F1", "validation ROC-AUC", "lower validation loss", "earlier epoch"],
            "candidate_order": ["validation balanced accuracy", "validation macro F1", "validation ROC-AUC", "fewer intervals", "larger q"],
            "candidate_seed_policy": "same seed for every q within an outer fold",
        },
        "final_refit": {
            "fresh_initialization": True,
            "bounds_source": "all outer-training diagrams only",
            "epochs": "selected validation-best epoch",
        },
        "model": {
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "quad_points": config.quad_points,
            "point_chunk": config.point_chunk,
            "lambda_ce": config.lambda_ce,
            "lambda_within": config.lambda_within,
            "lambda_radius": config.lambda_radius,
            "lambda_between": config.lambda_between,
            "margin_between": config.margin_between,
            "max_grad_norm": config.max_grad_norm,
        },
        "randomness": {
            "model_seed": config.model_seed,
            "inner_split_seed": config.inner_split_seed,
            "deterministic": config.deterministic,
        },
        "fingerprints": {
            "manifest_canonical_sha256": canonical_frame_sha256(manifest, manifest_columns),
            "label_csv_sha256": sha256_file(config.label_csv),
            "metadata_csv_sha256": sha256_file(config.metadata_csv),
            "pd_listing_sha256": canonical_frame_sha256(
                pd.DataFrame(
                    [
                        {"sample_id": sid, "path": str(path.resolve()), "sha256": sha256_file(path)}
                        for sid, path in sorted(bundle.pd_paths.items())
                    ]
                ),
                ["sample_id", "path", "sha256"],
            ),
        },
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "torch": torch.__version__,
            "sklearn": sklearn_version,
            "device": config.device,
        },
    }


def _compatible(existing: dict[str, Any], current: dict[str, Any]) -> None:
    keys = ["schema_version", "task_folder", "class_order", "repeat", "expected_folds", "paths", "selection", "final_refit", "model", "randomness", "fingerprints"]
    mismatch = [key for key in keys if existing.get(key) != current.get(key)]
    if mismatch:
        raise RuntimeError(f"Incompatible existing task output; mismatched sections: {mismatch}")


def _inner_split(bundle: TaskBundle, outer_train: np.ndarray, fold: int, config: EngineConfig) -> tuple[np.ndarray, np.ndarray, int]:
    """Create a deterministic participant-grouped inner holdout containing every class.

    ``StratifiedGroupKFold`` can return different valid partitions across
    scikit-learn releases, and on small grouped fixtures a particular seed may
    yield no fold with every class on both sides even when such a split exists.
    We therefore search a deterministic sequence of seeds and accept the first
    all-class split.  No outcome information beyond the training labels and
    participant groups is used.
    """
    labels = np.asarray(bundle.labels[outer_train], dtype=int)
    all_groups = np.asarray(bundle.participant_ids, dtype=str)
    groups = all_groups[outer_train]
    all_classes = set(range(len(config.task.class_order)))

    participant_class_counts: dict[str, int] = {}
    for class_index, class_name in enumerate(config.task.class_order):
        participant_class_counts[str(class_name)] = int(
            len(set(groups[labels == class_index].tolist()))
        )
    insufficient = {
        class_name: count for class_name, count in participant_class_counts.items() if count < 2
    }
    if insufficient:
        raise RuntimeError(
            f"Cannot create an all-class inner train/validation split for {config.task.folder} fold {fold}; "
            f"fewer than two outer-training participants in classes {insufficient}"
        )

    base_seed = int(config.inner_split_seed + config.repeat * 10_007 + fold * 101)
    attempts = max(1, int(config.inner_split_attempts))
    for attempt in range(attempts):
        seed = int(base_seed + attempt)
        splitter = StratifiedGroupKFold(
            n_splits=config.inner_splits,
            shuffle=True,
            random_state=seed,
        )
        try:
            candidates = splitter.split(np.zeros(len(labels)), labels, groups)
            for train_relative, validation_relative in candidates:
                train_relative = np.asarray(train_relative, dtype=int)
                validation_relative = np.asarray(validation_relative, dtype=int)
                if set(labels[train_relative].tolist()) != all_classes:
                    continue
                if set(labels[validation_relative].tolist()) != all_classes:
                    continue
                train = outer_train[train_relative]
                validation = outer_train[validation_relative]
                if set(all_groups[train].tolist()) & set(all_groups[validation].tolist()):
                    raise RuntimeError("Participant leakage in inner split")
                return train, validation, seed
        except ValueError:
            # Some sklearn versions reject a seed/fold arrangement before
            # yielding candidates. Continue through the audited seed sequence.
            continue

    raise RuntimeError(
        f"Could not create an inner split containing all classes for {config.task.folder} fold {fold} "
        f"after {attempts} deterministic seed attempts; participant counts by class="
        f"{participant_class_counts}, inner_splits={config.inner_splits}, base_seed={base_seed}"
    )


def _evaluate(
    model: AlphaPiModel,
    loader,
    ce_loss,
    bundle: TaskBundle,
    device: torch.device,
    config: EngineConfig,
    return_latents: bool,
) -> dict[str, Any]:
    model.eval()
    total_examples = 0
    totals = {"loss": 0.0, "ce": 0.0, "within": 0.0, "radius": 0.0, "between": 0.0}
    true_parts: list[np.ndarray] = []
    probability_parts: list[np.ndarray] = []
    centroid_parts: list[np.ndarray] = []
    sample_ids: list[str] = []
    participant_ids: list[str] = []
    global_parts: list[np.ndarray] = []
    z_parts: list[np.ndarray] = []
    contribution_parts: list[np.ndarray] = []
    logits_parts: list[np.ndarray] = []
    with torch.no_grad():
        for deaths, bins, mask, labels, sids, pids, global_indices in loader:
            deaths = deaths.to(device)
            bins = bins.to(device)
            mask = mask.to(device)
            labels = labels.to(device)
            z, logits = model(deaths, bins, mask)
            ce = ce_loss(logits, labels)
            parts = model.alpha_parts(z, labels, config.margin_between)
            loss = full_loss(ce, parts, config)
            probabilities = F.softmax(logits, dim=1)
            centroid_probabilities = F.softmax(-model.distances_to_centers(z), dim=1)
            batch_size = int(labels.size(0))
            total_examples += batch_size
            totals["loss"] += float(loss.item()) * batch_size
            totals["ce"] += float(ce.item()) * batch_size
            for key in ("within", "radius", "between"):
                totals[key] += float(parts[key].item()) * batch_size
            true_parts.append(labels.cpu().numpy())
            probability_parts.append(probabilities.cpu().numpy())
            centroid_parts.append(centroid_probabilities.cpu().numpy())
            sample_ids.extend(sids)
            participant_ids.extend(pids)
            global_parts.append(np.asarray(global_indices, dtype=int))
            if return_latents:
                # sample x class x interval logit contributions (bias excluded)
                contribution = z[:, None, :] * model.clf_head.weight[None, :, :]
                z_parts.append(z.cpu().numpy())
                contribution_parts.append(contribution.cpu().numpy())
                logits_parts.append(logits.cpu().numpy())
    if not total_examples:
        raise RuntimeError("Evaluation loader was empty")
    y_true = np.concatenate(true_parts)
    probabilities = np.vstack(probability_parts)
    centroid_probabilities = np.vstack(centroid_parts)
    y_pred = np.argmax(probabilities, axis=1)
    metrics = probability_metrics(y_true, probabilities, config.task.class_order)
    centroid_metrics = probability_metrics(y_true, centroid_probabilities, config.task.class_order)
    prediction_frame = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "participant_id": participant_ids,
            "global_index": np.concatenate(global_parts),
            "true_idx": y_true,
            "true_label": [config.task.class_order[index] for index in y_true],
            "pred_idx": y_pred,
            "pred_label": [config.task.class_order[index] for index in y_pred],
        }
    )
    for index, label in enumerate(config.task.class_order):
        prediction_frame[f"probability_{label}"] = probabilities[:, index]
        prediction_frame[f"centroid_probability_{label}"] = centroid_probabilities[:, index]
    result: dict[str, Any] = {
        "metrics": metrics,
        "centroid_metrics": centroid_metrics,
        "mean_loss": totals["loss"] / total_examples,
        "mean_ce": totals["ce"] / total_examples,
        "mean_within": totals["within"] / total_examples,
        "mean_radius": totals["radius"] / total_examples,
        "mean_between": totals["between"] / total_examples,
        "predictions": prediction_frame,
    }
    if return_latents:
        result["z"] = np.vstack(z_parts)
        result["class_interval_contributions"] = np.concatenate(contribution_parts, axis=0)
        result["logits"] = np.vstack(logits_parts)
    return result


def _train_epoch(model, loader, optimizer, ce_loss, bundle, device, config) -> dict[str, float]:
    model.train()
    total_examples = 0
    totals = {"loss": 0.0, "ce": 0.0, "within": 0.0, "radius": 0.0, "between": 0.0, "grad_norm": 0.0}
    true_parts: list[np.ndarray] = []
    probability_parts: list[np.ndarray] = []
    for deaths, bins, mask, labels, _sids, _pids, _global in loader:
        deaths = deaths.to(device)
        bins = bins.to(device)
        mask = mask.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        z, logits = model(deaths, bins, mask)
        ce = ce_loss(logits, labels)
        parts = model.alpha_parts(z, labels, config.margin_between)
        loss = full_loss(ce, parts, config)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite training loss")
        loss.backward()
        if config.max_grad_norm > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        else:
            gradient_square = torch.zeros((), device=device)
            for parameter in model.parameters():
                if parameter.grad is not None:
                    gradient_square += (parameter.grad.detach() ** 2).sum()
            grad_norm = torch.sqrt(gradient_square)
        optimizer.step()
        probabilities = F.softmax(logits, dim=1)
        batch_size = int(labels.size(0))
        total_examples += batch_size
        totals["loss"] += float(loss.item()) * batch_size
        totals["ce"] += float(ce.item()) * batch_size
        totals["grad_norm"] += float(grad_norm.item()) * batch_size
        for key in ("within", "radius", "between"):
            totals[key] += float(parts[key].item()) * batch_size
        true_parts.append(labels.detach().cpu().numpy())
        probability_parts.append(probabilities.detach().cpu().numpy())
    metrics = probability_metrics(np.concatenate(true_parts), np.vstack(probability_parts), config.task.class_order)
    return {
        **metrics,
        **{f"mean_{key}": value / total_examples for key, value in totals.items()},
    }


def _epoch_better(candidate: dict[str, Any], best: dict[str, Any] | None) -> bool:
    if best is None:
        return True
    key = (
        _safe(candidate["balanced_accuracy"]),
        _safe(candidate["macro_f1"]),
        _safe(candidate["roc_auc"]),
        -float(candidate["loss"]),
        -int(candidate["epoch"]),
    )
    best_key = (
        _safe(best["balanced_accuracy"]),
        _safe(best["macro_f1"]),
        _safe(best["roc_auc"]),
        -float(best["loss"]),
        -int(best["epoch"]),
    )
    return key > best_key


def _create_model(bounds: np.ndarray, config: EngineConfig, device: torch.device) -> AlphaPiModel:
    return AlphaPiModel(
        n_intervals=len(bounds) - 1,
        n_classes=len(config.task.class_order),
        bounds=bounds,
        sigma_log=sigma_from_bounds(bounds),
        quad_points=config.quad_points,
        point_chunk=config.point_chunk,
        device=device,
    ).to(device)


def _fit_candidate(
    bundle: TaskBundle,
    inner_train: np.ndarray,
    validation: np.ndarray,
    q_step: float,
    fold: int,
    candidate_dir: Path,
    device: torch.device,
    config: EngineConfig,
) -> dict[str, Any]:
    completion = candidate_dir / "CANDIDATE_COMPLETE.json"
    result_path = candidate_dir / "candidate_result.json"
    if config.resume and completion.exists() and result_path.exists():
        print(f"[resume] {config.task.folder} fold={fold} q={q_step:g}", flush=True)
        return json.loads(result_path.read_text())
    candidate_dir.mkdir(parents=True, exist_ok=True)
    bounds = build_bounds(bundle, inner_train, q_step, config.min_interval_width)
    seed = _candidate_seed(config, fold)
    set_all_seeds(seed, config.deterministic)
    train_loader = make_loader(bundle, inner_train, bounds, config.batch_size, True, seed, config.num_workers)
    validation_loader = make_loader(bundle, validation, bounds, config.batch_size, False, seed, config.num_workers)
    model = _create_model(bounds, config, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    ce_loss = class_weighted_ce(bundle.labels[inner_train], len(config.task.class_order), device)
    best_state = None
    best: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []
    for epoch in range(1, config.epochs + 1):
        train = _train_epoch(model, train_loader, optimizer, ce_loss, bundle, device, config)
        validation_result = _evaluate(model, validation_loader, ce_loss, bundle, device, config, False)
        validation_metrics = validation_result["metrics"]
        row = {
            "epoch": epoch,
            **{f"train_{key}": value for key, value in train.items()},
            **{f"validation_{key}": value for key, value in validation_metrics.items()},
            "validation_loss": validation_result["mean_loss"],
            "validation_ce": validation_result["mean_ce"],
        }
        history.append(row)
        candidate_epoch = {
            "epoch": epoch,
            "balanced_accuracy": validation_metrics["balanced_accuracy"],
            "macro_f1": validation_metrics["macro_f1"],
            "roc_auc": validation_metrics["roc_auc"],
            "loss": validation_result["mean_loss"],
        }
        if _epoch_better(candidate_epoch, best):
            best = candidate_epoch
            best_state = copy.deepcopy(model.state_dict())
        if epoch == 1 or epoch == config.epochs or epoch % config.print_every == 0:
            print(
                f"[candidate {config.task.folder} r01 f{fold} q={q_step:g}] "
                f"epoch={epoch:03d} train_bacc={train['balanced_accuracy']:.3f} "
                f"val_bacc={validation_metrics['balanced_accuracy']:.3f} "
                f"val_f1={validation_metrics['macro_f1']:.3f} val_auc={validation_metrics['roc_auc']:.3f}",
                flush=True,
            )
    if best_state is None or best is None:
        raise RuntimeError("Candidate training did not save a best state")
    model.load_state_dict(best_state)
    validation_result = _evaluate(model, validation_loader, ce_loss, bundle, device, config, False)
    history_frame = pd.DataFrame(history)
    atomic_csv(history_frame, candidate_dir / "training_history.csv", index=False)
    if config.make_plots:
        plot_training(history_frame, f"{config.task.report_name}, fold {fold}, q={q_step:g}", candidate_dir / "training_curves.png")
    alpha = model.alpha_positive().detach().cpu().numpy()
    atomic_npy(candidate_dir / "adaptive_bounds_inner_train.npy", bounds)
    atomic_npy(candidate_dir / "alpha_best.npy", alpha)
    atomic_torch(
        candidate_dir / "candidate_model.pt",
        {
            "state_dict": model.state_dict(),
            "bounds": bounds,
            "class_order": config.task.class_order,
            "best_epoch": int(best["epoch"]),
            "q_step": float(q_step),
            "seed": seed,
        },
    )
    result = {
        "q_step": float(q_step),
        "n_intervals": int(len(bounds) - 1),
        "best_epoch": int(best["epoch"]),
        "validation_accuracy": float(validation_result["metrics"]["accuracy"]),
        "validation_balanced_accuracy": float(validation_result["metrics"]["balanced_accuracy"]),
        "validation_macro_f1": float(validation_result["metrics"]["macro_f1"]),
        "validation_roc_auc": float(validation_result["metrics"]["roc_auc"]),
        "validation_loss": float(validation_result["mean_loss"]),
        "candidate_seed": seed,
    }
    atomic_json(result_path, result)
    atomic_json(completion, {"completed_utc": now_iso(), **result})
    return result


def _choose_candidate(results: list[dict[str, Any]]) -> dict[str, Any]:
    # Larger q means fewer nominal intervals; n_intervals is the authoritative tie-break.
    return max(
        results,
        key=lambda item: (
            _safe(item["validation_balanced_accuracy"]),
            _safe(item["validation_macro_f1"]),
            _safe(item["validation_roc_auc"]),
            -int(item["n_intervals"]),
            float(item["q_step"]),
        ),
    )


def _interpretation_curves(model: AlphaPiModel, bounds: np.ndarray, config: EngineConfig, device: torch.device) -> pd.DataFrame:
    grid = np.geomspace(1e-6, 0.999, config.common_grid_size).astype(np.float32)
    upper = np.nextafter(np.float32(bounds[-1]), np.float32(bounds[0]), dtype=np.float32)
    clipped = np.clip(grid, bounds[0], upper)
    bins = np.searchsorted(bounds, clipped, side="right") - 1
    bins = np.clip(bins, 0, len(bounds) - 2).astype(np.int64)
    with torch.no_grad():
        deaths = torch.tensor(clipped[:, None], dtype=torch.float32, device=device)
        bins_t = torch.tensor(bins[:, None], dtype=torch.long, device=device)
        mask = torch.ones_like(deaths)
        z, logits = model(deaths, bins_t, mask)
        bias = model.clf_head.bias.view(1, -1)
        contribution = (logits - bias).cpu().numpy()
        centered = contribution - contribution.mean(axis=1, keepdims=True)
    alpha = model.alpha_positive().detach().cpu().numpy()
    # Alpha is piecewise constant by construction; map each common-grid death
    # directly to its fold-specific adaptive interval rather than interpolating
    # between interval centres. This is faithful at both endpoints and avoids
    # artificial all-NaN regions when folds have different first/last centres.
    alpha_interp = alpha[bins]
    normalized = alpha / max(float(np.mean(alpha)), 1e-12)
    normalized_interp = normalized[bins]
    frame = pd.DataFrame(
        {
            "death_value": grid,
            "alpha_interpolated": alpha_interp,
            "alpha_normalized_interpolated": normalized_interp,
        }
    )
    for index, label in enumerate(config.task.class_order):
        frame[f"logit_contribution_{label}"] = contribution[:, index]
        frame[f"centered_logit_influence_{label}"] = centered[:, index]
    if len(config.task.class_order) == 2:
        frame["signed_logit_influence_class1_minus_class0"] = contribution[:, 1] - contribution[:, 0]
    return frame


def _fit_final(
    bundle: TaskBundle,
    outer_train: np.ndarray,
    outer_test: np.ndarray,
    selected: dict[str, Any],
    fold: int,
    fold_dir: Path,
    device: torch.device,
    config: EngineConfig,
) -> dict[str, Any]:
    final_dir = fold_dir / "final_outer_model"
    final_dir.mkdir(parents=True, exist_ok=True)
    bounds = build_bounds(bundle, outer_train, float(selected["q_step"]), config.min_interval_width)
    seed = _final_seed(config, fold)
    set_all_seeds(seed, config.deterministic)
    train_loader = make_loader(bundle, outer_train, bounds, config.batch_size, True, seed, config.num_workers)
    train_eval_loader = make_loader(bundle, outer_train, bounds, config.batch_size, False, seed, config.num_workers)
    test_loader = make_loader(bundle, outer_test, bounds, config.batch_size, False, seed, config.num_workers)
    model = _create_model(bounds, config, device)
    initial_state = copy.deepcopy(model.state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    ce_loss = class_weighted_ce(bundle.labels[outer_train], len(config.task.class_order), device)
    history: list[dict[str, Any]] = []
    epochs = int(selected["best_epoch"])
    for epoch in range(1, epochs + 1):
        train = _train_epoch(model, train_loader, optimizer, ce_loss, bundle, device, config)
        history.append({"epoch": epoch, **{f"train_{key}": value for key, value in train.items()}})
        if epoch == 1 or epoch == epochs or epoch % config.print_every == 0:
            print(
                f"[final {config.task.folder} r01 f{fold}] epoch={epoch:03d}/{epochs:03d} "
                f"train_bacc={train['balanced_accuracy']:.3f} train_auc={train['roc_auc']:.3f}",
                flush=True,
            )
    history_frame = pd.DataFrame(history)
    atomic_csv(history_frame, final_dir / "training_history.csv", index=False)
    if config.make_plots:
        plot_training(history_frame, f"{config.task.report_name}, fold {fold}: final outer-training refit", final_dir / "training_curves.png")
    train_result = _evaluate(model, train_eval_loader, ce_loss, bundle, device, config, False)
    test_result = _evaluate(model, test_loader, ce_loss, bundle, device, config, True)
    sample_predictions = test_result["predictions"]
    participants = participant_predictions(sample_predictions, config.task.class_order)
    participant_metrics = metrics_from_prediction_frame(participants, config.task.class_order)
    atomic_csv(sample_predictions, final_dir / "test_predictions_sample.csv", index=False)
    atomic_csv(participants, final_dir / "test_predictions_participant.csv", index=False)
    for level, prediction_frame in (("sample", sample_predictions), ("participant", participants)):
        matrix = confusion_matrix(
            prediction_frame["true_idx"],
            prediction_frame["pred_idx"],
            labels=np.arange(len(config.task.class_order)),
        )
        atomic_csv(
            pd.DataFrame(matrix, index=config.task.class_order, columns=config.task.class_order),
            final_dir / f"test_confusion_{level}.csv",
        )
        if config.make_plots:
            plot_confusion(
                matrix,
                config.task.class_order,
                f"{config.task.report_name}, fold {fold}: held-out {level} confusion",
                final_dir / f"test_confusion_{level}.png",
            )
        report = classification_report(
            prediction_frame["true_idx"],
            prediction_frame["pred_idx"],
            labels=np.arange(len(config.task.class_order)),
            target_names=config.task.class_order,
            zero_division=0,
            digits=4,
        )
        (final_dir / f"test_classification_report_{level}.txt").write_text(report)
    atomic_npz(
        final_dir / "test_latents_and_contributions.npz",
        z=test_result["z"],
        class_interval_contributions=test_result["class_interval_contributions"],
        logits=test_result["logits"],
        sample_ids=np.asarray(sample_predictions["sample_id"].tolist(), dtype=str),
        class_names=np.asarray(config.task.class_order, dtype=str),
    )
    alpha = model.alpha_positive().detach().cpu().numpy()
    atomic_npy(final_dir / "adaptive_bounds_outer_train.npy", bounds)
    atomic_npy(final_dir / "alpha_raw.npy", alpha)
    atomic_npy(final_dir / "alpha_normalized_mean1.npy", alpha / max(float(alpha.mean()), 1e-12))
    curves = _interpretation_curves(model, bounds, config, device)
    atomic_csv(curves, final_dir / "interpretation_curves.csv", index=False)
    if config.make_plots:
        plot_curve(
            curves["death_value"].to_numpy(dtype=float),
            curves["alpha_interpolated"].to_numpy(dtype=float),
            None,
            "learned nonnegative alpha",
            f"{config.task.report_name}, fold {fold}: learned alpha",
            final_dir / "alpha_curve.png",
        )
        fold_influence_means = {
            label: curves[f"centered_logit_influence_{label}"].to_numpy(dtype=float)
            for label in config.task.class_order
        }
        fold_influence_sds = {label: np.zeros(len(curves), dtype=float) for label in config.task.class_order}
        plot_multiclass_curves(
            curves["death_value"].to_numpy(dtype=float),
            fold_influence_means,
            fold_influence_sds,
            "centered class-logit influence of one H0 death",
            f"{config.task.report_name}, fold {fold}: H0 logit influence",
            final_dir / "centered_logit_influence_curve.png",
        )
        if len(config.task.class_order) == 2:
            plot_curve(
                curves["death_value"].to_numpy(dtype=float),
                curves["signed_logit_influence_class1_minus_class0"].to_numpy(dtype=float),
                None,
                f"logit influence: {config.task.class_order[1]} minus {config.task.class_order[0]}",
                f"{config.task.report_name}, fold {fold}: signed H0 influence",
                final_dir / "signed_logit_influence_curve.png",
            )
    atomic_csv(
        pd.DataFrame(
            model.clf_head.weight.detach().cpu().numpy(),
            index=config.task.class_order,
            columns=[f"interval_{index}" for index in range(len(bounds) - 1)],
        ),
        final_dir / "classifier_head_coefficients.csv",
    )
    atomic_csv(
        pd.DataFrame(model.centers.detach().cpu().numpy(), index=config.task.class_order),
        final_dir / "class_centers_parameter.csv",
    )
    atomic_torch(
        final_dir / "final_outer_model.pt",
        {
            "state_dict": model.state_dict(),
            "initial_state_dict": initial_state,
            "bounds": bounds,
            "class_order": config.task.class_order,
            "selected_q_step": float(selected["q_step"]),
            "selected_epoch": epochs,
            "seed": seed,
            "fresh_initialization": True,
        },
    )
    result = {
        "task_folder": config.task.folder,
        "task": config.task.report_name,
        "repeat": config.repeat,
        "fold": fold,
        "n_outer_train_samples": int(len(outer_train)),
        "n_outer_test_samples": int(len(outer_test)),
        "n_outer_train_participants": int(len(set(np.asarray(bundle.participant_ids)[outer_train]))),
        "n_outer_test_participants": int(len(set(np.asarray(bundle.participant_ids)[outer_test]))),
        "selected_q_step": float(selected["q_step"]),
        "selected_n_intervals_inner": int(selected["n_intervals"]),
        "selected_n_intervals_final": int(len(bounds) - 1),
        "selected_epoch": epochs,
        **{f"test_sample_{key}": float(value) for key, value in test_result["metrics"].items()},
        **{f"test_participant_{key}": float(value) for key, value in participant_metrics.items()},
        **{f"test_centroid_{key}": float(value) for key, value in test_result["centroid_metrics"].items()},
        **{f"train_sample_{key}": float(value) for key, value in train_result["metrics"].items()},
        "test_mean_loss": float(test_result["mean_loss"]),
        "test_mean_ce": float(test_result["mean_ce"]),
        "final_seed": seed,
    }
    atomic_json(final_dir / "final_fold_result.json", result)
    return result


def _write_split_membership(
    path: Path,
    bundle: TaskBundle,
    fold: int,
    outer_train: np.ndarray,
    outer_test: np.ndarray,
    inner_train: np.ndarray,
    validation: np.ndarray,
    split_seed: int,
    inner_seed: int,
) -> None:
    role_map: dict[int, str] = {}
    for index in inner_train:
        role_map[int(index)] = "inner_train"
    for index in validation:
        role_map[int(index)] = "validation"
    for index in outer_test:
        role_map[int(index)] = "outer_test"
    rows = []
    for index in np.concatenate([outer_train, outer_test]):
        index = int(index)
        rows.append(
            {
                "repeat": 1,
                "fold": fold,
                "sample_id": bundle.sample_ids[index],
                "participant_id": bundle.participant_ids[index],
                "label": int(bundle.labels[index]),
                "label_name": bundle.label_names[index],
                "role": role_map[index],
                "outer_split_seed": split_seed,
                "inner_split_seed": inner_seed,
            }
        )
    atomic_csv(pd.DataFrame(rows), path, index=False)


def _run_fold(
    manifest: pd.DataFrame,
    bundle: TaskBundle,
    fold: int,
    task_dir: Path,
    device: torch.device,
    config: EngineConfig,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fold_dir = task_dir / "folds" / f"repeat_01_fold_{fold}"
    completion = fold_dir / "FOLD_COMPLETE.json"
    if config.resume and completion.exists():
        final_dir = fold_dir / "final_outer_model"
        required = [
            final_dir / "final_fold_result.json",
            final_dir / "test_predictions_sample.csv",
            final_dir / "test_predictions_participant.csv",
            final_dir / "interpretation_curves.csv",
            fold_dir / "candidate_summary.csv",
        ]
        if not all(path.exists() for path in required):
            raise RuntimeError(f"Corrupt completion marker in {fold_dir}")
        print(f"[resume] {config.task.folder} repeat_01_fold_{fold}", flush=True)
        return (
            json.loads(required[0].read_text()),
            pd.read_csv(required[1], dtype={"sample_id": str, "participant_id": str}),
            pd.read_csv(required[2], dtype={"participant_id": str}),
            pd.read_csv(required[3]),
            pd.read_csv(required[4]),
        )
    fold_dir.mkdir(parents=True, exist_ok=True)
    outer_train, outer_test, split_seed = fold_indices(manifest, bundle, fold)
    inner_train, validation, inner_seed = _inner_split(bundle, outer_train, fold, config)
    _write_split_membership(
        fold_dir / "split_membership.csv",
        bundle,
        fold,
        outer_train,
        outer_test,
        inner_train,
        validation,
        split_seed,
        inner_seed,
    )
    print(
        f"\n[outer {config.task.folder} repeat_01_fold_{fold}] "
        f"inner_train={len(inner_train)} samples/{len(set(np.asarray(bundle.participant_ids)[inner_train]))} participants, "
        f"validation={len(validation)} samples/{len(set(np.asarray(bundle.participant_ids)[validation]))} participants, "
        f"outer_test={len(outer_test)} samples/{len(set(np.asarray(bundle.participant_ids)[outer_test]))} participants",
        flush=True,
    )
    candidates = []
    for q_step in config.percentile_steps:
        candidates.append(
            _fit_candidate(
                bundle,
                inner_train,
                validation,
                q_step,
                fold,
                fold_dir / "candidates" / _percentile_tag(q_step),
                device,
                config,
            )
        )
    selected = _choose_candidate(candidates)
    candidate_frame = pd.DataFrame(candidates)
    candidate_frame["selected"] = np.isclose(candidate_frame["q_step"], float(selected["q_step"]))
    atomic_csv(candidate_frame, fold_dir / "candidate_summary.csv", index=False)
    atomic_json(fold_dir / "selected_candidate.json", selected)
    result = _fit_final(bundle, outer_train, outer_test, selected, fold, fold_dir, device, config)
    final_dir = fold_dir / "final_outer_model"
    sample_predictions = pd.read_csv(final_dir / "test_predictions_sample.csv", dtype={"sample_id": str, "participant_id": str})
    participants = pd.read_csv(final_dir / "test_predictions_participant.csv", dtype={"participant_id": str})
    curves = pd.read_csv(final_dir / "interpretation_curves.csv")
    atomic_json(
        completion,
        {
            "completed_utc": now_iso(),
            "task_folder": config.task.folder,
            "repeat": 1,
            "fold": fold,
            "selected_q_step": selected["q_step"],
            "selected_epoch": selected["best_epoch"],
        },
    )
    return result, sample_predictions, participants, curves, candidate_frame


def _aggregate_task(
    task_dir: Path,
    bundle: TaskBundle,
    results: list[dict[str, Any]],
    samples: list[pd.DataFrame],
    participants: list[pd.DataFrame],
    curves: list[pd.DataFrame],
    candidates: list[pd.DataFrame],
    config: EngineConfig,
) -> dict[str, Any]:
    result_frame = pd.DataFrame(results).sort_values("fold")
    sample_frame = pd.concat(samples, ignore_index=True).sort_values(["fold", "sample_id"]) if "fold" in samples[0] else pd.concat(samples, ignore_index=True)
    participant_frame = pd.concat(participants, ignore_index=True)
    # Add fold/repeat if older readers omitted them.
    if "fold" not in sample_frame:
        pieces = []
        for fold, frame in enumerate(samples, start=1):
            copy_frame = frame.copy()
            copy_frame.insert(0, "fold", fold)
            copy_frame.insert(0, "repeat", 1)
            pieces.append(copy_frame)
        sample_frame = pd.concat(pieces, ignore_index=True)
    if "fold" not in participant_frame:
        pieces = []
        for fold, frame in enumerate(participants, start=1):
            copy_frame = frame.copy()
            copy_frame.insert(0, "fold", fold)
            copy_frame.insert(0, "repeat", 1)
            pieces.append(copy_frame)
        participant_frame = pd.concat(pieces, ignore_index=True)
    if sample_frame["sample_id"].duplicated().any() or set(sample_frame["sample_id"]) != set(bundle.sample_ids):
        raise RuntimeError(f"Task {config.task.folder} does not have exactly one OOF prediction per sample")
    if participant_frame["participant_id"].duplicated().any() or set(participant_frame["participant_id"]) != set(bundle.participant_ids):
        raise RuntimeError(f"Task {config.task.folder} does not have exactly one OOF prediction per participant")
    atomic_csv(result_frame, task_dir / "fold_results.csv", index=False)
    atomic_csv(sample_frame, task_dir / "all_oof_sample_predictions.csv", index=False)
    atomic_csv(participant_frame, task_dir / "all_oof_participant_predictions.csv", index=False)
    atomic_csv(pd.concat(candidates, ignore_index=True), task_dir / "all_candidate_validation_results.csv", index=False)
    sample_metrics = metrics_from_prediction_frame(sample_frame, config.task.class_order)
    participant_metrics = metrics_from_prediction_frame(participant_frame, config.task.class_order)
    pooled = pd.DataFrame(
        [
            {"level": "sample", **sample_metrics},
            {"level": "participant", **participant_metrics},
        ]
    )
    atomic_csv(pooled, task_dir / "pooled_oof_metrics.csv", index=False)
    metric_suffixes = ["accuracy", "balanced_accuracy", "macro_f1", "roc_auc", "brier_multiclass", "log_loss"]
    summary: dict[str, Any] = {
        "task_folder": config.task.folder,
        "task": config.task.report_name,
        "n_folds": config.expected_folds,
        "n_samples": len(bundle.sample_ids),
        "n_participants": len(set(bundle.participant_ids)),
    }
    for suffix in metric_suffixes:
        summary.update(describe(result_frame[f"test_sample_{suffix}"], f"sample_{suffix}"))
        summary.update(describe(result_frame[f"test_participant_{suffix}"], f"participant_{suffix}"))
    class_recall_rows = []
    for label in config.task.class_order:
        sample_column = f"test_sample_recall_{label}"
        participant_column = f"test_participant_recall_{label}"
        if sample_column in result_frame.columns:
            sample_description = describe(result_frame[sample_column], f"sample_recall_{label}")
            summary.update(sample_description)
            class_recall_rows.append({"level": "sample", "class": label, **sample_description})
        if participant_column in result_frame.columns:
            participant_description = describe(result_frame[participant_column], f"participant_recall_{label}")
            summary.update(participant_description)
            class_recall_rows.append({"level": "participant", "class": label, **participant_description})
    if class_recall_rows:
        atomic_csv(pd.DataFrame(class_recall_rows), task_dir / "per_class_recall_mean_sd_across_folds.csv", index=False)
    summary.update(describe(result_frame["selected_q_step"], "selected_q_step"))
    summary.update(describe(result_frame["selected_n_intervals_final"], "selected_intervals"))
    atomic_csv(pd.DataFrame([summary]), task_dir / "summary_mean_sd_across_folds.csv", index=False)

    for level, frame in (("sample", sample_frame), ("participant", participant_frame)):
        cm = confusion_matrix(frame["true_idx"], frame["pred_idx"], labels=np.arange(len(config.task.class_order)))
        atomic_csv(pd.DataFrame(cm, index=config.task.class_order, columns=config.task.class_order), task_dir / f"aggregated_confusion_{level}.csv")
        if config.make_plots:
            plot_confusion(cm, config.task.class_order, f"{config.task.report_name}: pooled OOF {level} confusion", task_dir / f"aggregated_confusion_{level}.png")
        report = classification_report(
            frame["true_idx"],
            frame["pred_idx"],
            labels=np.arange(len(config.task.class_order)),
            target_names=config.task.class_order,
            zero_division=0,
            digits=4,
        )
        (task_dir / f"classification_report_{level}.txt").write_text(report)

    curve_stack = pd.concat(
        [frame.assign(fold=index) for index, frame in enumerate(curves, start=1)],
        ignore_index=True,
    )
    atomic_csv(curve_stack, task_dir / "interpretation_curves_all_folds.csv", index=False)
    grid = curves[0]["death_value"].to_numpy(dtype=float)
    alpha_stack = np.vstack([frame["alpha_interpolated"].to_numpy(dtype=float) for frame in curves])
    alpha_norm_stack = np.vstack([frame["alpha_normalized_interpolated"].to_numpy(dtype=float) for frame in curves])
    alpha_mean = np.nanmean(alpha_stack, axis=0)
    alpha_sd = np.nanstd(alpha_stack, axis=0, ddof=1)
    alpha_norm_mean = np.nanmean(alpha_norm_stack, axis=0)
    alpha_norm_sd = np.nanstd(alpha_norm_stack, axis=0, ddof=1)
    alpha_summary = pd.DataFrame(
        {
            "death_value": grid,
            "alpha_mean": alpha_mean,
            "alpha_sd": alpha_sd,
            "alpha_normalized_mean": alpha_norm_mean,
            "alpha_normalized_sd": alpha_norm_sd,
        }
    )
    for fold, values in enumerate(alpha_stack, start=1):
        alpha_summary[f"alpha_fold_{fold}"] = values
    atomic_csv(alpha_summary, task_dir / "alpha_common_grid_mean_sd.csv", index=False)
    if config.make_plots:
        plot_curve(grid, alpha_mean, alpha_sd, "learned nonnegative alpha", f"{config.task.report_name}: mean alpha across five folds", task_dir / "mean_alpha_across_folds.png")
        plot_curve(grid, alpha_norm_mean, alpha_norm_sd, "alpha normalized to fold mean 1", f"{config.task.report_name}: normalized alpha across five folds", task_dir / "mean_alpha_normalized_across_folds.png")

    influence_means: dict[str, np.ndarray] = {}
    influence_sds: dict[str, np.ndarray] = {}
    influence_summary = pd.DataFrame({"death_value": grid})
    for label in config.task.class_order:
        column = f"centered_logit_influence_{label}"
        stack = np.vstack([frame[column].to_numpy(dtype=float) for frame in curves])
        influence_means[label] = np.mean(stack, axis=0)
        influence_sds[label] = np.std(stack, axis=0, ddof=1)
        influence_summary[f"{label}_mean"] = influence_means[label]
        influence_summary[f"{label}_sd"] = influence_sds[label]
    atomic_csv(influence_summary, task_dir / "centered_logit_influence_common_grid_mean_sd.csv", index=False)
    if config.make_plots:
        plot_multiclass_curves(
            grid,
            influence_means,
            influence_sds,
            "centered class-logit influence of one H0 death",
            f"{config.task.report_name}: H0 logit influence across five folds",
            task_dir / "mean_centered_logit_influence_across_folds.png",
        )
    if len(config.task.class_order) == 2:
        signed_stack = np.vstack([frame["signed_logit_influence_class1_minus_class0"].to_numpy(dtype=float) for frame in curves])
        signed_mean = signed_stack.mean(axis=0)
        signed_sd = signed_stack.std(axis=0, ddof=1)
        atomic_csv(
            pd.DataFrame({"death_value": grid, "signed_influence_mean": signed_mean, "signed_influence_sd": signed_sd}),
            task_dir / "signed_logit_influence_common_grid_mean_sd.csv",
            index=False,
        )
        if config.make_plots:
            plot_curve(
                grid,
                signed_mean,
                signed_sd,
                f"logit influence: {config.task.class_order[1]} minus {config.task.class_order[0]}",
                f"{config.task.report_name}: signed H0 influence across five folds",
                task_dir / "mean_signed_logit_influence_across_folds.png",
            )
    atomic_json(task_dir / "TASK_COMPLETE.json", {"completed_utc": now_iso(), **summary})
    return summary


def run_task(config: EngineConfig, validate_only: bool = False) -> dict[str, Any] | None:
    task_dir = config.output_dir / config.task.folder
    task_dir.mkdir(parents=True, exist_ok=True)
    manifest, metadata = load_task_manifest(config.split_manifest, config.task, config.repeat, config.expected_folds)
    bundle = load_task_bundle(config.pd_dir, config.label_csv, config.metadata_csv, metadata, config.task)
    current_config = _run_config(config, manifest, bundle)
    config_path = task_dir / "run_config.json"
    if config_path.exists() and config.resume:
        _compatible(json.loads(config_path.read_text()), current_config)
    else:
        atomic_json(config_path, current_config)
    print("=" * 112, flush=True)
    print(f"H0 Alpha-Pi 1x5 — {config.task.report_name}", flush=True)
    print(f"Samples: {len(bundle.sample_ids)}; participants: {len(set(bundle.participant_ids))}; classes: {config.task.class_order}", flush=True)
    print(f"Manifest: {config.split_manifest}; repeat={config.repeat}; folds={config.expected_folds}", flush=True)
    print("Adaptive bounds: inner-train only for selection; full outer-train for fresh final refit", flush=True)
    print("=" * 112, flush=True)
    if validate_only:
        return None
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    results = []
    samples = []
    participants = []
    curves = []
    candidates = []
    for fold in range(1, config.expected_folds + 1):
        result, sample_frame, participant_frame, curve_frame, candidate_frame = _run_fold(
            manifest, bundle, fold, task_dir, device, config
        )
        results.append(result)
        samples.append(sample_frame)
        participants.append(participant_frame)
        curves.append(curve_frame)
        candidates.append(candidate_frame.assign(fold=fold, repeat=1))
        atomic_json(
            task_dir / "progress.json",
            {
                "updated_utc": now_iso(),
                "completed_folds": len(results),
                "expected_folds": config.expected_folds,
                "task_folder": config.task.folder,
            },
        )
    summary = _aggregate_task(task_dir, bundle, results, samples, participants, curves, candidates, config)
    print(f"[complete] {config.task.report_name}: {task_dir / 'summary_mean_sd_across_folds.csv'}", flush=True)
    return summary
