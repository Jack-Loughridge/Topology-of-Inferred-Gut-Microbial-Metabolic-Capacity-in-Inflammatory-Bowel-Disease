from __future__ import annotations

import math
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy import sparse
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler

from .wkpi import DenseStandardizer, alpha_from_logits, fit_dense_standardizer


ProgressFunction = Callable[[str], None]
StateCallback = Callable[["ModelWarmStart"], None]


@dataclass(frozen=True)
class JointHyperparameters:
    n_intervals: int
    lambda_h0: float
    lambda_ricci: float


@dataclass
class AlternationRecord:
    alternation: int
    alpha_objective_before: float
    alpha_objective_after: float
    alpha_optimizer_success: bool
    alpha_optimizer_iterations: int
    logistic_iterations: int
    active_set_rounds: int
    active_ricci_features: int
    max_kkt_violation: float
    logistic_objective: float
    nonzero_h0: int
    nonzero_ricci: int
    elapsed_seconds: float
    polish_calls: int = 0
    polish_iterations: int = 0


@dataclass
class SolverDiagnostics:
    iterations: int
    active_set_rounds: int
    active_ricci_features: int
    max_kkt_violation: float
    objective: float
    smooth_loss: float
    penalty: float
    converged: bool
    step_size: float
    elapsed_seconds: float
    polish_calls: int = 0
    polish_iterations: int = 0


@dataclass
class ModelWarmStart:
    """Serializable state used for alpha alternations and penalty-path warm starts."""

    beta_h0: np.ndarray
    beta_ricci: np.ndarray
    intercept: np.ndarray
    alpha_logits: np.ndarray
    step_size: float = 1.0
    completed_alternations: int = 0

    def copy(self) -> "ModelWarmStart":
        return ModelWarmStart(
            beta_h0=np.array(self.beta_h0, dtype=np.float64, copy=True),
            beta_ricci=np.array(self.beta_ricci, dtype=np.float64, copy=True),
            intercept=np.array(self.intercept, dtype=np.float64, copy=True),
            alpha_logits=np.array(self.alpha_logits, dtype=np.float64, copy=True),
            step_size=float(self.step_size),
            completed_alternations=int(self.completed_alternations),
        )


@dataclass
class PreparedRicciSplit:
    train: np.ndarray
    evaluation: np.ndarray | None
    scaler: StandardScaler


@dataclass
class PreparedH0Split:
    train: np.ndarray
    evaluation: np.ndarray | None
    standardizer: DenseStandardizer


@dataclass
class JointSparseModel:
    classes: tuple[str, ...]
    hyperparameters: JointHyperparameters
    alpha: np.ndarray
    h0_standardizer: DenseStandardizer
    ricci_scaler: StandardScaler
    effective_beta_h0: np.ndarray
    effective_beta_ricci: np.ndarray
    effective_intercept: np.ndarray
    alpha_logits: np.ndarray
    solver_diagnostics: SolverDiagnostics
    alternation_history: list[AlternationRecord] = field(default_factory=list)

    def transform_h0(self, phi: np.ndarray) -> np.ndarray:
        standardised = self.h0_standardizer.transform(np.asarray(phi, dtype=np.float64))
        return np.ascontiguousarray(standardised * self.alpha[None, :], dtype=np.float64)

    def transform_ricci(self, matrix: np.ndarray | sparse.spmatrix) -> np.ndarray:
        dense = _as_dense64(matrix, copy=True)
        return np.ascontiguousarray(self.ricci_scaler.transform(dense, copy=False), dtype=np.float64)

    def predict_proba_prepared(
        self,
        phi_standardised: np.ndarray,
        ricci_standardised: np.ndarray,
    ) -> np.ndarray:
        x_h0 = np.asarray(phi_standardised, dtype=np.float64) * self.alpha[None, :]
        scores = (
            x_h0 @ self.effective_beta_h0.T
            + np.asarray(ricci_standardised, dtype=np.float64) @ self.effective_beta_ricci.T
            + self.effective_intercept[None, :]
        )
        return _softmax(scores)

    def predict_proba(self, phi: np.ndarray, ricci: np.ndarray | sparse.spmatrix) -> np.ndarray:
        phi_standardised = self.h0_standardizer.transform(np.asarray(phi, dtype=np.float64))
        ricci_standardised = self.transform_ricci(ricci)
        return self.predict_proba_prepared(phi_standardised, ricci_standardised)

    def predict(self, phi: np.ndarray, ricci: np.ndarray | sparse.spmatrix) -> np.ndarray:
        probabilities = self.predict_proba(phi, ricci)
        return np.asarray(
            [self.classes[index] for index in np.argmax(probabilities, axis=1)],
            dtype=object,
        )

    def warm_start(self, completed_alternations: int | None = None) -> ModelWarmStart:
        return ModelWarmStart(
            beta_h0=np.array(self.effective_beta_h0, copy=True),
            beta_ricci=np.array(self.effective_beta_ricci, copy=True),
            intercept=np.array(self.effective_intercept, copy=True),
            alpha_logits=np.array(self.alpha_logits, copy=True),
            step_size=float(self.solver_diagnostics.step_size),
            completed_alternations=(
                len(self.alternation_history)
                if completed_alternations is None
                else int(completed_alternations)
            ),
        )


def _as_dense64(matrix: np.ndarray | sparse.spmatrix, *, copy: bool) -> np.ndarray:
    if sparse.issparse(matrix):
        array = matrix.toarray(order="C")
    else:
        array = np.asarray(matrix, dtype=np.float64, order="C")
        if copy:
            array = np.array(array, dtype=np.float64, order="C", copy=True)
    if array.dtype != np.float64:
        array = array.astype(np.float64, copy=False)
    if not array.flags.c_contiguous:
        array = np.ascontiguousarray(array)
    bad = ~np.isfinite(array)
    if bad.any():
        if not copy:
            array = array.copy(order="C")
        array[bad] = 0.0
    return array


def prepare_ricci_split(
    train: np.ndarray | sparse.spmatrix,
    evaluation: np.ndarray | sparse.spmatrix | None = None,
) -> PreparedRicciSplit:
    """Fit the train-only Ricci scaler once and transform dense arrays in place."""
    train_dense = _as_dense64(train, copy=True)
    evaluation_dense = _as_dense64(evaluation, copy=True) if evaluation is not None else None
    scaler = StandardScaler(with_mean=False, with_std=True, copy=False)
    train_dense = np.ascontiguousarray(scaler.fit_transform(train_dense), dtype=np.float64)
    if evaluation_dense is not None:
        evaluation_dense = np.ascontiguousarray(
            scaler.transform(evaluation_dense, copy=False), dtype=np.float64
        )
    train_dense.setflags(write=False)
    if evaluation_dense is not None:
        evaluation_dense.setflags(write=False)
    return PreparedRicciSplit(train=train_dense, evaluation=evaluation_dense, scaler=scaler)


def prepare_h0_split(
    train: np.ndarray,
    evaluation: np.ndarray | None = None,
) -> PreparedH0Split:
    train = np.asarray(train, dtype=np.float64)
    standardizer = fit_dense_standardizer(train)
    train_standardised = np.ascontiguousarray(standardizer.transform(train), dtype=np.float64)
    evaluation_standardised = None
    if evaluation is not None:
        evaluation_standardised = np.ascontiguousarray(
            standardizer.transform(np.asarray(evaluation, dtype=np.float64)), dtype=np.float64
        )
    train_standardised.setflags(write=False)
    if evaluation_standardised is not None:
        evaluation_standardised.setflags(write=False)
    return PreparedH0Split(
        train=train_standardised,
        evaluation=evaluation_standardised,
        standardizer=standardizer,
    )


def _balanced_sample_weights(y: np.ndarray, classes: tuple[str, ...]) -> np.ndarray:
    counts = {label: int(np.sum(y == label)) for label in classes}
    n = len(y)
    k = len(classes)
    weights = {label: n / (k * max(counts[label], 1)) for label in classes}
    return np.asarray([weights[str(label)] for label in y], dtype=np.float64)


def _softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)


def _sigmoid(scores: np.ndarray) -> np.ndarray:
    output = np.empty_like(scores, dtype=np.float64)
    positive = scores >= 0
    output[positive] = 1.0 / (1.0 + np.exp(-scores[positive]))
    exp_scores = np.exp(scores[~positive])
    output[~positive] = exp_scores / (1.0 + exp_scores)
    return output


# ALPHA_SMOOTHNESS_RESOLUTION_NORMALISATION_V1


def resolution_normalised_alpha_smoothness_gamma(
    base_gamma: float,
    n_intervals: int,
    reference_intervals: int = 160,
) -> float:
    """Return a finite-difference smoothness weight comparable across M.

    The discrete roughness sum ``sum((alpha[j+1]-alpha[j])**2)`` scales
    approximately as ``1/(M-1)`` for samples of the same underlying smooth
    function. Multiplying gamma by ``(M-1)/(M_ref-1)`` therefore keeps the
    functional strength of the smoothness prior approximately invariant.
    """
    if base_gamma < 0:
        raise ValueError("base_gamma must be nonnegative.")
    if n_intervals < 2:
        raise ValueError("n_intervals must be at least 2.")
    if reference_intervals < 2:
        raise ValueError("reference_intervals must be at least 2.")
    return float(base_gamma) * (int(n_intervals) - 1) / (int(reference_intervals) - 1)


def _alpha_loss_and_gradient(
    logits: np.ndarray,
    phi_standardised: np.ndarray,
    ricci_scores: np.ndarray,
    beta_h0: np.ndarray,
    y_indices: np.ndarray,
    sample_weights: np.ndarray,
    smoothness_gamma: float,
) -> tuple[float, np.ndarray]:
    alpha = alpha_from_logits(logits)
    smoothness_gamma = resolution_normalised_alpha_smoothness_gamma(
        smoothness_gamma, len(alpha), reference_intervals=160
    )
    h0_scores = (phi_standardised * alpha[None, :]) @ beta_h0.T
    scores = ricci_scores + h0_scores
    probabilities = _softmax(scores)

    eps = 1e-12
    normaliser = float(np.sum(sample_weights))
    ce = -np.sum(
        sample_weights
        * np.log(np.clip(probabilities[np.arange(len(y_indices)), y_indices], eps, 1.0))
    ) / normaliser

    score_gradient = probabilities.copy()
    score_gradient[np.arange(len(y_indices)), y_indices] -= 1.0
    score_gradient *= sample_weights[:, None] / normaliser
    alpha_gradient = np.einsum(
        "ik,im,km->m", score_gradient, phi_standardised, beta_h0, optimize=True
    )

    if len(alpha) > 1:
        differences = np.diff(alpha)
        smoothness = smoothness_gamma * float(np.sum(differences * differences))
        scale = 2.0 * smoothness_gamma
        smooth_gradient = np.zeros_like(alpha)
        smooth_gradient[:-1] -= scale * differences
        smooth_gradient[1:] += scale * differences
        alpha_gradient += smooth_gradient
    else:
        smoothness = 0.0

    weighted_mean_gradient = float(np.sum(alpha * alpha_gradient) / len(alpha))
    logits_gradient = alpha * (alpha_gradient - weighted_mean_gradient)
    return float(ce + smoothness), logits_gradient


def _soft_threshold(values: np.ndarray, threshold: float) -> np.ndarray:
    return np.sign(values) * np.maximum(np.abs(values) - threshold, 0.0)


def _penalty_value(beta_h0: np.ndarray, beta_ricci: np.ndarray, rho_h0: float, rho_ricci: float) -> float:
    return float(rho_h0 * np.abs(beta_h0).sum() + rho_ricci * np.abs(beta_ricci).sum())


def _max_l1_kkt(beta: np.ndarray, gradient: np.ndarray, rho: float, zero_tol: float = 1e-12) -> float:
    nonzero = np.abs(beta) > zero_tol
    violations = np.empty_like(beta, dtype=np.float64)
    violations[nonzero] = np.abs(gradient[nonzero] + rho * np.sign(beta[nonzero]))
    violations[~nonzero] = np.maximum(np.abs(gradient[~nonzero]) - rho, 0.0)
    return float(np.max(violations)) if violations.size else 0.0


def _ricci_column_violations(beta: np.ndarray, gradient: np.ndarray, rho: float) -> np.ndarray:
    nonzero = np.abs(beta) > 1e-12
    violations = np.empty_like(beta, dtype=np.float64)
    violations[nonzero] = np.abs(gradient[nonzero] + rho * np.sign(beta[nonzero]))
    violations[~nonzero] = np.maximum(np.abs(gradient[~nonzero]) - rho, 0.0)
    if violations.ndim == 1:
        return violations
    return np.max(violations, axis=0)


def _estimate_design_lipschitz(
    x_h0: np.ndarray,
    x_ricci_active: np.ndarray,
    sample_weights: np.ndarray,
    curvature_bound: float,
    *,
    seed: int,
    n_power_iterations: int = 12,
) -> float:
    """Estimate curvature_bound * ||sqrt(W) [X,1]||_2^2 / sum(W)."""
    rng = np.random.default_rng(seed)
    vh = rng.normal(size=x_h0.shape[1])
    vr = rng.normal(size=x_ricci_active.shape[1])
    vb = float(rng.normal())
    total_norm = math.sqrt(float(vh @ vh + vr @ vr + vb * vb))
    if total_norm == 0:
        return 1.0
    vh /= total_norm
    vr /= total_norm
    vb /= total_norm
    normalised_weights = sample_weights / float(np.sum(sample_weights))
    eigenvalue = 1.0
    for _ in range(n_power_iterations):
        projected = x_h0 @ vh + x_ricci_active @ vr + vb
        weighted = normalised_weights * projected
        next_h = x_h0.T @ weighted
        next_r = x_ricci_active.T @ weighted
        next_b = float(np.sum(weighted))
        norm = math.sqrt(float(next_h @ next_h + next_r @ next_r + next_b * next_b))
        if norm <= 1e-30:
            return 1.0
        vh = next_h / norm
        vr = next_r / norm
        vb = next_b / norm
        eigenvalue = float(
            vh @ (x_h0.T @ (normalised_weights * (x_h0 @ vh + x_ricci_active @ vr + vb)))
            + vr @ (x_ricci_active.T @ (normalised_weights * (x_h0 @ vh + x_ricci_active @ vr + vb)))
            + vb * np.sum(normalised_weights * (x_h0 @ vh + x_ricci_active @ vr + vb))
        )
    return max(curvature_bound * eigenvalue * 1.05, 1e-8)


def _binary_smooth(
    x_h0: np.ndarray,
    x_ricci: np.ndarray,
    y01: np.ndarray,
    sample_weights: np.ndarray,
    beta_h0: np.ndarray,
    beta_ricci: np.ndarray,
    intercept: float,
    *,
    gradient: bool,
):
    scores = x_h0 @ beta_h0 + x_ricci @ beta_ricci + intercept
    normaliser = float(np.sum(sample_weights))
    smooth = float(
        np.sum(sample_weights * (np.logaddexp(0.0, scores) - y01 * scores)) / normaliser
    )
    if not gradient:
        return smooth
    residual = sample_weights * (_sigmoid(scores) - y01) / normaliser
    return (
        smooth,
        x_h0.T @ residual,
        x_ricci.T @ residual,
        float(np.sum(residual)),
    )


def _multiclass_smooth(
    x_h0: np.ndarray,
    x_ricci: np.ndarray,
    y_indices: np.ndarray,
    sample_weights: np.ndarray,
    beta_h0: np.ndarray,
    beta_ricci: np.ndarray,
    intercept: np.ndarray,
    *,
    gradient: bool,
):
    scores = x_h0 @ beta_h0.T + x_ricci @ beta_ricci.T + intercept[None, :]
    shifted = scores - np.max(scores, axis=1, keepdims=True)
    log_normaliser = np.log(np.exp(shifted).sum(axis=1))
    normaliser = float(np.sum(sample_weights))
    smooth = float(
        np.sum(sample_weights * (log_normaliser - shifted[np.arange(len(y_indices)), y_indices]))
        / normaliser
    )
    if not gradient:
        return smooth
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    probabilities[np.arange(len(y_indices)), y_indices] -= 1.0
    residual = probabilities * sample_weights[:, None] / normaliser
    return (
        smooth,
        residual.T @ x_h0,
        residual.T @ x_ricci,
        residual.sum(axis=0),
    )


def _restricted_binary_fista(
    x_h0: np.ndarray,
    x_ricci: np.ndarray,
    y01: np.ndarray,
    sample_weights: np.ndarray,
    beta_h0: np.ndarray,
    beta_ricci: np.ndarray,
    intercept: float,
    rho_h0: float,
    rho_ricci: float,
    *,
    tolerance: float,
    max_iter: int,
    seed: int,
    initial_step: float | None,
) -> tuple[np.ndarray, np.ndarray, float, int, float, float, float]:
    """Monotone FISTA with adaptive restart and backtracking.

    The accepted iterate is always objective-monotone.  Acceleration is reset
    when the extrapolated point points against the most recent proximal step.
    This is the O'Donoghue-Candès gradient restart criterion, evaluated using
    the incoming extrapolated point (not the newly constructed next point).
    """
    beta_h0 = np.array(beta_h0, dtype=np.float64, copy=True)
    beta_ricci = np.array(beta_ricci, dtype=np.float64, copy=True)
    intercept = float(intercept)

    y_h0 = beta_h0.copy()
    y_ricci = beta_ricci.copy()
    y_intercept = intercept
    t_value = 1.0

    lipschitz = _estimate_design_lipschitz(
        x_h0, x_ricci, sample_weights, 0.25, seed=seed, n_power_iterations=24
    )
    spectral_step = 1.0 / lipschitz
    if initial_step is not None and initial_step > 0:
        step = min(float(initial_step), 4.0 * spectral_step)
    else:
        step = spectral_step
    step = max(step, 1e-16)
    step_ceiling = max(8.0 * spectral_step, step)

    smooth_current = _binary_smooth(
        x_h0, x_ricci, y01, sample_weights,
        beta_h0, beta_ricci, intercept, gradient=False,
    )
    objective_current = smooth_current + _penalty_value(
        beta_h0, beta_ricci, rho_h0, rho_ricci
    )
    iterations = 0

    def proximal_step(
        base_h0: np.ndarray,
        base_ricci: np.ndarray,
        base_intercept: float,
        trial_step: float,
    ):
        smooth_base, grad_h, grad_r, grad_b = _binary_smooth(
            x_h0, x_ricci, y01, sample_weights,
            base_h0, base_ricci, base_intercept, gradient=True,
        )
        while True:
            candidate_h = _soft_threshold(
                base_h0 - trial_step * grad_h, trial_step * rho_h0
            )
            candidate_r = _soft_threshold(
                base_ricci - trial_step * grad_r, trial_step * rho_ricci
            )
            candidate_b = base_intercept - trial_step * grad_b
            delta_h = candidate_h - base_h0
            delta_r = candidate_r - base_ricci
            delta_b = candidate_b - base_intercept
            smooth_candidate = _binary_smooth(
                x_h0, x_ricci, y01, sample_weights,
                candidate_h, candidate_r, candidate_b, gradient=False,
            )
            quadratic = (
                smooth_base
                + float(grad_h @ delta_h + grad_r @ delta_r + grad_b * delta_b)
                + 0.5 * float(
                    delta_h @ delta_h + delta_r @ delta_r + delta_b * delta_b
                ) / trial_step
            )
            if smooth_candidate <= quadratic + 1e-12:
                objective_candidate = smooth_candidate + _penalty_value(
                    candidate_h, candidate_r, rho_h0, rho_ricci
                )
                mapping = max(
                    float(np.max(np.abs(delta_h))) if delta_h.size else 0.0,
                    float(np.max(np.abs(delta_r))) if delta_r.size else 0.0,
                    abs(float(delta_b)),
                ) / trial_step
                return (
                    candidate_h, candidate_r, float(candidate_b),
                    smooth_candidate, objective_candidate, mapping, trial_step,
                )
            trial_step *= 0.5
            if trial_step < 1e-16:
                raise RuntimeError(
                    "Binary proximal solver line search collapsed below 1e-16."
                )

    for iterations in range(1, max_iter + 1):
        trial_step = min(step * 1.08, step_ceiling)
        (
            candidate_h,
            candidate_r,
            candidate_b,
            smooth_candidate,
            objective_candidate,
            mapping,
            accepted_step,
        ) = proximal_step(y_h0, y_ricci, y_intercept, trial_step)

        # Monotone FISTA: an accelerated proposal may not replace a better
        # accepted iterate.  Restart at x_k and recompute the proximal step.
        if objective_candidate > objective_current + 1e-12:
            t_value = 1.0
            y_h0 = beta_h0.copy()
            y_ricci = beta_ricci.copy()
            y_intercept = intercept
            (
                candidate_h,
                candidate_r,
                candidate_b,
                smooth_candidate,
                objective_candidate,
                mapping,
                accepted_step,
            ) = proximal_step(
                y_h0, y_ricci, y_intercept, min(step, step_ceiling)
            )

        previous_h = beta_h0
        previous_r = beta_ricci
        previous_b = intercept
        incoming_y_h = y_h0
        incoming_y_r = y_ricci
        incoming_y_b = y_intercept

        beta_h0 = candidate_h
        beta_ricci = candidate_r
        intercept = float(candidate_b)
        smooth_current = smooth_candidate
        objective_current = objective_candidate
        step = accepted_step

        if mapping <= tolerance:
            break

        restart_dot = float(
            (incoming_y_h - beta_h0) @ (beta_h0 - previous_h)
            + (incoming_y_r - beta_ricci) @ (beta_ricci - previous_r)
            + (incoming_y_b - intercept) * (intercept - previous_b)
        )
        if restart_dot > 0.0:
            t_value = 1.0
            y_h0 = beta_h0.copy()
            y_ricci = beta_ricci.copy()
            y_intercept = intercept
        else:
            t_next = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t_value * t_value))
            momentum = (t_value - 1.0) / t_next
            y_h0 = beta_h0 + momentum * (beta_h0 - previous_h)
            y_ricci = beta_ricci + momentum * (beta_ricci - previous_r)
            y_intercept = intercept + momentum * (intercept - previous_b)
            t_value = t_next

    smooth, grad_h, grad_r, grad_b = _binary_smooth(
        x_h0, x_ricci, y01, sample_weights,
        beta_h0, beta_ricci, intercept, gradient=True,
    )
    restricted_kkt = max(
        _max_l1_kkt(beta_h0, grad_h, rho_h0),
        _max_l1_kkt(beta_ricci, grad_r, rho_ricci),
        abs(float(grad_b)),
    )
    objective = smooth + _penalty_value(beta_h0, beta_ricci, rho_h0, rho_ricci)
    return beta_h0, beta_ricci, intercept, iterations, step, restricted_kkt, objective


def _restricted_multiclass_fista(
    x_h0: np.ndarray,
    x_ricci: np.ndarray,
    y_indices: np.ndarray,
    sample_weights: np.ndarray,
    beta_h0: np.ndarray,
    beta_ricci: np.ndarray,
    intercept: np.ndarray,
    rho_h0: float,
    rho_ricci: float,
    *,
    tolerance: float,
    max_iter: int,
    seed: int,
    initial_step: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, float, float, float]:
    """Multinomial monotone FISTA with adaptive restart/backtracking."""
    beta_h0 = np.array(beta_h0, dtype=np.float64, copy=True)
    beta_ricci = np.array(beta_ricci, dtype=np.float64, copy=True)
    intercept = np.array(intercept, dtype=np.float64, copy=True)
    intercept -= np.mean(intercept)

    y_h0 = beta_h0.copy()
    y_ricci = beta_ricci.copy()
    y_intercept = intercept.copy()
    t_value = 1.0

    lipschitz = _estimate_design_lipschitz(
        x_h0, x_ricci, sample_weights, 0.5, seed=seed, n_power_iterations=24
    )
    spectral_step = 1.0 / lipschitz
    if initial_step is not None and initial_step > 0:
        step = min(float(initial_step), 4.0 * spectral_step)
    else:
        step = spectral_step
    step = max(step, 1e-16)
    step_ceiling = max(8.0 * spectral_step, step)

    smooth_current = _multiclass_smooth(
        x_h0, x_ricci, y_indices, sample_weights,
        beta_h0, beta_ricci, intercept, gradient=False,
    )
    objective_current = smooth_current + _penalty_value(
        beta_h0, beta_ricci, rho_h0, rho_ricci
    )
    iterations = 0

    def proximal_step(
        base_h0: np.ndarray,
        base_ricci: np.ndarray,
        base_intercept: np.ndarray,
        trial_step: float,
    ):
        smooth_base, grad_h, grad_r, grad_b = _multiclass_smooth(
            x_h0, x_ricci, y_indices, sample_weights,
            base_h0, base_ricci, base_intercept, gradient=True,
        )
        while True:
            candidate_h = _soft_threshold(
                base_h0 - trial_step * grad_h, trial_step * rho_h0
            )
            candidate_r = _soft_threshold(
                base_ricci - trial_step * grad_r, trial_step * rho_ricci
            )
            candidate_b = base_intercept - trial_step * grad_b
            candidate_b -= np.mean(candidate_b)
            delta_h = candidate_h - base_h0
            delta_r = candidate_r - base_ricci
            delta_b = candidate_b - base_intercept
            smooth_candidate = _multiclass_smooth(
                x_h0, x_ricci, y_indices, sample_weights,
                candidate_h, candidate_r, candidate_b, gradient=False,
            )
            quadratic = (
                smooth_base
                + float(
                    np.sum(grad_h * delta_h)
                    + np.sum(grad_r * delta_r)
                    + grad_b @ delta_b
                )
                + 0.5 * float(
                    np.sum(delta_h * delta_h)
                    + np.sum(delta_r * delta_r)
                    + delta_b @ delta_b
                ) / trial_step
            )
            if smooth_candidate <= quadratic + 1e-12:
                objective_candidate = smooth_candidate + _penalty_value(
                    candidate_h, candidate_r, rho_h0, rho_ricci
                )
                mapping = max(
                    float(np.max(np.abs(delta_h))) if delta_h.size else 0.0,
                    float(np.max(np.abs(delta_r))) if delta_r.size else 0.0,
                    float(np.max(np.abs(delta_b))) if delta_b.size else 0.0,
                ) / trial_step
                return (
                    candidate_h, candidate_r, candidate_b,
                    smooth_candidate, objective_candidate, mapping, trial_step,
                )
            trial_step *= 0.5
            if trial_step < 1e-16:
                raise RuntimeError(
                    "Multinomial proximal solver line search collapsed below 1e-16."
                )

    for iterations in range(1, max_iter + 1):
        trial_step = min(step * 1.08, step_ceiling)
        (
            candidate_h,
            candidate_r,
            candidate_b,
            smooth_candidate,
            objective_candidate,
            mapping,
            accepted_step,
        ) = proximal_step(y_h0, y_ricci, y_intercept, trial_step)

        if objective_candidate > objective_current + 1e-12:
            t_value = 1.0
            y_h0 = beta_h0.copy()
            y_ricci = beta_ricci.copy()
            y_intercept = intercept.copy()
            (
                candidate_h,
                candidate_r,
                candidate_b,
                smooth_candidate,
                objective_candidate,
                mapping,
                accepted_step,
            ) = proximal_step(
                y_h0, y_ricci, y_intercept, min(step, step_ceiling)
            )

        previous_h = beta_h0
        previous_r = beta_ricci
        previous_b = intercept
        incoming_y_h = y_h0
        incoming_y_r = y_ricci
        incoming_y_b = y_intercept

        beta_h0 = candidate_h
        beta_ricci = candidate_r
        intercept = candidate_b
        smooth_current = smooth_candidate
        objective_current = objective_candidate
        step = accepted_step

        if mapping <= tolerance:
            break

        restart_dot = float(
            np.sum((incoming_y_h - beta_h0) * (beta_h0 - previous_h))
            + np.sum((incoming_y_r - beta_ricci) * (beta_ricci - previous_r))
            + (incoming_y_b - intercept) @ (intercept - previous_b)
        )
        if restart_dot > 0.0:
            t_value = 1.0
            y_h0 = beta_h0.copy()
            y_ricci = beta_ricci.copy()
            y_intercept = intercept.copy()
        else:
            t_next = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t_value * t_value))
            momentum = (t_value - 1.0) / t_next
            y_h0 = beta_h0 + momentum * (beta_h0 - previous_h)
            y_ricci = beta_ricci + momentum * (beta_ricci - previous_r)
            y_intercept = intercept + momentum * (intercept - previous_b)
            y_intercept -= np.mean(y_intercept)
            t_value = t_next

    smooth, grad_h, grad_r, grad_b = _multiclass_smooth(
        x_h0, x_ricci, y_indices, sample_weights,
        beta_h0, beta_ricci, intercept, gradient=True,
    )
    restricted_kkt = max(
        _max_l1_kkt(beta_h0, grad_h, rho_h0),
        _max_l1_kkt(beta_ricci, grad_r, rho_ricci),
        float(np.max(np.abs(grad_b))),
    )
    objective = smooth + _penalty_value(beta_h0, beta_ricci, rho_h0, rho_ricci)
    return beta_h0, beta_ricci, intercept, iterations, step, restricted_kkt, objective


def _elementwise_l1_violations(
    beta: np.ndarray,
    gradient: np.ndarray,
    rho: float,
    zero_tol: float = 1e-12,
) -> np.ndarray:
    nonzero = np.abs(beta) > zero_tol
    violations = np.empty_like(beta, dtype=np.float64)
    violations[nonzero] = np.abs(
        gradient[nonzero] + rho * np.sign(beta[nonzero])
    )
    violations[~nonzero] = np.maximum(np.abs(gradient[~nonzero]) - rho, 0.0)
    return violations


def _orthant_polish_binary(
    x_h0: np.ndarray,
    x_ricci: np.ndarray,
    y01: np.ndarray,
    sample_weights: np.ndarray,
    beta_h0: np.ndarray,
    beta_ricci: np.ndarray,
    intercept: float,
    rho_h0: float,
    rho_ricci: float,
    *,
    tolerance: float,
    max_outer: int = 12,
    max_iter: int = 2500,
) -> tuple[np.ndarray, np.ndarray, float, int, float, float]:
    """Polish a binary L1 solution on adaptively identified orthants.

    Once FISTA has identified a nearly stable sparse support, the L1 objective
    is smooth inside each orthant.  We optimize coefficient magnitudes with
    L-BFGS-B bounds z >= 0, then re-check the exact nonsmooth KKT conditions.
    Variables that should enter or reverse sign are incorporated on the next
    orthant round.  No approximate fit is accepted.
    """
    beta_h0 = np.array(beta_h0, dtype=np.float64, copy=True)
    beta_ricci = np.array(beta_ricci, dtype=np.float64, copy=True)
    intercept = float(intercept)
    total_iterations = 0
    objective = math.inf
    kkt = math.inf
    support_tol = 1e-10
    entry_threshold = max(0.5 * tolerance, 1e-10)

    for _ in range(max_outer):
        smooth, grad_h, grad_r, grad_b = _binary_smooth(
            x_h0, x_ricci, y01, sample_weights,
            beta_h0, beta_ricci, intercept, gradient=True,
        )
        violations_h = _elementwise_l1_violations(beta_h0, grad_h, rho_h0)
        violations_r = _elementwise_l1_violations(beta_ricci, grad_r, rho_ricci)
        kkt = max(
            float(np.max(violations_h)) if violations_h.size else 0.0,
            float(np.max(violations_r)) if violations_r.size else 0.0,
            abs(float(grad_b)),
        )
        objective = smooth + _penalty_value(
            beta_h0, beta_ricci, rho_h0, rho_ricci
        )
        if kkt <= tolerance:
            break

        support_h = (np.abs(beta_h0) > support_tol) | (
            violations_h > entry_threshold
        )
        support_r = (np.abs(beta_ricci) > support_tol) | (
            violations_r > entry_threshold
        )
        idx_h = np.flatnonzero(support_h)
        idx_r = np.flatnonzero(support_r)

        sign_h = np.sign(beta_h0[idx_h])
        zero_h = np.abs(sign_h) < 0.5
        sign_h[zero_h] = -np.sign(grad_h[idx_h][zero_h])
        sign_h[np.abs(sign_h) < 0.5] = 1.0

        sign_r = np.sign(beta_ricci[idx_r])
        zero_r = np.abs(sign_r) < 0.5
        sign_r[zero_r] = -np.sign(grad_r[idx_r][zero_r])
        sign_r[np.abs(sign_r) < 0.5] = 1.0

        n_h = len(idx_h)
        n_r = len(idx_r)
        initial = np.concatenate([
            np.abs(beta_h0[idx_h]),
            np.abs(beta_ricci[idx_r]),
            np.asarray([intercept], dtype=np.float64),
        ])
        bounds = [(0.0, None)] * (n_h + n_r) + [(None, None)]

        def objective_gradient(values: np.ndarray):
            candidate_h = np.zeros_like(beta_h0)
            candidate_r = np.zeros_like(beta_ricci)
            candidate_h[idx_h] = sign_h * values[:n_h]
            candidate_r[idx_r] = sign_r * values[n_h:n_h + n_r]
            candidate_b = float(values[-1])
            candidate_smooth, candidate_grad_h, candidate_grad_r, candidate_grad_b = _binary_smooth(
                x_h0, x_ricci, y01, sample_weights,
                candidate_h, candidate_r, candidate_b, gradient=True,
            )
            value = (
                candidate_smooth
                + rho_h0 * float(np.sum(values[:n_h]))
                + rho_ricci * float(np.sum(values[n_h:n_h + n_r]))
            )
            gradient = np.empty_like(values)
            gradient[:n_h] = sign_h * candidate_grad_h[idx_h] + rho_h0
            gradient[n_h:n_h + n_r] = (
                sign_r * candidate_grad_r[idx_r] + rho_ricci
            )
            gradient[-1] = candidate_grad_b
            return float(value), gradient

        result = minimize(
            objective_gradient,
            initial,
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            options={
                "maxiter": max_iter,
                "maxls": 50,
                "maxcor": 30,
                "ftol": 1e-15,
                "gtol": max(0.05 * tolerance, 1e-10),
            },
        )
        total_iterations += int(getattr(result, "nit", 0))
        values = np.asarray(result.x, dtype=np.float64)
        beta_h0.fill(0.0)
        beta_ricci.fill(0.0)
        beta_h0[idx_h] = sign_h * values[:n_h]
        beta_ricci[idx_r] = sign_r * values[n_h:n_h + n_r]
        intercept = float(values[-1])
        beta_h0[np.abs(beta_h0) <= support_tol] = 0.0
        beta_ricci[np.abs(beta_ricci) <= support_tol] = 0.0

    smooth, grad_h, grad_r, grad_b = _binary_smooth(
        x_h0, x_ricci, y01, sample_weights,
        beta_h0, beta_ricci, intercept, gradient=True,
    )
    kkt = max(
        _max_l1_kkt(beta_h0, grad_h, rho_h0),
        _max_l1_kkt(beta_ricci, grad_r, rho_ricci),
        abs(float(grad_b)),
    )
    objective = smooth + _penalty_value(beta_h0, beta_ricci, rho_h0, rho_ricci)
    return beta_h0, beta_ricci, intercept, total_iterations, kkt, objective


def _orthant_polish_multiclass(
    x_h0: np.ndarray,
    x_ricci: np.ndarray,
    y_indices: np.ndarray,
    sample_weights: np.ndarray,
    beta_h0: np.ndarray,
    beta_ricci: np.ndarray,
    intercept: np.ndarray,
    rho_h0: float,
    rho_ricci: float,
    *,
    tolerance: float,
    max_outer: int = 12,
    max_iter: int = 2500,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, float, float]:
    """Orthant-constrained L-BFGS-B polish for multinomial L1 heads."""
    beta_h0 = np.array(beta_h0, dtype=np.float64, copy=True)
    beta_ricci = np.array(beta_ricci, dtype=np.float64, copy=True)
    intercept = np.array(intercept, dtype=np.float64, copy=True)
    intercept -= np.mean(intercept)
    total_iterations = 0
    objective = math.inf
    kkt = math.inf
    support_tol = 1e-10
    entry_threshold = max(0.5 * tolerance, 1e-10)
    n_classes = beta_h0.shape[0]

    for _ in range(max_outer):
        smooth, grad_h, grad_r, grad_b = _multiclass_smooth(
            x_h0, x_ricci, y_indices, sample_weights,
            beta_h0, beta_ricci, intercept, gradient=True,
        )
        violations_h = _elementwise_l1_violations(beta_h0, grad_h, rho_h0)
        violations_r = _elementwise_l1_violations(beta_ricci, grad_r, rho_ricci)
        kkt = max(
            float(np.max(violations_h)) if violations_h.size else 0.0,
            float(np.max(violations_r)) if violations_r.size else 0.0,
            float(np.max(np.abs(grad_b))),
        )
        objective = smooth + _penalty_value(
            beta_h0, beta_ricci, rho_h0, rho_ricci
        )
        if kkt <= tolerance:
            break

        support_h = (np.abs(beta_h0) > support_tol) | (
            violations_h > entry_threshold
        )
        support_r = (np.abs(beta_ricci) > support_tol) | (
            violations_r > entry_threshold
        )
        rows_h, cols_h = np.nonzero(support_h)
        rows_r, cols_r = np.nonzero(support_r)

        sign_h = np.sign(beta_h0[rows_h, cols_h])
        zero_h = np.abs(sign_h) < 0.5
        sign_h[zero_h] = -np.sign(grad_h[rows_h, cols_h][zero_h])
        sign_h[np.abs(sign_h) < 0.5] = 1.0

        sign_r = np.sign(beta_ricci[rows_r, cols_r])
        zero_r = np.abs(sign_r) < 0.5
        sign_r[zero_r] = -np.sign(grad_r[rows_r, cols_r][zero_r])
        sign_r[np.abs(sign_r) < 0.5] = 1.0

        n_h = len(rows_h)
        n_r = len(rows_r)
        # The last class intercept is minus the sum of the first K-1, fixing
        # the softmax common-shift gauge without changing probabilities.
        intercept_free = intercept[:-1].copy()
        initial = np.concatenate([
            np.abs(beta_h0[rows_h, cols_h]),
            np.abs(beta_ricci[rows_r, cols_r]),
            intercept_free,
        ])
        bounds = (
            [(0.0, None)] * (n_h + n_r)
            + [(None, None)] * (n_classes - 1)
        )

        def objective_gradient(values: np.ndarray):
            candidate_h = np.zeros_like(beta_h0)
            candidate_r = np.zeros_like(beta_ricci)
            candidate_h[rows_h, cols_h] = sign_h * values[:n_h]
            candidate_r[rows_r, cols_r] = sign_r * values[n_h:n_h + n_r]
            free_b = values[n_h + n_r:]
            candidate_b = np.empty(n_classes, dtype=np.float64)
            candidate_b[:-1] = free_b
            candidate_b[-1] = -float(np.sum(free_b))
            candidate_smooth, candidate_grad_h, candidate_grad_r, candidate_grad_b = _multiclass_smooth(
                x_h0, x_ricci, y_indices, sample_weights,
                candidate_h, candidate_r, candidate_b, gradient=True,
            )
            value = (
                candidate_smooth
                + rho_h0 * float(np.sum(values[:n_h]))
                + rho_ricci * float(np.sum(values[n_h:n_h + n_r]))
            )
            gradient = np.empty_like(values)
            gradient[:n_h] = (
                sign_h * candidate_grad_h[rows_h, cols_h] + rho_h0
            )
            gradient[n_h:n_h + n_r] = (
                sign_r * candidate_grad_r[rows_r, cols_r] + rho_ricci
            )
            gradient[n_h + n_r:] = (
                candidate_grad_b[:-1] - candidate_grad_b[-1]
            )
            return float(value), gradient

        result = minimize(
            objective_gradient,
            initial,
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            options={
                "maxiter": max_iter,
                "maxls": 50,
                "maxcor": 30,
                "ftol": 1e-15,
                "gtol": max(0.05 * tolerance, 1e-10),
            },
        )
        total_iterations += int(getattr(result, "nit", 0))
        values = np.asarray(result.x, dtype=np.float64)
        beta_h0.fill(0.0)
        beta_ricci.fill(0.0)
        beta_h0[rows_h, cols_h] = sign_h * values[:n_h]
        beta_ricci[rows_r, cols_r] = sign_r * values[n_h:n_h + n_r]
        free_b = values[n_h + n_r:]
        intercept[:-1] = free_b
        intercept[-1] = -float(np.sum(free_b))
        beta_h0[np.abs(beta_h0) <= support_tol] = 0.0
        beta_ricci[np.abs(beta_ricci) <= support_tol] = 0.0

    smooth, grad_h, grad_r, grad_b = _multiclass_smooth(
        x_h0, x_ricci, y_indices, sample_weights,
        beta_h0, beta_ricci, intercept, gradient=True,
    )
    kkt = max(
        _max_l1_kkt(beta_h0, grad_h, rho_h0),
        _max_l1_kkt(beta_ricci, grad_r, rho_ricci),
        float(np.max(np.abs(grad_b))),
    )
    objective = smooth + _penalty_value(beta_h0, beta_ricci, rho_h0, rho_ricci)
    return beta_h0, beta_ricci, intercept, total_iterations, kkt, objective


def _binary_full_gradient(
    x_h0: np.ndarray,
    x_ricci: np.ndarray,
    y01: np.ndarray,
    sample_weights: np.ndarray,
    beta_h0: np.ndarray,
    beta_ricci: np.ndarray,
    intercept: float,
):
    return _binary_smooth(
        x_h0, x_ricci, y01, sample_weights, beta_h0, beta_ricci, intercept, gradient=True
    )


def _multiclass_full_gradient(
    x_h0: np.ndarray,
    x_ricci: np.ndarray,
    y_indices: np.ndarray,
    sample_weights: np.ndarray,
    beta_h0: np.ndarray,
    beta_ricci: np.ndarray,
    intercept: np.ndarray,
):
    return _multiclass_smooth(
        x_h0,
        x_ricci,
        y_indices,
        sample_weights,
        beta_h0,
        beta_ricci,
        intercept,
        gradient=True,
    )


def solve_block_l1_logistic(
    x_h0: np.ndarray,
    x_ricci: np.ndarray,
    y_indices: np.ndarray,
    sample_weights: np.ndarray,
    beta_h0_start: np.ndarray,
    beta_ricci_start: np.ndarray,
    intercept_start: np.ndarray,
    *,
    lambda_h0: float,
    lambda_ricci: float,
    c_value: float,
    tolerance: float,
    max_iter: int,
    initial_active_ricci: int,
    active_batch_size: int,
    max_active_rounds: int,
    seed: int,
    initial_step: float | None = None,
    progress: ProgressFunction | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, SolverDiagnostics]:
    """Solve the exact block-L1 problem with KKT-safe working sets.

    Phase 1 uses monotone accelerated proximal iterations to identify a sparse
    working set.  Phase 2 polishes the identified orthants with bound-constrained
    L-BFGS-B.  Every candidate is then checked against the full Ricci matrix;
    omitted violating columns are added and re-polished.  Thus the second-order
    phase changes only the numerical route, never the objective or acceptance
    criterion.
    """
    started = time.perf_counter()
    x_h0 = np.ascontiguousarray(x_h0, dtype=np.float64)
    x_ricci = np.asarray(x_ricci, dtype=np.float64, order="C")
    y_indices = np.asarray(y_indices, dtype=int)
    sample_weights = np.asarray(sample_weights, dtype=np.float64)
    beta_h0_start = np.asarray(beta_h0_start, dtype=np.float64)
    beta_ricci_start = np.asarray(beta_ricci_start, dtype=np.float64)
    intercept_start = np.asarray(intercept_start, dtype=np.float64)

    if c_value <= 0:
        raise ValueError("C must be positive.")
    if tolerance <= 0:
        raise ValueError("The KKT tolerance must be positive.")
    if max_iter < 1:
        raise ValueError("max_iter must be at least one.")

    weight_sum = float(np.sum(sample_weights))
    rho_h0 = float(lambda_h0) / (float(c_value) * weight_sum)
    rho_ricci = float(lambda_ricci) / (float(c_value) * weight_sum)
    n_classes = beta_h0_start.shape[0]
    n_ricci = x_ricci.shape[1]

    if n_classes == 2:
        beta_h0_internal = beta_h0_start[1] - beta_h0_start[0]
        beta_ricci_internal = beta_ricci_start[1] - beta_ricci_start[0]
        intercept_internal: float | np.ndarray = float(
            intercept_start[1] - intercept_start[0]
        )
        y_internal = (y_indices == 1).astype(np.float64)
        smooth, grad_h, grad_r, grad_b = _binary_full_gradient(
            x_h0, x_ricci, y_internal, sample_weights,
            beta_h0_internal, beta_ricci_internal, float(intercept_internal),
        )
    else:
        beta_h0_internal = np.array(beta_h0_start, copy=True)
        beta_ricci_internal = np.array(beta_ricci_start, copy=True)
        intercept_internal = np.array(intercept_start, copy=True)
        intercept_internal -= np.mean(intercept_internal)
        smooth, grad_h, grad_r, grad_b = _multiclass_full_gradient(
            x_h0, x_ricci, y_indices, sample_weights,
            beta_h0_internal, beta_ricci_internal,
            np.asarray(intercept_internal),
        )

    column_violations = _ricci_column_violations(
        beta_ricci_internal, grad_r, rho_ricci
    )
    if beta_ricci_internal.ndim == 1:
        active = set(
            np.flatnonzero(np.abs(beta_ricci_internal) > 1e-12).tolist()
        )
    else:
        active = set(
            np.flatnonzero(
                np.any(np.abs(beta_ricci_internal) > 1e-12, axis=0)
            ).tolist()
        )
    if initial_active_ricci > 0 and n_ricci:
        top = np.argsort(column_violations)[::-1][
            : min(initial_active_ricci, n_ricci)
        ]
        active.update(
            int(index) for index in top if column_violations[index] > 0
        )

    total_iterations = 0
    polish_calls = 0
    polish_iterations = 0
    step = float(initial_step) if initial_step and initial_step > 0 else None
    max_kkt = math.inf
    objective = float(
        smooth + _penalty_value(
            beta_h0_internal, beta_ricci_internal, rho_h0, rho_ricci
        )
    )
    converged = False
    fista_rounds = 0

    def full_gradient_and_kkt():
        nonlocal smooth, grad_h, grad_r, grad_b, max_kkt, objective, column_violations
        if n_classes == 2:
            smooth, grad_h, grad_r, grad_b = _binary_full_gradient(
                x_h0, x_ricci, y_internal, sample_weights,
                beta_h0_internal, beta_ricci_internal,
                float(intercept_internal),
            )
            kkt_h = _max_l1_kkt(beta_h0_internal, grad_h, rho_h0)
            kkt_r = _max_l1_kkt(beta_ricci_internal, grad_r, rho_ricci)
            kkt_b = abs(float(grad_b))
        else:
            smooth, grad_h, grad_r, grad_b = _multiclass_full_gradient(
                x_h0, x_ricci, y_indices, sample_weights,
                beta_h0_internal, beta_ricci_internal,
                np.asarray(intercept_internal),
            )
            kkt_h = _max_l1_kkt(beta_h0_internal, grad_h, rho_h0)
            kkt_r = _max_l1_kkt(beta_ricci_internal, grad_r, rho_ricci)
            kkt_b = float(np.max(np.abs(grad_b)))
        max_kkt = max(kkt_h, kkt_r, kkt_b)
        objective = float(
            smooth + _penalty_value(
                beta_h0_internal, beta_ricci_internal, rho_h0, rho_ricci
            )
        )
        column_violations = _ricci_column_violations(
            beta_ricci_internal, grad_r, rho_ricci
        )
        return max_kkt

    def omitted_violators() -> np.ndarray:
        excluded = np.ones(n_ricci, dtype=bool)
        if active:
            excluded[np.fromiter(active, dtype=int)] = False
        return np.flatnonzero(excluded & (column_violations > tolerance))

    # Phase 1: fast support identification.  As soon as no omitted Ricci
    # variable violates KKT, do not waste thousands of first-order iterations
    # polishing a fixed ill-conditioned support; hand it to the orthant solver.
    for fista_rounds in range(1, max_active_rounds + 1):
        active_indices = np.asarray(sorted(active), dtype=int)
        x_active = (
            x_ricci[:, active_indices]
            if active_indices.size
            else x_ricci[:, :0]
        )
        remaining = max_iter - total_iterations
        if remaining <= 0:
            break
        round_budget = min(750, remaining)

        if n_classes == 2:
            active_beta = beta_ricci_internal[active_indices]
            (
                beta_h0_internal,
                active_beta,
                intercept_internal,
                used_iterations,
                step,
                _,
                _,
            ) = _restricted_binary_fista(
                x_h0, x_active, y_internal, sample_weights,
                beta_h0_internal, active_beta, float(intercept_internal),
                rho_h0, rho_ricci,
                tolerance=max(tolerance * 0.25, 1e-8),
                max_iter=round_budget,
                seed=seed + fista_rounds,
                initial_step=step,
            )
            beta_ricci_internal[:] = 0.0
            beta_ricci_internal[active_indices] = active_beta
        else:
            active_beta = beta_ricci_internal[:, active_indices]
            (
                beta_h0_internal,
                active_beta,
                intercept_internal,
                used_iterations,
                step,
                _,
                _,
            ) = _restricted_multiclass_fista(
                x_h0, x_active, y_indices, sample_weights,
                beta_h0_internal, active_beta,
                np.asarray(intercept_internal), rho_h0, rho_ricci,
                tolerance=max(tolerance * 0.25, 1e-8),
                max_iter=round_budget,
                seed=seed + fista_rounds,
                initial_step=step,
            )
            beta_ricci_internal[:] = 0.0
            beta_ricci_internal[:, active_indices] = active_beta

        total_iterations += used_iterations
        full_gradient_and_kkt()
        if progress is not None:
            progress(
                f"active_round={fista_rounds} active_ricci={len(active)} "
                f"iterations={total_iterations} objective={objective:.8f} "
                f"max_kkt={max_kkt:.3e}"
            )
        if max_kkt <= tolerance:
            converged = True
            break

        candidates = omitted_violators()
        if candidates.size == 0:
            break
        order = candidates[
            np.argsort(column_violations[candidates])[::-1]
        ]
        active.update(
            int(index) for index in order[: min(active_batch_size, len(order))]
        )

    # Phase 2: exact active-orthant polishing and full KKT certification.
    # This phase has its own iteration allowance because its iterations are
    # quasi-Newton steps, not comparable to proximal-gradient iterations.
    if not converged:
        for polish_round in range(1, max_active_rounds + 1):
            active_indices = np.asarray(sorted(active), dtype=int)
            x_active = (
                x_ricci[:, active_indices]
                if active_indices.size
                else x_ricci[:, :0]
            )
            polish_calls += 1

            if n_classes == 2:
                active_beta = beta_ricci_internal[active_indices]
                (
                    beta_h0_internal,
                    active_beta,
                    intercept_internal,
                    used_polish,
                    _,
                    _,
                ) = _orthant_polish_binary(
                    x_h0, x_active, y_internal, sample_weights,
                    beta_h0_internal, active_beta,
                    float(intercept_internal), rho_h0, rho_ricci,
                    tolerance=max(tolerance * 0.5, 1e-8),
                )
                beta_ricci_internal[:] = 0.0
                beta_ricci_internal[active_indices] = active_beta
            else:
                active_beta = beta_ricci_internal[:, active_indices]
                (
                    beta_h0_internal,
                    active_beta,
                    intercept_internal,
                    used_polish,
                    _,
                    _,
                ) = _orthant_polish_multiclass(
                    x_h0, x_active, y_indices, sample_weights,
                    beta_h0_internal, active_beta,
                    np.asarray(intercept_internal), rho_h0, rho_ricci,
                    tolerance=max(tolerance * 0.5, 1e-8),
                )
                beta_ricci_internal[:] = 0.0
                beta_ricci_internal[:, active_indices] = active_beta

            polish_iterations += used_polish
            full_gradient_and_kkt()
            if progress is not None:
                progress(
                    f"polish_round={polish_round} active_ricci={len(active)} "
                    f"polish_iterations={polish_iterations} "
                    f"objective={objective:.8f} max_kkt={max_kkt:.3e}"
                )
            if max_kkt <= tolerance:
                converged = True
                break

            candidates = omitted_violators()
            if candidates.size == 0:
                # The complete working set has been polished but the exact KKT
                # certificate is still not met.  Another identical call cannot
                # legitimately rescue the fit, so fail clearly below.
                break
            order = candidates[
                np.argsort(column_violations[candidates])[::-1]
            ]
            active.update(
                int(index) for index in order[: min(active_batch_size, len(order))]
            )

    active_rounds = fista_rounds + polish_calls
    if n_classes == 2:
        beta_h0 = np.vstack([
            -0.5 * beta_h0_internal,
            0.5 * beta_h0_internal,
        ])
        beta_ricci = np.vstack([
            -0.5 * beta_ricci_internal,
            0.5 * beta_ricci_internal,
        ])
        intercept = np.asarray([
            -0.5 * float(intercept_internal),
            0.5 * float(intercept_internal),
        ])
    else:
        beta_h0 = np.asarray(beta_h0_internal, dtype=np.float64)
        beta_ricci = np.asarray(beta_ricci_internal, dtype=np.float64)
        intercept = np.asarray(intercept_internal, dtype=np.float64)

    diagnostics = SolverDiagnostics(
        iterations=int(total_iterations),
        active_set_rounds=int(active_rounds),
        active_ricci_features=int(len(active)),
        max_kkt_violation=float(max_kkt),
        objective=float(objective),
        smooth_loss=float(smooth),
        penalty=float(objective - smooth),
        converged=bool(converged),
        step_size=float(step or 1.0),
        elapsed_seconds=float(time.perf_counter() - started),
        polish_calls=int(polish_calls),
        polish_iterations=int(polish_iterations),
    )
    if not converged:
        raise RuntimeError(
            "Block-L1 solver did not satisfy the full KKT tolerance after "
            "monotone FISTA and orthant polishing: "
            f"max_kkt={max_kkt:.3e}, tolerance={tolerance:.3e}, "
            f"fista_iterations={total_iterations}, "
            f"polish_iterations={polish_iterations}, "
            f"active_ricci={len(active)}. Do not use this fit."
        )
    return beta_h0, beta_ricci, intercept, diagnostics

def _zero_warm_start(n_classes: int, n_h0: int, n_ricci: int) -> ModelWarmStart:
    return ModelWarmStart(
        beta_h0=np.zeros((n_classes, n_h0), dtype=np.float64),
        beta_ricci=np.zeros((n_classes, n_ricci), dtype=np.float64),
        intercept=np.zeros(n_classes, dtype=np.float64),
        alpha_logits=np.zeros(n_h0, dtype=np.float64),
        step_size=1.0,
        completed_alternations=0,
    )


def _validate_warm_start(
    state: ModelWarmStart | None,
    n_classes: int,
    n_h0: int,
    n_ricci: int,
) -> ModelWarmStart:
    if state is None:
        return _zero_warm_start(n_classes, n_h0, n_ricci)
    expected = ((n_classes, n_h0), (n_classes, n_ricci), (n_classes,), (n_h0,))
    actual = (
        state.beta_h0.shape,
        state.beta_ricci.shape,
        state.intercept.shape,
        state.alpha_logits.shape,
    )
    if actual != expected:
        raise ValueError(f"Warm-start shapes {actual} do not match expected shapes {expected}.")
    return state.copy()


def fit_joint_sparse_model_prepared(
    h0: PreparedH0Split,
    ricci: PreparedRicciSplit,
    y_train: np.ndarray,
    classes: tuple[str, ...],
    hyperparameters: JointHyperparameters,
    *,
    c_value: float = 0.02,
    logistic_max_iter: int = 8000,
    logistic_tolerance: float = 1e-4,
    n_alternations: int = 6,
    alpha_smoothness_gamma: float = 0.01,
    alpha_optimizer_max_iter: int = 100,
    seed: int = 13,
    initial_active_ricci: int = 512,
    active_batch_size: int = 512,
    max_active_rounds: int = 50,
    warm_start: ModelWarmStart | None = None,
    progress: ProgressFunction | None = None,
    state_callback: StateCallback | None = None,
) -> JointSparseModel:
    if hyperparameters.lambda_h0 <= 0 or hyperparameters.lambda_ricci <= 0:
        raise ValueError("Both block penalties must be positive.")
    y_train = np.asarray(y_train, dtype=object)
    class_to_index = {label: index for index, label in enumerate(classes)}
    y_indices = np.asarray([class_to_index[str(label)] for label in y_train], dtype=int)
    sample_weights = _balanced_sample_weights(y_train, classes)
    state = _validate_warm_start(
        warm_start,
        n_classes=len(classes),
        n_h0=h0.train.shape[1],
        n_ricci=ricci.train.shape[1],
    )
    history: list[AlternationRecord] = []
    last_diagnostics: SolverDiagnostics | None = None
    start_alternation = min(max(state.completed_alternations, 0), n_alternations)

    for alternation in range(start_alternation + 1, n_alternations + 1):
        alternation_started = time.perf_counter()
        alpha = alpha_from_logits(state.alpha_logits)
        x_h0 = np.ascontiguousarray(h0.train * alpha[None, :], dtype=np.float64)
        prefix = f"alternation={alternation}/{n_alternations}"
        beta_h0, beta_ricci, intercept, diagnostics = solve_block_l1_logistic(
            x_h0=x_h0,
            x_ricci=ricci.train,
            y_indices=y_indices,
            sample_weights=sample_weights,
            beta_h0_start=state.beta_h0,
            beta_ricci_start=state.beta_ricci,
            intercept_start=state.intercept,
            lambda_h0=hyperparameters.lambda_h0,
            lambda_ricci=hyperparameters.lambda_ricci,
            c_value=c_value,
            tolerance=logistic_tolerance,
            max_iter=logistic_max_iter,
            initial_active_ricci=initial_active_ricci,
            active_batch_size=active_batch_size,
            max_active_rounds=max_active_rounds,
            seed=seed + 1000 * alternation,
            initial_step=state.step_size,
            progress=(
                (lambda message, p=prefix: progress(f"{p} {message}"))
                if progress is not None
                else None
            ),
        )
        ricci_scores = ricci.train @ beta_ricci.T + intercept[None, :]
        before, _ = _alpha_loss_and_gradient(
            state.alpha_logits,
            h0.train,
            ricci_scores,
            beta_h0,
            y_indices,
            sample_weights,
            alpha_smoothness_gamma,
        )
        result = minimize(
            fun=lambda values: _alpha_loss_and_gradient(
                values,
                h0.train,
                ricci_scores,
                beta_h0,
                y_indices,
                sample_weights,
                alpha_smoothness_gamma,
            ),
            x0=state.alpha_logits,
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": alpha_optimizer_max_iter, "ftol": 1e-10, "gtol": 1e-7},
        )
        state = ModelWarmStart(
            beta_h0=beta_h0,
            beta_ricci=beta_ricci,
            intercept=intercept,
            alpha_logits=np.asarray(result.x, dtype=np.float64),
            step_size=diagnostics.step_size,
            completed_alternations=alternation,
        )
        history.append(
            AlternationRecord(
                alternation=alternation,
                alpha_objective_before=float(before),
                alpha_objective_after=float(result.fun),
                alpha_optimizer_success=bool(result.success),
                alpha_optimizer_iterations=int(getattr(result, "nit", 0)),
                logistic_iterations=diagnostics.iterations,
                active_set_rounds=diagnostics.active_set_rounds,
                active_ricci_features=diagnostics.active_ricci_features,
                max_kkt_violation=diagnostics.max_kkt_violation,
                logistic_objective=diagnostics.objective,
                nonzero_h0=int(np.count_nonzero(np.abs(beta_h0) > 1e-12)),
                nonzero_ricci=int(np.count_nonzero(np.abs(beta_ricci) > 1e-12)),
                elapsed_seconds=float(time.perf_counter() - alternation_started),
                polish_calls=diagnostics.polish_calls,
                polish_iterations=diagnostics.polish_iterations,
            )
        )
        last_diagnostics = diagnostics
        if state_callback is not None:
            state_callback(state.copy())
        if progress is not None:
            progress(
                f"{prefix} complete objective={diagnostics.objective:.8f} "
                f"max_kkt={diagnostics.max_kkt_violation:.3e} "
                f"active_ricci={diagnostics.active_ricci_features} "
                f"alpha_objective={float(result.fun):.8f}"
            )

    # The saved head must correspond to the final alpha profile. This solve is
    # warm-started from the preceding alternation rather than restarted from zero.
    alpha = alpha_from_logits(state.alpha_logits)
    final_x_h0 = np.ascontiguousarray(h0.train * alpha[None, :], dtype=np.float64)
    beta_h0, beta_ricci, intercept, final_diagnostics = solve_block_l1_logistic(
        x_h0=final_x_h0,
        x_ricci=ricci.train,
        y_indices=y_indices,
        sample_weights=sample_weights,
        beta_h0_start=state.beta_h0,
        beta_ricci_start=state.beta_ricci,
        intercept_start=state.intercept,
        lambda_h0=hyperparameters.lambda_h0,
        lambda_ricci=hyperparameters.lambda_ricci,
        c_value=c_value,
        tolerance=logistic_tolerance,
        max_iter=logistic_max_iter,
        initial_active_ricci=initial_active_ricci,
        active_batch_size=active_batch_size,
        max_active_rounds=max_active_rounds,
        seed=seed + 999_999,
        initial_step=state.step_size,
        progress=(
            (lambda message: progress(f"final_refit {message}"))
            if progress is not None
            else None
        ),
    )
    state = ModelWarmStart(
        beta_h0=beta_h0,
        beta_ricci=beta_ricci,
        intercept=intercept,
        alpha_logits=state.alpha_logits,
        step_size=final_diagnostics.step_size,
        completed_alternations=n_alternations,
    )
    if state_callback is not None:
        state_callback(state.copy())
    last_diagnostics = final_diagnostics
    return JointSparseModel(
        classes=classes,
        hyperparameters=hyperparameters,
        alpha=alpha,
        h0_standardizer=h0.standardizer,
        ricci_scaler=ricci.scaler,
        effective_beta_h0=beta_h0,
        effective_beta_ricci=beta_ricci,
        effective_intercept=intercept,
        alpha_logits=np.array(state.alpha_logits, copy=True),
        solver_diagnostics=last_diagnostics,
        alternation_history=history,
    )


def fit_joint_sparse_model(
    phi_train: np.ndarray,
    ricci_train: np.ndarray | sparse.spmatrix,
    y_train: np.ndarray,
    classes: tuple[str, ...],
    hyperparameters: JointHyperparameters,
    *,
    c_value: float = 0.02,
    logistic_max_iter: int = 8000,
    logistic_tolerance: float = 1e-4,
    n_alternations: int = 6,
    alpha_smoothness_gamma: float = 0.01,
    alpha_optimizer_max_iter: int = 100,
    seed: int = 13,
    n_jobs: int = 1,
    initial_active_ricci: int = 512,
    active_batch_size: int = 512,
    max_active_rounds: int = 50,
    warm_start: ModelWarmStart | None = None,
    progress: ProgressFunction | None = None,
    state_callback: StateCallback | None = None,
) -> JointSparseModel:
    del n_jobs  # Parallelism is across independent M paths, not inside one fit.
    h0 = prepare_h0_split(phi_train)
    ricci = prepare_ricci_split(ricci_train)
    return fit_joint_sparse_model_prepared(
        h0=h0,
        ricci=ricci,
        y_train=y_train,
        classes=classes,
        hyperparameters=hyperparameters,
        c_value=c_value,
        logistic_max_iter=logistic_max_iter,
        logistic_tolerance=logistic_tolerance,
        n_alternations=n_alternations,
        alpha_smoothness_gamma=alpha_smoothness_gamma,
        alpha_optimizer_max_iter=alpha_optimizer_max_iter,
        seed=seed,
        initial_active_ricci=initial_active_ricci,
        active_batch_size=active_batch_size,
        max_active_rounds=max_active_rounds,
        warm_start=warm_start,
        progress=progress,
        state_callback=state_callback,
    )


def save_warm_start(path: Path, state: ModelWarmStart, threshold: float = 1e-14) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    active = np.flatnonzero(np.any(np.abs(state.beta_ricci) > threshold, axis=0))
    with tempfile.NamedTemporaryFile(
        prefix=path.stem + ".", suffix=".npz", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(
            temporary,
            beta_h0=state.beta_h0,
            ricci_indices=active.astype(np.int64),
            beta_ricci_active=state.beta_ricci[:, active],
            n_ricci=np.asarray([state.beta_ricci.shape[1]], dtype=np.int64),
            intercept=state.intercept,
            alpha_logits=state.alpha_logits,
            step_size=np.asarray([state.step_size], dtype=np.float64),
            completed_alternations=np.asarray([state.completed_alternations], dtype=np.int64),
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_warm_start(path: Path, n_classes: int, n_h0: int, n_ricci: int) -> ModelWarmStart:
    with np.load(path, allow_pickle=False) as payload:
        saved_n_ricci = int(payload["n_ricci"][0])
        if saved_n_ricci != n_ricci:
            raise ValueError(
                f"Checkpoint has {saved_n_ricci} Ricci features, expected {n_ricci}: {path}"
            )
        beta_h0 = np.asarray(payload["beta_h0"], dtype=np.float64)
        beta_ricci = np.zeros((n_classes, n_ricci), dtype=np.float64)
        indices = np.asarray(payload["ricci_indices"], dtype=int)
        beta_ricci[:, indices] = np.asarray(payload["beta_ricci_active"], dtype=np.float64)
        state = ModelWarmStart(
            beta_h0=beta_h0,
            beta_ricci=beta_ricci,
            intercept=np.asarray(payload["intercept"], dtype=np.float64),
            alpha_logits=np.asarray(payload["alpha_logits"], dtype=np.float64),
            step_size=float(payload["step_size"][0]),
            completed_alternations=int(payload["completed_alternations"][0]),
        )
    return _validate_warm_start(state, n_classes, n_h0, n_ricci)


def model_sparsity(model: JointSparseModel, threshold: float = 1e-12) -> dict[str, Any]:
    h0_nonzero = int(np.count_nonzero(np.abs(model.effective_beta_h0) > threshold))
    ricci_nonzero = int(np.count_nonzero(np.abs(model.effective_beta_ricci) > threshold))
    return {
        "h0_nonzero_coefficients": h0_nonzero,
        "ricci_nonzero_coefficients": ricci_nonzero,
        "h0_total_coefficients": int(model.effective_beta_h0.size),
        "ricci_total_coefficients": int(model.effective_beta_ricci.size),
        "h0_nonzero_fraction": h0_nonzero / max(model.effective_beta_h0.size, 1),
        "ricci_nonzero_fraction": ricci_nonzero / max(model.effective_beta_ricci.size, 1),
        "alpha_min": float(np.min(model.alpha)),
        "alpha_max": float(np.max(model.alpha)),
        "alpha_mean": float(np.mean(model.alpha)),
        "alpha_roughness": float(np.mean(np.diff(model.alpha) ** 2)) if len(model.alpha) > 1 else 0.0,
        "solver_iterations": int(model.solver_diagnostics.iterations),
        "solver_active_set_rounds": int(model.solver_diagnostics.active_set_rounds),
        "solver_active_ricci_features": int(model.solver_diagnostics.active_ricci_features),
        "solver_max_kkt_violation": float(model.solver_diagnostics.max_kkt_violation),
        "solver_objective": float(model.solver_diagnostics.objective),
        "solver_converged": bool(model.solver_diagnostics.converged),
        "solver_polish_calls": int(model.solver_diagnostics.polish_calls),
        "solver_polish_iterations": int(model.solver_diagnostics.polish_iterations),
    }
