from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class AdaptiveIntervals:
    log_edges: np.ndarray
    sigma_log: float
    requested_intervals: int

    @property
    def n_intervals(self) -> int:
        return len(self.log_edges) - 1

    @property
    def log_centers(self) -> np.ndarray:
        return 0.5 * (self.log_edges[:-1] + self.log_edges[1:])

    @property
    def death_centers(self) -> np.ndarray:
        return np.exp(self.log_centers)


@dataclass(frozen=True)
class DenseStandardizer:
    mean_: np.ndarray
    scale_: np.ndarray

    def transform(self, matrix: np.ndarray) -> np.ndarray:
        return ((matrix - self.mean_) / self.scale_).astype(np.float64, copy=False)


def fit_dense_standardizer(matrix: np.ndarray, eps: float = 1e-12) -> DenseStandardizer:
    matrix = np.asarray(matrix, dtype=np.float64)
    mean = np.mean(matrix, axis=0)
    scale = np.std(matrix, axis=0, ddof=0)
    scale = np.where(scale > eps, scale, 1.0)
    return DenseStandardizer(mean_=mean, scale_=scale)


def _strictly_increasing(values: np.ndarray) -> np.ndarray:
    out = np.asarray(values, dtype=np.float64).copy()
    for index in range(1, len(out)):
        if out[index] <= out[index - 1]:
            out[index] = np.nextafter(out[index - 1], np.inf)
    return out


def fit_train_only_adaptive_intervals(
    train_deaths: Sequence[np.ndarray],
    n_intervals: int,
    epsilon: float = 1e-12,
) -> AdaptiveIntervals:
    """Fit equal-mass log-death intervals using only the supplied training diagrams.

    The pooled training deaths define quantile boundaries.  This is the leak-safe
    adaptive construction used inside every inner and outer training split.
    """
    if n_intervals < 2:
        raise ValueError("n_intervals must be at least 2")
    pooled = np.concatenate([
        np.asarray(values, dtype=np.float64)[np.asarray(values, dtype=np.float64) > 0]
        for values in train_deaths
    ])
    pooled = pooled[np.isfinite(pooled)]
    if pooled.size < n_intervals:
        raise ValueError(
            f"Only {pooled.size} positive finite H0 deaths available for {n_intervals} intervals."
        )
    log_deaths = np.log(np.maximum(pooled, epsilon))
    quantiles = np.linspace(0.0, 1.0, n_intervals + 1)
    edges = np.quantile(log_deaths, quantiles, method="linear")
    edges = _strictly_increasing(edges)

    widths = np.diff(edges)
    positive_widths = widths[widths > np.finfo(float).eps]
    if positive_widths.size == 0:
        raise ValueError("Training H0 deaths do not span a usable log-death range.")
    sigma_log = float(np.median(positive_widths))
    sigma_log = max(sigma_log, 1e-6)
    return AdaptiveIntervals(
        log_edges=edges,
        sigma_log=sigma_log,
        requested_intervals=n_intervals,
    )


def _gaussian_mixture_density(
    evaluation_points: np.ndarray,
    log_deaths: np.ndarray,
    sigma: float,
    death_chunk: int = 1024,
) -> np.ndarray:
    evaluation_points = np.asarray(evaluation_points, dtype=np.float64)
    log_deaths = np.asarray(log_deaths, dtype=np.float64)
    normaliser = 1.0 / (np.sqrt(2.0 * np.pi) * sigma)
    density = np.zeros(evaluation_points.shape, dtype=np.float64)
    for start in range(0, len(log_deaths), death_chunk):
        chunk = log_deaths[start : start + death_chunk]
        z = (evaluation_points[:, None] - chunk[None, :]) / sigma
        density += normaliser * np.exp(-0.5 * z * z).sum(axis=1)
    return density


def transform_diagram(
    deaths: np.ndarray,
    intervals: AdaptiveIntervals,
    quad_points: int = 5,
    epsilon: float = 1e-12,
) -> np.ndarray:
    """Compute the unweighted 1-D WKPI basis with Gauss-Legendre quadrature."""
    deaths = np.asarray(deaths, dtype=np.float64)
    deaths = deaths[np.isfinite(deaths) & (deaths > 0)]
    if deaths.size == 0:
        return np.zeros(intervals.n_intervals, dtype=np.float64)
    log_deaths = np.log(np.maximum(deaths, epsilon))

    nodes, weights = np.polynomial.legendre.leggauss(quad_points)
    left = intervals.log_edges[:-1]
    right = intervals.log_edges[1:]
    mid = 0.5 * (left + right)
    half = 0.5 * (right - left)
    points = (mid[:, None] + half[:, None] * nodes[None, :]).reshape(-1)
    density = _gaussian_mixture_density(points, log_deaths, intervals.sigma_log)
    density = density.reshape(intervals.n_intervals, quad_points)
    integrals = half * np.sum(density * weights[None, :], axis=1)
    return integrals.astype(np.float64, copy=False)


def transform_many(
    diagrams: Sequence[np.ndarray],
    intervals: AdaptiveIntervals,
    quad_points: int = 5,
) -> np.ndarray:
    output = np.empty((len(diagrams), intervals.n_intervals), dtype=np.float64)
    for row, deaths in enumerate(diagrams):
        output[row] = transform_diagram(deaths, intervals, quad_points=quad_points)
    return output


def normalise_alpha(alpha: np.ndarray, floor: float = 1e-8) -> np.ndarray:
    alpha = np.asarray(alpha, dtype=np.float64)
    alpha = np.maximum(alpha, floor)
    return alpha / np.mean(alpha)


def alpha_from_logits(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - np.max(logits)
    positive = np.exp(shifted)
    return normalise_alpha(positive)


def interpolate_alpha(
    centers: np.ndarray,
    alpha: np.ndarray,
    common_grid: np.ndarray,
) -> np.ndarray:
    centers = np.asarray(centers, dtype=np.float64)
    alpha = np.asarray(alpha, dtype=np.float64)
    order = np.argsort(centers)
    return np.interp(
        common_grid,
        centers[order],
        alpha[order],
        left=alpha[order][0],
        right=alpha[order][-1],
    )
