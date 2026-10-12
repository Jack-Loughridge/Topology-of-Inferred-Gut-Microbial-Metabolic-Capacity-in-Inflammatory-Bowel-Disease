import numpy as np

from h0_ricci_joint_sparse.model import (
    _alpha_loss_and_gradient,
    resolution_normalised_alpha_smoothness_gamma,
)


def test_resolution_normalised_alpha_smoothness_gamma():
    gamma = 0.01
    assert np.isclose(resolution_normalised_alpha_smoothness_gamma(gamma, 160), gamma)
    assert np.isclose(
        resolution_normalised_alpha_smoothness_gamma(gamma, 64),
        gamma * 63.0 / 159.0,
    )
    assert np.isclose(
        resolution_normalised_alpha_smoothness_gamma(gamma, 96),
        gamma * 95.0 / 159.0,
    )
    assert resolution_normalised_alpha_smoothness_gamma(gamma, 64) < gamma


def test_resolution_normalisation_rejects_invalid_arguments():
    import pytest
    with pytest.raises(ValueError):
        resolution_normalised_alpha_smoothness_gamma(-0.1, 160)
    with pytest.raises(ValueError):
        resolution_normalised_alpha_smoothness_gamma(0.01, 1)
    with pytest.raises(ValueError):
        resolution_normalised_alpha_smoothness_gamma(0.01, 160, reference_intervals=1)


def _roughness_component(n_intervals: int) -> float:
    x = np.linspace(0.0, 1.0, n_intervals)
    alpha = 1.0 + 0.25 * np.sin(2.0 * np.pi * x) + 0.10 * np.cos(4.0 * np.pi * x)
    logits = np.log(alpha)
    phi = np.zeros((6, n_intervals), dtype=float)
    ricci_scores = np.zeros((6, 2), dtype=float)
    beta_h0 = np.zeros((2, n_intervals), dtype=float)
    y_indices = np.asarray([0, 1, 0, 1, 0, 1], dtype=int)
    sample_weights = np.ones(6, dtype=float)
    loss, gradient = _alpha_loss_and_gradient(
        logits, phi, ricci_scores, beta_h0, y_indices, sample_weights, 0.01
    )
    assert np.isfinite(gradient).all()
    return float(loss - np.log(2.0))


def test_alpha_loss_uses_resolution_normalisation_functionally():
    roughness = np.asarray([_roughness_component(m) for m in (64, 96, 160)])
    # Discretisation leaves a small residual difference, but the functional
    # roughness should be nearly invariant rather than scaling as 1/(M-1).
    assert np.ptp(roughness) / np.mean(roughness) < 0.01
