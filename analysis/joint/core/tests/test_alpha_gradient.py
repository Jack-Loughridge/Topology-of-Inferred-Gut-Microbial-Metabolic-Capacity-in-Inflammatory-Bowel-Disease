import numpy as np

from h0_ricci_joint_sparse.model import _alpha_loss_and_gradient


def test_alpha_analytic_gradient_matches_finite_difference():
    rng = np.random.default_rng(13)
    logits = rng.normal(size=5)
    phi = rng.normal(size=(7, 5))
    ricci_scores = rng.normal(size=(7, 3))
    beta_h0 = rng.normal(size=(3, 5))
    y_indices = np.array([0, 1, 2, 0, 1, 2, 0])
    weights = np.ones(7)
    _, analytic = _alpha_loss_and_gradient(
        logits, phi, ricci_scores, beta_h0, y_indices, weights, 0.03
    )
    epsilon = 1e-6
    numerical = np.empty_like(logits)
    for index in range(len(logits)):
        plus = logits.copy(); plus[index] += epsilon
        minus = logits.copy(); minus[index] -= epsilon
        plus_loss, _ = _alpha_loss_and_gradient(
            plus, phi, ricci_scores, beta_h0, y_indices, weights, 0.03
        )
        minus_loss, _ = _alpha_loss_and_gradient(
            minus, phi, ricci_scores, beta_h0, y_indices, weights, 0.03
        )
        numerical[index] = (plus_loss - minus_loss) / (2 * epsilon)
    assert np.allclose(analytic, numerical, atol=2e-5, rtol=2e-5)
