import numpy as np


def test_block_scaling_gives_separate_l1_penalties():
    rng = np.random.default_rng(13)
    beta_h = rng.normal(size=5)
    beta_r = rng.normal(size=7)
    lambda_h = 0.5
    lambda_r = 2.0

    theta_h = lambda_h * beta_h
    theta_r = lambda_r * beta_r
    scaled_penalty = np.abs(np.concatenate([theta_h, theta_r])).sum()
    separate_penalty = lambda_h * np.abs(beta_h).sum() + lambda_r * np.abs(beta_r).sum()
    assert np.isclose(scaled_penalty, separate_penalty)

    x_h = rng.normal(size=(4, 5))
    x_r = rng.normal(size=(4, 7))
    original_scores = x_h @ beta_h + x_r @ beta_r
    scaled_scores = (x_h / lambda_h) @ theta_h + (x_r / lambda_r) @ theta_r
    assert np.allclose(original_scores, scaled_scores)
