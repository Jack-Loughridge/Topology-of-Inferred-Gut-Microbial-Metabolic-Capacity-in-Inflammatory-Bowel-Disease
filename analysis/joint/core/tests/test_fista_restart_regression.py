import numpy as np

from h0_ricci_joint_sparse.model import (
    _balanced_sample_weights,
    solve_block_l1_logistic,
)


def test_multiclass_acceleration_is_not_restarted_every_iteration():
    """Regression test for the former always-positive restart dot product."""
    rng = np.random.default_rng(2026)
    n_samples = 180
    n_classes = 3

    latent = rng.normal(size=(n_samples, 8))
    x_h0 = latent @ rng.normal(size=(8, 10)) + 0.05 * rng.normal(size=(n_samples, 10))
    x_ricci = latent @ rng.normal(size=(8, 80)) + 0.05 * rng.normal(size=(n_samples, 80))
    class_scores = latent @ rng.normal(size=(8, n_classes))
    y_indices = np.argmax(
        class_scores + 0.3 * rng.normal(size=class_scores.shape), axis=1
    )

    classes = tuple(f"C{index}" for index in range(n_classes))
    labels = np.asarray([classes[index] for index in y_indices], dtype=object)
    weights = _balanced_sample_weights(labels, classes)

    _, _, _, diagnostics = solve_block_l1_logistic(
        x_h0=x_h0,
        x_ricci=x_ricci,
        y_indices=y_indices,
        sample_weights=weights,
        beta_h0_start=np.zeros((n_classes, x_h0.shape[1])),
        beta_ricci_start=np.zeros((n_classes, x_ricci.shape[1])),
        intercept_start=np.zeros(n_classes),
        lambda_h0=1.0,
        lambda_ricci=2.0,
        c_value=0.05,
        tolerance=1e-6,
        max_iter=1000,
        initial_active_ricci=16,
        active_batch_size=16,
        max_active_rounds=20,
        seed=13,
    )

    assert diagnostics.converged
    assert diagnostics.max_kkt_violation <= 1e-6
    # The fixed implementation converges in roughly 200 iterations on this
    # deterministic ill-conditioned problem. The old always-restart code did
    # not satisfy this bound.
    assert diagnostics.iterations < 500
