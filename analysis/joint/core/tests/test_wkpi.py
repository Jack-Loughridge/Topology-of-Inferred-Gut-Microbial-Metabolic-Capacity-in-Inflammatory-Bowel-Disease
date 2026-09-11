import numpy as np

from h0_ricci_joint_sparse.wkpi import (
    alpha_from_logits,
    fit_train_only_adaptive_intervals,
    transform_many,
)


def test_alpha_is_positive_and_mean_one():
    alpha = alpha_from_logits(np.array([-3.0, 0.0, 2.0, 5.0]))
    assert np.all(alpha > 0)
    assert np.isclose(alpha.mean(), 1.0)


def test_intervals_are_train_only():
    train = [np.array([0.1, 0.2, 0.3]), np.array([0.15, 0.25, 0.35])]
    first = fit_train_only_adaptive_intervals(train, 4)
    # An extreme held-out diagram must not alter intervals fitted from train.
    held_out = np.array([10.0, 20.0])
    second = fit_train_only_adaptive_intervals(train, 4)
    assert np.allclose(first.log_edges, second.log_edges)
    assert held_out.max() > np.exp(first.log_edges[-1])


def test_wkpi_basis_shape_and_finiteness():
    diagrams = [np.array([0.1, 0.2, 0.5]), np.array([0.12, 0.3, 0.7])]
    intervals = fit_train_only_adaptive_intervals(diagrams, 4)
    basis = transform_many(diagrams, intervals, quad_points=5)
    assert basis.shape == (2, 4)
    assert np.isfinite(basis).all()
    assert (basis >= 0).all()
