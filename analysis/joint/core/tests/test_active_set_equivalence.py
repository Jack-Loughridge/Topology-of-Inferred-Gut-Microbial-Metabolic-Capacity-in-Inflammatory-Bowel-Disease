import numpy as np
from sklearn.linear_model import LogisticRegression

from h0_ricci_joint_sparse.model import (
    _balanced_sample_weights,
    _softmax,
    solve_block_l1_logistic,
)


def _sklearn_probabilities(x_h0, x_ricci, labels, classes, lambda_h0, lambda_ricci, c_value):
    design = np.hstack([x_h0 / lambda_h0, x_ricci / lambda_ricci])
    kwargs = dict(
        C=c_value,
        solver="saga",
        class_weight="balanced",
        fit_intercept=True,
        max_iter=50_000,
        tol=1e-9,
        random_state=13,
    )
    # Supported by both the VM's sklearn 1.7 and newer versions.
    kwargs["penalty"] = "l1"
    estimator = LogisticRegression(**kwargs).fit(design, labels)
    raw = estimator.predict_proba(design)
    output = np.zeros((len(labels), len(classes)))
    for source_index, label in enumerate(estimator.classes_):
        output[:, classes.index(str(label))] = raw[:, source_index]
    return output


def _run_case(n_classes):
    rng = np.random.default_rng(13 + n_classes)
    n_samples = 72
    x_h0 = rng.normal(size=(n_samples, 6))
    x_ricci = rng.normal(size=(n_samples, 18))
    y_indices = np.tile(np.arange(n_classes), n_samples // n_classes + 1)[:n_samples]
    rng.shuffle(y_indices)
    classes = tuple(f"C{index}" for index in range(n_classes))
    labels = np.asarray([classes[index] for index in y_indices], dtype=object)
    weights = _balanced_sample_weights(labels, classes)
    lambda_h0 = 1.0
    lambda_ricci = 2.0
    c_value = 0.2

    beta_h0, beta_ricci, intercept, diagnostics = solve_block_l1_logistic(
        x_h0=x_h0,
        x_ricci=x_ricci,
        y_indices=y_indices,
        sample_weights=weights,
        beta_h0_start=np.zeros((n_classes, x_h0.shape[1])),
        beta_ricci_start=np.zeros((n_classes, x_ricci.shape[1])),
        intercept_start=np.zeros(n_classes),
        lambda_h0=lambda_h0,
        lambda_ricci=lambda_ricci,
        c_value=c_value,
        tolerance=1e-7,
        max_iter=5000,
        initial_active_ricci=4,
        active_batch_size=4,
        max_active_rounds=30,
        seed=13,
    )
    active_probabilities = _softmax(
        x_h0 @ beta_h0.T + x_ricci @ beta_ricci.T + intercept
    )
    saga_probabilities = _sklearn_probabilities(
        x_h0, x_ricci, labels, classes, lambda_h0, lambda_ricci, c_value
    )
    assert diagnostics.converged
    assert diagnostics.max_kkt_violation <= 1e-7
    assert np.allclose(active_probabilities, saga_probabilities, atol=3e-5, rtol=3e-5)


def test_binary_active_set_matches_saga():
    _run_case(2)


def test_multinomial_active_set_matches_saga():
    _run_case(3)
