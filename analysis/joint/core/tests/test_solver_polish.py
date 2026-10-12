import numpy as np

from h0_ricci_joint_sparse.model import solve_block_l1_logistic


def _correlated_design(seed: int, n: int, n_h0: int, n_ricci: int, latent_dim: int):
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=(n, latent_dim))
    h0 = latent @ (rng.normal(size=(latent_dim, n_h0)) / np.sqrt(latent_dim))
    ricci = latent @ (rng.normal(size=(latent_dim, n_ricci)) / np.sqrt(latent_dim))
    h0 += 0.04 * rng.normal(size=h0.shape)
    ricci += 0.04 * rng.normal(size=ricci.shape)
    h0 /= np.maximum(h0.std(axis=0), 1e-8)
    ricci /= np.maximum(ricci.std(axis=0), 1e-8)
    return rng, latent, h0, ricci


def test_binary_orthant_polish_certifies_when_fista_budget_is_tiny():
    rng, latent, h0, ricci = _correlated_design(31, 240, 48, 520, 12)
    score = 1.8 * latent[:, 0] - 1.3 * latent[:, 1] + 0.4 * rng.normal(size=240)
    y = (score > np.median(score)).astype(int)

    _, _, _, diagnostics = solve_block_l1_logistic(
        x_h0=h0,
        x_ricci=ricci,
        y_indices=y,
        sample_weights=np.ones(len(y)),
        beta_h0_start=np.zeros((2, h0.shape[1])),
        beta_ricci_start=np.zeros((2, ricci.shape[1])),
        intercept_start=np.zeros(2),
        lambda_h0=0.5,
        lambda_ricci=0.5,
        c_value=0.02,
        tolerance=1e-4,
        max_iter=8,
        initial_active_ricci=96,
        active_batch_size=96,
        max_active_rounds=20,
        seed=13,
    )

    assert diagnostics.converged
    assert diagnostics.max_kkt_violation <= 1e-4
    assert diagnostics.polish_calls >= 1
    assert diagnostics.polish_iterations >= 1


def test_multiclass_orthant_polish_certifies_when_fista_budget_is_tiny():
    rng, latent, h0, ricci = _correlated_design(32, 270, 42, 420, 12)
    class_scores = np.column_stack([
        1.4 * latent[:, 0] - 0.5 * latent[:, 1],
        1.2 * latent[:, 1] - 0.4 * latent[:, 2],
        1.1 * latent[:, 2] - 0.5 * latent[:, 0],
    ])
    class_scores += 0.25 * rng.normal(size=class_scores.shape)
    y = np.argmax(class_scores, axis=1)

    _, _, _, diagnostics = solve_block_l1_logistic(
        x_h0=h0,
        x_ricci=ricci,
        y_indices=y,
        sample_weights=np.ones(len(y)),
        beta_h0_start=np.zeros((3, h0.shape[1])),
        beta_ricci_start=np.zeros((3, ricci.shape[1])),
        intercept_start=np.zeros(3),
        lambda_h0=0.5,
        lambda_ricci=0.5,
        c_value=0.02,
        tolerance=1e-4,
        max_iter=12,
        initial_active_ricci=96,
        active_batch_size=96,
        max_active_rounds=20,
        seed=13,
    )

    assert diagnostics.converged
    assert diagnostics.max_kkt_violation <= 1e-4
    assert diagnostics.polish_calls >= 1
    assert diagnostics.polish_iterations >= 1
