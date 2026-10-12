import numpy as np
from scipy import sparse

from h0_ricci_joint_sparse.model import JointHyperparameters, fit_joint_sparse_model


def test_joint_model_smoke():
    rng = np.random.default_rng(13)
    n = 36
    phi = rng.normal(size=(n, 8))
    ricci_dense = rng.normal(size=(n, 14))
    labels = np.array(["nonIBD"] * 18 + ["IBD"] * 18, dtype=object)
    phi[18:, 2:4] += 1.0
    ricci_dense[18:, 5:8] += 0.8
    ricci = sparse.csr_matrix(ricci_dense)

    model = fit_joint_sparse_model(
        phi_train=phi,
        ricci_train=ricci,
        y_train=labels,
        classes=("nonIBD", "IBD"),
        hyperparameters=JointHyperparameters(8, 1.0, 1.0),
        c_value=0.5,
        logistic_max_iter=500,
        n_alternations=2,
        alpha_optimizer_max_iter=20,
        seed=13,
    )
    probabilities = model.predict_proba(phi, ricci)
    assert probabilities.shape == (n, 2)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert np.all(model.alpha > 0)
    assert np.isclose(model.alpha.mean(), 1.0)
