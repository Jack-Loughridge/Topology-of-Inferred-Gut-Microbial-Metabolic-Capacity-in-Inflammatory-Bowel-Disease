from pathlib import Path

import numpy as np

from h0_ricci_joint_sparse.model import ModelWarmStart, load_warm_start, save_warm_start


def test_sparse_warm_start_checkpoint_roundtrip(tmp_path: Path):
    beta_ricci = np.zeros((3, 20))
    beta_ricci[:, [2, 9, 17]] = np.arange(9).reshape(3, 3)
    state = ModelWarmStart(
        beta_h0=np.arange(12, dtype=float).reshape(3, 4),
        beta_ricci=beta_ricci,
        intercept=np.array([-0.2, 0.0, 0.2]),
        alpha_logits=np.array([0.1, -0.2, 0.3, 0.4]),
        step_size=0.7,
        completed_alternations=3,
    )
    path = tmp_path / "state.npz"
    save_warm_start(path, state)
    loaded = load_warm_start(path, 3, 4, 20)
    assert np.allclose(loaded.beta_h0, state.beta_h0)
    assert np.allclose(loaded.beta_ricci, state.beta_ricci)
    assert np.allclose(loaded.intercept, state.intercept)
    assert np.allclose(loaded.alpha_logits, state.alpha_logits)
    assert loaded.step_size == state.step_size
    assert loaded.completed_alternations == state.completed_alternations
