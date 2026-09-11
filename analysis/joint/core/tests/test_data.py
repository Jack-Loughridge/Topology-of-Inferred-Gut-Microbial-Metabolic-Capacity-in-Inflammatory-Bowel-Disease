import numpy as np

from h0_ricci_joint_sparse.data import extract_h0_deaths


def test_extract_h0_deaths_accepts_n_by_2_and_2_by_n():
    n_by_2 = np.array([[0.0, 0.1], [0.0, 0.4], [0.0, np.inf]])
    two_by_n = np.array([[0.0, 0.0, 0.0], [0.1, 0.4, np.inf]])
    expected = np.array([0.1, 0.4])
    assert np.allclose(extract_h0_deaths(n_by_2), expected)
    assert np.allclose(extract_h0_deaths(two_by_n), expected)
