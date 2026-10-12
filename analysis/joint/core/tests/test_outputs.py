from h0_ricci_joint_sparse.outputs import format_mean_sd, latex_escape


def test_latex_math_is_preserved_and_text_is_escaped():
    assert latex_escape("$\\lambda_H$") == "$\\lambda_H$"
    assert latex_escape(format_mean_sd(0.5, 0.1)) == "0.500 $\\pm$ 0.100"
    assert latex_escape("non_IBD & UC") == r"non\_IBD \& UC"
