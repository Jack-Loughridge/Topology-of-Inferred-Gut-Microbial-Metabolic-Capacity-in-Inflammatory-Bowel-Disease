import pytest

from install_core_patch import (
    PATCH_MARKER,
    SUPPORTED_UNPATCHED_MODELS,
    patched_text,
    source_profile,
    synthetic_patcher_test,
)


def test_synthetic_patcher_regression() -> None:
    synthetic_patcher_test()


def test_patch_is_single_idempotent_and_resolution_normalised() -> None:
    source = '''from __future__ import annotations\n\ndef alpha_from_logits(x):\n    return x\n\ndef _alpha_loss_and_gradient(\n    logits, phi_standardised, ricci_scores, beta_h0, y_indices, sample_weights, smoothness_gamma\n):\n    alpha = alpha_from_logits(logits)\n    h0_scores = (phi_standardised * alpha[None, :]) @ beta_h0.T\n    return smoothness_gamma, h0_scores\n'''
    result = patched_text(source)
    assert PATCH_MARKER in result
    assert result.count("resolution_normalised_alpha_smoothness_gamma") >= 2
    assert result.count("smoothness_gamma = resolution_normalised_alpha_smoothness_gamma") == 1
    assert patched_text(result) == result
    compile(result, "<patched>", "exec")


def test_exact_audited_source_allowlist() -> None:
    current_hash = "b0eff25de9bf1400d5d1e3a5d3576ba1233ffdeda3420596cf24de65ed8e8e8e"
    assert source_profile(current_hash) == "core-v1.2.0-orthant-polish"
    assert len(SUPPORTED_UNPATCHED_MODELS) == 2
    with pytest.raises(RuntimeError, match="Refusing a blind patch"):
        source_profile("0" * 64)
