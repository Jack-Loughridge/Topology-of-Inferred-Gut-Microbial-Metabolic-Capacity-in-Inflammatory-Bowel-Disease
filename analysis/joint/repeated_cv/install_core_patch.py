#!/usr/bin/env python3
"""Install the audited resolution-normalised alpha-smoothness patch.

The validated joint solver historically applied ``gamma * sum(diff(alpha)**2)``
with the same numerical gamma at every number of H0 intervals. Because the
finite-difference sum scales approximately as ``1 / (M - 1)`` for samples of a
fixed smooth function, this made high-resolution alpha curves effectively less
regularised.

This installer changes only the alpha smoothness coefficient used inside the
existing core model:

    gamma_eff(M) = gamma * (M - 1) / (160 - 1)

The objective at M=160 is exactly unchanged. The installer recognises an exact
allow-list of audited unpatched core sources, creates a timestamped backup,
patches idempotently, checks the deterministic patched hash when available,
runs an explicit functional normalisation test, and then runs the complete core
pytest suite. Any failure triggers rollback.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Final


PATCH_MARKER: Final = "ALPHA_SMOOTHNESS_RESOLUTION_NORMALISATION_V1"
REFERENCE_INTERVALS: Final = 160
PATCH_RECORD: Final = "ALPHA_SMOOTHNESS_RESOLUTION_PATCH.json"
FORMULA: Final = "gamma_eff(M) = gamma * (M - 1) / (160 - 1)"
PATCH_INSTALLER_VERSION: Final = "2.0"

# Exact source allow-list. The first source is the original post-FISTA v1.1.1
# core. The second is the later v1.2.0 core with exact orthant-constrained
# L-BFGS-B polishing and additional solver diagnostics. The alpha-loss function
# and patch insertion point are identical in the two audited variants.
SUPPORTED_UNPATCHED_MODELS: Final[dict[str, str]] = {
    "4f83b45a04943589dd2887e2d9f41ea488a760f18ffa08438a8215301bbf8e8d": (
        "core-v1.1.1-post-FISTA"
    ),
    "b0eff25de9bf1400d5d1e3a5d3576ba1233ffdeda3420596cf24de65ed8e8e8e": (
        "core-v1.2.0-orthant-polish"
    ),
}

# Deterministic patched hash established from the uploaded v1.2.0 source.
# The v1.1.1 transformation remains protected by its exact input hash, syntax
# compilation, functional test and full core test suite.
EXPECTED_PATCHED_HASHES: Final[dict[str, str]] = {
    "b0eff25de9bf1400d5d1e3a5d3576ba1233ffdeda3420596cf24de65ed8e8e8e": (
        "fc6d0a8ec94172900c9ac25e431e5dfb601f0eddf0674c4f389c0993d6d0a230"
    ),
}

HELPER = '''\n\ndef resolution_normalised_alpha_smoothness_gamma(\n    base_gamma: float,\n    n_intervals: int,\n    reference_intervals: int = 160,\n) -> float:\n    """Return a finite-difference smoothness weight comparable across M.\n\n    The discrete roughness sum ``sum((alpha[j+1]-alpha[j])**2)`` scales\n    approximately as ``1/(M-1)`` for samples of the same underlying smooth\n    function. Multiplying gamma by ``(M-1)/(M_ref-1)`` therefore keeps the\n    functional strength of the smoothness prior approximately invariant.\n    """\n    if base_gamma < 0:\n        raise ValueError("base_gamma must be nonnegative.")\n    if n_intervals < 2:\n        raise ValueError("n_intervals must be at least 2.")\n    if reference_intervals < 2:\n        raise ValueError("reference_intervals must be at least 2.")\n    return float(base_gamma) * (int(n_intervals) - 1) / (int(reference_intervals) - 1)\n\n\n'''

TEST_TEXT = '''import numpy as np\n\nfrom h0_ricci_joint_sparse.model import (\n    _alpha_loss_and_gradient,\n    resolution_normalised_alpha_smoothness_gamma,\n)\n\n\ndef test_resolution_normalised_alpha_smoothness_gamma():\n    gamma = 0.01\n    assert np.isclose(resolution_normalised_alpha_smoothness_gamma(gamma, 160), gamma)\n    assert np.isclose(\n        resolution_normalised_alpha_smoothness_gamma(gamma, 64),\n        gamma * 63.0 / 159.0,\n    )\n    assert np.isclose(\n        resolution_normalised_alpha_smoothness_gamma(gamma, 96),\n        gamma * 95.0 / 159.0,\n    )\n    assert resolution_normalised_alpha_smoothness_gamma(gamma, 64) < gamma\n\n\ndef test_resolution_normalisation_rejects_invalid_arguments():\n    import pytest\n    with pytest.raises(ValueError):\n        resolution_normalised_alpha_smoothness_gamma(-0.1, 160)\n    with pytest.raises(ValueError):\n        resolution_normalised_alpha_smoothness_gamma(0.01, 1)\n    with pytest.raises(ValueError):\n        resolution_normalised_alpha_smoothness_gamma(0.01, 160, reference_intervals=1)\n\n\ndef _roughness_component(n_intervals: int) -> float:\n    x = np.linspace(0.0, 1.0, n_intervals)\n    alpha = 1.0 + 0.25 * np.sin(2.0 * np.pi * x) + 0.10 * np.cos(4.0 * np.pi * x)\n    logits = np.log(alpha)\n    phi = np.zeros((6, n_intervals), dtype=float)\n    ricci_scores = np.zeros((6, 2), dtype=float)\n    beta_h0 = np.zeros((2, n_intervals), dtype=float)\n    y_indices = np.asarray([0, 1, 0, 1, 0, 1], dtype=int)\n    sample_weights = np.ones(6, dtype=float)\n    loss, gradient = _alpha_loss_and_gradient(\n        logits, phi, ricci_scores, beta_h0, y_indices, sample_weights, 0.01\n    )\n    assert np.isfinite(gradient).all()\n    return float(loss - np.log(2.0))\n\n\ndef test_alpha_loss_uses_resolution_normalisation_functionally():\n    roughness = np.asarray([_roughness_component(m) for m in (64, 96, 160)])\n    # Discretisation leaves a small residual difference, but the functional\n    # roughness should be nearly invariant rather than scaling as 1/(M-1).\n    assert np.ptp(roughness) / np.mean(roughness) < 0.01\n'''


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_profile(model_hash: str) -> str:
    try:
        return SUPPORTED_UNPATCHED_MODELS[model_hash]
    except KeyError as error:
        supported = "\n".join(
            f"  {digest}  ({profile})"
            for digest, profile in SUPPORTED_UNPATCHED_MODELS.items()
        )
        raise RuntimeError(
            "model.py does not match any audited unpatched core source. "
            "Refusing a blind patch.\n"
            f"Current SHA-256: {model_hash}\n"
            f"Supported exact sources:\n{supported}"
        ) from error


def patched_text(source: str) -> str:
    if PATCH_MARKER in source:
        return source
    anchor = "def _alpha_loss_and_gradient(\n"
    if source.count(anchor) != 1:
        raise RuntimeError(f"Expected exactly one {anchor!r} anchor, found {source.count(anchor)}.")
    source = source.replace(anchor, f"# {PATCH_MARKER}\n" + HELPER + anchor, 1)
    alpha_anchor = "    alpha = alpha_from_logits(logits)\n    h0_scores = (phi_standardised * alpha[None, :]) @ beta_h0.T\n"
    replacement = (
        "    alpha = alpha_from_logits(logits)\n"
        "    smoothness_gamma = resolution_normalised_alpha_smoothness_gamma(\n"
        "        smoothness_gamma, len(alpha), reference_intervals=160\n"
        "    )\n"
        "    h0_scores = (phi_standardised * alpha[None, :]) @ beta_h0.T\n"
    )
    if source.count(alpha_anchor) != 1:
        raise RuntimeError(
            "Could not locate the unique alpha-loss insertion point. "
            f"Found {source.count(alpha_anchor)} matches."
        )
    return source.replace(alpha_anchor, replacement, 1)


def synthetic_patcher_test() -> None:
    synthetic = '''from __future__ import annotations\n\ndef alpha_from_logits(x):\n    return x\n\ndef _alpha_loss_and_gradient(\n    logits, phi_standardised, ricci_scores, beta_h0, y_indices, sample_weights, smoothness_gamma\n):\n    alpha = alpha_from_logits(logits)\n    h0_scores = (phi_standardised * alpha[None, :]) @ beta_h0.T\n    return smoothness_gamma, h0_scores\n'''
    result = patched_text(synthetic)
    assert PATCH_MARKER in result
    assert "resolution_normalised_alpha_smoothness_gamma" in result
    assert result.count("smoothness_gamma = resolution_normalised_alpha_smoothness_gamma") == 1
    assert patched_text(result) == result
    compile(result, "<synthetic_model>", "exec")


def _validate_existing_record(record: dict, current_hash: str) -> None:
    original_hash = str(record.get("original_model_sha256", ""))
    profile = source_profile(original_hash)
    expected = {
        "patch": PATCH_MARKER,
        "reference_intervals": REFERENCE_INTERVALS,
        "formula": FORMULA,
        "source_profile": profile,
    }
    mismatches = {
        key: {"observed": record.get(key), "expected": value}
        for key, value in expected.items()
        if record.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"Core patch record is incompatible: {mismatches}")
    recorded_hash = str(record.get("patched_model_sha256", ""))
    if current_hash != recorded_hash:
        raise RuntimeError(
            "Patched model.py does not match its patch record. Refusing to continue.\n"
            f"Current SHA-256:  {current_hash}\n"
            f"Recorded SHA-256: {recorded_hash}"
        )
    deterministic = EXPECTED_PATCHED_HASHES.get(original_hash)
    if deterministic is not None and current_hash != deterministic:
        raise RuntimeError(
            "Patched model.py differs from the audited deterministic output.\n"
            f"Observed: {current_hash}\nExpected: {deterministic}"
        )


def install(repo: Path) -> None:
    repo = repo.expanduser().resolve()
    model = repo / "src" / "h0_ricci_joint_sparse" / "model.py"
    test_file = repo / "tests" / "test_alpha_smoothness_resolution_normalisation.py"
    record_file = repo / PATCH_RECORD
    python = repo / ".venv" / "bin" / "python"
    for path in (model, python):
        if not path.exists():
            raise FileNotFoundError(path)
    if subprocess.run(
        ["pgrep", "-af", "[r]un_joint_sparse.py"], capture_output=True, text=True
    ).returncode == 0:
        raise RuntimeError(
            "A joint-classifier process is running. Stop it before changing the core source."
        )

    original_text = model.read_text(encoding="utf-8")
    current_hash = sha256(model)
    if PATCH_MARKER in original_text:
        print("[already patched] Resolution normalisation marker found.")
        if not record_file.exists():
            raise RuntimeError("Model is patched but the patch record is missing.")
        record = json.loads(record_file.read_text(encoding="utf-8"))
        _validate_existing_record(record, current_hash)
        subprocess.run([str(python), "-m", "pytest", "-q"], cwd=repo, check=True)
        return

    profile = source_profile(current_hash)
    new_text = patched_text(original_text)
    compile(new_text, str(model), "exec")
    expected_patched_hash = EXPECTED_PATCHED_HASHES.get(current_hash)
    if expected_patched_hash is not None:
        observed_patched_hash = hashlib.sha256(new_text.encode("utf-8")).hexdigest()
        if observed_patched_hash != expected_patched_hash:
            raise RuntimeError(
                "Deterministic patch output does not match the audited hash.\n"
                f"Observed: {observed_patched_hash}\nExpected: {expected_patched_hash}"
            )

    stamp = __import__("datetime").datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = repo / ".patch_backups" / f"alpha_smoothness_resolution_{stamp}"
    backup_model = backup / "src" / "h0_ricci_joint_sparse" / "model.py"
    backup_test = backup / "tests" / test_file.name
    backup_model.parent.mkdir(parents=True, exist_ok=True)
    backup_test.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(model, backup_model)
    if test_file.exists():
        shutil.copy2(test_file, backup_test)
    backup_record = backup / PATCH_RECORD
    if record_file.exists():
        shutil.copy2(record_file, backup_record)

    def rollback() -> None:
        shutil.copy2(backup_model, model)
        if backup_test.exists():
            shutil.copy2(backup_test, test_file)
        else:
            test_file.unlink(missing_ok=True)
        if backup_record.exists():
            shutil.copy2(backup_record, record_file)
        else:
            record_file.unlink(missing_ok=True)

    try:
        model.write_text(new_text, encoding="utf-8")
        test_file.write_text(TEST_TEXT, encoding="utf-8")
        patched_hash = sha256(model)
        if expected_patched_hash is not None and patched_hash != expected_patched_hash:
            raise RuntimeError(
                "Written patched model differs from the audited deterministic hash.\n"
                f"Observed: {patched_hash}\nExpected: {expected_patched_hash}"
            )
        record = {
            "patch": PATCH_MARKER,
            "patch_installer_version": PATCH_INSTALLER_VERSION,
            "reference_intervals": REFERENCE_INTERVALS,
            "formula": FORMULA,
            "source_profile": profile,
            "original_model_sha256": current_hash,
            "patched_model_sha256": patched_hash,
            "backup": str(backup),
        }
        record_file.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        subprocess.run(
            [
                str(python),
                "-c",
                "from h0_ricci_joint_sparse.model import "
                "resolution_normalised_alpha_smoothness_gamma as f; "
                "assert abs(f(0.01,160)-0.01)<1e-15; "
                "assert abs(f(0.01,64)-0.01*63/159)<1e-15",
            ],
            cwd=repo,
            check=True,
        )
        subprocess.run(
            [
                str(python),
                "-m",
                "pytest",
                "-q",
                str(test_file),
            ],
            cwd=repo,
            check=True,
        )
        subprocess.run([str(python), "-m", "pytest", "-q"], cwd=repo, check=True)
    except Exception:
        rollback()
        raise

    print("[SUCCESS] Resolution-normalised alpha smoothness patch installed.")
    print(f"[PROFILE] {profile}")
    print(f"[MODEL]   {model}")
    print(f"[SHA256]  {sha256(model)}")
    print(f"[BACKUP]  {backup}")
    print(f"[RECORD]  {record_file}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--joint-repo",
        type=Path,
        default=Path.home() / "Real_Data" / "h0_ricci_joint_sparse",
    )
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    synthetic_patcher_test()
    if args.self_test:
        print("PATCHER SELF-TEST PASSED")
        return
    install(args.joint_repo)


if __name__ == "__main__":
    main()
