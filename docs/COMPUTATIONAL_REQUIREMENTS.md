# Computational requirements

## Audited production platform

The source freeze was collected on 1 September 2026 from an Azure x86-64 Linux VM with:

```text
Operating system: Ubuntu 22.04 family
Kernel:           Linux 6.8.0-1052-azure
Python:           3.10.12
Processor:        Intel Xeon Platinum 8272CL at 2.60 GHz
Logical CPUs:     4
Memory:           15 GiB RAM and 23 GiB swap
Filesystem:       248 GiB root volume
```

The full recorded environment is retained in the Phase 0 audit archive. The principal observed Python packages were:

| Package | Observed version |
|---|---:|
| NumPy | 2.2.6 |
| pandas | 2.3.3 |
| SciPy | 1.15.3 |
| scikit-learn | 1.7.2 |
| PyTorch | 2.9.1+cpu |
| NetworkX | 3.4.2 |
| XGBoost | 3.2.0 |
| PyArrow | 22.0.0 |
| Matplotlib | 3.10.8 |
| openpyxl | 3.1.5 |
| fastparquet | 2024.11.0 |
| tqdm | 4.67.1 |

These versions document the audited VM state. Component `requirements.txt` and
`pyproject.toml` files record their direct requirements. The clean-tested,
resolved Linux x86-64/Python 3.10 version lock is
`environment/locks/requirements-linux-x86_64-py310.lock`; consolidated
specifications and provenance are under `environment/`.

## Environment separation

The repository maintains three distinct environment records:

1. **Modern analysis environment:** graph construction, H0, Ricci, joint models, species benchmarks, interpretation, and portable tests.
2. **Current external-profiling environment:** KneadData 0.12.4, HUMAnN 3.9, MetaPhlAn 4.0.6, and their sequence-search dependencies.
3. **Historical MetaPhlAn2 environment:** the separate MetaPhlAn 2.6.0 and `mpa_v20_m200` path used to reconstruct the frozen-support external species representation.

These environments must not be conflated. The current HUMAnN workflow explicitly uses the `mpa_vJun23_CHOCOPhlAnSGB_202307` MetaPhlAn index, whereas the historical workflow uses `mpa_v20_m200`. Reference databases are part of the computation and require independent version and checksum records.

The audited current external environment contains a recorded anomaly: the DIAMOND executable reports version 2.0.15 while its Conda metadata reports package version 2.2.2. The executable SHA-256 is recorded under `environment/provenance/`; clean reconstruction must resolve or reproduce this mismatch before raw-data equivalence is claimed.

## Runtime classes

| Class | Intended work | Release expectation |
|---|---|---|
| Lightweight | Integrity checks, unit tests, synthetic self-tests, configuration and manifest validation | Suitable for GitHub Actions or a laptop. |
| Moderate | Result aggregation, plotting, vectorisation from completed features | Expected to run on a workstation. |
| Heavy CPU | Graph construction, full Ricci computation, repeated benchmark fitting | Run on a multicore VM; resumable checkpoints required. |
| Heavy model fitting | Repeated H0 Alpha-Pi and nested joint H0-Ricci analyses | Long-running VM workload; GPU may accelerate PyTorch stages but is not assumed by the source freeze. |
| Storage intensive | Raw FASTQ processing and complete generated graph/feature collections | External storage required; not suitable for GitHub. |

The Phase 6A audit recorded 216 GiB used and 33 GiB available on the 248 GiB
root volume while the final Ricci runs were active. CPU model, logical CPU
count, memory, swap, and filesystem capacity are therefore fixed provenance.
Per-stage peak memory and exact wall-clock timing were not instrumented in the
historical production runs and must not be inferred retrospectively. Runtime
classes and completion-marker timestamps provide the available operational
record.

## Portable checks

The root verification entry point provides cumulative static, unit, and full synthetic levels:

```bash
python run_checks.py static
python run_checks.py unit
python run_checks.py full
```

Static and unit checks run automatically through GitHub Actions. The longer full synthetic suite is manually triggerable. None of these checks requires human microbiome data.

## Clean-environment reconstruction

On 3 September 2026, a fresh clone of commit
`06e9b9db738552253e9645d8be9e023d4d96f578` was installed in an isolated
Python 3.10.12 virtual environment on Linux x86-64. `pip check` reported no
broken requirements, the environment validator passed, 46 scientific unit
tests passed, and all six synthetic component self-tests passed. The resolved
46-package lock has SHA-256
`6938bab7cffdea4e0e9cd0cbaaf083319001064cf22c4994a82d0e35dcc40bf5`.

The lock records exact package versions but not wheel hashes. It is therefore a
platform-specific resolved version lock, not a guarantee of byte-identical
package artifacts. The reconstruction recipe and scope are in
`environment/provenance/clean_environment_validation_20260903.md`.

## Thread control

Many launchers deliberately constrain numerical libraries while multiple analyses share the VM:

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
```

The root test runner applies the same constraints. Thread counts are operational settings unless a locked run configuration states otherwise. Scientific seeds, split manifests, preprocessing rules, model grids, and feature parameters must remain unchanged when resuming.

## Determinism and portability

- Every production run records explicit random seeds or consumes locked split manifests.
- Participant boundaries and output membership are verified independently of aggregate metrics.
- The joint solver performs convergence and KKT checks rather than silently accepting incomplete fits.
- Floating-point results may vary slightly across BLAS, PyTorch, operating-system, and processor versions; release tests use tolerances while exact membership, shapes, configurations, and hashes remain strict.
- Portable verification should use the clean-tested lock on Linux x86-64 with
  Python 3.10. Historical-result comparison must also retain the recorded input,
  split, configuration, and output hashes.

## Release deliverables still required

The repository now contains direct modern and external environment
specifications, an observed production-version record, a clean-tested resolved
modern-environment lock, and automated portable checks. Before release it still
requires:

- representative analysis replay evidence using the released derived-data archive;
- final all-results manifest, release tag, and archive DOI.
