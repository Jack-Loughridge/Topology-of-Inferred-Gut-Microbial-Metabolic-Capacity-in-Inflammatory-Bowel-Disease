# Computational requirements

## Audited production platform

The source freeze was collected on 1 September 2026 from an Azure x86-64 Linux VM with:

```text
Operating system: Ubuntu 22.04 family
Kernel:           Linux 6.8.0-1052-azure
Python:           3.10.12
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

These versions document the audited VM state; they are not the final portable dependency lock. Component `requirements.txt` and `pyproject.toml` files record their direct requirements. Consolidated specifications and provenance are under `environment/`.

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

Exact peak memory, disk usage, CPU model, core count, and wall-clock timings remain to be recorded from the completed production runs before release.

## Portable checks

The root verification entry point provides cumulative static, unit, and full synthetic levels:

```bash
python run_checks.py static
python run_checks.py unit
python run_checks.py full
```

Static and unit checks run automatically through GitHub Actions. The longer full synthetic suite is manually triggerable. None of these checks requires human microbiome data.

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
- Full numerical reproduction should use the final locked environment or container once supplied.

## Release deliverables still required

The repository now contains direct modern and external environment specifications, an observed production-version record, and automated portable checks. Before release it still requires:

- a fully resolved Linux dependency lock generated and verified from a clean environment;
- completed database and executable checksum manifests for both external preprocessing paths;
- clean-run test logs and representative analysis replay evidence;
- final CPU, memory, storage, and wall-clock measurements for the heavy production stages.

A clean environment must execute the root unit suite and the full synthetic workflow before `v1.0.0` is tagged.
