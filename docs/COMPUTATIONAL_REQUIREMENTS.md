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

These versions document the audited VM state; they are not yet the final portable dependency lock. Component `requirements.txt` and `pyproject.toml` files record their direct requirements.

## Environment separation

The final release will define at least two environments:

1. **Modern analysis environment:** graph construction, H0, Ricci, joint models, species benchmarks, interpretation, and portable tests.
2. **Legacy external preprocessing environment:** the historical MetaPhlAn2 v2.6/HUMAnN-era tools needed to reproduce the external profiles.

The legacy tools must not be forced into the modern classifier environment. Their databases, command versions, and database identifiers must be recorded separately because the reference database is part of the computation.

## Runtime classes

| Class | Intended work | Release expectation |
|---|---|---|
| Lightweight | Unit tests, synthetic self-tests, configuration and manifest validation | Suitable for GitHub Actions or a laptop. |
| Moderate | Result aggregation, plotting, vectorisation from completed features | Expected to run on a workstation. |
| Heavy CPU | Graph construction, full Ricci computation, repeated benchmark fitting | Run on a multicore VM; resumable checkpoints required. |
| Heavy model fitting | Repeated H0 Alpha-Pi and nested joint H0-Ricci analyses | Long-running VM workload; GPU may accelerate PyTorch stages but is not assumed by the source freeze. |
| Storage intensive | Raw FASTQ processing and complete generated graph/feature collections | External storage required; not suitable for GitHub. |

Exact peak memory, disk usage, CPU model, core count, and wall-clock timings remain to be recorded from the completed production runs before release.

## Thread control

Many launchers deliberately constrain numerical libraries while multiple analyses share the VM:

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
```

Thread counts are operational settings unless a locked run configuration states otherwise. Scientific seeds, split manifests, preprocessing rules, model grids, and feature parameters must remain unchanged when resuming.

## Determinism and portability

- Every production run records explicit random seeds or consumes locked split manifests.
- Participant boundaries and output membership are verified independently of aggregate metrics.
- The joint solver performs convergence and KKT checks rather than silently accepting incomplete fits.
- Floating-point results may vary slightly across BLAS, PyTorch, operating-system, and processor versions; the release tests should use tolerances while exact membership, shapes, configurations, and hashes remain strict.
- Full numerical reproduction should use the final locked environment or container once supplied.

## Release deliverables still required

The `environment/` directory will receive:

```text
environment.yml                 Modern analysis environment
requirements-lock.txt           Fully resolved Python package snapshot
metaphlan2_v260_environment.yml  Legacy external profiling environment
README.md                       Installation and database acquisition notes
```

A clean environment must execute the root CI test suite and at least one end-to-end synthetic workflow before `v1.0.0` is tagged.
