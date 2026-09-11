# Clean modern-environment validation — 3 September 2026

## Scope

A fresh clone of the publication branch was reconstructed in a new Python
virtual environment. No human microbiome data or production outputs were used;
the scientific checks used only repository unit fixtures and generated
synthetic inputs.

## Tested source

| Field | Value |
|---|---|
| Repository branch | `publication/rebuild-2026` |
| Tested commit | `06e9b9db738552253e9645d8be9e023d4d96f578` |
| Clone state | synchronized with the corresponding remote branch |
| Audit created | `2026-09-03T11:42:23Z` |
| Audit archive SHA-256 | `1ad0a6ef4abf1edcb9a96f4d4ebfa23be02065911dd2a954a4fa39e8b4dc6abf` |

## Platform and installation

| Field | Value |
|---|---|
| Operating system | Ubuntu 22.04 family, Linux kernel 6.8.0-1052-azure |
| Architecture | x86-64 |
| Python | 3.10.12 |
| pip | 26.2.1 |
| Numerical thread limit | 1 |
| Process niceness | 15 |

Installation order:

1. create a fresh Python 3.10 virtual environment;
2. upgrade `pip`, `setuptools`, and `wheel`;
3. install CPU-only PyTorch from `https://download.pytorch.org/whl/cpu`;
4. install `environment/requirements-ci.txt`;
5. run `pip check` and the repository validators;
6. export the complete resolved package version set.

The resulting 46-package lock is
`environment/locks/requirements-linux-x86_64-py310.lock`, with SHA-256
`6938bab7cffdea4e0e9cd0cbaaf083319001064cf22c4994a82d0e35dcc40bf5`.

## Results

| Check | Result |
|---|---|
| Package dependency consistency | PASS — no broken requirements |
| `environment/validate_analysis_environment.py` | PASS |
| Production source manifest | PASS — 191 files |
| Python syntax | PASS — 110 files |
| Bash syntax | PASS — 23 files |
| Scientific unit tests | PASS — 46 tests |
| Data-free synthetic component self-tests | PASS — all six |
| Root `python run_checks.py full` | PASS |

## Interpretation

This establishes that the publication code and portable test workflows can be
installed and executed from a clean clone using the recorded resolved package
versions. It does not establish bitwise reproduction of historical production
outputs across different hardware or library builds. Final numerical replay
also requires the released derived inputs, locked splits and configurations,
and result checksums.
