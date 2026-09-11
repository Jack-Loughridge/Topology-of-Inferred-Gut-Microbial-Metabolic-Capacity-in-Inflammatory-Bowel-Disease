# Reproducible software environments

This directory separates the software used for statistical/topological analysis
from the software used to preprocess the external sequencing cohort. The modern
analysis environment has both portable direct specifications and a clean-tested,
platform-specific resolved version lock. The external preprocessing files remain
environment and provenance specifications rather than cross-platform locks.

## Environment map

| File | Purpose | Status |
|---|---|---|
| `environment.yml` | Modern graph, topology, modelling, interpretation, and test environment | Direct specification; clean-install validation required |
| `requirements-direct.txt` | Pip-compatible direct requirements for the modern environment | Direct specification, not a lock |
| `locks/requirements-linux-x86_64-py310.lock` | Complete resolved package versions from the clean modern-environment reconstruction | Validated on Linux x86-64 with Python 3.10 |
| `observed-production-versions.tsv` | Principal versions observed on the audited Azure production VM | Provenance record, not an install file |
| `external-profiling-environment.yml` | KneadData, HUMAnN 3, and MetaPhlAn 4 environment used for the current external preprocessing workflow | Conda metadata reconstruction candidate |
| `metaphlan2_v260_environment.yml` | Runtime scaffold for the separate historical MetaPhlAn 2.6.0 workflow | Historical executable and database files hash verified |
| `validate_analysis_environment.py` | Lightweight import/version validation for the modern environment | Portable verification utility |
| `provenance/external_profiling_runtime.tsv` | Audited runtime and Conda metadata observations | Production provenance |
| `provenance/metaphlan2_v260_database_inventory.tsv` | Historical MetaPhlAn database filename, size, and SHA-256 inventory | Verified production provenance |
| `provenance/clean_environment_validation_20260903.md` | Clean-clone reconstruction commands, platform, commit, hashes, and test outcome | Passed validation record |

## Modern analysis environment

Create the environment with Micromamba or Conda:

```bash
micromamba create -f environment/environment.yml
micromamba activate tda-ibd-analysis
python environment/validate_analysis_environment.py
```

Alternatively, in an isolated Python 3.10 environment:

```bash
python -m pip install -r environment/requirements-direct.txt
python environment/validate_analysis_environment.py
```

For the exact clean-tested Linux x86-64/Python 3.10 package set, use a fresh
virtual environment and install CPU-only PyTorch before the resolved lock:

```bash
python3.10 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install --index-url https://download.pytorch.org/whl/cpu \
  'torch==2.14.0+cpu'
python -m pip install -r \
  environment/locks/requirements-linux-x86_64-py310.lock
python -m pip check
python environment/validate_analysis_environment.py
python run_checks.py full
```

This reconstruction passed at commit
`06e9b9db738552253e9645d8be9e023d4d96f578`. The package set differs from
several versions observed on the long-lived production VM, so the lock supports
portable code and synthetic-test verification; it does not retroactively claim
bitwise identity with every historical production result. Full evidence and
scope are recorded in
`provenance/clean_environment_validation_20260903.md`.

## Current external preprocessing environment

The `external-profiling-environment.yml` file records the Conda package metadata
observed for the environment used by
`external_validation/serrano_gomez/scripts/process_ge50_kneaddata_humann.sh`.
That workflow used:

- Python 3.12.13;
- KneadData 0.12.4;
- HUMAnN 3.9;
- MetaPhlAn 4.0.6 with the `mpa_vJun23_CHOCOPhlAnSGB_202307` index;
- Bowtie2 2.5.5;
- UniRef90 translated search.

The production DIAMOND executable reported version 2.0.15, while the Conda
metadata recorded package version 2.2.2. The executable is therefore recorded
by SHA-256 in `provenance/external_profiling_runtime.tsv`. A freshly solved
environment must not be treated as numerically equivalent until this mismatch
has been tested and resolved.

Database locations in production were machine-specific and are intentionally
not embedded as portable defaults. The workflow script explicitly selected the
MetaPhlAn database and index. The nucleotide, protein, utility-mapping, host
decontamination, and MetaPhlAn database releases must all be recorded and
checksummed in the final data manifest.

## Historical MetaPhlAn 2.6.0 workflow

The historical MetaPhlAn2 workflow is scientifically distinct from the HUMAnN
3 environment. Production invoked a standalone `metaphlan2.py` reporting
MetaPhlAn 2.6.0 (19 August 2016) with the `mpa_v20_m200` database. Do not install
modern `metaphlan` into this scaffold and assume equivalence.

`metaphlan2_v260_environment.yml` supplies a conservative interpreter and
Bowtie2 scaffold. The audited executable reported MetaPhlAn 2.6.0 (19 August
2016); its SHA-256 and the hashes of all seven `mpa_v20_m200` database files are
recorded in `provenance/input_file_inventory.tsv` and
`environment/provenance/metaphlan2_v260_database_inventory.tsv`. These records
identify the production runtime; they do not claim that modern package solvers
can reconstruct the historical environment without archived third-party
artifacts.

## Lock and release status

The modern environment completed these gates on 3 September 2026:

- fresh Git clone at the recorded commit;
- isolated Python 3.10 virtual environment;
- dependency consistency check and environment validator;
- 46 scientific unit tests and all six data-free synthetic component self-tests;
- export and SHA-256 verification of the complete resolved version set.

Before tagging `v1.0.0`, replay at least one representative analysis from the
released derived-data archive and update the documentation with the permanent
software and data-archive DOI. The external profiling environments must retain
their separate provenance and database records.

Do not call a direct dependency list or an environment export a complete lock
unless it contains the full resolved dependency graph and has passed the clean
reconstruction checks.
