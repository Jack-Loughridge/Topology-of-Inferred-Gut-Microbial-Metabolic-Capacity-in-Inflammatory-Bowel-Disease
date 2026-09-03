# Reproducibility guide

## Scope

This repository supports three levels of reproduction:

1. **Portable verification:** run integrity, unit, and synthetic integration tests without human microbiome data.
2. **Analysis replay:** reproduce classifiers and manuscript tables from released derived diagrams, feature matrices, split manifests, and metadata.
3. **Full reconstruction:** begin with public cohort products or raw sequence data and AGORA2 SBML files, then rebuild graphs, topology, models, and manuscript outputs.

The final release must support levels 1 and 2 directly. Level 3 is computationally expensive and depends on separately distributed third-party resources, but every author-generated transformation must be documented and available.

## Portable verification

The repository has one root entry point with three cumulative levels:

```bash
python run_checks.py static
python run_checks.py unit
python run_checks.py full
```

- `static` verifies the production source manifest, citation and environment metadata, locked JSON configurations, canonical Python and Bash syntax, and tracked-file hygiene.
- `unit` also executes the H0 Alpha-Pi and joint H0-Ricci pytest suites.
- `full` also executes all six data-free synthetic component self-tests for H0, joint, Ricci, and species analyses.

For static checks alone, install PyYAML into Python 3.10 or later. For unit or full checks, create the modern environment described in `environment/README.md` and validate it first:

```bash
python environment/validate_analysis_environment.py
```

For the clean-tested Linux x86-64/Python 3.10 package set, install
`environment/locks/requirements-linux-x86_64-py310.lock` using the ordered
commands in `environment/README.md`. A clean clone at commit
`06e9b9db738552253e9645d8be9e023d4d96f578` passed the validator, all 46 unit
tests, and every data-free synthetic component self-test.

The root `.github/workflows/ci.yml` workflow runs static and unit verification on pushes and pull requests. `.github/workflows/full-synthetic.yml` exposes the longer full suite through a manual workflow dispatch. Both workflows use CPU-only PyTorch and constrain numerical-library threads.

## Integrity first

From the repository root, verify the imported production source and configuration lock:

```bash
sha256sum -c provenance/source_import_manifest.sha256
```

The publication rebuild began from GitHub commit `9a20daf38bfad2c00ab5b18792e1b2b119d74e6a`. The production source baseline is described in `provenance/IMPORT_NOTES.md`.

Do not modify hash-locked production sources when improving documentation, wrappers, tests, or portability. A scientific source change requires an explicit new provenance record, rationale, tests, and updated checksum rather than a silent manifest rewrite.

## Canonical workflow

### 1. Acquire and verify inputs

- Obtain the exact IBDMDB/HMP2 metadata and taxonomic-profile products recorded in `provenance/INPUT_PROVENANCE.md`.
- Obtain the AGORA2 version 2.01 SBML distribution from the Virtual Metabolic Human resource.
- For external validation, construct the selected `PRJEB42155` manifest and verify downloaded FASTQ MD5 values.
- Confirm all filenames and hashes against the release input manifest.

Verified public URLs, local basenames, sizes, and hashes are recorded in `provenance/input_file_inventory.tsv`. The historical AGORA2 archive checksum and acquisition date were not retained; the resulting reaction tables are hash pinned and the limitation is explicit in `provenance/INPUT_PROVENANCE.md`.

### 2. Construct the metabolic reaction table and graphs

Canonical sources:

```text
pipeline/graph_construction/sanitize_agora_sbml.py
pipeline/graph_construction/extract_agora_reactions.py
pipeline/graph_construction/build_microbiome_graphs.py
```

The previously undocumented graph-input transformations are now encoded in `pipeline/graph_construction/canonicalize_graph_inputs.py`. The species and reaction operations verify the audited production input hashes and expected dimensions when invoked with `--require-production-hash`; exact rules, reference hashes, commands, and the privacy-preserving audit chain are recorded in `pipeline/graph_construction/README.md` and `provenance/CANONICALIZATION.md`.

### 3. Derive topological representations

Canonical sources:

```text
pipeline/topology/build_persistence_diagrams.py
pipeline/topology/compute_ricci_faithful_pairwise_active.py
```

The faithful Ricci production configuration uses active edges and records parameters including `epsilon_dist=0.0001`, `c_single_out=0.001`, and `beta=1.4`. Exact run-specific values and source hashes are in `provenance/locked_run_configs/`.

### 4. Establish locked participant-grouped splits

The repeated joint-analysis preflight constructs and audits task-specific split manifests. Those exact manifests are consumed by the repeated Ricci and species analyses rather than independently regenerating folds.

Canonical orchestration:

```text
analysis/joint/repeated_cv/
```

Every split must keep all samples from a participant on one side of each train/test boundary. Inner model selection is restricted to outer-training participants.

### 5. Run source-cohort analyses

| Analysis | Canonical location | Design summary |
|---|---|---|
| Standalone H0 IBD repeated analysis | `analysis/h0_alpha_pi/ibd_repeated_cv/` | Corrected train-only adaptive intervals using locked outer folds. |
| Standalone H0 five-task analysis | `analysis/h0_alpha_pi/five_task_1x5/` | One locked five-fold repetition across the five tasks, with component-specific selection and refit safeguards. |
| Ricci IBD C-path | `analysis/ricci/ibd_cpath/` | Five fixed C values over repeated participant-grouped folds. |
| Ricci five-task repeated analysis | `analysis/ricci/five_task_repeated_cv/` | Locked repeated folds and faithful `[B | K0]` features. |
| Joint H0-Ricci | `analysis/joint/core/` and `analysis/joint/repeated_cv/` | Three-fold grouped inner selection within repeated five-fold outer evaluation. |
| Species benchmarks | `analysis/species/ibd_complete_case/` and `analysis/species/remaining_tasks/` | Train-only CLR, variance filtering, scaling, and fixed comparator models. |

Each component contains its own README, validation command, test or self-test, launcher, expected completion marker, and output schema. Use the locked configuration JSON rather than relying on launcher defaults alone.

### 6. Run external validation

External processing sources are under:

```text
external_validation/serrano_gomez/scripts/
```

Maintain the distinction between:

- v1 frozen IBDMDB representation and frozen model deployment;
- v2 external-normalised expanded-community structural replication.

External labels must not influence graph construction, scaling, feature vocabulary, parameter selection, or model fitting. Cohort labels are used only for the prespecified final evaluation and labelled summaries.

Corrected publication-stage deployment entry points and their relationship to
the historical source snapshot are recorded in
`provenance/PUBLICATION_CORRECTIONS.md`. Canonical aggregate external tables and
machine-readable figure-source data are under `results/`.

### 7. Generate manuscript-facing outputs

Canonical interpretation scripts are under:

```text
analysis/interpretation/
```

The final release must record the exact command and source output for every manuscript table and figure in `docs/MANUSCRIPT_CODE_MAP.md`. Compact generated tables belong under `results/`; large artifacts belong in the permanent derived-results archive.

## Reproducibility safeguards already present

- participant-grouped outer and inner evaluation;
- train-only preprocessing and adaptive H0 bin construction;
- locked split manifests shared across model families;
- source, configuration, input, and output fingerprints;
- resume checks that reject incompatible output directories;
- explicit completion markers before aggregation or result display;
- sample-level and participant-level evaluation;
- repetition-level pooled out-of-fold uncertainty rather than treating overlapping folds as independent experiments;
- frozen external model and feature-order deployment for v1.

## Items still required before release

- the remaining completed internal Ricci result and figure-source tables;
- a clean-clone analysis replay using the release archive;
- a tagged release and permanent DOI.
