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

- Obtain the selected IBDMDB/HMP2 products associated with BioProject `PRJNA398089`.
- Obtain the exact AGORA2 SBML distribution recorded by the final release.
- For external validation, construct the selected `PRJEB42155` manifest and verify downloaded FASTQ MD5 values.
- Confirm all filenames and hashes against the release input manifest.

The final acquisition commands and checksums will be recorded in `docs/DATA_AVAILABILITY.md` and the permanent archive.

### 2. Construct the metabolic reaction table and graphs

Canonical sources:

```text
pipeline/graph_construction/sanitize_agora_sbml.py
pipeline/graph_construction/extract_agora_reactions.py
pipeline/graph_construction/build_microbiome_graphs.py
```

The two canonicalisation transformations identified in `provenance/IMPORT_NOTES.md` remain release blockers. Full reconstruction is not considered closed until executable steps replace those undocumented transformations.

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

- a fully resolved environment lock and clean-install report;
- executable canonicalisation steps for the two provenance gaps;
- final result and figure-source tables;
- a clean-clone analysis replay using the release archive;
- a tagged release and permanent DOI.
