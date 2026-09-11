# Topology of Inferred Gut Microbial Metabolic Capacity in Inflammatory Bowel Disease

This repository contains the code and reproducibility materials for constructing abundance-weighted directed microbial metabolic graphs and analysing their topology in inflammatory bowel disease (IBD).

The workflow combines microbial species profiles with genome-scale metabolic reconstructions, constructs sample-specific directed graphs, derives H0 persistence and faithful active-edge Ricci curvature representations, evaluates topology-based classifiers against species-abundance benchmarks, and performs external validation in an independent shotgun-metagenomic cohort.

> **Release status:** publication-preparation branch. Production source code,
> external-validation aggregate results, public-input provenance, and reference
> database hashes are recorded. A clean-clone reconstruction of the modern
> analysis environment and its resolved Linux/Python 3.10 lock are also recorded.
> The primary IBD Ricci feature-block ablation and its prediction-level audit
> are now recorded. The remaining internal Ricci summaries,
> immutable archive identifier, and preferred manuscript citation will be added
> at the final result freeze.

## Study components

| Component | Location | Purpose |
|---|---|---|
| Graph construction | `pipeline/graph_construction/` | Extract and sanitise AGORA2 reactions, reproduce canonical graph inputs, and build sample-specific weighted directed graphs. |
| Graph diagnostics | `analysis/graph_diagnostics/` | Compute active-vertex directed clustering and sequential minimax-path edge-removal diagnostics with sample-to-participant aggregation. |
| Topological features | `pipeline/topology/` | Construct H0 persistence diagrams and faithful active-edge Ricci curvature features. |
| Standalone H0 analyses | `analysis/h0_alpha_pi/` | Train-only adaptive H0 Alpha-Pi classification and sensitivity analyses. |
| Standalone Ricci analyses | `analysis/ricci/` | Repeated participant-grouped Ricci classification and C-path analyses. |
| Primary Ricci feature-block ablation | `analysis/ricci/ibd_block_ablation/` | Compare active-edge indicators, active-edge Ricci curvature, and their combined representation at the prespecified primary-task regularisation value. |
| Joint H0-Ricci analyses | `analysis/joint/` | Nested participant-grouped sparse joint models with resolution-normalised alpha smoothness. |
| Species benchmarks | `analysis/species/` | L1 logistic regression, random forest, and XGBoost comparators on species abundances. |
| Interpretation | `analysis/interpretation/` | Generate coefficient, contribution, process, and manuscript-facing interpretation tables. |
| External validation | `external_validation/serrano_gomez/` | Process the independent cohort and evaluate frozen-support and expanded-community representations. |
| Environments | `environment/` | Modern analysis, current external-profiling, and historical MetaPhlAn2 environment records. |
| Provenance | `provenance/` | Source checksums, locked production configurations, and import notes. |
| Publication results | `results/` | Compact result and figure-source tables added at the final result freeze. |

Historical files from the pre-publication flat repository are isolated under `archive/legacy_root/`. They are retained for provenance and are not canonical execution entry points unless explicitly identified in the manuscript-to-code map.

## Scientific design

Five classification tasks are represented throughout the repository:

1. IBD versus non-IBD;
2. non-IBD versus ulcerative colitis (UC) versus Crohn's disease (CD);
3. non-IBD versus UC;
4. non-IBD versus CD;
5. UC versus CD.

Participant identifiers define every cross-validation grouping boundary. Fold-internal transformations are fitted using training data only. The repeated Ricci, species, and joint analyses use locked task-specific split manifests; component READMEs document the exact repetition counts, inner-selection procedures, model grids, and output schemas.

The external study has two deliberately distinct representations:

- **v1 frozen support:** projects external samples into the structural representation and feature order learned from IBDMDB, without external-cohort refitting;
- **v2 expanded community:** rebuilds an external-normalised expanded-community structural representation and is interpreted as structural replication or exploratory transfer, not as the same frozen predictor.

## Quick verification

The static checks require Python 3.10 or later and PyYAML, but no microbiome data:

```bash
python -m pip install PyYAML
python run_checks.py static
```

After installing the modern analysis environment, run the existing scientific unit suites through the root entry point:

```bash
python run_checks.py unit
```

The complete synthetic end-to-end suite is more computationally intensive:

```bash
python run_checks.py full
```

`static` verifies the frozen 191-file production source manifest, the separate
publication-stage Ricci-ablation source and result manifests, citation and
environment metadata, locked JSON configurations, Python and Bash syntax, and
tracked-file hygiene. `unit` additionally runs graph-diagnostic, H0, and
joint-model tests. `full` additionally runs all seven data-free synthetic
component self-tests. GitHub Actions runs the static and unit levels on pushes
and pull requests; the full suite is available as a manually triggered
workflow.

## Reproducibility

Start with:

- [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) for the ordered workflow and reproduction levels;
- [`docs/DATA_AVAILABILITY.md`](docs/DATA_AVAILABILITY.md) for public sources, redistribution boundaries, and the planned permanent archive;
- [`docs/COMPUTATIONAL_REQUIREMENTS.md`](docs/COMPUTATIONAL_REQUIREMENTS.md) for environments and resource classes;
- [`docs/MANUSCRIPT_CODE_MAP.md`](docs/MANUSCRIPT_CODE_MAP.md) for analysis-to-code traceability;
- [`docs/RELEASE_CHECKLIST.md`](docs/RELEASE_CHECKLIST.md) for the remaining freeze and archival gates;
- [`environment/README.md`](environment/README.md) for installation and environment provenance;
- [`environment/locks/requirements-linux-x86_64-py310.lock`](environment/locks/requirements-linux-x86_64-py310.lock) for the clean-tested modern-environment lock;
- [`pipeline/graph_construction/README.md`](pipeline/graph_construction/README.md) for the audited graph-input canonicalisation commands.
- [`analysis/graph_diagnostics/README.md`](analysis/graph_diagnostics/README.md) for the corrected clustering and recovered sequential-removal protocols.

Production source files and locked configurations can also be checked directly with:

```bash
sha256sum -c provenance/source_import_manifest.sha256
sha256sum -c provenance/ricci_ibd_feature_block_ablation/source_manifest_20260911.sha256
sha256sum -c provenance/ricci_ibd_feature_block_ablation/result_manifest_20260911.sha256
```

The original source manifest records 191 files imported from the production
Azure VM source freeze. The later Ricci feature-block ablation has separate
source and result manifests so that the original freeze remains immutable.
Generated data, fitted checkpoints, caches, identifier-bearing predictions,
and large results are deliberately excluded from Git.

## Data and code availability

The source cohort is the public IBDMDB/HMP2 study (BioProject `PRJNA398089`). The external Serrano-Gomez processing scripts use ENA study `PRJEB42155`. The graph pipeline uses AGORA2 version 2.01, obtained separately from the Virtual Metabolic Human resource and subject to its original distribution terms. Exact public-product and production-input hashes are recorded in `provenance/INPUT_PROVENANCE.md`.

Raw sequence data and large intermediate matrices are not redistributed in this Git repository. The final release will link a permanent archive containing the compact derived materials needed to verify every reported table and figure, together with a checksum manifest.

## License and citation

Author-generated code is provided under the MIT License. Third-party data and resources retain their original terms. Citation metadata are provided in `CITATION.cff`; the final manuscript citation and software DOI will be inserted at release.
