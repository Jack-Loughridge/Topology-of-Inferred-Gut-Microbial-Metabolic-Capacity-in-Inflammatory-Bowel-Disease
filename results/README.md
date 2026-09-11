# Publication result files

This directory contains compact, aggregate outputs intended to support the
manuscript and its figures. It does not contain raw cohort data, sample-level
predictions, participant identifiers, fitted model binaries, or large
intermediate artifacts.

## Layout

- `manuscript_tables/`: LaTeX tables that can be copied into the manuscript.
- `figure_source_data/`: machine-readable aggregate metrics used for tables or
  figures.
- `provenance/`: run-completion records and portable deployment provenance.
- `external_validation_manifest.sha256`: checksums for the external-validation
  files in the three directories above.
- `../provenance/ricci_ibd_feature_block_ablation/result_manifest_20260911.sha256`:
  checksums for the compact primary IBD Ricci feature-block ablation result,
  table, locked configurations, and audit files.

The strict v1 analyses use frozen IBDMDB coordinates and models. Expanded v2
analyses use the full external cohort and external-cohort structural support;
they are exploratory replications, not second strict external validations.

Verify this result set from the repository root with:

```bash
sha256sum -c results/external_validation_manifest.sha256
sha256sum -c provenance/ricci_ibd_feature_block_ablation/result_manifest_20260911.sha256
```
