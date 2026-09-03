# External-validation result record

## Analysis roles

The repository distinguishes two external analysis roles:

1. **Strict external validation (v1):** models, feature coordinates,
   preprocessing, and thresholds were fixed from IBDMDB before evaluation in
   Serrano--Gomez. This cohort contains 90 samples from 78 host subjects.
2. **Exploratory expanded-community replication (v2):** frozen prediction
   models were applied to 184 samples from 154 host subjects after constructing
   the expanded external-community representation. Because the structural
   support is normalized using the external cohort, these analyses are not a
   second strict external validation.

External labels were not used for feature mapping, preprocessing, fitting,
calibration, model selection, or threshold selection. Participant-level
probabilities are arithmetic means across samples assigned to the same host
subject and are thresholded at 0.5.

## Duplicate-policy correction

The source species workbook contains 1,360 rows representing 1,317 bare
external identifiers, with 41 duplicated identifier groups and 43 rows beyond
the first occurrence. Graph construction retained the last workbook occurrence
for a duplicated identifier, whereas the historical species benchmark pipeline
had averaged duplicate rows. The publication benchmark was therefore rerun
with the last-occurrence policy so that the abundance baseline and topological
representations use the same effective profiles.

The corrected external species deployment used those last-occurrence models.
On the paired 90-sample cohort, probabilities agreed with the independent
strict-v1 deployment to a maximum absolute difference of
`1.1102230246251565e-16`, below the prespecified `1e-12` tolerance. All paired
aggregate metrics and confusion matrices were exactly equal after loading the
saved outputs.

The H0 sample-level predictions did not change. Its participant aggregation was
corrected to use the host-subject field, changing the strict cohort count from
79 to 78 participants.

## Canonical manuscript tables

- `results/manuscript_tables/external_validation/species_benchmarks_strict_v1.tex`
- `results/manuscript_tables/external_validation/species_benchmarks_expanded_v2.tex`
- `results/manuscript_tables/external_validation/h0_alphapi_strict_and_expanded.tex`

The expanded species table replaces any table generated from the historical
mean-aggregation benchmark models. The H0 table replaces any version reporting
79 strict-v1 participants. The strict species values did not change in the
postflight comparison, but its canonical table is included to keep all current
external benchmark values together.

## Public result scope

Only aggregate metrics, confusion matrices, run markers, and portable
provenance are tracked. Sample-level probabilities and identifiers remain in
the controlled analysis workspace. Paths in public provenance use
`DATA_ROOT/` and `REPOSITORY_ROOT/` placeholders; hashes preserve the identity
of the original files and model artifacts.

