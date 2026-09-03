# Publication-stage external-validation corrections

The original 2026-09-01 production-source import remains unchanged and is still
verified by `provenance/source_import_manifest.sha256`. The scripts listed in
`provenance/publication_correction_manifest.sha256` were added during the
publication audit to make corrected external analyses explicit and
reproducible without rewriting the historical source snapshot.

## Added entry points

- `apply_frozen_species_benchmarks_external_ibd_last_occurrence_v1.py` applies
  the corrected last-occurrence IBDMDB species benchmark models to strict
  structural external v1.
- `apply_frozen_species_benchmarks_external_v2_last_occurrence_all184.py`
  applies the same frozen models to all 184 expanded-v2 samples and the paired
  90-sample subset, including an independent paired-probability cross-check.
- `apply_frozen_h0_alphapi_ensemble_external_ibd_v1_host_subject.py` preserves
  sample-level H0 inference and groups participant results by host subject.
- `apply_frozen_ricci_ibd_model_expanded_v2.py` is the standalone frozen Ricci
  deployment entry point used for expanded-v2 result generation.

## Verification outcome

The corrected external result postflight passed. Strict species metrics were
unchanged relative to the preceding last-occurrence deployment. Expanded-v2
species results differ from results made with the historical mean-aggregation
models, as expected. Paired-v2 probabilities match strict-v1 probabilities to
within `1e-12`. H0 sample metrics are unchanged, while correct host-subject
grouping gives 78 strict-v1 and 154 expanded-v2 participants.

Aggregate public outputs are recorded under `results/`; raw predictions,
identifiers, private data, and model binaries are excluded.
