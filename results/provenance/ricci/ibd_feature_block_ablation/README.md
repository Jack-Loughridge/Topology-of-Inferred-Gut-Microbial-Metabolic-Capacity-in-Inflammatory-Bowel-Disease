# IBD Ricci feature-block ablation provenance

This directory records the completed `B only` and `K0 only` runs, their
comparison with the existing `[B|K0]` result at `C=0.02`, and the
publication-stage audit of the prediction-level evidence.

- `ABLATION_RUN_COMPLETE.json` embeds the scientific lock, input hashes,
  split hashes, completion status, and block reports.
- `b_only_ABLATION_PROVENANCE.json` and
  `k0_only_ABLATION_PROVENANCE.json` are the corresponding embedded block
  reports serialised in the exact format written by the production driver.
- the three `*_RUN_COMPLETE.json` files record completion of the two
  ablations and the matched combined reference.
- `ablation_output_manifest.json` is the production output manifest.
- `raw_bundle_file_manifest.sha256` is the 54-file internal manifest from the
  audited transfer bundle, including the untracked prediction-level evidence.
- `raw_oof_audit_20260911.json` records the independent reconstruction checks
  and the boundary between files retained in Git and files reserved for the
  permanent derived-results archive.

The original block-specific provenance files were not separately selected by
the transfer filter. Their complete JSON values were retained inside
`ABLATION_RUN_COMPLETE.json`; the two files here were deterministically
recreated from those embedded values and checked against the production
driver's sorted, indented JSON serialisation.

The raw sample- and participant-level prediction tables contain study
identifiers and are intentionally not tracked in Git. They were nevertheless
included in the audited transfer bundle, verified against its internal
manifest, and used to reconstruct participant probabilities, all repetition
metrics, the aggregate performance table, and every paired delta.
