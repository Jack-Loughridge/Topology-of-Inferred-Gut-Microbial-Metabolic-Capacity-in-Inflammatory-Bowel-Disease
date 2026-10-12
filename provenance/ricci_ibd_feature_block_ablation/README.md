# Publication-stage Ricci feature-block ablation

The primary IBD Ricci feature-block ablation was added after the original
191-file production-source import. It is therefore recorded separately rather
than silently modifying `provenance/source_import_manifest.sha256`.

`source_manifest_20260911.sha256` covers the five-file source package under
`analysis/ricci/ibd_block_ablation/`. The driver imports and hash-verifies the
existing canonical IBD C-path engine; the feature block is the only scientific
change between the `B only`, `K0 only`, and matched `[B|K0]` analyses.

`result_manifest_20260911.sha256` covers the compact non-identifying source
data, manuscript table, run-completion records, locked block configurations,
and raw-evidence audit retained in Git. Prediction-level evidence and
identifier-bearing split manifests remain outside Git and are assigned to the
permanent derived-results archive.

Verify both publication-stage manifests from the repository root:

```bash
sha256sum -c provenance/ricci_ibd_feature_block_ablation/source_manifest_20260911.sha256
sha256sum -c provenance/ricci_ibd_feature_block_ablation/result_manifest_20260911.sha256
```
