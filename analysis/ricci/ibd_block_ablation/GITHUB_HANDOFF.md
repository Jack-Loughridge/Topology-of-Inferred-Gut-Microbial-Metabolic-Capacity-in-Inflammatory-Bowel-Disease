# GitHub handoff

Add this directory to the publication branch at:

```text
analysis/ricci/ibd_block_ablation/
```

The ablation driver imports, and hash-verifies, the existing production source:

```text
analysis/ricci/ibd_cpath/repeated_ricci_ibd_cpath.py
SHA-256: 88b4110b482cdde7d1a77803835ca3d5b8b414ee956c5987ce5acddeabb66682
```

Do not modify that hash-locked production engine merely to add the ablation.
The new driver changes only the input feature block and corrects block-specific
reporting around the imported engine.

## Repository integration

1. Add all five package files under `analysis/ricci/ibd_block_ablation/`.
2. Add its `self_test.py` to the root `run_checks.py` synthetic suite.
3. Add the ablation to `docs/MANUSCRIPT_CODE_MAP.md`.
4. Add an explicit ablation completion/result gate to
   `docs/RELEASE_CHECKLIST.md`.
5. Mention the ablation in the Ricci section of the root `README.md`.
6. Record these publication-stage files in a new provenance manifest or the
   publication-correction manifest. Do not silently rewrite the original
   production-source snapshot.
7. Run:

   ```bash
   python3 analysis/ricci/ibd_block_ablation/self_test.py
   python3 run_checks.py static
   python3 run_checks.py unit
   python3 run_checks.py full
   ```

8. Keep the PR in draft until the real ablation completes and its compact
   aggregate outputs are audited.

## Result integration after the VM run

From `~/Real_Data/Ricci_IBD_FeatureBlock_Ablation_C002`, inspect and release the
compact non-identifying files needed for the manuscript, including:

```text
ablation_performance_summary.csv
ablation_paired_deltas.csv
ricci_feature_block_ablation.tex
ABLATION_RUN_COMPLETE.json
ablation_output_manifest.json
B_only/ABLATION_PROVENANCE.json
K0_only/ABLATION_PROVENANCE.json
```

The repetition-level aggregate table can also be released if disclosure review
confirms that it contains no sample or participant identifiers. Do not commit
fold predictions, participant identifiers, fitted model binaries, or large
feature matrices.

Place manuscript tables and machine-readable figure data under the existing
`results/` hierarchy, extend the final result checksum manifest, and map every
released item to its exact command, configuration, input hash, output hash,
commit and release tag.

## Scientific constraints

- Primary task only: IBD versus non-IBD.
- Fixed `C=0.02`; do not tune C separately for B or K0.
- Same 20 repetitions and five participant-grouped folds as `[B|K0]`.
- Same model seeds, training-fold-only scaler, L1 SAGA logistic regression,
  balanced class weights, threshold, metrics and participant aggregation.
- Treat paired-repeat quantiles and proportions as descriptive split
  sensitivity, not independent-cohort confidence intervals or p-values.
