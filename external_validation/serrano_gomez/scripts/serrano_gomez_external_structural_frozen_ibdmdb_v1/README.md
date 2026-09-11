# Serrano-Gomez external structural validation — frozen IBDMDB representation

This package projects each completed external shotgun sample into the **frozen IBDMDB structural representation** used by the manuscript.

## What is frozen

- `AGORA_reactions_canon.parquet`
- exact original multi-input gate (`seed=13`, `tau=0.0001`, `50` roots/input)
- IBDMDB edge-specific **positive-support geometric-mean** baselines from `~/Real_Data/out_graphs/global_baselines.csv`
- graph equation `w = exp(-E_external/(Ehat_IBDMDB + 1e-8))`
- inactive/template edges at `w=1`
- H0 minimax filtration on the underlying undirected graph
- exact existing `14_compute_ricci_faithful_pairwise_active.py`, run with the production corrected settings (`n_paths=250`, `epsilon_dist=1e-4`, `c_single_out=0.001`, `beta=1.4`, active edges `w < 1-1e-12`)
- Ricci vector column order from `~/Real_Data/Ricci_Classifier_Faithful_Eps0001_n250_v3/edge_metadata.csv`, with `[B | K0]`

## What is deliberately NOT used

- no external-cohort reaction baseline
- no HUMAnN gene-family species proxy for the primary validation
- no renormalisation of the mapped external species subset back to 100%
- no new Ricci implementation
- no re-fitted Ricci feature vocabulary

The external numerator uses MetaPhlAn **species relative abundance percentages**. If a profile is stored as fractions summing to approximately 1, the code converts units to percent by multiplying by 100; this is unit conversion, not cohort normalization.

## Installation location on the VM

Copy this directory to:

`~/Real_Data/external_validation/serrano_gomez_ibd/ge50_external_validation/scripts/serrano_gomez_external_structural_frozen_ibdmdb_v1`

## 1. Mandatory preflight

```bash
cd ~/Real_Data/external_validation/serrano_gomez_ibd/ge50_external_validation/scripts/serrano_gomez_external_structural_frozen_ibdmdb_v1
python3 external_structural_validation.py preflight
```

It will abort unless all of these are true:

1. the current AGORA reaction table plus the exact original gate reconstructs the same directed edge set as frozen `global_baselines.csv`;
2. sampled existing IBDMDB GraphML files have that exact scaffold;
3. their stored `baseline`, `support`, and `weight` attributes satisfy the original graph equation;
4. the H0 union-find implementation reproduces existing training `*_H0.npy` non-trivial deaths;
5. the frozen training Ricci matrix has exactly `2 * len(edge_metadata)` columns;
6. the exact faithful active Ricci source file is present and passes source-signature checks.

Only after the terminal prints:

`STRICT EXTERNAL STRUCTURAL PREFLIGHT: PASSED`

should external samples be built.

## 2. Build all currently completed graph + H0 samples

```bash
python3 external_structural_validation.py build
```

The command scans the existing per-sample HUMAnN directories for parseable MetaPhlAn profiles. It is incremental: rerun the same command later as sample curation progresses. Unchanged samples are skipped. If a MetaPhlAn profile changes, that sample is rebuilt and any stale Ricci result for it is invalidated.

By default a sample fails rather than being silently forced into the model when less than 50% of its species-level MetaPhlAn abundance maps conservatively to the frozen training species axis. Mapping diagnostics are written under `audits/` and `species_projection/`.

## 3. Run exact production Ricci on the graphs available now

```bash
bash launch_ricci_tmux.sh
```

The source is not copied or rewritten. The helper finds and invokes the exact existing `14_compute_ricci_faithful_pairwise_active.py`. Its native done markers make the computation resumable. When additional external graphs appear, rerun the launcher after the current Ricci job finishes; already completed samples are skipped by the exact Ricci source.

Monitor:

```bash
tail -f ~/Real_Data/external_validation/serrano_gomez_ibd/ge50_external_validation/structural_frozen_ibdmdb/logs/ricci_n250.log
```

## 4. Produce frozen-order `[B | K0]` vectors

This can be rerun whenever more Ricci samples finish:

```bash
python3 external_structural_validation.py vectorize
```

Output:

`~/Real_Data/external_validation/serrano_gomez_ibd/ge50_external_validation/structural_frozen_ibdmdb/ricci_features_frozen_training_order/`

The feature matrix has exactly the frozen training order:

`[B_0,...,B_(m-1), K0_0,...,K0_(m-1)]`

where `m` is the number of rows in the training `edge_metadata.csv`. An external active edge absent from the training feature vocabulary is **not given a new classifier coordinate**, but it is retained in the external graph and therefore may affect H0 and the Ricci neighbourhood geometry of represented edges. This is the correct frozen-model behaviour.

## 5. Check progress

```bash
python3 external_structural_validation.py status
```

## Output layout

- `graphs/<sample>.graphml` — full frozen scaffold, including inactive `w=1` template edges
- `edges/<sample>_edges.csv.gz` — support, frozen denominator and weight audit
- `h0/<sample>_H0.npy` — Nx2 `[0, death]` persistence pairs
- `species_projection/<sample>_species_projection.csv.gz` — exact taxon mapping audit
- `sample_markers/<sample>.json` — profile hash and per-sample provenance
- `ricci_raw/<sample>_ricci.csv.gz` — exact faithful active Ricci output
- `ricci_features_frozen_training_order/feature_matrix_B_K0.npz` — external feature matrix
- `provenance/REFERENCE_SIGNATURE.json` — hashes and frozen reference parameters
- `audits/` — discovery/build diagnostics
