# Manuscript and publication code compatibility audit

Date: 11 October 2026.

## Scope

The manuscript baseline is `Jack-Loughridge/Final-tda-classifier` at `d0f720a11bfc68c4c9380e9fddb76ccbcb4ed1bd`. The publication-code baseline is this repository's `publication/rebuild-2026` at `effc7bf0661cbf36dda4ed82c44a54a455ad60a4`. The default `main` branch remains an older protected snapshot. Updating this review branch does not by itself update the default branch or produce an immutable release.

This is a source, configuration, aggregate-output and software-test audit. No cohort-wide graph reconstruction, Ricci recalculation, classifier fitting or external deployment was performed. Passing unit tests is not validation of every manuscript number. The machine-readable repository inventory records the paths inspected by the automated scan, while the statement-level checks below identify actual source observations.

## Resolved in this revision

The final joint entry point now selects the exact delivered v8 package, not the historical nested joint model. The new manuscript Methods, Results, abstract, Discussion and supporting tables describe fixed M32, IBD only, five locked repetitions, explicit .001 H0 ridge, Ricci C=.02, source-weighted alpha and local 1e-5 numerical checks. They no longer attach the v5 70/90-digit, 100-extra-refit protocol to v8 results. All numerical table values are regenerated from the supplied aggregates.

The old joint core remains immutable for historical reproduction and the 191-file source lock. Current root documentation and the manuscript map supersede it explicitly. Its old internal README is not evidence that it generated v8.

Source-backed corrections also align S1/S6 with the main article and code. `BinnedDataset` assigns an internal-boundary death to its right interval, so the interval convention is [left,right). The between-class loss is evaluated over centroid pairs, not sampled observations. The graph builder retains unsupported candidate edges at weight one and defines an empty positive-support baseline as zero. Ricci walks use active edges with the stated weight floor. Written SBML directions, not a complete reversible reconstruction, are used.

The mapped H0 generator replaces nonpositive edge weights by 1e-9 before sorting. The Methods now distinguish this from the descriptive active-graph summaries that exclude nonpositive weights. The audit did not count how many saved classifier deaths arose from this safeguard.

## Compatibility and remaining release work

| Component | What the source supports | Remaining boundary or action |
|---|---|---|
| Graph extraction and construction | Written reactants/products, distinct catalysts per edge, positive-support geometric-mean reference, exp(-support/(reference+1e-8)), zero-support weight one, fixed multi-input gate | Missing reverse support is a documented limitation. Do not silently repair graphs underlying frozen fits. The incidence of name-based currency filtering and underflow must be interpreted from the audit outputs, not defaults alone. |
| H0 classifier diagrams | Kruskal-style merging with births zero, direction ignored and nonpositive-weight floor | This is not the separate descriptive entropy/density generator. Nonpositive flooring and omitted essential bars need their documented conventions. |
| Standalone H0 model | Source-weighted kernels, train-only adaptive bins, logit head, centroid/radius/between/CE losses | S3 explicitly retains historical plots/table pending reconciliation with the final five-task one-repetition export. A source match alone does not close that plot provenance. |
| Ricci construction | Active graphs, pair-specific reachability and LP transport, epsilon 1e-4, single-outgoing product decrement .001 | Script defaults include 1,000 walks, while the reported source analysis used 250. The production command/config override, not the default, must substantiate 250. Do not alter historical code merely to align its default. |
| Ricci classifier and B/K0 ablations | Saved [B|K0], locked participant folds, training scaling, class-balanced sparse logistic models | B is finite-curvature availability in the source builder, not all active-edge presence. Existing root terminology is corrected. Post hoc complete-presence analyses are different controls. |
| Species comparators | Dedicated and remaining-task runners with their documented complete-case policy, CLR, scaling and fixed classifier configurations | Confirm all final displayed tables against the matching exported task/configuration, not one runner's default settings. Sample exclusions differ from graph cohort totals. |
| External models | Separate strict frozen and expanded reconstruction/deployment entry points | Source and original external presence encoding differ as disclosed in the manuscript. Expanded reconstruction is not strict validation. No new v8 external model was run. |
| Graph diagnostics | Canonical path, concentration, clustering, outdegree, weight-density and sequential-removal generators and manifests are mapped | Later entropy/death-density summaries are not covered by the current main map. Obtain and hash-lock their exact generator/output provenance rather than infer it from H0 classification code. |
| Later post hoc manuscript analyses | Manuscript reports direction census, validity controls and local reverse/cost challenges | These later analyses are not mapped to canonical generation/output rows in the existing manuscript map. The repository inventory must be checked for recoverable scripts, and exact source/configuration/result hashes are still required. Absence from the map is not proof that code does not exist elsewhere. |
| Binary and frozen all-task follow-up tables | Existing classification infrastructure can run the tasks | The later full export and its table generators need final per-table mapping and immutable archive references. Do not substitute generic classifier code for evidence of the actual generated tables. |
| Final joint result | Exact delivered package, 50 converged records, all aggregate means, SDs and paired differences verified | The public selection lacks individual probabilities, exact split membership, realised interval boundaries and the complete executed VM source manifest. These remain on the VM and must be retained. |
| Release | MIT code licence, environments, historical source locks and many static checks exist | Final release tag, DOI, author/citation metadata, derived-data access arrangements and promotion of publication branch remain unfinished. |

## Not defects to be silently repaired

The missing fixed-alpha control and continuous reaction-strength comparator, lack of external joint validation, absence of joint subtype evaluation, and restriction to five repetitions are limitations of the completed study. They are not implemented results awaiting a different label. No new experiment was introduced to eliminate them.

## Verification boundary

The v8 source package was tested twice, including a fresh publication copy. The publication aggregate checker recomputes means, sample SDs, paired differences and confusion-derived metrics without observations. It does not independently reconstruct AUC ranks, individual log likelihoods, production gradients or original participant mappings. The source package matches the delivered version, while the exported VM integrity flags do not expose its complete source manifest. These limitations remain explicit in the manuscript's verification paragraph and `provenance/joint_v8/REPRODUCIBILITY_BOUNDARIES.md`.
