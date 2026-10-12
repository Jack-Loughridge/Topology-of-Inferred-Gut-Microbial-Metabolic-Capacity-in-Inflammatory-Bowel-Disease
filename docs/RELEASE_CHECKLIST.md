# Publication release checklist

This checklist separates repository work that is already complete from the
items that depend on the remaining Ricci computations.

## Complete

- [x] Preserve and hash-lock the production source snapshot.
- [x] Separate canonical production code from the historical flat layout.
- [x] Provide repository-level static checks, scientific unit tests, and
      synthetic end-to-end checks.
- [x] Record modern, current external-profiling, and historical MetaPhlAn2
      environments.
- [x] Recover and test graph-input canonicalisation.
- [x] Audit the duplicated source identifiers and rerun species benchmarks
      under the graph-consistent last-occurrence policy.
- [x] Correct participant grouping in external H0 summaries.
- [x] Commit corrected strict-v1 and expanded-v2 aggregate external results.
- [x] Pin public IBDMDB products, production input hashes, MetaPhlAn2 executable,
      and all seven `mpa_v20_m200` database files.
- [x] Record the production VM CPU, memory, swap, and filesystem capacity.
- [x] Update GitHub Actions to Node 24-compatible, commit-pinned action releases.
- [x] Recover the full sequential routing-attack generator, make its GraphML
      parser self-contained, and track its aggregate source data and LaTeX tables.
- [x] Promote the sampled-path generators for Tables 1--5 and remove their
      dependency on untracked home-directory source files.
- [x] Supersede the faulty archived edge-concentration, outdegree, and
      edge-weight GraphML parsers with tested canonical implementations.
- [x] Complete and audit the primary IBD `B only` and `K0 only`
      feature-block ablations at the prespecified `C=0.02`, retain compact
      non-identifying results, and hash-lock their publication-stage source.

## Independent pre-release gates

- [x] Create and validate a fully resolved Linux dependency lock from a clean
      environment.
- [x] Run the complete synthetic workflow on the release candidate.
- [ ] Open and review the publication branch pull request; require the
      publication checks before merge.
- [ ] Add final author metadata, ORCID identifiers if applicable, manuscript
      citation, release version, and release date to `CITATION.cff`.
- [x] Run the corrected active-vertex directed-clustering implementation,
      review valid/total counts, and track the replacement manuscript table.
- [x] Re-aggregate Tables 1--5 with valid/total counts and track their final
      LaTeX and aggregate source CSV files.
- [x] Rerun the corrected outdegree and edge-weight figure generators, compare
      against the manuscript copies, and track the final aggregate data and
      PNG/PDF files.

## Gates waiting only for the remaining Ricci computations

- [ ] Confirm every requested C-path has 100 completed outer folds and a valid
      run-completion marker.
- [ ] Select/report C values according to the prespecified analysis rule.
- [ ] Generate the remaining internal Ricci aggregate, confusion-matrix,
      coefficient-stability, and manuscript tables.
- [ ] Add each final table and figure to `docs/MANUSCRIPT_CODE_MAP.md`.
- [ ] Generate a single checksum manifest over all released result and
      figure-source files.

## Final archive and submission gates

- [ ] Tag the reviewed commit as `v1.0.0`.
- [ ] Archive the tagged release and compact derived-result package in a
      DOI-issuing repository.
- [ ] Insert the DOI in `README.md`, `CITATION.cff`, and the manuscript Data
      Availability Statement.
- [ ] Verify a fresh clone and downloaded archive using the documented commands.
- [ ] Confirm that every reported manuscript number is present in a tracked
      table or figure-source file and matches the release commit.

No scientific claim should cite the moving publication branch after release;
the manuscript should cite the immutable tag and DOI.
