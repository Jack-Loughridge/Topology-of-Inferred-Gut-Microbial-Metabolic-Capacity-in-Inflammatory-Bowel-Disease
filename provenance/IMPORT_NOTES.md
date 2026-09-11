# Production source import

This branch began from GitHub commit 9a20daf38bfad2c00ab5b18792e1b2b119d74e6a.

Production analysis source was imported from the Azure VM source freeze on 2026-09-01 without methodological refactoring.
Imported source files: 191
Manifest SHA-256: bc137b4996a1d40bd6ae466a41e40a61f2143bbe11ab0dbbf3961bd4bebb6567

Generated results, datasets, model artifacts, caches, nested Git metadata and historical backup copies were excluded.
The excluded historical copies remain recoverable under ~/publication_repo_excluded_20260901.

The two graph-input canonicalisation gaps identified during import were resolved by privacy-preserving audits on 2026-09-02. The recovered transformations, audited hashes, executable implementation, and regression tests are documented in `provenance/CANONICALIZATION.md` and `pipeline/graph_construction/README.md`.

Remaining release provenance includes exact public-input and AGORA2 distribution pins, final result tables and hashes, clean replay evidence, and permanent archive identifiers.
