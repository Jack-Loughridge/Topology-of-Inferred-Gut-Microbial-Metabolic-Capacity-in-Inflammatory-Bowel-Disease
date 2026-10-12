# Legacy root snapshot

This directory preserves files from the repository layout that existed before the publication rebuild.

The files are retained for provenance and historical comparison. They are not canonical production entry points unless a later release explicitly identifies one in `docs/MANUSCRIPT_CODE_MAP.md`.

- `species_benchmark_package/` is the former flat benchmark package. Its root-level README, citation file, manifest, launchers, and scripts described only that component and were superseded by the structured production sources under `analysis/species/`.
- `historical_scripts/` contains earlier graph, persistence, Ricci, clustering, path, and diagnostic scripts that previously lacked conventional extensions or portable entry points.

Canonical publication sources are under `pipeline/`, `analysis/`, and `external_validation/`. The historical filenames and file contents were preserved during the move so Git can record them as renames.

Before the final release, each historical script will be classified as either superseded, exploratory, or still required for a manuscript item. Any required code will receive a documented canonical location and executable entry point outside this archive.
