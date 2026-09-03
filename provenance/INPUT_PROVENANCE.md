# Input and reference-database provenance

This record identifies the public cohort products, derived production inputs,
and historical reference database files used by the publication pipeline. It
contains no sample identifiers or abundance values.

## IBDMDB/HMP2 inputs

The public metadata product was downloaded from:

```text
https://g-227ca.190ebd.75bc.data.globus.org/ibdmdb/metadata/hmp2_metadata_2018-08-20.csv
```

Its SHA-256 is
`656b7bd97660ddb875548805e30bede31f2d1208293f7170d2d5755e33862ec9`.
The public file was confirmed byte-identical to the production
`hmp2_metadata.csv`.

The taxonomic-profile product was downloaded from:

```text
https://g-227ca.190ebd.75bc.data.globus.org/ibdmdb/products/HMP2/MGX/2018-05-04/taxonomic_profiles_3.tsv.gz
```

Its SHA-256 is
`d790ff15e46d61ca0cadc55d9f918de4e3415d7f97c992ac37610aaee02117ed`.
A privacy-preserving occurrence audit matched all 1,360 rows of the derived
species workbook to official profiles under the recorded identifier
normalisation. The effective graph input contains 1,317 identifiers because
graph construction retains the final workbook occurrence for duplicated bare
identifiers. The correction and sensitivity analysis are described in
`provenance/PUBLICATION_CORRECTIONS.md`.

## AGORA2

The metabolic reconstruction resource is AGORA2 version 2.01, described by
Heinken et al. (2023), DOI `10.1038/s41587-022-01628-0`, and distributed by the
Virtual Metabolic Human resource at:

```text
https://vmh.life/files/reconstructions/AGORA2/version2.01/
```

The provider distributes individual and zipped SBML reconstructions. The
repository does not redistribute those third-party files. The historical raw
download archive was not retained, so an archive-level checksum cannot be
recovered retrospectively. Reproducibility is instead anchored to the exact
author-generated extraction and canonicalisation code and the verified hashes
of the two reaction tables entering graph construction:

- `AGORA_reactions.parquet`: `799fe755c0c9ede07572742442a799a8de1ff4e7165f6ecb2812b6699955cf77`;
- `AGORA_reactions_canon.parquet`: `ec906cd5f08448991b3958112fd4367f17141e0bbbe7d0333c8382a5c36af755`.

The exact acquisition date of the historical AGORA2 download was not logged.
This limitation must remain explicit; it must not be replaced by an invented
date or archive checksum.

## Historical MetaPhlAn 2.6 database

The separate historical external-profiling route used MetaPhlAn 2.6.0 and the
seven-file `mpa_v20_m200` database. Exact filenames, sizes, and SHA-256 hashes
are recorded in
`environment/provenance/metaphlan2_v260_database_inventory.tsv`.

## Portable use

Machine-specific paths in locked run configurations preserve the audited
production invocation and must not be silently rewritten. When replaying the
pipeline, set a local `DATA_ROOT` and substitute that prefix for
`/home/Jack/Real_Data` while preserving filenames, configurations, hashes,
seeds, and split manifests. `provenance/input_file_inventory.tsv` provides the
portable basename-to-hash mapping.

## Audit record

These records were captured on 3 September 2026 by the read-only Phase 6A
audit. The audit archive
`phase6a_input_provenance_audit_20260903T110459Z.tar.gz` has SHA-256
`4ef659312ac65913d3d566e3d0fbfebdff558672d748ece704c4eecafb68250c`;
all seven files listed by its internal manifest verified successfully. The
repository was clean at commit `d379939` when the audit ran.
