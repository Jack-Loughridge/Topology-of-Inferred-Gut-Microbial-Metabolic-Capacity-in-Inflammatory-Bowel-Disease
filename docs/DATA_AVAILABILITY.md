# Data and code availability

## Policy objective

The final release will make all author-generated code underlying the reported findings publicly accessible and will provide the compact data products needed to verify every table and figure. Raw public data and third-party reconstructions will remain in their original repositories and will be accompanied by acquisition and integrity instructions.

## Primary source cohort

The source cohort is the Inflammatory Bowel Disease Multi'omics Database / integrative Human Microbiome Project (IBDMDB/HMP2).

- Study portal: <https://ibdmdb.org/>
- NCBI BioProject: `PRJNA398089`
- Study publication: Lloyd-Price et al., *Nature* (2019), DOI `10.1038/s41586-019-1237-9`

The Git repository does not redistribute raw human sequence data or the complete source metadata tables. The production inputs were traced to these exact public products:

- metadata: `hmp2_metadata_2018-08-20.csv`, SHA-256 `656b7bd97660ddb875548805e30bede31f2d1208293f7170d2d5755e33862ec9`;
- taxonomic profiles: `taxonomic_profiles_3.tsv.gz` from the HMP2/MGX 2018-05-04 product directory, SHA-256 `d790ff15e46d61ca0cadc55d9f918de4e3415d7f97c992ac37610aaee02117ed`.

Direct URLs, the derived-input hashes, and the privacy-preserving source-match evidence are recorded in `provenance/INPUT_PROVENANCE.md` and `provenance/input_file_inventory.tsv`.

## External validation cohort

The independent external cohort is the Serrano-Gomez shotgun-metagenomic IBD study.

- ENA study used by the acquisition scripts: `PRJEB42155`
- Study publication: Serrano-Gomez et al., *Computational and Structural Biotechnology Journal* (2021), DOI `10.1016/j.csbj.2021.11.037`
- Download script: `external_validation/serrano_gomez/scripts/download_ge50_fastqs.sh`
- Integrity checker: `external_validation/serrano_gomez/scripts/check_ge50_md5.py`

The final derived-data archive must include the exact selected-run manifest, sample-to-participant mapping used for evaluation, inclusion/exclusion reasons, source URLs, and source MD5 values. It must not include information prohibited by the source repository's terms.

## Metabolic reconstruction resource

The graph-construction pipeline uses AGORA2 genome-scale microbial metabolic reconstructions obtained separately from the Virtual Metabolic Human resource.

- Resource: AGORA2 version 2.01, <https://vmh.life/files/reconstructions/AGORA2/version2.01/>
- AGORA2 description: Heinken et al., *Nature Biotechnology* (2023), DOI `10.1038/s41587-022-01628-0`
- Reaction extraction: `pipeline/graph_construction/extract_agora_reactions.py`
- SBML sanitisation: `pipeline/graph_construction/sanitize_agora_sbml.py`

AGORA2 files are not redistributed here. The historical raw archive and its acquisition date were not retained, so an archive-level checksum cannot be reconstructed. This is recorded transparently in `provenance/INPUT_PROVENANCE.md`. The exact reaction tables entering graph construction are pinned by SHA-256, and all author-generated extraction, sanitisation, and canonicalisation code is included.

## Derived files excluded from Git

Large or generated materials are deliberately excluded from Git, including:

- raw FASTQ files and intermediate KneadData/HUMAnN/MetaPhlAn products;
- full sample graph collections;
- persistence-diagram arrays and full Ricci tables;
- sparse feature matrices;
- model checkpoints and per-fold fitted artifacts;
- large intermediate or cached analysis directories.

These exclusions keep the source repository reviewable; they do not remove the obligation to provide the materials necessary to verify the published findings.

## Planned permanent publication archive

The final tagged GitHub release will be archived in a DOI-issuing repository such as Zenodo. A companion derived-results archive will contain, where licensing and participant protections allow:

1. cohort and exclusion manifests using public study identifiers;
2. all locked participant-grouped split manifests;
3. graph and feature-axis metadata;
4. configuration and provenance JSON files;
5. pooled out-of-fold sample and participant predictions;
6. repetition-level and aggregate performance tables;
7. coefficient, feature-stability, and interpretation tables supporting the manuscript;
8. external-cohort predictions and evaluation tables;
9. figure-source tables;
10. SHA-256 checksums for every archived file.

The archive DOI is intentionally left unset until the final result freeze.

## Canonicalisation provenance

The transformations producing `Real_Species_Abundances_canon.xlsx` and `AGORA_reactions_canon.parquet` have been recovered through privacy-preserving audits and encoded in `pipeline/graph_construction/canonicalize_graph_inputs.py`. Exact input and historical output hashes, selection and aggregation rules, production dimensions, and the audit chain are recorded in `provenance/CANONICALIZATION.md`. No sample identifiers or sample-level abundance values are stored in Git.

## Outstanding release blockers

The following provenance items must still be resolved before release:

1. add the remaining completed internal Ricci summaries and their manuscript-facing tables;
2. create the final all-results manifest, tagged release, and permanent DOI.

## Provisional manuscript Data Availability Statement

> All author-generated code supporting this study is available in the project GitHub repository and will be archived with a version-specific DOI. The source IBDMDB/HMP2 data are publicly available through the IBDMDB portal and NCBI BioProject PRJNA398089; exact products and SHA-256 checksums are recorded in the repository. External-validation sequence data are available through ENA study PRJEB42155. AGORA2 version 2.01 metabolic reconstructions are available from the Virtual Metabolic Human resource under the provider's terms. Compact derived data supporting the reported results, including locked split manifests, aggregate performance outputs, external evaluation tables, figure-source data, configurations, and checksums, will be deposited in the permanent release archive; the DOI will be inserted before submission or publication, according to the chosen repository workflow.

This statement remains provisional until the archive exists and all source-version fields above are closed.
