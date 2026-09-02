# Data and code availability

## Policy objective

The final release will make all author-generated code underlying the reported findings publicly accessible and will provide the compact data products needed to verify every table and figure. Raw public data and third-party reconstructions will remain in their original repositories and will be accompanied by acquisition and integrity instructions.

## Primary source cohort

The source cohort is the Inflammatory Bowel Disease Multi'omics Database / integrative Human Microbiome Project (IBDMDB/HMP2).

- Study portal: <https://ibdmdb.org/>
- NCBI BioProject: `PRJNA398089`
- Study publication: Lloyd-Price et al., *Nature* (2019), DOI `10.1038/s41586-019-1237-9`

The Git repository does not redistribute raw human sequence data or the complete source metadata tables. The final release must state precisely which public IBDMDB products were downloaded, their download date or release identifier, and the checksums of the files entering the pipeline.

## External validation cohort

The independent external cohort is the Serrano-Gomez shotgun-metagenomic IBD study.

- ENA study used by the acquisition scripts: `PRJEB42155`
- Study publication: Serrano-Gomez et al., *Computational and Structural Biotechnology Journal* (2021), DOI `10.1016/j.csbj.2021.11.037`
- Download script: `external_validation/serrano_gomez/scripts/download_ge50_fastqs.sh`
- Integrity checker: `external_validation/serrano_gomez/scripts/check_ge50_md5.py`

The final derived-data archive must include the exact selected-run manifest, sample-to-participant mapping used for evaluation, inclusion/exclusion reasons, source URLs, and source MD5 values. It must not include information prohibited by the source repository's terms.

## Metabolic reconstruction resource

The graph-construction pipeline uses AGORA2 genome-scale microbial metabolic reconstructions obtained separately from the Virtual Metabolic Human resource.

- Resource: <https://www.vmh.life/>
- AGORA2 description: Heinken et al., *Nature Biotechnology* (2023), DOI `10.1038/s41587-022-01628-0`
- Reaction extraction: `pipeline/graph_construction/extract_agora_reactions.py`
- SBML sanitisation: `pipeline/graph_construction/sanitize_agora_sbml.py`

AGORA2 files are not redistributed here. Before release, the exact AGORA2 version, downloaded archive name, acquisition date, archive checksum, and applicable terms must be recorded.

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

1. pin and hash the exact public IBDMDB input files and AGORA2 distribution;
2. create the final derived-results manifest and permanent DOI.

## Provisional manuscript Data Availability Statement

> All author-generated code supporting this study is available in the project GitHub repository and will be archived with a version-specific DOI upon acceptance. The source IBDMDB/HMP2 data are publicly available through the IBDMDB portal and NCBI BioProject PRJNA398089. External-validation sequence data are available through ENA study PRJEB42155. AGORA2 metabolic reconstructions are available from the Virtual Metabolic Human resource under the provider's terms. Compact derived data supporting all reported results, including locked split manifests, out-of-fold predictions, external predictions, summary tables, figure-source data, configurations, and checksums, will be deposited in the permanent release archive; the DOI will be inserted before publication.

This statement remains provisional until the archive exists and all source-version fields above are closed.
