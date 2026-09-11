# Canonical graph-input provenance

## Status

The two transformations previously recorded as publication blockers are now specified by executable code in `pipeline/graph_construction/canonicalize_graph_inputs.py` and protected by data-free regression tests.

No source or canonical human-microbiome tables are stored in Git.

## Audited production files

| Role | Filename | Rows | Columns | SHA-256 |
|---|---:|---:|---:|---|
| Species source | `Real_Species_Abundances.xlsx` | 1,360 | 933 | `991087eb7105b85ad60b0917fae895bb966ab4463709de092f7297fda6d22cb6` |
| Historical species canonical | `Real_Species_Abundances_canon.xlsx` | 1,360 | 509 | `087a5bd0dd001906b348fdedd433d911edad73fef3bde5af47ca0a0fef5cee69` |
| Reaction source | `AGORA_reactions.parquet` | 12,582,264 | 6 | `799fe755c0c9ede07572742442a799a8de1ff4e7165f6ecb2812b6699955cf77` |
| Historical reaction canonical | `AGORA_reactions_canon.parquet` | 12,582,264 | 6 | `ec906cd5f08448991b3958112fd4367f17141e0bbbe7d0333c8382a5c36af755` |

The first column of both abundance workbooks is `External ID`. Its sequence and row order are unchanged. Identifiers and sample-level abundance values were not exported from the audit environment.

## Recovered species rule

For each abundance header, take the terminal taxonomy component after the final `|`.

- retain every terminal species-rank (`s__`) component;
- retain a terminal genus-rank (`g__`) component when its label contains an underscore;
- remove the rank prefix;
- retain the first two underscore-delimited tokens;
- sum source columns that yield the same name;
- sort names lexicographically; and
- preserve the identifier column and row order.

For the audited input this selects 578 species-rank and 13 multi-token genus-rank columns. After aggregation, the output contains 508 organism columns plus `External ID`.

The independent selection-variant audit found that this rule reproduced the complete historical canonical table with zero changed cells at `rtol=1e-12, atol=1e-12`. Alternative rules based on species-only selection or only adding genus labels ending `_unclassified` did not reproduce the table.

## Recovered reaction rule

The source `catalysts` sequence is identical to `model_id`. The canonical table:

1. preserves all six columns and all 12,582,264 rows in their original order;
2. preserves `model_id`, `species_file`, `reaction_id`, `inputs`, and `outputs` exactly; and
3. replaces `catalysts` by removing a leading `M_` from `model_id` and retaining its first two underscore-delimited tokens.

All 7,302 distinct source model identifiers followed this rule, producing 1,369 canonical catalyst names.

## Audit chain

The privacy-preserving audit archives were checked before these rules were encoded:

| Audit archive | SHA-256 | Purpose |
|---|---|---|
| `publication_canonicalization_audit_20260902T104711Z.tar.gz` | `9ac4af89613e2ffeb9a121a732720b5a15e5b30529cb126ce926f93e49d2ced5` | File, schema, row-key, positional, and text-change inventory |
| `publication_canonicalization_followup_20260902T111826Z.tar.gz` | `bc7ebf54f7edcfa95ee8297061d376bd865e05a2e795798d0eedbb2615f5590d` | Name-aware species-rank comparison |
| `publication_canonicalization_full_rule_20260902T113213Z.tar.gz` | `64f1907e676dfd01638212443d864579fb745efac935d84b90d2312e4f2a4770` | AGORA-vocabulary hypothesis and aggregation check |

The final selection-variant result was generated on 2026-09-02 and identified `species_plus_all_multitoken_genus` as an exact candidate at `1e-12`. Its compact report should be retained with the final private audit record and included in the release provenance archive if its taxonomy-only contents pass the project disclosure review.

## Byte identity and semantic identity

Excel and Parquet writers can encode metadata, compression, and floating-point summation order differently across software versions. Consequently, a newly generated container need not have the same file SHA-256 as the historical canonical file. The production script records both hashes. Acceptance requires the audited input hash, expected dimensions and counts, identical identifiers and column order, and numerical equality at `1e-12`; historical byte identity is reported but is not assumed.
