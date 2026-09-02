# Graph-input construction

This directory contains the canonical sources for sanitising AGORA2 SBML, extracting reactions, canonicalising the reaction and abundance tables, and constructing sample-specific weighted directed graphs.

## Canonicalisation

`canonicalize_graph_inputs.py` closes the two historical preprocessing gaps recorded during the publication audit. It has two explicit operations.

### Species abundances

The verified historical rule:

1. preserves the first identifier column and row order;
2. retains every terminal MetaPhlAn `s__` column;
3. additionally retains terminal `g__` columns whose label contains an underscore;
4. removes the rank prefix and retains the first two underscore-delimited tokens;
5. sums columns mapping to the same organism name; and
6. sorts the resulting organism columns lexicographically.

For the hash-locked production workbook, this maps 591 selected source columns—578 species-rank and 13 multi-token genus-rank columns—onto 508 organism columns.

```bash
python pipeline/graph_construction/canonicalize_graph_inputs.py species \
  --input /path/to/Real_Species_Abundances.xlsx \
  --output /path/to/Real_Species_Abundances_canon.xlsx \
  --provenance /path/to/species_canonicalization.json \
  --require-production-hash
```

### AGORA reactions

The reaction operation preserves row order and every field except `catalysts`. It verifies that the source `catalysts` field equals `model_id`, strips the leading `M_` from `model_id`, and retains the first two underscore-delimited tokens.

```bash
python pipeline/graph_construction/canonicalize_graph_inputs.py reactions \
  --input /path/to/AGORA_reactions.parquet \
  --output /path/to/AGORA_reactions_canon.parquet \
  --provenance /path/to/reaction_canonicalization.json \
  --require-production-hash
```

Reaction processing is streamed in bounded batches. Both operations refuse to overwrite an existing output unless `--force` is supplied. `--require-production-hash` verifies the audited source hash and expected production dimensions before accepting the result.

The generated provenance JSON reports input/output hashes, counts, software versions, and whether the bytes happen to match the historical canonical file. Excel and Parquet container bytes may differ across writer versions even when the decoded tables are semantically identical; the recovered transformations were validated at a numerical tolerance of `1e-12`.

## Tests

The synthetic regression tests require pandas, NumPy, openpyxl, PyArrow, and pytest:

```bash
python -m pytest -q pipeline/graph_construction/tests
```

The tests exercise selection, renaming, aggregation, ordering, reaction-field preservation, invariants, and production-hash rejection without using microbiome data.
