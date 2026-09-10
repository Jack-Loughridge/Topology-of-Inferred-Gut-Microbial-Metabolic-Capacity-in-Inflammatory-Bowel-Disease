# Graph-diagnostics source recovery and clustering correction

## Scope

This record addresses two repository-audit findings. It does not change graph
construction, H0 persistence, Ricci curvature, classification, or external
validation.

## Sequential routing-attack analysis

The complete historical generator was recovered as
`~/Real_Data/sequential_routing_attack.py` (SHA-256
`5fd6c2ffc3332965f55cb65e3ec42b7ddf9ec8d30aa3d795014d59107f78edde`).
Its production run depended on the GraphML parser in
`~/Real_Data/path_backbone_longrange_diagnostics.py` (SHA-256
`7178c33bd17f1eea46624d6ac48bfca210f2967a6aec524df0abc711836943e9`),
which was outside the repository. The publication implementation embeds the
validated parser and removes that external runtime dependency without changing
the analysis protocol.

The recovered condition summary has SHA-256
`5932016b519b60d1f9361b692bccc36bf94901742d4e03214221e717a478f57d`.
The recovered combined LaTeX output has SHA-256
`1e43b235a36c06677791cffe7d766fd6c294f69d08c68302a0a1a00f27281966`
and matches the four sequential-removal tables in the manuscript. Those table
values therefore do not require numerical replacement.

The run matched 1,317 graphs and 106 participants. Eleven graphs had no active
edge and were recorded in `failed_samples.csv`; 1,306 graphs were analyzed and
all 106 participants remained represented. This must be disclosed in the
Methods/results caption or supporting text.

## Directed clustering

The attempted participant-average output at
`~/Real_Data/Ricci_Summary/directed_clustering_active_participant_average_summary.csv`
(SHA-256
`49081e1b10f1c9349423bbc98d4ad355c2a36e8b35a65d4d2728334fefb64efb`)
is invalid: all rows report zero active vertices and zero active edges and all
clustering coefficients are missing.

Two independent defects caused or affected that attempted replacement:

1. its streaming GraphML parser cleared each `data` element before processing
   the parent `edge`, so it recovered no weights; and
2. its sparse Fagiolo implementation divided the triangle numerator by two and
   also used the conventional denominator containing the factor of two. This
   returns half the correct coefficient (for example, 0.5 instead of 1.0 for a
   complete bidirected triangle).

The canonical implementation fixes both defects, uses active vertices only,
uses raw strength `s = 1 - w`, and performs the manuscript-specified
sample-to-participant aggregation. The manuscript clustering table must be
regenerated from this implementation before release. Participant-union and
participant-mean graph outputs estimate different quantities and must not be
used as replacements.

## Provenance policy

The original import manifest is retained unchanged because it records the
source freeze. New canonical diagnostic sources and their final outputs should
be recorded in the final release/result manifest after the corrected
clustering run has been reviewed.
