# Changelog

## 1.2.0

- Added exact orthant-constrained L-BFGS-B polishing after active-set FISTA support identification.
- Retained full-feature KKT certification and exact block-L1 objective.
- Added polish diagnostics and end-to-end five-task nested-CV verification.

## 1.0.0 — 2026-07-14

- Recreated the final joint train-only WKPI + faithful Ricci sparse classifier.
- Added nested participant-grouped validation for all five disease tasks.
- Added separate H0 and Ricci L1 penalties through exact block scaling.
- Added alternating positive mean-one alpha learning and final-head refitting.
- Added canonical input audits, sparsity diagnostics, coefficient tables, alpha profiles, confusion matrices and LaTeX outputs.
- Added tmux launch/monitor scripts, packaging metadata, CI and tests.
