# V8 verification and limitations

The original v7 review checksum and all 17 listed payload checksums match. It
records failure of IBD repeat 1, fold 1 because training stability was not
confirmed. It contains no fitted outer arrays or detailed late optimisation
history. No finer diagnosis of that particular trajectory is inferred.

V8 deliberately changes the model by adding H0 ridge 0.001 in the weighted-mean
loss convention. It also replaces the old mandatory stability/refit protocol
with numerical first-order checks and a predefined finite-budget rule. It does
not claim to reproduce the old unpenalised predictor.

## Tests

The software suite has 73 tests. It covers source-before-integration weighting,
training-only boundaries and scales, independent finite-difference gradients,
penalty units, exact and near aliases, separable H0 examples, finite coefficient
bounds, reference probabilities against converged SAGA, save/reload, explicit
approximation labels, invalid-state refusal, original split protection, no
partial complete summaries, and privacy of review exports.

One end-to-end fixture runs all 25 synthetic joint/control pairs with two
workers. A separate test exercises the main entry point through all 25 pairs
and verifies no-refit resume. These fixtures use small tensors and reduced
numerical sizes. They are orchestration checks, not real-cohort cross-validation.

## Available real training inputs

Seven numerical fits passed their actual 1e-5 targets, with no budget
approximation, on supplied real arrays or deliberately constructed subsets:

* The full 704-row, 31-interval case with all 33,864 Ricci columns.
* Two row subsets of that case, with 530 and 410 training rows.
* Its matched full-dimensional Ricci-only control.
* A 1,056-row resampled stress design from the same data.
* A strongly imbalanced 494-row subset.
* The separately supplied 704-row, 62-interval M64 numerical case.

These are numerical tests, not independent clinical replications. No held-out
classification scores were used to select the ridge or numerical settings.
Individual predictions were used only for arithmetic and reload checks.

The first full real case reached head KKT 1.13e-7 and conditional alpha gap
4.47e-6. Its largest H0 coefficient was 0.676 and its H0 coefficient norm 1.932.
It fitted locally in about 22 seconds from its precomputed tensor. The M64
numerical stress case reached head KKT 2.70e-7 and alpha gap 3.95e-6 in about
68 seconds. These are local completed measurements, not VM forecasts. The
model is different from the old unpenalised model, so the objective values
are not presented as evidence of a better solution to the old objective.

The exact failed v7 outer-training observations were not supplied in its review
archive. Consequently that particular fit and the other actual outer folds
were not run locally. The recorded VM software is not upgraded by this
package. Local tests used Python 3.13.5, NumPy 2.3.5, SciPy 1.17.0, pandas 2.2.3
and scikit-learn 1.8.0. The VM has its already audited older versions. The
used optimisation interfaces exist in those versions, but identical numerical
paths or timings are not asserted.

## Acceptance semantics

CONVERGED means the actual head KKT and conditional alpha gap met the declared
targets. It is not a global optimality certificate or the previous coupled
stability certificate. FINITE_BUDGET_APPROXIMATION means a safe best fitted
state was retained at a declared limit without those targets being met. The
warning and residuals are exported. A complete evaluation with such estimates
is labelled COMPLETE_WITH_OPTIMISATION_WARNINGS, never all-converged.

Nonfinite data, nonfinite or invalid predictions, no completed optimisation
step, incompatible source identities, or corrupted result receipts still stop
execution. No failure is converted into a favourable clinical result.

All old source inputs, results and manuscript files are preserved. The ZIP
contains no training matrices, labels, fitted parameter arrays, individual
probabilities, compiler binaries or fonts. The native multiprecision helpers
and old solver hierarchy are not dependencies of v8.
