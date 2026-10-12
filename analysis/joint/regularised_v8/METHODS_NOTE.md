# Proposed Methods wording for the v8 analysis

This note describes the implemented procedure, not a claim that the production
25-pair evaluation has already finished. It is not applied automatically.

**We defined an exploratory joint comparison for IBD versus non-IBD using a fixed
requested resolution of 32 intervals and the first five locked repetitions of
five participant grouped outer folds. The coarser resolution and reduced
schedule were selected for numerical feasibility rather than favourable held
out performance. Interval boundaries, bandwidth, centring, scaling and class
weights were fitted using the relevant outer training observations. Alpha
weighted each death's Gaussian contribution according to its source interval
before integration on the original death axis. Its entries were at least
10^-8 and had mean one. The two feature blocks were combined in one fitted
binary logistic model. There was no inner resolution search.**

**To control the large coefficient values encountered without H0 regularisation,
we added a fixed quadratic penalty on the standardised H0 coefficients. The
objective was the class balanced mean cross entropy plus
||beta_R||_1/(0.02 W) + 0.001 ||beta_H||_2^2/2, where W was the sum of the training
sample weights. Intercepts and alpha were not penalised. This was a regularised
complementarity analysis, not a completed refit of the original unpenalised
model. The H0 ridge does not enforce sparsity and its coefficient was not
selected using held out accuracy.**

**The head and alpha were optimised jointly on a working set of Ricci
coordinates, with all Ricci gradients checked before expanding or exchanging
that set. Safeguarded head and alpha updates supplied additional numerical
refinement when needed. The target head KKT residual and conditional alpha loss
gap were both 10^-5. These are numerical local criteria rather than a proof of
a global optimum. The preceding 100-refit stability procedure was not used.
Fitting was bounded by 30 working-set rounds and the recorded computation
allowance. A valid best finite state retained at a prescribed limit was labelled
an approximate finite-budget fit rather than declared converged. Nonfinite or
invalid models were not scored, and no fold was omitted to create a complete
performance summary.**

**A Ricci only control used exactly the same outer observations, transformations,
class weights, C and head criterion, without the H0 block. Within each
repetition, held out probabilities were pooled across folds and averaged within
participant before participant level scoring. Metrics were summarised as mean
and sample SD across the five repetitions, with joint-minus-control differences
formed within matching repetitions. Consensus results after averaging across
repetitions remained separate. The existing shared source graph reference was
not reconstructed within folds.**

Complete this note with the achieved fit counts, residuals and any optimisation
warnings after reviewing the actual output. Do not describe v8 as unpenalised,
nested resolution selection, 20 repetitions, or the old 100-refit method.
