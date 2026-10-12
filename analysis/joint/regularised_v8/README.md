# Regularised IBD joint comparison v8

This is the small, feasibility motivated follow-up to the failed fixed M32 v7
experiment. It is a **different, explicitly regularised model**, not a new
successful result from the original unpenalised model.

## What runs

IBD versus non-IBD only. Fixed requested M=32. The first five original locked
repetitions, each with five participant grouped outer folds, give **25 joint
fits and 25 matched Ricci controls**. There is no inner search, gate sweep or
manual approval stage. The first outer pair runs alone, after which the
remaining pairs run with two workers. Completed pairs are retained on resume.

## The concessions

The joint weighted mean objective is

    CE + ||beta_R||_1 / (0.02 * sum_w) + 0.001/2 * ||beta_H||_2^2.

The H0 ridge is on the original, training-standardised H0 output coefficients.
It is not L1 sparsity. There is no alpha penalty, alpha smoothness penalty or
intercept penalty. Ricci keeps the same L1 penalty convention. Ridge strength
is fixed, not selected by held-out scores. It should not be called negligibly
small in scientific effect just because its coefficient is small.

Starting from zero head coefficients gives objective log(2). The algorithm
retains nonincreasing finite candidate objectives. Consequently every retained
H0 coefficient vector satisfies ||beta_H||_2 <= sqrt(2*log(2)/0.001), about 37.23,
up to numerical acceptance slack. This addresses the earlier gigantic free
coefficients instead of increasing precision around them indefinitely.

The second concession is an ordinary finite optimisation procedure. Full Ricci
and H0 head KKT residuals and the conditional alpha Frank-Wolfe loss-gap bound
are targeted at 1e-5. The old 1000-alternation plus 100-refit stability protocol
is NOT used. Passing these checks is local first-order numerical evidence,
not a global nonconvex optimum or a prediction-stability certificate.

There are at most 30 working-set rounds, each with at most 1000 SLSQP iterations.
Convex head/alpha substeps are bounded fallbacks. A per-fit 1800-second numerical
budget is checked during objective evaluations. Library/native calls may delay
a deadline check, so this is not an exact wall-time promise.

When a fit reaches its prescribed budget with a valid best finite fitted state,
that state is saved and evaluated, with **FINITE_BUDGET_APPROXIMATION** prominently
recorded. It is not falsely declared converged. This is the predefined finite
training algorithm for this exploratory comparison, not selective omission of
failed folds. No completed optimisation step, nonfinite scores, corrupt inputs,
invalid probabilities or incompatible checkpoints still cause a hard stop.

All seven numerical tests on the available real arrays reached the actual
convergence targets rather than the approximate-fit route. The supplied v7
review did not contain its failed outer model's arrays, so that exact first
outer fit was not replayed locally.

## Representation and validation retained

* Source-interval alpha weighting before Gaussian integration, never an output
  coordinate multiplier. Alpha is shared across classes, at least 1e-8, and sums
  to the realised interval count.
* Original death-axis integration of the log-shaped Gaussian, using five-point
  quadrature. Finite 0<d<1-1e-12 deaths retain multiplicities. Training quantiles
  and the 1e-8 minimum-gap rule determine realised intervals. Ties are merged.
* H0 centring/scaling is estimated at uniform alpha and then fixed. Ricci
  centring/scaling and sample class weights are also training-only.
* The stored source graph reference and Ricci axis remain shared upstream.
* The complete Ricci gradient is scanned after each working-set solve. Zero
  coordinates can re-enter. Numerical coefficient zeroing uses 1e-10, followed
  by re-evaluation, not a permanent feature exclusion.
* Every new outer fit starts at alpha=1 and zero head. Only checkpoints from the
  exact same fit may resume. Previous models and favourable test outcomes do
  not initialise or select a fit.
* No new outer splits are generated. The existing legacy input audit checks all
  old manifests before selecting the first five IBD repetitions. Historical
  inner memberships are audited but NO inner models are fitted here.
* Original expressions are evaluated in float64 with extended-precision
  accumulation for returned predictions. Scalers and intervals remain frozen.
  Saved-model reload must reproduce predictions. No C++ compiler is required.

## Install

Upload `joint_ibd_regularised_v8.zip` and `install_joint_regularised_v8.py` into
`~/Real_Data`, leaving the ZIP compressed, then run in the VM Bash terminal:

```bash
cd "$HOME/Real_Data" &&
python3 install_joint_regularised_v8.py joint_ibd_regularised_v8.zip &&
bash "$HOME/Real_Data/joint_ibd_regularised_v8/launch.sh"
```

The original Python environment is reused without installing or upgrading
anything. The executable symlink remains inside that virtual environment.
Previous directories and results remain unchanged. Do not run older jobs beside
this one. Package checks and 73 tests run before production preflight.

```text
Code:    ~/Real_Data/joint_ibd_regularised_v8
Results: ~/Real_Data/H0_Ricci_IBD_Regularised_v8
Log:     ~/Real_Data/logs/joint_ibd_regularised_v8.log
Session: joint_ibd_regularised_v8
```

## Monitor and resume

```bash
tail -f "$HOME/Real_Data/logs/joint_ibd_regularised_v8.log"
bash "$HOME/Real_Data/joint_ibd_regularised_v8/status.sh"
```

After tests and inputs are verified:

```text
V8 PREFLIGHT PASSED. Fixed M32, five IBD repetitions, explicit H0 ridge=0.001.
```

Resume using the same launcher. Completed pairs are checksum-verified, not
refitted. An interrupted fit resumes from its most recent completed working-set
round. Work since that checkpoint can be repeated. There is no administrative
24-hour pause or authorisation token. Keep the VM running. Closing SSH does not
stop the tmux job.

The two successful complete-evaluation statuses are
`COMPLETE_ALL_CONVERGED` and `COMPLETE_WITH_OPTIMISATION_WARNINGS`. The latter
means every planned pair was evaluated but some estimates used the finite-budget
rule. Review the warnings, residuals and loss histories before interpreting the
comparison. These statuses never imply clinical superiority or a global optimum.

Unexpected numerical, data or execution errors remain hard stops. The new review
archive includes recorded optimisation progress, not just a generic stop message.
Do not delete receipt files or manually modify coefficients to get a pass.

## Results

The main repeated summaries require all 25 pairs. Sample and participant metrics
are calculated by pooling the five test folds within each repetition. Participant
probabilities average that participant's sample probabilities. Binary ties go to
non-IBD. Mean and sample SD across the five repetitions, paired differences and
separate consensus results are provided. These SDs are not independent-cohort
confidence intervals. Log loss is reported with the same class order.

Saved model and individual prediction records remain on the VM. Return:

```text
~/Real_Data/H0_Ricci_IBD_Regularised_v8/joint_ibd_regularised_v8_review.tar.gz
~/Real_Data/H0_Ricci_IBD_Regularised_v8/joint_ibd_regularised_v8_review.tar.gz.sha256
```

This allowlisted review archive excludes training arrays, model coefficients,
participant/sample IDs and per-observation predictions. Keep the full output
folder private. A startup error may precede export. Keep the log then.

The manuscript still needs its joint subsection revised before any v8 result is
used. See `METHODS_NOTE.md`. This installer does not alter GitHub or manuscript
files, and historical v3-v7 results are not replaced.
