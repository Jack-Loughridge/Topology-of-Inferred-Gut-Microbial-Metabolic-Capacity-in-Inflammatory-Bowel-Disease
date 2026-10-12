# Orthant-polish solver patch (v1.2.0)

The scientific objective and validation design are unchanged.

## Numerical changes

- Correct monotone FISTA with the adaptive-restart test evaluated at the incoming extrapolated point.
- Backtracking and train-split spectral step initialization.
- Automatic transition from support identification to orthant-constrained L-BFGS-B polishing.
- Full 33,864-feature Ricci KKT verification after every polish.
- Omitted violating columns are added and re-polished; uncertified fits still fail.
- Per-fit polish call and iteration diagnostics.

## Verification completed before packaging

- Binary and multinomial probability equivalence with converged scikit-learn SAGA.
- Binary and multinomial ill-conditioned stress tests with deliberately tiny FISTA budgets.
- A 600 x 33,864 Ricci-dimensional stress fit, certified at KKT residual 1.63e-6.
- Complete synthetic execution of all five tasks, 25 outer folds, 5x3 grouped nested CV, all global outputs, and a second-pass 25-fold resume.
