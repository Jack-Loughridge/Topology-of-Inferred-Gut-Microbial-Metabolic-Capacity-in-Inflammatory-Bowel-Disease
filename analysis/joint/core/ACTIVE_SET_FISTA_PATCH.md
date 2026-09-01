# Active-set FISTA convergence patch

Version 1.1.1 corrects the restricted logistic optimizer without changing the
statistical model or validation design.

## Corrected numerical behavior

- Fixes an adaptive-restart expression that restarted every accelerated step.
- Uses the incoming FISTA extrapolated point in the gradient-restart test.
- Uses a 30-step split-specific spectral Lipschitz estimate.
- Uses monotone FISTA: an objective-increasing accelerated proposal is rejected
  and recomputed from the last accepted point.
- Uses backtracking as the correctness guard and cautiously expands accepted
  local step sizes.
- Evaluates true restricted KKT residuals periodically and retains the existing
  full-feature KKT acceptance test.
- Adds a deterministic regression test that fails if acceleration is
  effectively disabled again.

No folds, tasks, features, penalties, alpha updates, grids, or selection rules
were changed.
