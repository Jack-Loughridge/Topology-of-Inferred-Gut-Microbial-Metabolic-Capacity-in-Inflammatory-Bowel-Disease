# Methodology

## H0 block

For sample diagram deaths `d > 0`, the method works in `log(d)` space. Within each training split, pooled training deaths define equal-mass quantile intervals. No validation or test deaths enter the interval construction.

For each interval, the unweighted WKPI basis integrates the sample's log-Gaussian mixture over that interval using five-point Gauss-Legendre quadrature. The training basis is standardised, and the same training transformation is applied to validation/test samples.

The learned alpha vector is positive and normalised to mean one. This removes arbitrary scale non-identifiability while preserving an interpretable relative weighting over H0 death regions.

## Ricci block

The classifier loads `feature_matrix_B_K0.npz` from the faithful `epsilon_dist=0.0001`, `n=250`, v3 run. The first half is the B active-edge block and the second half is the K0 curvature block. Absent edges remain zero. Sparse standardisation is fitted inside each training split.

The original-style all-task output directory is not used as a replacement feature matrix. Its coefficient tables are scanned only for reaction/process annotations that can be merged by `feature_index`.

## Joint sparse head

With fixed alpha, the design is

`[alpha * standardised_WKPI | standardised_B_K0]`.

The objective contains separate H0 and Ricci L1 penalties. The implementation scales the blocks before fitting one SAGA L1 logistic regression:

`[X_H / lambda_H | X_R / lambda_R]`.

If the fitted scaled coefficients are `theta_H` and `theta_R`, the coefficients in original units are

`beta_H = theta_H / lambda_H` and `beta_R = theta_R / lambda_R`.

Consequently the single L1 norm is exactly

`lambda_H ||beta_H||_1 + lambda_R ||beta_R||_1`.

## Alpha update

With the sparse logistic head fixed, alpha is updated against inverse-frequency weighted cross-entropy plus the exact first-difference penalty `gamma * sum((alpha[j+1] - alpha[j])^2)`. The positive mean-one constraint is imposed through an exponentiated normalised parameterisation. The alpha step has an analytic gradient and is solved deterministically with L-BFGS-B. The logistic and alpha steps alternate six times, followed by a final logistic refit for the saved alpha.

## Nested grouped validation

The outer loop uses five shuffled `StratifiedGroupKFold` splits with seed 13. Groups are participant IDs. Within each outer-training set, three grouped inner folds select:

- interval count: 96, 160 or 224;
- H0 L1 multiplier: 0.5, 1 or 2;
- Ricci L1 multiplier: 0.5, 1 or 2.

The default selection metric is ROC-AUC. Binary AUC uses the task's stated positive class. Three-way AUC is macro one-vs-rest. Accuracy, balanced accuracy and macro F1 are always reported.
