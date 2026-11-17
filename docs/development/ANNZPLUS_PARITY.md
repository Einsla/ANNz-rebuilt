# ANNzPlus C++ reference and Python rebuild status

## C++ source layout

The upstream repository contains one implementation directory per activation function: `Tanh`, `Sigmoid`, `Leaky_ReLU`, `ReLU`, `Mish`, `Softplus`, and `SiLU`. The core files are repeated across these directories.

- `network.h/.cpp`: fully connected network, forward propagation, backpropagation derivatives, Jacobians, architecture and weight-file parsing.
- `train_network.h/.cpp`: training-network operations, per-pattern squared error and gradients with respect to weights.
- `train.h/.cpp`: training/validation data loading, normalization, variable-metric quasi-Newton optimization, weight decay, validation-RMS model selection and network-state writing.
- `test_network.h/.cpp`: model-state loading, input/output normalization and input-to-output Jacobians.
- `test.h/.cpp`: committee inference, committee scatter and propagated input measurement-error estimates.
- `util.h/.cpp`: parsing of whitespace-delimited data and parameter files.

## Confirmed C++ behaviour

- Adjacent layers are fully connected; no skip connections are created.
- Hidden layers use the selected activation; output nodes are linear.
- Every non-input node has a bias weight.
- Seven activation choices are available: Tanh, Sigmoid, Leaky ReLU (negative slope 0.01), ReLU, Mish, Softplus and SiLU.
- Training uses squared error with a weight-decay term and a variable-metric quasi-Newton/BFGS-style minimizer.
- The best validation-RMS weights are saved.
- C++ input records contain N input values, N input measurement-error values, and optionally target value(s). Training parses but does not use the measurement-error values; committee testing uses them for uncertainty propagation.
- Test mode can average multiple trained networks and combine committee variance with propagated input measurement errors.

## Current Python implementation

- Streamlit app with CSV/TSV input and named columns.
- Chunked catalogue analysis, row validation, deterministic train/validation/test split assignment, and scaling fitted on training rows only.
- Fully connected PyTorch regression network with bias terms, configurable hidden-layer widths, seven activation choices and linear outputs.
- **Adam:** mini-batch training with a user-selected learning rate. Mini-batches are shuffled within source chunks with a deterministic epoch-specific seed.
- **L-BFGS:** PyTorch full-batch quasi-Newton training with a strong-Wolfe line search. This is conceptually similar to, but not an exact reproduction of, the C++ `vmmin()` optimiser.
- Validation MSE is monitored during training. Automatic architecture search ranks candidates by validation loss and supports early stopping; the test split is not used for model selection.
- Checkpoints preserve model weights, optimiser state, epoch history and a signature of the architecture/data-preparation settings.
- Prediction export is chunked and limits inference batch size on the selected device.
- Evaluation reports MAE, RMSE, split-relative R2 and external predictive Q2. For conventional redshift target names (`z`, `redshift`, `z_spec`, `zspec`, `specz`, `spec_z`), it also reports the three ANNz+ photo-z metrics.
- Portable project ZIPs preserve architecture, split/scaling settings, optimiser configuration, search results, training history and checkpoints.

## Important differences from C++

1. Python fits scaling on training rows only to avoid validation leakage. The C++ code calculates its normalization using both training and validation rows.
2. Python currently does not include the C++ weight-decay term.
3. Python offers Adam and PyTorch L-BFGS; it does not reproduce the exact legacy `vmmin()` implementation or all of its stopping/step-selection details.
4. Python uses named CSV/TSV columns rather than the original whitespace-delimited ANNz+ record format.
5. Input measurement-error columns are not yet used for uncertainty propagation.
6. Committee inference, committee scatter, and propagated input-error estimates are not yet implemented.
7. The split is random rather than redshift-stratified or grouped by survey field/object. The appropriate split strategy depends on the intended scientific generalisation test.
8. No K-fold or leave-one-out cross-validation is currently implemented. The validation split is used for model selection, and the test split is reserved for final assessment.
9. The Python photo-z `sigma_68` is currently an unweighted empirical percentile calculation. Weighted variants require a corresponding weighting implementation.

## Metrics

For a recognised redshift target:

`delta_z = (z_phot - z_spec) / (1 + z_spec)`

- `sigma_RMS = sqrt(mean(delta_z^2))`
- `sigma_68 = (quantile_84.1(delta_z) - quantile_15.9(delta_z)) / 2`
- `outlier_fraction_percent = 100 * mean(abs(delta_z) >= 0.15)`

Predictive Q2 is calculated as:

`Q2 = 1 - PRESS / sum((y_i - mean_train(y))^2)`

It uses the training-target mean as its reference. R2 uses the evaluated split's own mean. Q2 is not Pearson correlation squared and is not LOOCV.

The optimisation objective is mean squared error in the scaled target space. The reported MAE/RMSE are converted back to the original target units.

## Parity work still required

1. Add a documented ANNz+ input adapter for measurement columns and optional target columns.
2. Implement committee inference and the C++ uncertainty calculation.
3. Implement and validate the legacy weight-decay objective and compare it with the Python objective.
4. Compare normalization and activation outputs/derivatives against C++ using controlled fixtures.
5. Run matched train/validation/test catalogues and report prediction and uncertainty differences. A successful import or training run alone does not demonstrate scientific equivalence.
