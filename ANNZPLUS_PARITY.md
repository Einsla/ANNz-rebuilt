# ANNz+ C++ reference and Python rebuild status

## C++ source layout

The upstream repository contains one implementation directory per activation
function: `Tanh`, `Sigmoid`, `Leaky_ReLU`, `ReLU`, `Mish`, `Softplus`,
and `SiLU`. The core files are repeated across these directories.

- `network.h/.cpp`: fully connected network, forward propagation, backpropagation
  derivatives, Jacobians, architecture and weight-file parsing.
- `train_network.h/.cpp`: training-network operations, per-pattern squared error
  and gradients with respect to weights.
- `train.h/.cpp`: training/validation data loading, normalization, variable-metric
  quasi-Newton optimization, weight decay, validation-RMS model selection and
  network-state writing.
- `test_network.h/.cpp`: model-state loading, input/output normalization and
  input-to-output Jacobians.
- `test.h/.cpp`: committee inference, committee scatter and propagated input
  measurement-error estimates.
- `util.h/.cpp`: parsing of whitespace-delimited data and parameter files.

## Confirmed C++ behavior

- Adjacent layers are fully connected; no skip connections are created.
- Hidden layers use the selected activation; output nodes are linear.
- Every non-input node has a bias weight.
- The seven activation choices are Tanh, Sigmoid, Leaky ReLU (negative slope
  0.01), ReLU, Mish, Softplus and SiLU.
- Training uses squared error with a weight-decay term and a variable-metric
  quasi-Newton/BFGS-style minimizer.
- The best validation-RMS weights are saved.
- C++ input records contain N input values, N input measurement-error values,
  and optionally the target value(s). Training parses but does not use the
  measurement-error values; committee testing uses them for uncertainty
  propagation.
- Test mode can average multiple trained networks and combine committee
  variance with propagated input measurement errors.

## Current Python implementation

- CSV/TSV catalogues are scanned with pandas chunk iteration.
- Only selected columns are converted to numeric arrays.
- Prepared arrays are yielded in bounded mini-batches and transferred to the
  selected PyTorch device; the full catalogue is not copied to GPU memory.
- Scaling is fitted on the training split only. This avoids validation leakage,
  but differs from the C++ code, which calculates its normalization using both
  training and validation rows.
- The Python model is a fully connected network with bias terms, selected
  hidden activation and linear outputs.
- Training currently uses mini-batch Adam, not the C++ quasi-Newton optimizer.
- A single checkpoint is trained and used; committee inference and propagated
  input-error estimates are not implemented yet.
- Python currently uses named CSV/TSV columns rather than the original
  whitespace-delimited ANNz+ record format.

## Parity work still required

1. Add a documented ANNz+ input adapter for measurement columns and optional
   target columns.
2. Implement a committee abstraction and the C++ uncertainty calculation.
3. Add a selectable optimizer/parity mode and compare it with the C++ training
   objective, weight decay and validation-best model selection.
4. Compare normalization and activation outputs/derivatives against C++ using
   controlled fixtures.
5. Run matched train/validation/test catalogues and report prediction and
   uncertainty differences. A successful import or training run alone does not
   demonstrate scientific equivalence.

## Prediction-quality metrics added

The evaluation report now retains MAE, RMSE and split-relative R2, and adds
external predictive Q2:

`Q2 = 1 - PRESS / sum((y_i - mean_train(y))^2)`

The denominator is referenced to the target mean learned from the training
split, rather than the mean of the evaluation split. This makes Q2 a
predictive score for validation/test rows. It is not Pearson correlation
squared and is not LOOCV; the selected validation or test split supplies the
held-out predictions.

For conventional redshift target names (`z`, `redshift`, `z_spec`,
`zspec`, `specz`, `spec_z`), evaluation also reports the three ANNz+
paper metrics using
`delta_z = (z_phot - z_spec) / (1 + z_spec)`:

- `sigma_RMS = sqrt(mean(delta_z^2))`
- `sigma_68 = (quantile_84.1(delta_z) - quantile_15.9(delta_z)) / 2`
- `outlier_fraction_percent = 100 * mean(abs(delta_z) >= 0.15)`

The current sigma_68 is an unweighted empirical percentile calculation.
These photo-z metrics are omitted for targets that are not recognized as
redshift columns. They are computed over the selected evaluation split.
