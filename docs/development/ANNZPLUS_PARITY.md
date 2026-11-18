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
- Fully connected PyTorch regression network with bias terms, configurable hidden-layer widths, seven activation choices assignable independently to each hidden layer, and linear outputs.
- **Adam:** mini-batch training with a user-selected learning rate. Mini-batches are shuffled within source chunks with a deterministic epoch-specific seed.
- **legacy vmmin:** A Python translation of the original dense variable-metric update, line-search acceptance, reset rules and periodic weight-decay hyperparameter update in Tanh/train.cpp.
- Validation MSE is monitored during training. The optimiser keeps the user-described layer count and neuron widths fixed, exhaustively tests the selected activation-function combinations independently across hidden layers, and ranks candidates by validation loss. Adam candidates support early stopping; the test split is not used for model selection. The search is capped at 24 candidates.
- Checkpoints preserve model weights, optimiser state, epoch history and a signature of the architecture/data-preparation settings.
- Prediction export is chunked and limits inference batch size on the selected device.
- Committee training creates independently initialised members using the same architecture and fixed data splits; each member retains its best validation-loss checkpoint. Committee predictions are the arithmetic mean of member predictions.
- Committee inference reproduces `Tanh/test.cpp`: for more than two members it uses the sample-variance correction `M/(M-1) * (mean(y_member^2) - mean(y_member)^2)`; for one or two members the committee scatter term is zero, as in the C++ implementation.
- Input measurement errors can be mapped to model input columns at prediction time in both single-network and committee modes. For a single network, PyTorch autograd computes its Jacobian and propagates independent input errors in quadrature. For a committee, the implementation converts member Jacobians to original input/output units, averages the Jacobians, and applies the legacy ANNz+ propagation rule. Error columns are never added to the neural-network input layer.
- Single-network prediction exports the propagated input uncertainty for each target. Committee prediction exports committee disagreement, propagated input uncertainty, and their quadrature total as separate columns.
- The committee uncertainty path follows the original C++ Tanh/test.cpp implementation. Single-network Jacobian propagation is a mathematically valid extension of the same first-order error-propagation principle; the original C++ committee code is the reference for the ensemble path.
- Error-column mappings are stored in the model bundle and restored when the model is loaded. Committee checkpoints are embedded in the existing .annzmodel bundle, so a saved committee can be reloaded for prediction without retraining.
- Evaluation reports MAE, RMSE, split-relative R2 and external predictive Q2. For conventional redshift target names (`z`, `redshift`, `z_spec`, `zspec`, `specz`, `spec_z`), it also reports the three ANNz+ photo-z metrics.
- Portable project ZIPs preserve architecture, split/scaling settings, optimiser configuration, search results, training history and checkpoints.

## Important differences from C++

1. Python fits scaling on training rows only to avoid validation leakage. The C++ code calculates its normalization using both training and validation rows.
2. Python currently does not include the C++ weight-decay term.
3. Python now ports the legacy vmmin update and line-search procedure. End-to-end parity still depends on matching data normalisation, weight initialisation, objective conventions and floating-point execution.
4. Python uses named CSV/TSV columns rather than the original whitespace-delimited ANNz+ record format.
5. The Python implementation accepts named input/error-column mappings at prediction time rather than the original fixed-width whitespace record layout. Input errors are assumed independent; correlated measurement errors would require a covariance matrix. The single-network Jacobian path is an extension; the committee path follows the C++ ensemble equations.
6. Committee inference and uncertainty propagation now follow the original `Tanh/test.cpp` equations. Numerical end-to-end parity against compiled C++ fixtures is still outstanding.
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

## Current integrated checkpoint (19 November 2026)

The committee and input measurement-error propagation work is merged into `main` at commit `7b1634c`. Single-network Jacobian propagation and committee uncertainty are both available in prediction mode, with named input-to-error-column mapping. Mappings are persisted in model bundles and project configuration. The full milestone, verification results and resume instructions are recorded in [the dated checkpoint](../checkpoints/2026-11-19-committee-and-input-error-propagation.md).

The automated suite currently passes 27 tests, with Ruff lint/format checks and Python compilation also passing. This confirms software regression coverage, not numerical scientific parity with C++.

## Parity work still required

1. Add a documented ANNz+ input adapter for measurement columns and optional target columns.
2. Validate committee and uncertainty outputs against controlled fixtures from the compiled C++ implementation.
3. Implement and validate the legacy weight-decay objective and compare it with the Python objective.
4. Compare normalization and activation outputs/derivatives against C++ using controlled fixtures.
5. Run matched train/validation/test catalogues and report prediction and uncertainty differences. A successful import or training run alone does not demonstrate scientific equivalence.


## Legacy vmmin Python port (feature branch)

The Python implementation ports the algorithm in Tanh/train.cpp rather than
using torch.optim.LBFGS. The translated procedure includes:

- Dense symmetric inverse-Hessian approximation, initialised to identity.
- Search direction -Gg and the original step acceptance test (acctol=1e-4).
- Step-length backtracking by 0.2.
- The source's weight-change test (reltest=10), objective tolerances, reset
  behaviour and periodic reset after more than 2 * number_of_weights steps.
- The lower-triangle BFGS inverse-Hessian update.
- The original periodic (every >29 accepted iterations) "quick and dirty"
  weight-decay update: alpha = number_of_weights / sum(weights**2) and
  beta = number_of_training_rows / sum_squared_training_residuals.

The objective and gradient are
E = 0.5 * beta * sum_squared_residuals + 0.5 * alpha * sum(weights**2) and
grad(E) = 0.5 * beta * grad(sum_squared_residuals) + alpha * weights.
The implementation uses PyTorch autograd for network derivatives and NumPy
float64 for the legacy dense variable-metric matrix.

### Important constraints and parity gaps

- This is a port of the Tanh implementation. The UI restricts vmmin optimisation
  candidates to tanh in every hidden layer. Adam retains all supported activations.
- The inverse-Hessian approximation is dense, requiring O(P**2) CPU memory for
  P network weights. A safety limit prevents excessive matrix allocation.
  Adam is recommended for larger networks.
- Training rows are streamed in bounded chunks for each complete objective and
  gradient evaluation. This is not mini-batch optimisation: weights change
  only after a full training-set objective/gradient has been evaluated.
- Legacy vmmin uses its own convergence/reset rules; validation loss is
  recorded for model selection but does not drive patience-based early stopping.
- Python data preparation currently fits scaling on training rows, while C++
  normalise() computes normalisation using both training and validation sets.
  Initial weight generation and floating-point execution also differ.
  Therefore this is an algorithm/objective port, not end-to-end parity.
- The periodic hyperparameter update changes the objective during optimisation;
  this behaviour is retained, including objective reevaluation.
- Existing pre-port quasi-Newton checkpoints remain readable for model weights, but continuing
  with vmmin starts a fresh variable-metric state.
