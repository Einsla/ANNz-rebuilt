# Legacy ANNzPlus C++ vs Python rebuild

## Purpose and scope

This document distinguishes behaviour reproduced from the original [ANNzPlus C++ repository](https://github.com/imdadmpt/ANNzPlus) from Python implementation choices and work that still needs scientific validation. "Implemented" means the feature exists in the Python app; it does not automatically mean bit-for-bit or scientifically identical output.

## At-a-glance comparison

| Area | Legacy C++ ANNzPlus | Current Python app |
|---|---|---|
| Interface | Command-line / parameter-file workflow | Streamlit graphical interface |
| Catalogue format | Whitespace-delimited records and parameter files | CSV/TSV with named columns |
| Data loading | Legacy file-reading workflow | Chunked catalogue scanning and prediction |
| Network | Fully connected feed-forward network with biases and linear outputs | PyTorch fully connected network with biases and linear outputs |
| Hidden activations | Tanh, Sigmoid, Leaky ReLU, ReLU, Mish, Softplus and SiLU, implemented in activation-specific source directories | Same seven activation choices; each hidden layer can have its own activation |
| Architecture selection | User configures and compares network settings through legacy workflow | User describes layer widths; optimisation tests selected activation combinations while keeping the described topology fixed |
| Optimisation | C++ variable-metric vmmin-style optimiser | Adam mini-batch training or a Python port of legacy vmmin |
| vmmin activation support | Activation-specific C++ implementations exist for the seven activation variants | vmmin can train with all seven supported activations |
| vmmin memory | Dense inverse-Hessian approximation | Dense inverse-Hessian approximation, O(P^2) CPU memory for P trainable parameters |
| Adam | Not the legacy vmmin algorithm | Additional modern optimiser; mini-batch updates |
| Normalisation | Legacy training code calculates normalisation using training and validation data | Scaling statistics are fitted on training rows only to avoid validation leakage |
| Data splits | Legacy training and validation files; test workflow is separate | Deterministic train/validation/test split |
| Validation/model selection | Validation RMS is used to select saved network state | Validation loss ranks candidate configurations and checkpoints; Adam can use patience-based early stopping |
| Weight decay | Included in legacy vmmin objective and periodically updated by legacy procedure | Legacy-style periodic weight-decay update is ported in vmmin; Adam follows its own training path |
| Committee prediction | Averages predictions from multiple trained networks | Independently trains committee members and averages their predictions |
| Committee scatter | Legacy sample-variance correction; zero for one or two members | Implements the same correction |
| Input measurement errors | Input rows include measurement-error values; used by test/committee uncertainty calculation, not as extra network inputs | User maps named error columns at prediction time; errors are not added to network inputs |
| Error propagation | Uses network Jacobians and committee rules | Uses PyTorch Jacobians and propagates independent errors in quadrature; committee path follows legacy equations |
| Evaluation | Legacy photo-z outputs and metrics | MAE, RMSE, R-squared, predictive Q-squared and recognised photo-z metrics |
| Model persistence | Legacy network state/weight files | Model bundles and project ZIPs preserve weights, architecture, scaling, settings, history and available optimiser/checkpoint state |

## Optimisers in the Python app

### Adam

- Uses PyTorch Adam with mini-batch updates.
- The learning rate is user-configurable.
- The app automatically selects CUDA when available and otherwise uses CPU.
- Adam is the practical default for larger networks because it does not allocate a dense P-by-P inverse-Hessian matrix.
- Adam is an additional Python option, not a reproduction of the legacy vmmin optimiser.

### Legacy vmmin port

- Implements the legacy variable-metric approach in Python rather than using `torch.optim.LBFGS`.
- Uses PyTorch autograd to calculate network gradients and NumPy float64 for the dense inverse-Hessian state.
- Evaluates the full training objective and gradient by streaming through catalogue chunks. Chunk streaming limits memory use; it is not mini-batch optimisation. Parameters are updated after a full objective/gradient evaluation.
- Includes the dense inverse-Hessian update, line search, reset/convergence rules and periodic legacy weight-decay update.
- Supports the seven activation functions listed above. The original C++ project has separate activation-specific implementations; vmmin is not limited to Tanh.
- Memory for the dense inverse-Hessian matrix grows approximately as 8P^2 bytes (before other process/model memory), so large networks can exceed the safety limit. Adam is recommended when P is large.
- Validation is recorded for model selection, but vmmin uses its own convergence rules rather than Adam-style patience-based early stopping.

## Important scientific differences and limitations

1. **Normalisation:** Python fits scaling on training rows only. Legacy C++ uses training plus validation data when calculating normalisation. This can change predictions even when architecture and weights otherwise match.
2. **Initialisation and numerical execution:** Python and C++ initial weight generation, tensor operations and floating-point execution are not yet demonstrated to match exactly.
3. **Input representation:** Python uses named CSV/TSV columns and explicit mappings from each input to its 1-sigma error column. The legacy format uses a fixed record layout containing inputs, input errors and optional targets.
4. **Uncertainty assumptions:** Python's input-error propagation assumes independent input errors. Correlated measurement errors would require a covariance matrix. Single-network Jacobian propagation is a first-order extension; the committee equations follow the legacy implementation.
5. **Data splitting:** Python currently uses deterministic random splitting, not survey-field/object grouping or a redshift-stratified strategy. The best scientific split depends on the generalisation question.
6. **Metrics:** Python reports predictive Q-squared using the training-target mean as the reference denominator. Its R-squared uses the evaluated split's own mean. Predictive Q-squared is not LOOCV Q-squared and is not Pearson correlation squared.
7. **Parity evidence:** Passing software tests confirms code paths and regression behaviour, not scientific equivalence. Matched C++/Python fixtures and catalogue-level comparisons are still required.

## What is already implemented

- Chunked CSV/TSV analysis and prediction.
- Automatic CUDA/CPU selection.
- Configurable fully connected architectures and per-hidden-layer activations.
- Adam and legacy vmmin training paths.
- Activation-combination search with the described hidden-layer count and widths held fixed.
- Train/validation/test workflow, metrics and model evaluation.
- Committee training, mean prediction and legacy committee-scatter calculation.
- Named input-error mapping and Jacobian-based uncertainty propagation.
- Model bundles and project ZIP persistence.

## Remaining parity work

- Build controlled input/output fixtures from the compiled C++ implementation for every activation and compare forward values and derivatives.
- Compare vmmin objective, gradient, line-search decisions, convergence and weight-decay updates against C++ on small deterministic networks.
- Match or explicitly quantify normalisation and initialisation differences.
- Compare committee predictions, committee scatter and propagated input uncertainties against C++ fixtures.
- Run matched real catalogues and report prediction residuals and uncertainty differences.
- Document any intentional Python-only behaviour separately from legacy-compatible behaviour.

## Primary reference

- [ANNzPlus C++ source repository](https://github.com/imdadmpt/ANNzPlus)
- Detailed implementation/parity notes: [ANNZPLUS_PARITY.md](ANNZPLUS_PARITY.md)
