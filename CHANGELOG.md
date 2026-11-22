# Changelog

All notable user-facing changes to ANNzPlus Python Rebuild are recorded here.

## [2.0.0] - 2026-11-22

Version 2 is a major feature release integrating the committee/error-propagation
work and the subsequent training, optimisation, and checkpoint improvements.

### Added
- Committee training with independently initialised network members and
  averaged predictions.
- Committee disagreement, input-measurement-error propagation through network
  Jacobians, and combined prediction uncertainty.
- Named mapping of input variables to their 1-sigma measurement-error columns.
- Seven hidden activation functions supported by both Adam and legacy vmmin:
  Tanh, Sigmoid, Leaky ReLU, ReLU, Mish, Softplus, and SiLU.
- Legacy vmmin optimiser port with dense variable-metric updates, line search,
  reset rules, and periodic weight-decay parameter updates.
- Reusable model bundles and project ZIP workflows for saving and restoring
  architectures, preprocessing settings, training history, and checkpoints.

### Improved
- Training validation is shared consistently by Adam and vmmin.
- Best-validation weights are stored with their matching optimiser state;
  complete training history and monotonic continuation counters are preserved.
- vmmin checkpoints retain the weight-decay alpha/beta values associated with
  the selected weights.
- Optimisation keeps the user-described network topology fixed while searching
  supported activation combinations; Adam additionally searches learning rates.
- Prediction, evaluation, and catalogue processing use bounded chunks/batches.
- Windows setup and launch documentation describes automatic CUDA detection
  and CPU fallback.

### Compatibility and scientific notes
- Model bundle format version 2 is supported, with backward loading of format
  version 1. Project ZIP formats 1 through 5 are supported.
- This release does not claim bit-for-bit or end-to-end scientific parity with
  the original C++ ANNzPlus. Differences in normalization, initialisation,
  floating-point execution, and numerical end-to-end comparisons remain
  documented in [the parity assessment](docs/development/ANNZPLUS_PARITY.md).
- vmmin uses a dense inverse-Hessian approximation with O(P^2) CPU memory;
  Adam remains the recommended optimiser for larger networks.
