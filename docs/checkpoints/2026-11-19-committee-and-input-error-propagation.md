# Project checkpoint — 19 November 2026

## Checkpoint identity

- Integration branch: `main`
- Feature branch merged: `feature/committee-error-propagation`
- Merge commit: `7b1634c601fac3895cc818c6dc9cdb14414024c6`
- Feature commit: `6d35fe6`
- Git tag: `checkpoint-committee-input-errors-2026-11-19`

This checkpoint records the committee prediction and input measurement-error propagation milestone in the Python ANNzPlus rebuild.

## Implemented

### Prediction uncertainty

- Single-network inference computes the model Jacobian with PyTorch autograd and propagates independent input measurement errors: `sigma_y^2 = sum_i((dy/dx_i * sigma_x_i)^2)`.
- Jacobians are converted from normalized coordinates to the original physical input and target units before propagation.
- Committee inference retains the legacy ANNzPlus method: arithmetic mean prediction, mean member Jacobian for input-error propagation, and the C++ committee scatter correction for more than two members. The scatter term is zero for one or two members.
- Committee disagreement, input-propagated uncertainty, and combined uncertainty are exported separately.
- In single-network mode, each target receives an `<target>_input_error_std` output column when at least one error mapping is supplied.

### Input measurement-error mapping

- Prediction UI exposes a mapping from each model input to its 1-sigma measurement-error column in both single-network and committee modes.
- Mapped errors must be finite, non-negative and expressed in the same physical units as their corresponding input.
- Error columns are used only for uncertainty propagation; they are not additional neural-network features and are not used in training.
- Error-column mappings are preserved in portable model bundles and project training configuration.

### Portability and documentation

- Portable `.annzmodel` export/import retains model configuration, committee metadata, evaluation information and error-column mappings.
- Project ZIP training configuration preserves prediction error mappings.
- User and developer documentation distinguishes the legacy committee implementation from the single-network Jacobian extension.

## Verification at checkpoint

The local verification run completed successfully:

- `python -m pytest -q`: **27 passed**
- `python -m ruff check .`: passed
- `python -m ruff format --check .`: passed
- `python -m compileall -q src app.py`: passed
- `git diff --check`: passed
- Streamlit application startup smoke test: passed

## Known scientific limitations and next work

1. Numerical end-to-end comparison of prediction and uncertainty outputs against controlled fixtures from the compiled C++ ANNzPlus implementation is still outstanding. Matching equations in code is not by itself proof of numerical parity.
2. Single-network Jacobian uncertainty is a mathematically consistent extension of first-order propagation; the legacy C++ implementation applies the documented input-error calculation through committee inference.
3. Input errors are assumed independent. Correlated input errors require covariance-matrix propagation.
4. The Python rebuild still differs from C++ in training-data normalization, random weight initialization and floating-point execution. The legacy weight-decay objective and full scientific parity require further work as described in the parity assessment.
5. The application upload flow holds the uploaded source file in memory, although catalogue processing and prediction are chunked.

## Resume instructions

- Start from the `main` branch at this checkpoint.
- The checkpoint tag is `checkpoint-committee-input-errors-2026-11-19`.
- Run the verification commands above before further development.
- Use `docs/development/ANNZPLUS_PARITY.md` as the authoritative list of confirmed behaviour and remaining C++ parity tasks.
