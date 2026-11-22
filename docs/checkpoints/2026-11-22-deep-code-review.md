# Project checkpoint — 22 November 2026

## Scope

Deep review of the Python ANNzPlus application and the review fixes originally
developed on `feature/separate-network-structure-activation-search`. The work
was subsequently merged into `main` as part of the v2.0.0 release. This
checkpoint records the review fixes and verified behaviour; it does not claim
scientific equivalence with the original C++ implementation.

## Review findings and changes

### Training checkpoint consistency

- Adam and legacy vmmin select the model weights associated with the lowest
  validation loss when a validation split is available.
- The selected weights are now packaged with their matching optimiser state.
  For vmmin, the matching periodic weight-decay alpha/beta values are retained.
- The checkpoint's completed epoch/iteration counter and embedded history retain
  the entire completed run, even when the selected weights came from an earlier
  best-validation step. Continuation therefore uses the next step number and
  does not lose the previous run's history.
- The best-validation step is stored separately in checkpoint metadata and
  survives subsequent continuation.
- The vmmin iteration callback now runs after the periodic hyperparameter update,
  so a selected checkpoint records the current alpha/beta values.

### Application and maintenance

- Removed an unreachable duplicate optimiser error branch after optimiser
  validation in the training dispatcher.
- Removed a redundant activation-choice check in the optimisation UI.
- Corrected the vmmin optimisation summary so it no longer describes an
  Adam-style learning-rate search.
- Replaced corrupted page-icon and preview-separator text with explicit Unicode
  escapes; corrected the project ZIP version range in the local-app guide.
- Updated the development and user documentation to describe the checkpoint
  selection and continuation behaviour accurately.

## Verification

- Full automated test suite: 40 passed.
- Ruff lint: passed.
- Ruff format check: passed.
- Python compilation: passed.
- Streamlit source parse smoke test: passed.
- Git whitespace/diff check: passed.

These checks demonstrate software regression coverage. They do not prove
numerical parity with compiled C++ ANNzPlus. Controlled C++/Python comparisons
for activations, vmmin, normalization, initialisation, predictions and
uncertainty propagation remain outstanding.

## Branch and release history

- At the time of the review, the work was on
  `feature/separate-network-structure-activation-search`.
- Review commit: `35edc65` (`Fix resumable best-model checkpoints and review app`)
  was included in the subsequent release branch history.
- The feature was later merged into `main` in PR #2, titled
  `ANNzPlus Python v2.0.0`.
- Repository state checked on 22 November 2026: local `main` and
  `origin/main` both point to `7286f1e` (release merge commit), tagged
  `v2.0.0`. The feature branch remains available as historical development
  context; it is not the current checked-out branch.
