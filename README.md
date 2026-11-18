# ANNzPlus Python Rebuild

## Project overview

This project investigates and rebuilds the legacy [ANNzPlus](https://github.com/imdadmpt/ANNzPlus) photometric-redshift workflow in Python. It provides a local Streamlit interface for catalogue analysis, neural-network architecture design, training, evaluation, prediction, and reusable project archives.

This is a research rebuild, not a claim of bit-for-bit or scientific equivalence with the original C++ implementation. See [the parity assessment](docs/development/ANNZPLUS_PARITY.md) for confirmed differences and remaining work.

## Install on Windows

Requirements: Windows 10/11 (64-bit), Python 3.9+, and internet access for initial setup. An NVIDIA GPU is optional.

1. Download the repository using GitHub **Code → Download ZIP**, or clone it.
2. Extract the complete folder.
3. Double-click `Install_ANNzPlus.bat` in the repository root.
4. Wait for the isolated `.venv`, PyTorch, and application dependencies to install.
5. Launch **ANNzPlus Python** from the Desktop/Start Menu shortcut, or double-click `Run_ANNzPlus.bat`.

The installer detects an NVIDIA GPU using `nvidia-smi`. It installs a CUDA 12.8 PyTorch wheel when detected and a CPU-only wheel otherwise. Large, hardware-specific PyTorch wheels are downloaded during installation rather than committed to GitHub.

See [installation/README.md](installation/README.md) and the [detailed Windows guide](installation/windows/README.md) for manual setup and troubleshooting.

## Manual installation

From the repository root:

```powershell
py -3 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe installation\windows\install.py
.venv\Scripts\python.exe -m streamlit run app.py
```

For development:

```powershell
.venv\Scripts\python.exe -m pip install -e ".[dev]"
.venv\Scripts\python.exe -m pytest
.venv\Scripts\python.exe -m ruff check .
```

## Application workflow

1. Upload a CSV or TSV galaxy catalogue with a header row.
2. Select input variables and target variable(s).
3. Scan the catalogue in chunks for missing, non-numeric, infinite, and finite values; review summary statistics and exploratory correlations.
4. Prepare deterministic training, validation, and test splits. Invalid selected rows are excluded, and scaling is fitted on training rows only.
5. Describe the network by choosing hidden-layer widths and an activation function.
6. Optimise candidate architectures using validation loss. Automatic and manual candidate layouts are supported.
7. Train or continue with Adam mini-batches or legacy vmmin. Validation loss ranks candidate networks; patience-based early stopping applies to Adam, while vmmin follows its own legacy convergence rules.
8. Evaluate using MAE, RMSE, predictive Q-squared, R-squared, and ANNz+ photo-z metrics for recognised redshift targets.
9. Generate predictions for another catalogue and export them as CSV. Map each input variable to its 1-sigma error column to export Jacobian-propagated input uncertainty. Committee mode additionally reports committee disagreement and the combined uncertainty.
10. Save a reusable settings ZIP or a full project ZIP with optional source data.

The test split is held out from architecture and hyperparameter selection. It should be used for final assessment after model choices have been fixed.

## Optimisers

- **Adam:** mini-batch gradient optimisation with an explicit learning-rate setting. Each epoch passes through the full training split.
- **Legacy vmmin:** Python port of the original ANNzPlus Tanh/train.cpp variable-metric optimiser, including its dense inverse-Hessian update, line search, reset rules and periodic weight-decay update. It uses the full training-set objective and gradient but streams the catalogue in chunks between evaluations. It is restricted to tanh hidden activations and has O(number_of_weights**2) CPU-memory cost. Use smaller networks; Adam is recommended for larger catalogues or architectures.

The vmmin port is not yet end-to-end scientific parity: data normalisation, random initialisation and floating-point execution still differ from C++. See the parity assessment in docs/development/ANNZPLUS_PARITY.md.

## Repository map

- `app.py`: Streamlit application entry point.
- `src/annz_rebuilt/`: reusable application and scientific-computing modules.
- `tests/`: automated regression and workflow tests.
- `docs/user-guide/`: local usage and project workflow documentation.
- `docs/development/`: implementation notes and C++ parity assessment.
- `docs/research/`: research notes and meeting outcomes.
- `installation/`: platform installation resources and guides.
- `pyproject.toml`: package metadata and dependency declarations.
- `Install_ANNzPlus.bat`, `Run_ANNzPlus.bat`: Windows setup and launch entry points.

## Project ZIP

A standalone `.annzmodel` artifact can be downloaded after training. It contains the trained network weights, architecture, fitted input/target scaling, optimiser state when available, training configuration, epoch/iteration loss history, and optimisation results. It intentionally excludes the original catalogue. In a later session, load the `.annzmodel` file and upload a new catalogue with the saved input columns; the app runs prediction using the saved weights without retraining. The input catalogue is required for prediction, but it does not need the training targets.

A full project ZIP can contain:

- `project.json`: architecture, split settings, training configuration, and scaling values.
- `data/`: original uploaded data, if included.
- `analysis/`: data-quality and exploratory reports.
- `training/checkpoint.pt`: model weights and optimiser state.
- `training/history.csv`: training and validation loss history.
- `training/optimisation_results.csv`: ranked candidate results.

Open a project ZIP in the app to restore its settings and continue training. If source data was not included, upload the identical file; the app uses its SHA-256 fingerprint to protect saved scaling, split assignments, and checkpoints from accidental reuse with a different catalogue.

## Quality checks

Run automated tests with `python -m pytest` and static checks with `python -m ruff check .`.

## Research references

- Legacy software: [imdadmpt/ANNzPlus](https://github.com/imdadmpt/ANNzPlus)
- Collister & Lahav (2004), [The Astronomical Journal](https://iopscience.iop.org/article/10.1086/383254)
- Mahmud Pathi et al. (2025), [JCAP](https://iopscience.iop.org/article/10.1088/1475-7516/2025/01/097/meta)
