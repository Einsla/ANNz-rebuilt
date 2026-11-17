# ANNzPlus Python Rebuild

## Project overview

This project investigates and rebuilds the legacy [ANNzPlus](https://github.com/imdadmpt/ANNzPlus) photometric-redshift workflow in Python. It provides a local Streamlit interface for catalogue analysis, neural-network architecture design, training, evaluation, prediction, and reusable project archives.

This is a research rebuild, not a claim of bit-for-bit or scientific equivalence with the original C++ implementation. See [ANNZPLUS_PARITY.md](ANNZPLUS_PARITY.md) for confirmed differences and remaining parity work.

## Windows installation (recommended)

Requirements:

- Windows 10 or 11, 64-bit.
- Python 3.9 or newer, with the Python Launcher recommended.
- Internet access during installation.
- An NVIDIA GPU is optional; CPU execution is supported.

1. Download the repository using GitHub's **Code → Download ZIP**, or clone it.
2. Extract the complete folder to a location you can write to.
3. Double-click **Install_ANNzPlus.bat**.
4. Wait for the installer to create a local `.venv`, install PyTorch and the app dependencies, and create desktop and Start Menu shortcuts.
5. Launch **ANNzPlus Python** from the desktop, or double-click **Run_ANNzPlus.bat** in the project folder.

The installer detects an NVIDIA GPU using `nvidia-smi`. It installs a CUDA 12.8 PyTorch wheel when an NVIDIA GPU is detected and a CPU-only PyTorch wheel otherwise. The actual wheel is selected by pip for the installed Python version. No Python packages or virtual-environment files are committed to GitHub; this avoids shipping multi-gigabyte, hardware-specific binaries.

If shortcut creation is blocked by Windows policy, the app remains launchable through `Run_ANNzPlus.bat`.

## Manual installation

From the repository root:

```powershell
py -3 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe install.py
.venv\Scripts\python.exe -m streamlit run app.py
```

For development and tests:

```powershell
.venv\Scripts\python.exe -m pip install -e ".[dev]"
.venv\Scripts\python.exe -m pytest
.venv\Scripts\python.exe -m ruff check .
```

## Current workflow

1. Upload a CSV or TSV galaxy catalogue with a header row.
2. Select input variables and target variable(s).
3. Scan the catalogue in chunks for missing, non-numeric, infinite, and finite values; review summary statistics and exploratory correlations.
4. Prepare deterministic training, validation, and test splits. Invalid selected rows are excluded, and scaling is fitted on training rows only.
5. Describe the network by choosing hidden-layer widths and an activation function.
6. Optimise candidate architectures using validation loss. Automatic and manual candidate layouts are supported.
7. Train or continue with Adam mini-batches or full-batch L-BFGS. Validation loss is monitored for candidate selection and early stopping.
8. Evaluate using MAE, RMSE, predictive Q-squared, R-squared, and ANNz+ photo-z metrics for recognised redshift targets.
9. Generate predictions for another catalogue and export them as CSV.
10. Save a reusable settings ZIP or a full project ZIP with optional source data.

The test split is held out from architecture and hyperparameter selection. It should be used for final assessment after model choices have been fixed.

## Optimisers

- **Adam:** mini-batch gradient optimisation with an explicit learning-rate setting. Each epoch passes through the full training split; batches are shuffled within source chunks using a reproducible epoch-specific seed.
- **L-BFGS:** full-batch quasi-Newton optimisation with an internal line search. It materialises the training and validation splits and can require substantially more memory. It is conceptually similar to the legacy ANNzPlus variable-metric optimiser, but it is not an exact port of the original `vmmin()` implementation.

The installer selects a CUDA-enabled PyTorch build for NVIDIA systems and a CPU-only build otherwise. The app can also be configured to use CPU if GPU memory is insufficient.

## Project ZIP

A project ZIP can contain:

- `project.json`: architecture, split settings, training configuration, and scaling values.
- `data/`: original uploaded data, if included.
- `analysis/`: data-quality and exploratory reports.
- `training/checkpoint.pt`: model weights and optimiser state.
- `training/history.csv`: training and validation loss history.
- `training/optimisation_results.csv`: ranked candidate results.

Open a project ZIP in the app to restore its settings and continue training. If the source data was not included, upload the identical file; the app uses its SHA-256 fingerprint to protect saved scaling, split assignments, and checkpoints from accidental reuse with a different catalogue.

## Quality checks

The repository includes automated tests for architecture validation, deterministic splits and shuffling, multi-target training-loss reporting, Adam and L-BFGS training, checkpoint continuation, metrics, prediction export, and project ZIP validation.

Run them with `python -m pytest`. Run static checks with `python -m ruff check .`.

## Research references

- Legacy software: [imdadmpt/ANNzPlus](https://github.com/imdadmpt/ANNzPlus)
- Collister & Lahav (2004), [The Astronomical Journal](https://iopscience.iop.org/article/10.1086/383254)
- Mahmud Pathi et al. (2025), [Journal of Cosmology and Astroparticle Physics](https://iopscience.iop.org/article/10.1088/1475-7516/2025/01/097/meta)
