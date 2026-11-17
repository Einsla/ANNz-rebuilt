# Local app quick start

## Windows installation

1. Download and extract the complete repository.
2. Double-click `Install_ANNzPlus.bat` in the repository root.
3. The installer creates an isolated `.venv`, installs the hardware-appropriate PyTorch wheel and project dependencies, and creates Desktop and Start Menu shortcuts.
4. Launch **ANNzPlus Python** from the Desktop, or double-click `Run_ANNzPlus.bat`.

Python 3.9 or newer and an internet connection are required for first installation. NVIDIA systems use the CUDA 12.8 PyTorch package when `nvidia-smi` detects a GPU; otherwise the CPU-only package is installed. See [the Windows installation guide](../../installation/windows/README.md).

## Manual setup

Run from the repository root:

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

## Workflow

1. Upload a CSV or TSV file with a header row.
2. Choose one or more input columns and target columns.
3. Analyze the catalogue to review invalid values, summary statistics, and exploratory correlations.
4. Prepare deterministic training, validation, and test splits. Invalid rows are excluded; input and target scaling is fitted from training rows only.
5. Configure a manual network, or choose **Automatic** in the optimisation section to generate candidate hidden-layer layouts. Choose **Manual** to supply your own layouts.
6. Choose Adam or L-BFGS (quasi-Newton), then compare layouts and activation functions. Adam candidates can also compare learning rates. Each candidate starts from fresh weights, is ranked using validation loss, and uses early stopping. The search is capped at 24 candidates.
7. Train or continue the selected network. Adam uses mini-batches and epochs; L-BFGS uses full-batch optimisation iterations.
8. Evaluate on validation or test data using MAE, RMSE, predictive Q-squared, R-squared, and ANNz+ photo-z metrics when the target is a recognised redshift column.
9. Upload a catalogue with the selected input columns to write predictions as CSV.
10. Download a reusable settings ZIP or a full project ZIP.

**Important:** Use validation results for model selection. Keep the test split for final assessment after architecture and hyperparameters have been chosen. The optimiser does not inspect the test split.

## Optimisers

Adam is the default mini-batch optimiser and uses a learning-rate setting. Training batches are shuffled within source chunks using a reproducible but epoch-specific seed. L-BFGS is a full-batch quasi-Newton option; it chooses step lengths internally and therefore does not use the Adam learning-rate field. L-BFGS can require substantially more memory because it evaluates the full training split together. If CUDA runs out of memory, select CPU or use Adam. The L-BFGS implementation is PyTorch's implementation and is conceptually similar to, but not an exact reproduction of, legacy ANNzPlus `vmmin()`.

## Reusable architecture/settings ZIP

Choose **Download reusable settings ZIP** to save a portable configuration without copying the original catalogue. It contains:

- Winning architecture: input/target columns, hidden-layer layout, and activation.
- Data-preparation settings: split fractions, random seed, batch sizes, and training-only scaling values.
- Training settings: optimiser, learning rate when applicable, compute-device preference, and epoch/iteration setting.
- Optimisation recipe: automatic/manual mode, selected optimiser, candidate layouts, activations, learning rates when using Adam, maximum epochs/iterations, early-stopping patience, and validation-based selection metric.
- Ranked optimisation results, training history, and the model checkpoint when available.
- Original catalogue filename and SHA-256 fingerprint, but not the catalogue itself.

To reuse after restarting the app:

1. Start Streamlit again.
2. Upload the saved ZIP using **Open a saved project ZIP**.
3. If the ZIP does not contain the catalogue, upload the original catalogue file. For saved scaling, split and checkpoint to remain valid, it must be the same file (matching SHA-256 fingerprint).
4. The app restores the selected architecture, optimisation settings/results, training settings, and checkpoint.

## Full project ZIP

Choose **Download full project ZIP** to save reusable settings plus optional original catalogue data and available analysis reports. If source data is excluded, upload the matching catalogue when reopening the ZIP.

The archive contains a versioned `project.json` manifest, a README, and optional `data/`, `analysis/`, and `training/` members. Older project ZIP formats (versions 1–4) remain readable.

## Prediction output

Generated predictions are saved in:

```
<Documents>/Galaxy Neural Network Analysis/
```

The CSV contains the source row number and one predicted column for each target variable. Rows with missing, non-numeric, or infinite selected inputs are skipped.

## Current limitations

The browser upload is still held in memory, although catalogue scans, model training, evaluation, and prediction process data in bounded batches. For extremely large source files, the input upload may use substantial memory. The implementation has not yet been compared scientifically with legacy ANNzPlus.
