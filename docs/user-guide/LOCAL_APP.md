# Local app quick start

For the latest reviewed implementation state and known parity gaps, see the [22 November 2026 code-review checkpoint](../checkpoints/2026-11-22-deep-code-review.md).

## Windows installation

1. Download and extract the complete repository.
2. Double-click `Install_ANNzPlus.bat` in the repository root.
3. The installer creates an isolated `.venv`, installs the hardware-appropriate PyTorch wheel and project dependencies, and creates Desktop and Start Menu shortcuts.
4. Launch **ANNzPlus Python** from the Desktop, or double-click `Run_ANNzPlus.bat`.

Python 3.9 or newer and an internet connection are required for first installation. The app automatically uses a supported CUDA GPU when available and otherwise falls back to CPU; no device selection is required. NVIDIA systems use the CUDA 12.8 PyTorch package when `nvidia-smi` detects a GPU; otherwise the CPU-only package is installed. See [the Windows installation guide](../../installation/windows/README.md).

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
6. Choose Adam or legacy vmmin, then compare layouts and activation functions. Adam candidates can also compare learning rates. Each candidate starts from fresh weights and is ranked using validation loss. Adam uses early stopping; vmmin uses its original convergence rules. The search is capped at 24 candidates.
7. Train or continue the selected network. Adam uses mini-batches and epochs; legacy vmmin uses full-objective optimisation iterations while streaming source chunks.
8. Evaluate on validation or test data using MAE, RMSE, predictive Q-squared, R-squared, and ANNz+ photo-z metrics when the target is a recognised redshift column.
9. Upload a catalogue with the selected input columns to write predictions as CSV.
10. Download a reusable settings ZIP or a full project ZIP.

**Important:** Use validation results for model selection. Keep the test split for final assessment after architecture and hyperparameters have been chosen. The optimiser does not inspect the test split.

## Optimisers

Adam is the default mini-batch optimiser and uses a learning-rate setting.
Legacy vmmin ports the original ANNzPlus variable-metric quasi-Newton
algorithm and supports all seven implemented hidden activations: Tanh, Sigmoid,
Leaky ReLU, ReLU, Mish, Softplus and SiLU. It evaluates the complete training-set objective and gradient,
streaming the catalogue in bounded chunks between evaluations; it is not
mini-batch SGD. It uses a dense inverse-Hessian approximation (O(P**2) CPU
memory for P weights), and follows
the legacy convergence/reset rules rather than validation-patience early
stopping. Use smaller networks with vmmin and Adam for larger architectures.
Existing L-BFGS checkpoints can restore their model weights, but vmmin starts
a fresh variable-metric quasi-Newton state when continuing from them.

The port reproduces the optimiser mechanics and weight-decay schedule, but
full scientific parity still requires matching C++ normalisation, weight
initialisation and numerical execution. See the development parity notes.

## Reusable architecture/settings ZIP

Choose **Download reusable settings ZIP** to save a portable configuration without copying the original catalogue. It contains:

- Winning architecture: input/target columns, hidden-layer layout, and activation.
- Data-preparation settings: split fractions, random seed, batch sizes, and training-only scaling values.
- Training settings: optimiser, learning rate when applicable, automatic compute-device setting (CUDA when available, otherwise CPU), and epoch/iteration setting.
- Checkpoint consistency: best-validation weights and their matching optimiser state are saved together with the full run history and a monotonic continuation counter. Continuing training therefore does not duplicate epoch/iteration numbers or discard the previous run history.
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

The archive contains a versioned `project.json` manifest, a README, and optional `data/`, `analysis/`, and `training/` members. Project ZIP formats 1 through 5 are supported.

## Prediction output

Generated predictions are saved in:

```
<Documents>/Galaxy Neural Network Analysis/
```

The CSV contains the source row number and one predicted column for each target variable. Rows with missing, non-numeric, or infinite selected inputs are skipped. When measurement-error columns are mapped in the prediction section, single-network mode adds one target_input_error_std column per target using the network Jacobian. Committee mode adds target_committee_std, target_input_error_std, and target_total_std. Error columns must contain finite, non-negative 1-sigma uncertainties in the same physical units as their corresponding inputs. They are not extra neural-network features.

## Current limitations

The browser upload is still held in memory, although catalogue scans, model training, evaluation, and prediction process data in bounded batches. For extremely large source files, the input upload may use substantial memory. The implementation has not yet been compared scientifically with legacy ANNzPlus.
