# Local app quick start

This Streamlit application runs on your computer. Install the dependencies and start it from the repository folder:

    python -m pip install -r requirements.txt
    python -m streamlit run app.py

## Workflow

1. Upload a CSV or TSV file with a header row.
2. Choose one or more input variables and target variables.
3. Analyze the catalogue to review invalid values, summary statistics, and exploratory correlations.
4. Prepare deterministic training, validation, and test splits. Invalid rows are excluded; input and target scaling is fitted from training rows only.
5. Configure a manual network, or choose **Automatic** in the optimisation section to generate candidate hidden-layer layouts. Choose **Manual** if you want to supply your own layouts.
6. Compare layouts, activation functions, and learning rates. Each candidate starts from fresh weights, is ranked using validation loss, and uses early stopping. The search is capped at 24 candidates.
7. Train or continue the selected network in mini-batches, if desired.
8. Evaluate on the validation or test split using MAE, RMSE, predictive Q-squared, R-squared, and the ANNz+ photo-z metrics when the target is a recognised redshift column.
9. Upload a catalogue with the selected input columns to write predictions as a CSV.
10. Download a reusable settings ZIP or a full project ZIP.

**Important:** Use validation results for model selection. Keep the test split for the final assessment after the architecture and hyperparameters have been chosen. The optimiser does not inspect the test split.

PyTorch chooses a CUDA GPU when one is available in the Python environment. The CPU can be selected as a fallback.

## Reusable architecture/settings ZIP

Choose **Download reusable settings ZIP** to save a portable configuration without copying the original catalogue. It contains:

- Winning architecture: input/target columns, hidden-layer layout, and activation.
- Data-preparation settings: split fractions, random seed, batch sizes, and training-only scaling values.
- Training settings: optimizer, learning rate, compute-device preference, and epoch setting.
- Optimisation recipe: automatic/manual mode, candidate layouts, activations, learning rates, maximum epochs, early-stopping patience, and validation-based selection metric.
- Ranked optimisation results, training history, and the model checkpoint when available.
- Original catalogue filename and SHA-256 fingerprint, but not the catalogue itself.

To reuse it after closing and reopening the app:

1. Start Streamlit again.
2. Upload the saved ZIP using **Open a saved project ZIP** at the top.
3. If the ZIP does not contain the catalogue, upload the original catalogue file. For the saved scaling, split and checkpoint to remain valid, it must be the same file (matching SHA-256 fingerprint).
4. The app restores the selected architecture, optimisation settings/results, training settings, and checkpoint. You can evaluate, predict, or continue training.

## Full project ZIP

Choose **Download full project ZIP** to save the same reusable settings plus optional original catalogue data and available analysis reports. If you uncheck the option to include source data, you will need to upload the matching catalogue when reopening the ZIP.

The archive contains a versioned `project.json` manifest, a README, and optional `data/`, `analysis/`, and `training/` members. Older project ZIP formats (versions 1–4) remain readable.

## Prediction output

Generated predictions are saved in:

    <Documents>/Galaxy Neural Network Analysis/

The CSV contains the source row number and one predicted column for each target variable. Rows with missing, non-numeric, or infinite selected inputs are skipped.

## Current limitations

The browser upload is still held in memory, although catalogue scans, model training, evaluation, and prediction process data in bounded batches. For extremely large source files, the input upload may use substantial memory. The implementation has not yet been compared scientifically with legacy ANNzPlus.
