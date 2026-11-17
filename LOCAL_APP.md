# Local app quick start

This Streamlit application runs on your computer. Install the dependencies and start it from the repository folder:

    python -m pip install -r requirements.txt
    python -m streamlit run app.py

## Workflow

1. Upload a CSV or TSV file with a header row.
2. Choose one or more input variables and target variables.
3. Analyze the catalogue to review invalid values, summary statistics, and exploratory correlations.
4. Prepare deterministic training, validation, and test splits. Invalid rows are excluded; input and target scaling is fitted from training rows only.
5. Configure hidden-layer sizes and an activation function.
6. Train the network in mini-batches. Change the epoch count to continue training from the current checkpoint.
7. Evaluate on a selected data split using MAE, RMSE, and R-squared.
8. Upload a catalogue with the selected input columns to write predictions as a CSV.
9. Download a project ZIP to preserve settings and resumable training progress.

PyTorch chooses a CUDA GPU when one is available in the Python environment. The CPU can be selected as a fallback.

## Project ZIP contents

    project.json
    README.txt
    data/<original-file-name>                  (optional)
    analysis/data_quality_summary.csv          (when analysis has been run)
    analysis/sample_correlations.csv            (when available)
    training/checkpoint.pt                     (after training starts)
    training/history.csv                       (after training starts)

The project ZIP can include the source data, preparation settings, analysis reports, model weights, optimizer state, and training history. Open the ZIP in the app to restore it. If the ZIP excludes its source data, upload the same file to continue from the saved checkpoint.

## Prediction output

Generated predictions are saved in:

    <Documents>/Galaxy Neural Network Analysis/

The CSV contains the source row number and one predicted column for each target variable. Rows with missing, non-numeric, or infinite selected inputs are skipped.

## Current limitations

The browser upload is still held in memory, although catalogue scans, model training, evaluation, and prediction process data in bounded batches. For extremely large source files, the input upload may use substantial memory. The implementation has not yet been compared scientifically with legacy ANNzPlus.
