# Local app quick start

This is a local Streamlit app for exploring galaxy catalogue data and preparing a neural-network project.

## Run the app

From this folder, install dependencies and launch Streamlit:

    python -m pip install -r requirements.txt
    python -m streamlit run app.py

## Workflow

1. Upload a CSV or TSV file with a header row.
2. Choose Input variables and Target variable(s).
3. Use Analyze data to scan the complete file in batches.
4. Review missing, non-numeric, infinite, and finite values, numeric summary statistics, and exploratory correlations from a sample.
5. Choose hidden-layer sizes and an activation function.
6. Download architecture settings or a project ZIP.

## Project ZIP contents

    project.json
    README.txt
    data/<original-file-name>                  (optional)
    analysis/data_quality_summary.csv          (when analysis has been run)
    analysis/sample_correlations.csv            (when available)

The data folder contains the original uploaded source. The analysis folder contains generated catalogue analysis results. Trained model weights and resumable training checkpoints are not available yet.

## Current limitations

Analysis reads CSV/TSV data in batches, but the browser upload is still held in memory. Very large catalogues may need a local-disk loading option. Model training, prediction, and resumable checkpoints are not implemented yet.
