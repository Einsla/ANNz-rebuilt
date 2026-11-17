# ANNzPlus Python Rebuild

## Project overview

This project investigates and rebuilds the legacy [ANNzPlus](https://github.com/imdadmpt/ANNzPlus) photometric-redshift workflow in Python. ANNzPlus extends ANNz for estimating galaxy photometric redshifts with configurable neural networks.

The rebuild is a local research prototype. It includes batched catalogue analysis and preparation, configurable fully connected regression networks, mini-batch training, validation/test metrics, prediction export, and portable project ZIP files with resumable training checkpoints. Matching legacy behavior and scientific performance still requires controlled comparison against the original software and suitable reference data.

## Research objective

Determine how faithfully the core ANNzPlus photometric-redshift workflow can be reproduced in Python, then compare predictions under matched input variables, data splits, and model settings. Report the results using defined accuracy metrics and document differences from the legacy system.

## Current local workflow

1. Upload a CSV/TSV galaxy catalogue and choose the input and target variables.
2. Scan the catalogue in chunks for data quality and exploratory summary statistics.
3. Prepare deterministic training, validation, and test splits. Invalid selected rows are skipped, and scaling is fitted from training rows only.
4. Configure hidden-layer sizes and an activation function.
5. Train in mini-batches, review training/validation loss, and continue from a saved checkpoint.
6. Evaluate using MAE, RMSE, and R-squared, and write predictions for another catalogue.
7. Save the settings, optional source data, analysis reports, training history, and checkpoint in a project ZIP.

## Run the local app

From the repository root:

    python install.py
    python -m streamlit run app.py

The installer uses the current Python interpreter and installs the project plus
its declared dependencies from `pyproject.toml`. This is the recommended
hardware-aware installation route. If you have already selected the correct
PyTorch build for your machine, `python -m pip install .` installs the project
and resolves any remaining runtime libraries (NumPy, pandas, PyTorch, and
Streamlit).

The installer detects an NVIDIA GPU and selects a CUDA-enabled PyTorch wheel; otherwise it installs the CPU build. The app uses CUDA when PyTorch can access a compatible GPU and falls back to CPU when requested or when CUDA is unavailable. Catalogue processing defaults to 50,000-row CPU chunks and 512-row model mini-batches; adjust these values for your available system memory and GPU VRAM.

## Project ZIP

A project ZIP can contain:

- `project.json`: selected variables, network configuration, split settings, and training-only scaling.
- `data/<filename>`: original uploaded data, if included.
- `analysis/`: data-quality and exploratory correlation reports, if available.
- `training/checkpoint.pt`: network weights, optimizer state, and epoch history, if training has started.
- `training/history.csv`: training and validation loss by epoch.

Open a project ZIP in the app to restore its settings and continue training. If the source data was not included, upload the identical file to reuse its saved preparation settings and checkpoint.

## Current limitations

The local Streamlit uploader holds the uploaded source bytes in memory, even though catalogue processing and model training use bounded batches. Very large files may need a future direct-from-disk input flow. Prediction output is written to the user's Documents folder under `Galaxy Neural Network Analysis`. Neural-network results have not yet been validated against legacy ANNzPlus. The app is a research prototype and does not establish scientific equivalence by itself.

## Research references

- Legacy software: [imdadmpt/ANNzPlus](https://github.com/imdadmpt/ANNzPlus)
- Collister & Lahav (2004), [The Astronomical Journal](https://iopscience.iop.org/article/10.1086/383254)
- Mahmud Pathi et al. (2025), [Journal of Cosmology and Astroparticle Physics](https://iopscience.iop.org/article/10.1088/1475-7516/2025/01/097/meta)
