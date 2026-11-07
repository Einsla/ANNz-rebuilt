# ANNzPlus Python Rebuild

## Project overview

This project investigates and rebuilds the legacy [ANNzPlus](https://github.com/imdadmpt/ANNzPlus) photometric-redshift software in Python. The legacy repository describes ANNzPlus as an extension of ANNz for estimating galaxy photometric redshifts, with support for multiple neural-network activation functions.

The rebuild will document the legacy behavior, implement a maintainable Python workflow, and compare its outputs with the legacy implementation on suitable data. It is a research and software engineering project; numerical equivalence and any performance claims must be established through documented experiments.

## Research objective

To determine how faithfully the core ANNzPlus photometric-redshift workflow can be reproduced in Python, and to evaluate the rebuilt implementation against the legacy software using controlled, reproducible data, configuration, and validation metrics.

## Research questions

1. Which inputs, preprocessing steps, model settings, activation functions, and outputs define the legacy ANNzPlus workflow?
2. Can those behaviors be implemented clearly and reproducibly in Python?
3. Under matched data splits and model settings, how closely do Python and legacy predictions agree?
4. How do supported activation functions and selected training settings affect photometric-redshift accuracy and runtime?
5. What limitations or differences need to be documented for users migrating from the legacy software?

## Scope

- Inspect the legacy source, manual, examples, and available validation material.
- Record the expected input schema, preprocessing, training and inference behavior, configuration, and output format.
- Build the Python implementation in small, documented components.
- Preserve or explicitly document behavior that cannot be matched exactly.
- Compare implementations using the same data, splits, and evaluation procedure where practical.
- Report accuracy, prediction agreement, runtime, reproducibility, and limitations.

The initial scope is the core workflow needed to train and evaluate photometric-redshift models. Additional features should be prioritized after the legacy behavior and project requirements are reviewed.

## Proposed methodology

1. **Legacy audit:** inventory source files and dependencies; extract the algorithm, data format, configuration, and activation-function behavior from the legacy repository and manual.
2. **Specification:** write down expected inputs, preprocessing, model behavior, outputs, and known edge cases before implementation.
3. **Python design:** select a modular structure for data loading, preprocessing, model configuration, training, prediction, metrics, and command-line use.
4. **Implementation:** implement the agreed core behavior and record intentional differences from the legacy program.
5. **Validation:** use documented datasets and fixed train/validation/test splits; avoid data leakage; compare predictions and report appropriate photometric-redshift metrics (for example bias, scatter, outlier fraction, and runtime).
6. **Reproducibility:** record software versions, configuration, random seeds, data provenance, and commands needed to reproduce results.

## Initial deliverables

- Legacy behavior and requirements notes.
- A Python package or application implementing the agreed core workflow.
- Reproducible example configuration and usage documentation.
- Validation scripts and a results report comparing the Python rebuild with the legacy baseline where the required baseline and data are available.
- A migration and limitations guide.

## Initial success criteria

- The documented core workflow can be run from a clean Python environment using the published instructions.
- Inputs, configuration, outputs, and failure cases are documented.
- Validation runs are reproducible from recorded data references, splits, settings, and software versions.
- Prediction agreement and accuracy are reported with defined metrics and an identified comparison baseline.
- Any unsupported behavior or deviation from the legacy implementation is clearly stated.

## Project status

Planning stage. The repository is currently being initialized. Research decisions, implementation choices, datasets, and results should be added as they are confirmed; do not present proposed work as completed findings.

## Reference

- Legacy software: [imdadmpt/ANNzPlus](https://github.com/imdadmpt/ANNzPlus)
- Original ANNz reference cited by the legacy project: Collister & Lahav (2004), [The Astronomical Journal](https://iopscience.iop.org/article/10.1086/383254)
- ANNz+ publication cited by the legacy project: Mahmud Pathi et al. (2025), [Journal of Cosmology and Astroparticle Physics](https://iopscience.iop.org/article/10.1088/1475-7516/2025/01/097/meta)

## Run the local prototype

The early Streamlit interface lets a user upload a CSV or TSV catalogue, choose **Input variables** (measurements used by the model) and **Target variable(s)** (values to predict), and configure hidden-layer sizes and an activation function. It validates selected numeric columns, previews inputs and targets separately, and downloads the choices as `architecture_config.json`.

From the repository root, install the dependencies and launch it locally:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

This prototype prepares data selections; model training, full project save/open, resumable checkpoints, and batch loading for very large catalogues are not implemented yet. Legacy `.net` and positional catalogue exports are not part of this standalone Python workflow.
