"""Local Streamlit app for galaxy data preparation and neural-network analysis."""

from __future__ import annotations

import hashlib
import io
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from annz_rebuilt import architecture_from_selections  # noqa: E402
from annz_rebuilt.architecture import SUPPORTED_ACTIVATIONS  # noqa: E402
from annz_rebuilt.data_analysis import analyze_catalogue_csv  # noqa: E402
from annz_rebuilt.data_preparation import DataPreparation, fit_data_preparation  # noqa: E402
from annz_rebuilt.neural_network import evaluate_model, train_model, write_predictions_csv  # noqa: E402
from annz_rebuilt.project import create_project_zip, open_project_zip  # noqa: E402


st.set_page_config(page_title="Galaxy Neural Network Analysis", page_icon="🌌", layout="wide")
st.title("Galaxy Neural Network Analysis")
st.write(
    "Prepare galaxy catalogue data, train a configurable neural network, evaluate it, "
    "make predictions, and save the project so training can continue later."
)


def _model_signature(architecture: dict, preparation: DataPreparation) -> str:
    return json.dumps(
        {"architecture": architecture, "preparation": preparation.to_config_dict()},
        sort_keys=True,
        separators=(",", ":"),
    )


def restore_project(zip_bytes: bytes) -> None:
    """Restore app settings, source data, reports, split settings, and checkpoint."""
    project = open_project_zip(zip_bytes)
    config = project.architecture
    st.session_state["loaded_project_fingerprint"] = hashlib.sha256(zip_bytes).hexdigest()
    st.session_state["loaded_data_source_name"] = project.data_source_name
    st.session_state["loaded_data_source_bytes"] = project.data_source_bytes
    st.session_state["loaded_data_source_sha256"] = project.data_source_sha256
    st.session_state["saved_quality_summary"] = project.quality_summary
    st.session_state["saved_sample_correlations"] = project.sample_correlations
    st.session_state["inputs"] = config.get("input_columns", [])
    st.session_state["targets"] = config.get("target_columns", [])
    layers = config.get("hidden_layers", [10])
    st.session_state["layer_count"] = len(layers)
    st.session_state["activation"] = config.get("activation", "leaky_relu")
    for index, size in enumerate(layers):
        st.session_state[f"layer_{index}"] = int(size)

    preparation_config = project.data_preparation
    preparation = (
        DataPreparation.from_config_dict(preparation_config)
        if preparation_config is not None
        else None
    )
    st.session_state["data_preparation"] = preparation
    if preparation is not None:
        st.session_state["train_share_percent"] = round(preparation.train_fraction * 100)
        st.session_state["validation_share_percent"] = round(preparation.validation_fraction * 100)
        st.session_state["rows_per_batch"] = preparation.rows_per_batch
        st.session_state["training_batch_size"] = preparation.batch_size
        st.session_state["random_seed"] = preparation.random_seed
        st.session_state["data_preparation_signature"] = (
            project.data_source_sha256,
            preparation.input_columns,
            preparation.target_columns,
            preparation.train_fraction,
            preparation.validation_fraction,
            preparation.random_seed,
            preparation.rows_per_batch,
            preparation.batch_size,
        )
    else:
        st.session_state["data_preparation_signature"] = None
    st.session_state["trained_checkpoint"] = project.checkpoint_bytes
    st.session_state["training_history"] = project.training_history
    history = project.training_history
    st.session_state["last_epoch"] = (
        int(history["epoch"].max())
        if history is not None and not history.empty and "epoch" in history
        else 0
    )
    st.session_state["evaluation_report"] = None
    st.session_state["checkpoint_signature"] = (
        _model_signature(config, preparation)
        if project.checkpoint_bytes is not None and preparation is not None
        else None
    )


project_upload = st.file_uploader("Open a saved project ZIP (optional)", type=["zip"])
if project_upload is not None:
    project_bytes = project_upload.getvalue()
    fingerprint = hashlib.sha256(project_bytes).hexdigest()
    if st.session_state.get("loaded_project_fingerprint") != fingerprint:
        try:
            restore_project(project_bytes)
            st.success("Project settings and any saved training progress were loaded.")
        except (ValueError, KeyError, TypeError) as exc:
            st.error(f"Could not open project: {exc}")
            st.stop()

uploaded = st.file_uploader(
    "Upload a galaxy catalogue (CSV or TSV with a header row)",
    type=["csv", "tsv"],
    key="data_source_upload",
)
saved_data = st.session_state.get("loaded_data_source_bytes")
if saved_data is not None:
    data_source_bytes = saved_data
    data_source_name = st.session_state.get("loaded_data_source_name") or "galaxies.csv"
    st.caption(f"Using the data source saved in the project: {data_source_name}")
elif uploaded is not None:
    data_source_bytes = uploaded.getvalue()
    data_source_name = uploaded.name
else:
    st.info("Upload a galaxy catalogue or open a project ZIP that contains its data source.")
    st.stop()

data_fingerprint = hashlib.sha256(data_source_bytes).hexdigest()
if st.session_state.get("data_source_fingerprint") != data_fingerprint:
    st.session_state["data_source_fingerprint"] = data_fingerprint
    st.session_state["catalogue_analysis"] = None
    if st.session_state.get("loaded_data_source_sha256") != data_fingerprint:
        st.session_state["data_preparation"] = None
        st.session_state["data_preparation_signature"] = None
        st.session_state["trained_checkpoint"] = None
        st.session_state["checkpoint_signature"] = None
        st.session_state["training_history"] = None
        st.session_state["evaluation_report"] = None
        st.session_state["saved_quality_summary"] = None
        st.session_state["saved_sample_correlations"] = None

separator = "\t" if data_source_name.lower().endswith(".tsv") else ","
try:
    preview = pd.read_csv(io.BytesIO(data_source_bytes), sep=separator, nrows=100)
    preview.columns = [str(column).lstrip("#").strip() for column in preview.columns]
except Exception as exc:
    st.error(f"Could not read the file as a headered CSV/TSV table: {exc}")
    st.stop()
if len(set(preview.columns)) != len(preview.columns):
    st.error("Column names must be unique after removing a leading #.")
    st.stop()
columns = list(preview.columns)
for key in ("inputs", "targets"):
    if key in st.session_state:
        st.session_state[key] = [name for name in st.session_state[key] if name in columns]

st.caption(
    f"Previewing up to {len(preview):,} rows · {len(columns)} variables · "
    f"source: {data_source_name}"
)
st.dataframe(preview.head(10), use_container_width=True)

st.subheader("1. Choose the variables")
inputs = st.multiselect(
    "Input variables — what information should the model use?",
    columns,
    key="inputs",
    help="For example: veldisp, deVAB_u, and deVRad_u.",
)
targets = st.multiselect(
    "Target variable(s) — what should the model learn to predict?",
    columns,
    key="targets",
    help="For example: z. Inputs and targets must be separate.",
)

st.subheader("2. Analyze the galaxy data")
st.write(
    "Scan the full file in chunks for missing, non-numeric, infinite, and finite values, "
    "then review summary statistics and exploratory correlations."
)
rows_per_batch = st.number_input(
    "Rows analyzed per batch",
    min_value=1_000,
    max_value=1_000_000,
    value=50_000,
    step=1_000,
    key="rows_per_batch",
)
if st.button("Analyze data"):
    try:
        with st.spinner("Scanning the catalogue in batches..."):
            st.session_state["catalogue_analysis"] = analyze_catalogue_csv(
                io.BytesIO(data_source_bytes),
                separator=separator,
                rows_per_batch=int(rows_per_batch),
                sample_columns=[*inputs, *targets],
            )
    except (ValueError, KeyError, OSError) as exc:
        st.error(f"Could not analyze this data source: {exc}")

analysis = st.session_state.get("catalogue_analysis")
quality_summary = None
sample_correlations = None
if analysis is not None:
    quality_summary, sample, analyzed_rows = analysis
    st.success(f"Analyzed {analyzed_rows:,} rows.")
    st.dataframe(quality_summary, use_container_width=True)
    selected = list(dict.fromkeys([*inputs, *targets]))
    selected = [name for name in selected if name in sample.columns]
    sample_numbers = sample.loc[:, selected].apply(pd.to_numeric, errors="coerce")
    sample_numbers = sample_numbers.replace([float("inf"), float("-inf")], float("nan")).dropna()
    if len(selected) >= 2 and not sample_numbers.empty:
        sample_correlations = sample_numbers.corr()
        st.markdown("**Exploratory correlation matrix**")
        st.dataframe(sample_correlations, use_container_width=True)
        st.caption("Correlation values use a sample of up to 20,000 rows.")
elif st.session_state.get("saved_quality_summary") is not None:
    quality_summary = st.session_state["saved_quality_summary"]
    sample_correlations = st.session_state.get("saved_sample_correlations")
    st.caption("Showing analysis reports saved in the opened project.")
    st.dataframe(quality_summary, use_container_width=True)
    if sample_correlations is not None:
        st.dataframe(sample_correlations, use_container_width=True)

st.subheader("3. Prepare training data")
st.write(
    "Invalid selected rows are skipped. Scaling values are learned from the training split "
    "only, then applied to validation and test data."
)
a, b = st.columns(2)
with a:
    train_percent = st.slider(
        "Training share (%)", 50, 90, 70, step=5, key="train_share_percent"
    )
max_validation = min(40, 100 - train_percent)
if st.session_state.get("validation_share_percent", 15) > max_validation:
    st.session_state["validation_share_percent"] = max_validation
with b:
    validation_percent = st.slider(
        "Validation share (%)",
        0,
        max_validation,
        min(15, max_validation),
        step=5,
        key="validation_share_percent",
    )
test_percent = 100 - train_percent - validation_percent
st.caption(f"Test share: {test_percent}%")
seed_col, batch_col = st.columns(2)
with seed_col:
    random_seed = st.number_input(
        "Random split seed", 0, 2_147_483_647, 42, step=1, key="random_seed"
    )
with batch_col:
    training_batch_size = st.number_input(
        "Training mini-batch size", 128, 65_536, 512, step=128,
        key="training_batch_size",
        help="Rows fed to the neural network at a time. Smaller batches use less GPU memory.",
    )

current_preparation_signature = (
    data_fingerprint,
    tuple(inputs),
    tuple(targets),
    train_percent / 100,
    validation_percent / 100,
    int(random_seed),
    int(rows_per_batch),
    int(training_batch_size),
)
if st.session_state.get("data_preparation_signature") not in {
    None,
    current_preparation_signature,
}:
    st.session_state["data_preparation"] = None
    st.session_state["trained_checkpoint"] = None
    st.session_state["checkpoint_signature"] = None
    st.session_state["training_history"] = None
if st.session_state.get("data_preparation_signature") != current_preparation_signature:
    st.session_state["data_preparation"] = None
    st.session_state["data_preparation_signature"] = current_preparation_signature

if st.button("Prepare data splits"):
    try:
        with st.spinner("Scanning rows and fitting training-only scaling..."):
            preparation = fit_data_preparation(
                io.BytesIO(data_source_bytes),
                inputs,
                targets,
                separator=separator,
                rows_per_batch=int(rows_per_batch),
                batch_size=int(training_batch_size),
                train_fraction=train_percent / 100,
                validation_fraction=validation_percent / 100,
                random_seed=int(random_seed),
            )
        st.session_state["data_preparation"] = preparation
        st.session_state["data_preparation_signature"] = current_preparation_signature
    except (ValueError, KeyError, OSError) as exc:
        st.error(f"Could not prepare this data source: {exc}")

preparation = st.session_state.get("data_preparation")
if preparation is not None:
    st.success(
        f"{preparation.usable_rows:,} usable rows, {preparation.dropped_rows:,} skipped."
    )
    st.dataframe(
        pd.DataFrame(
            {"split": preparation.rows_by_split.keys(), "rows": preparation.rows_by_split.values()}
        ),
        hide_index=True,
        use_container_width=True,
    )
    scaling = []
    for role, names, means, scales in (
        ("input", preparation.input_columns, preparation.input_mean, preparation.input_scale),
        ("target", preparation.target_columns, preparation.target_mean, preparation.target_scale),
    ):
        scaling.extend(
            {"variable": n, "role": role, "training_mean": m, "training_scale": s}
            for n, m, s in zip(names, means, scales)
        )
    st.markdown("**Scaling learned from training rows**")
    st.dataframe(pd.DataFrame(scaling), hide_index=True, use_container_width=True)
else:
    st.info("Prepare the data splits before training.")

st.subheader("4. Describe the network")
left, right = st.columns(2)
with left:
    layer_count = st.number_input(
        "Hidden layers", 1, 10, 1, step=1, key="layer_count"
    )
with right:
    activation = st.selectbox(
        "Activation function", sorted(SUPPORTED_ACTIVATIONS), key="activation"
    )
neurons = [
    int(
        st.number_input(
            f"Neurons in hidden layer {layer + 1}",
            1,
            4096,
            10,
            step=1,
            key=f"layer_{layer}",
        )
    )
    for layer in range(int(layer_count))
]
architecture = None
try:
    architecture = architecture_from_selections(inputs, targets, neurons, activation)
    st.caption(
        f"Network: {architecture.n_inputs} input(s) → "
        f"{' → '.join(map(str, architecture.hidden_layers))} → "
        f"{architecture.n_outputs} output(s)"
    )
except ValueError as exc:
    st.info(f"Choose valid inputs and targets to configure the network: {exc}")

if architecture is not None and preparation is not None:
    model_signature = _model_signature(architecture.to_config_dict(), preparation)
    if st.session_state.get("checkpoint_signature") not in {None, model_signature}:
        st.session_state["trained_checkpoint"] = None
        st.session_state["training_history"] = None
        st.session_state["evaluation_report"] = None
    st.session_state["checkpoint_signature"] = model_signature

    st.subheader("5. Train the network")
    train_left, train_mid, train_right = st.columns(3)
    with train_left:
        epochs = st.number_input("Additional epochs", 1, 500, 10, step=1, key="epochs")
    with train_mid:
        learning_rate = st.number_input(
            "Learning rate", min_value=0.000001, max_value=1.0, value=0.001,
            format="%.6f", key="learning_rate",
        )
    with train_right:
        device_choice = st.selectbox(
            "Compute device", ["auto", "cpu", "cuda"], index=0, key="compute_device"
        )
    st.caption(
        "Training rereads the source in bounded chunks. If a checkpoint exists, "
        "training continues from its saved weights and optimizer state."
    )
    if st.button("Start / continue training", type="primary"):
        progress = st.progress(0)
        status = st.empty()
        progress_state = {"epochs_this_run": 0}

        def update_progress(event):
            progress_state["epochs_this_run"] += 1
            st.session_state["trained_checkpoint"] = event["checkpoint_bytes"]
            st.session_state["last_epoch"] = event["epoch"]
            status.write(
                f"Epoch {event['epoch']}: training loss {event['training_loss']:.6g}; "
                f"validation loss {event['validation_loss'] if event['validation_loss'] is not None else 'not available'} "
                f"({event['device']})"
            )
            progress.progress(progress_state["epochs_this_run"] / int(epochs))

        try:
            result = train_model(
                io.BytesIO(data_source_bytes),
                architecture,
                preparation,
                epochs=int(epochs),
                learning_rate=float(learning_rate),
                checkpoint_bytes=st.session_state.get("trained_checkpoint"),
                separator=separator,
                device=device_choice,
                on_epoch=update_progress,
            )
            st.session_state["trained_checkpoint"] = result["checkpoint_bytes"]
            st.session_state["checkpoint_signature"] = model_signature
            st.session_state["training_history"] = result["history"]
            st.session_state["last_epoch"] = result["completed_epoch"]
            st.session_state["evaluation_report"] = None
            progress.progress(1.0)
            st.success(
                f"Training reached epoch {result['completed_epoch']} on {result['device']}."
            )
        except (ValueError, RuntimeError, OSError) as exc:
            st.error(f"Training did not finish: {exc}")

    history = st.session_state.get("training_history")
    if history is not None and not history.empty:
        st.markdown("**Training history**")
        st.dataframe(history, hide_index=True, use_container_width=True)
        st.line_chart(history.set_index("epoch")[["training_loss", "validation_loss"]])

    checkpoint = st.session_state.get("trained_checkpoint")
    if checkpoint:
        st.subheader("6. Evaluate the model")
        split = st.selectbox(
            "Evaluation data split",
            ["test", "validation", "train"],
            key="evaluation_split",
        )
        if st.button("Evaluate model"):
            try:
                report = evaluate_model(
                    io.BytesIO(data_source_bytes),
                    architecture,
                    preparation,
                    checkpoint,
                    split=split,
                    separator=separator,
                    device=device_choice,
                )
                st.session_state["evaluation_report"] = report
            except (ValueError, RuntimeError, OSError) as exc:
                st.error(f"Could not evaluate this model: {exc}")
        report = st.session_state.get("evaluation_report")
        if report is not None:
            st.dataframe(report, hide_index=True, use_container_width=True)
            st.caption(
                "Q2 is the external predictive coefficient of determination: "
                "1 - PRESS / sum((observed target - training-set mean)^2). "
                "R2 uses the evaluated split's own mean. For targets named z, "
                "redshift, z_spec, zspec, specz, or spec_z, the report also includes "
                "the ANNz+ photo-z metrics: normalized sigma_RMS, sigma_68, and "
                "outlier fraction (|delta z| >= 0.15). MAE/RMSE use original target units."
            )

        st.subheader("7. Generate predictions")
        prediction_upload = st.file_uploader(
            "Upload a catalogue containing the selected input columns",
            type=["csv", "tsv"],
            key="prediction_upload",
        )
        if prediction_upload is not None and st.button("Write prediction file"):
            prediction_name = prediction_upload.name
            prediction_separator = "\t" if prediction_name.lower().endswith(".tsv") else ","
            output_dir = Path.home() / "Documents" / "Galaxy Neural Network Analysis"
            output_file = output_dir / (
                f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.csv"
            )
            try:
                result = write_predictions_csv(
                    io.BytesIO(prediction_upload.getvalue()),
                    output_file,
                    architecture,
                    preparation,
                    checkpoint,
                    separator=prediction_separator,
                    device=device_choice,
                )
                st.success(
                    f"Saved {result['predicted_rows']:,} predictions to {result['path']}. "
                    f"Skipped {result['skipped_rows']:,} rows with invalid inputs."
                )
            except (ValueError, RuntimeError, OSError) as exc:
                st.error(f"Could not generate predictions: {exc}")

st.subheader("8. Save the project")
include_data = st.checkbox(
    "Include the original data source in the project ZIP", value=True
)
if st.button("Prepare project ZIP"):
    if architecture is None:
        st.error("Choose at least one input and target and valid network settings first.")
    else:
        project_zip = create_project_zip(
            architecture.to_config_dict(),
            data_preparation_config=preparation.to_config_dict() if preparation else None,
            data_source_name=data_source_name,
            data_source_bytes=data_source_bytes if include_data else None,
            data_source_sha256=data_fingerprint,
            quality_summary=quality_summary,
            sample_correlations=sample_correlations,
            checkpoint_bytes=st.session_state.get("trained_checkpoint"),
            training_history=st.session_state.get("training_history"),
        )
        col_a, col_b = st.columns(2)
        with col_a:
            st.download_button(
                "Download architecture settings",
                json.dumps(architecture.to_config_dict(), indent=2),
                file_name="architecture_config.json",
                mime="application/json",
            )
        with col_b:
            st.download_button(
                "Download project ZIP",
                project_zip,
                file_name="galaxy_analysis_project.zip",
                mime="application/zip",
            )
        st.info(
            "The ZIP includes the network and data settings, available analysis reports, "
            "optional source data, and a resumable model checkpoint when training has begun."
        )
