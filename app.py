"""Local Streamlit app for galaxy catalogue analysis and training-data setup."""

from __future__ import annotations

import hashlib
import io
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from annz_rebuilt import architecture_from_selections  # noqa: E402
from annz_rebuilt.architecture import SUPPORTED_ACTIVATIONS  # noqa: E402
from annz_rebuilt.data_analysis import analyze_catalogue_csv  # noqa: E402
from annz_rebuilt.data_preparation import (  # noqa: E402
    DataPreparation,
    fit_data_preparation,
)
from annz_rebuilt.project import create_project_zip, open_project_zip  # noqa: E402


st.set_page_config(
    page_title="Galaxy Neural Network Analysis",
    page_icon="🌌",
    layout="wide",
)
st.title("Galaxy Neural Network Analysis")
st.write(
    "Review galaxy data, choose the variables the model should use and predict, "
    "prepare training splits, and save the project. Model training comes next."
)


def restore_project(zip_bytes: bytes) -> None:
    """Restore project settings, source data, analysis reports, and preparation."""
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
    if preparation_config is not None:
        preparation = DataPreparation.from_config_dict(preparation_config)
        st.session_state["data_preparation"] = preparation
        st.session_state["train_share_percent"] = round(
            preparation.train_fraction * 100
        )
        st.session_state["validation_share_percent"] = round(
            preparation.validation_fraction * 100
        )
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
        st.session_state["data_preparation"] = None
        st.session_state["data_preparation_signature"] = None


project_upload = st.file_uploader("Open a saved project ZIP (optional)", type=["zip"])
if project_upload is not None:
    project_bytes = project_upload.getvalue()
    fingerprint = hashlib.sha256(project_bytes).hexdigest()
    if st.session_state.get("loaded_project_fingerprint") != fingerprint:
        try:
            restore_project(project_bytes)
            st.success("Project settings loaded.")
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
    data_source_name = (
        st.session_state.get("loaded_data_source_name") or "galaxies.csv"
    )
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
    expected_fingerprint = st.session_state.get("loaded_data_source_sha256")
    if expected_fingerprint != data_fingerprint:
        st.session_state["data_preparation"] = None
        st.session_state["data_preparation_signature"] = None
        st.session_state["saved_quality_summary"] = None
        st.session_state["saved_sample_correlations"] = None

separator = "\t" if data_source_name.lower().endswith(".tsv") else ","
try:
    preview = pd.read_csv(
        io.BytesIO(data_source_bytes),
        sep=separator,
        nrows=100,
    )
    preview.columns = [str(column).lstrip("#").strip() for column in preview.columns]
except Exception as exc:
    st.error(f"Could not read the file as a headered CSV/TSV table: {exc}")
    st.stop()

if len(set(preview.columns)) != len(preview.columns):
    st.error("Column names must be unique after removing a leading #.")
    st.stop()

columns = list(preview.columns)
for state_key in ("inputs", "targets"):
    if state_key in st.session_state:
        st.session_state[state_key] = [
            name for name in st.session_state[state_key] if name in columns
        ]

st.caption(
    f"Showing up to {len(preview):,} preview rows · {len(columns)} variables · "
    f"Source file: {data_source_name}"
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
    help="For example: z. Input and target variables must be separate.",
)

st.subheader("2. Analyze the galaxy data")
st.write(
    "The quality report scans the complete file in batches. It reports missing, "
    "non-numeric, infinite, and finite values, with numeric summary statistics."
)
rows_per_batch = st.number_input(
    "Rows analyzed per batch",
    min_value=1_000,
    max_value=1_000_000,
    value=100_000,
    step=1_000,
    key="rows_per_batch",
)
if st.button("Analyze data"):
    try:
        with st.spinner("Scanning the data source in batches..."):
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
    selected_variables = list(dict.fromkeys([*inputs, *targets]))
    valid_sample = sample.loc[
        :, [name for name in selected_variables if name in sample.columns]
    ].apply(pd.to_numeric, errors="coerce")
    valid_sample = valid_sample.replace(
        [float("inf"), float("-inf")], float("nan")
    ).dropna()
    if len(selected_variables) >= 2 and not valid_sample.empty:
        sample_correlations = valid_sample.corr()
        st.markdown("**Exploratory correlation matrix**")
        st.dataframe(sample_correlations, use_container_width=True)
        st.caption(
            "Correlations use a sample of up to 20,000 rows. They are exploratory "
            "estimates, not exact full-data correlations."
        )
elif st.session_state.get("saved_quality_summary") is not None:
    quality_summary = st.session_state["saved_quality_summary"]
    sample_correlations = st.session_state.get("saved_sample_correlations")
    st.caption("Showing analysis reports saved in this project ZIP.")
    st.dataframe(quality_summary, use_container_width=True)
    if sample_correlations is not None:
        st.markdown("**Saved exploratory correlation matrix**")
        st.dataframe(sample_correlations, use_container_width=True)

st.subheader("3. Prepare training data")
st.write(
    "Rows with missing, non-numeric, or infinite selected values are dropped. "
    "Input and target values are scaled using training rows only."
)
split_left, split_right = st.columns(2)
with split_left:
    train_percent = st.slider(
        "Training share (%)",
        min_value=50,
        max_value=90,
        value=70,
        step=5,
        key="train_share_percent",
    )
with split_right:
    validation_percent = st.slider(
        "Validation share (%)",
        min_value=0,
        max_value=40,
        value=15,
        step=5,
        key="validation_share_percent",
    )
test_percent = 100 - train_percent - validation_percent
st.caption(f"Test share: {test_percent}% (must be at least 0%).")
seed, batch = st.columns(2)
with seed:
    random_seed = st.number_input(
        "Random split seed",
        min_value=0,
        max_value=2_147_483_647,
        value=42,
        step=1,
        key="random_seed",
    )
with batch:
    training_batch_size = st.number_input(
        "Training batch size",
        min_value=128,
        max_value=65_536,
        value=4_096,
        step=128,
        key="training_batch_size",
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
if st.session_state.get("data_preparation_signature") != current_preparation_signature:
    st.session_state["data_preparation"] = None
    st.session_state["data_preparation_signature"] = current_preparation_signature

if st.button("Prepare data splits"):
    try:
        with st.spinner("Checking selected variables and fitting training-only scaling..."):
            prepared = fit_data_preparation(
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
        st.session_state["data_preparation"] = prepared
        st.session_state["data_preparation_signature"] = current_preparation_signature
    except (ValueError, KeyError, OSError) as exc:
        st.error(f"Could not prepare this data source: {exc}")

preparation = st.session_state.get("data_preparation")
preparation_config = None
if preparation is not None:
    preparation_config = preparation.to_config_dict()
    st.success(
        f"Prepared {preparation.usable_rows:,} usable rows out of "
        f"{preparation.total_rows:,}; dropped {preparation.dropped_rows:,}."
    )
    split_summary = pd.DataFrame(
        {
            "split": list(preparation.rows_by_split),
            "rows": list(preparation.rows_by_split.values()),
        }
    )
    st.dataframe(split_summary, hide_index=True, use_container_width=True)
    scaling_summary = pd.DataFrame(
        [
            {
                "variable": name,
                "role": role,
                "training_mean": mean,
                "training_scale": scale,
            }
            for role, names, means, scales in (
                (
                    "input",
                    preparation.input_columns,
                    preparation.input_mean,
                    preparation.input_scale,
                ),
                (
                    "target",
                    preparation.target_columns,
                    preparation.target_mean,
                    preparation.target_scale,
                ),
            )
            for name, mean, scale in zip(names, means, scales)
        ]
    )
    st.markdown("**Scaling fitted on training rows**")
    st.dataframe(scaling_summary, hide_index=True, use_container_width=True)
elif st.session_state.get("saved_data_preparation") is not None:
    preparation_config = st.session_state["saved_data_preparation"]
    st.caption("Saved train/validation/test split and scaling settings were restored.")

st.subheader("4. Describe the network")
left, right = st.columns(2)
with left:
    layer_count = st.number_input(
        "Number of hidden layers",
        min_value=1,
        max_value=10,
        value=1,
        step=1,
        key="layer_count",
    )
with right:
    activation = st.selectbox(
        "Activation function",
        sorted(SUPPORTED_ACTIVATIONS),
        key="activation",
    )
neurons = []
for layer in range(int(layer_count)):
    neurons.append(
        int(
            st.number_input(
                f"Neurons in hidden layer {layer + 1}",
                min_value=1,
                max_value=4096,
                value=10,
                step=1,
                key=f"layer_{layer}",
            )
        )
    )

if st.button("Review and save project", type="primary"):
    try:
        architecture = architecture_from_selections(
            inputs,
            targets,
            neurons,
            activation,
        )
        framed_inputs, framed_targets = architecture.frame_training_data(preview)
        st.success(
            f"Selections are valid: {architecture.n_inputs} input variable(s), "
            f"{architecture.n_outputs} target variable(s)."
        )
        st.markdown("**Input variables**")
        st.dataframe(framed_inputs.head(10), use_container_width=True)
        st.markdown("**Target variable(s)**")
        st.dataframe(framed_targets.head(10), use_container_width=True)

        include_data = st.checkbox(
            "Include the original data source in the project ZIP",
            value=True,
        )
        project_zip = create_project_zip(
            architecture.to_config_dict(),
            data_preparation_config=preparation_config,
            data_source_name=data_source_name,
            data_source_bytes=data_source_bytes if include_data else None,
            data_source_sha256=data_fingerprint,
            quality_summary=quality_summary,
            sample_correlations=sample_correlations,
        )
        first, second = st.columns(2)
        with first:
            st.download_button(
                "Download architecture settings",
                json.dumps(architecture.to_config_dict(), indent=2),
                file_name="architecture_config.json",
                mime="application/json",
            )
        with second:
            st.download_button(
                "Download project ZIP",
                project_zip,
                file_name="galaxy_analysis_project.zip",
                mime="application/zip",
            )
        st.info(
            "The ZIP contains project settings, optional source data under data/, "
            "analysis reports under analysis/, and preparation settings when ready. "
            "It does not contain trained weights or training progress yet."
        )
    except (ValueError, KeyError) as exc:
        st.error(str(exc))

