"""Local Streamlit interface for choosing data variables and a network shape."""

from __future__ import annotations

import io
import json
import hashlib
import sys
from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from annz_rebuilt import architecture_from_selections  # noqa: E402
from annz_rebuilt.architecture import SUPPORTED_ACTIVATIONS  # noqa: E402
from annz_rebuilt.project_bundle import create_project_bundle, read_project_bundle  # noqa: E402


st.set_page_config(page_title="Galaxy Neural Network Analysis", page_icon="🌌", layout="wide")
st.title("Galaxy Neural Network Analysis")
st.write(
    "Choose which catalogue variables the model will use as inputs and which "
    "variable(s) it should learn to predict. Then describe the network shape."
)

project_upload = st.file_uploader(
    "Open a saved project ZIP (optional)", type=["zip"], key="project_zip_upload"
)
if project_upload is not None:
    project_bytes = project_upload.getvalue()
    project_fingerprint = hashlib.sha256(project_bytes).hexdigest()
    if st.session_state.get("loaded_project_fingerprint") != project_fingerprint:
        try:
            config, catalogue_name, catalogue_bytes = read_project_bundle(project_bytes)
            st.session_state["loaded_project_fingerprint"] = project_fingerprint
            st.session_state["loaded_project_config"] = config
            st.session_state["loaded_catalogue_name"] = catalogue_name
            st.session_state["loaded_catalogue_bytes"] = catalogue_bytes
            st.session_state["inputs"] = config.get("input_variables", [])
            st.session_state["targets"] = config.get("target_variables", [])
            saved_layers = config.get("hidden_layers", [10])
            st.session_state["layer_count"] = len(saved_layers)
            st.session_state["activation"] = config.get("activation", "leaky_relu")
            for layer_index, size in enumerate(saved_layers):
                st.session_state[f"layer_{layer_index}"] = int(size)
            st.success("Project settings loaded.")
        except (ValueError, KeyError) as exc:
            st.error(f"Could not open project: {exc}")
            st.stop()

loaded_catalogue = st.session_state.get("loaded_catalogue_bytes")
uploaded = st.file_uploader(
    "Upload a CSV or TSV catalogue with a header row",
    type=["csv", "tsv", "txt"],
    key="catalogue_upload",
)
if loaded_catalogue is not None:
    catalogue_bytes = loaded_catalogue
    catalogue_name = st.session_state.get("loaded_catalogue_name") or "catalogue.csv"
    st.caption(f"Using catalogue saved in project: {catalogue_name}")
elif uploaded is not None:
    catalogue_bytes = uploaded.getvalue()
    catalogue_name = uploaded.name
else:
    st.info(
        "Upload a catalogue or open a project ZIP that contains one. "
        "The table's first row should contain column names."
    )
    st.stop()

try:
    separator = "\t" if catalogue_name.lower().endswith(".tsv") else None
    data = pd.read_csv(io.BytesIO(catalogue_bytes), sep=separator, engine="python")
    data.columns = [str(column).lstrip("#").strip() for column in data.columns]
except Exception as exc:
    st.error(f"Could not read this file as a headered table: {exc}")
    st.stop()

if len(set(data.columns)) != len(data.columns):
    st.error("The file has duplicate column names after removing a leading #. Please make the headers unique.")
    st.stop()

st.caption(f"{len(data):,} rows · {len(data.columns)} variables")
st.dataframe(data.head(10), use_container_width=True)
columns = list(data.columns)
if "inputs" in st.session_state:
    st.session_state["inputs"] = [name for name in st.session_state["inputs"] if name in columns]
if "targets" in st.session_state:
    st.session_state["targets"] = [name for name in st.session_state["targets"] if name in columns]

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
    help="For example: z. Keep target variables separate from inputs.",
)

st.subheader("2. Choose the network shape")
left, right = st.columns(2)
with left:
    layer_count = st.number_input(
        "Number of hidden layers", min_value=1, max_value=10, value=1, step=1,
        key="layer_count",
    )
with right:
    activation = st.selectbox(
        "Activation function", sorted(SUPPORTED_ACTIVATIONS), index=0,
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

if st.button("Review selections", type="primary"):
    try:
        architecture = architecture_from_selections(
            inputs, targets, neurons, activation
        )
        framed_inputs, framed_targets = architecture.frame_training_data(data)
        st.success(
            f"Selections are valid: {architecture.n_inputs} input variable(s), "
            f"{architecture.n_outputs} target variable(s)."
        )
        st.markdown("**Input variables**")
        st.dataframe(framed_inputs.head(10), use_container_width=True)
        st.markdown("**Target variable(s)**")
        st.dataframe(framed_targets.head(10), use_container_width=True)
        st.download_button(
            "Download architecture configuration",
            json.dumps(architecture.to_config_dict(), indent=2),
            file_name="architecture_config.json",
            mime="application/json",
        )
        st.markdown("**Save this session**")
        include_catalogue = st.checkbox(
            "Include the uploaded catalogue in the project ZIP", value=True,
            help="Turn this off to create a smaller ZIP. You will need to upload the same catalogue when reopening it.",
        )
        bundle = create_project_bundle(
            architecture.to_config_dict(),
            catalogue_name=catalogue_name if include_catalogue else None,
            catalogue_bytes=catalogue_bytes if include_catalogue else None,
        )
        st.download_button(
            "Download project ZIP",
            bundle,
            file_name="galaxy_analysis_project.zip",
            mime="application/zip",
        )
        st.info(
            "The ZIP saves the architecture settings and, if selected, the original "
            "catalogue. Training results and resumable model checkpoints are not "
            "available until model training is implemented."
        )
    except (ValueError, KeyError) as exc:
        st.error(str(exc))
