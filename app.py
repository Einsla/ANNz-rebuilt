"""Local Streamlit interface for choosing data variables and a network shape."""

from __future__ import annotations

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


st.set_page_config(page_title="Galaxy Neural Network Analysis", page_icon="🌌", layout="wide")
st.title("Galaxy Neural Network Analysis")
st.write(
    "Choose which catalogue variables the model will use as inputs and which "
    "variable(s) it should learn to predict. Then describe the network shape."
)

uploaded = st.file_uploader(
    "Upload a CSV or TSV catalogue with a header row", type=["csv", "tsv", "txt"]
)
if uploaded is None:
    st.info(
        "Start with a table whose first row contains column names, such as "
        "z, veldisp, deVAB_u, and deVMag_u."
    )
    st.stop()

try:
    separator = "\t" if uploaded.name.lower().endswith(".tsv") else None
    data = pd.read_csv(io.BytesIO(uploaded.getvalue()), sep=separator, engine="python")
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
        "Number of hidden layers", min_value=1, max_value=10, value=1, step=1
    )
with right:
    activation = st.selectbox(
        "Activation function", sorted(SUPPORTED_ACTIVATIONS), index=0
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
        st.info(
            "This prototype checks the selections and prepares the input and target "
            "tables. Model training, project save/open, and resumable checkpoints "
            "are later features."
        )
    except (ValueError, KeyError) as exc:
        st.error(str(exc))
