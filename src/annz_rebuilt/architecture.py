"""Architecture and data-framing helpers for the ANNzPlus Python rebuild.

The legacy ANNzPlus programs read architecture and catalogue data by position.
This module keeps the user-selected column mapping alongside the network shape,
then exports the small architecture file and numeric training rows expected by
the legacy format.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Sequence

import pandas as pd


SUPPORTED_ACTIVATIONS = {
    "leaky_relu",
    "relu",
    "tanh",
    "sigmoid",
    "mish",
    "softplus",
    "silu",
}


@dataclass(frozen=True)
class Architecture:
    """User-selected model shape and data-column mapping.

    Parameters
    ----------
    input_columns:
        Feature columns, in the exact order the network will receive them.
    uncertainty_columns:
        One measurement-uncertainty column for each feature, in matching order.
        The legacy catalogue parser requires these columns even though its
        training routine does not use them in the training loss.
    target_columns:
        Known target columns used during training, in output order.
    hidden_layers:
        Number of neurons in each hidden layer, from first to last.
    activation:
        Activation function selection. The legacy .net architecture format
        does not encode this; legacy builds select it in compiled code.
    """

    input_columns: tuple[str, ...]
    uncertainty_columns: tuple[str, ...]
    target_columns: tuple[str, ...]
    hidden_layers: tuple[int, ...]
    activation: str = "leaky_relu"

    def __post_init__(self) -> None:
        # Normalize list-like input so the model stays immutable and ordered.
        object.__setattr__(self, "input_columns", tuple(self.input_columns))
        object.__setattr__(self, "uncertainty_columns", tuple(self.uncertainty_columns))
        object.__setattr__(self, "target_columns", tuple(self.target_columns))
        object.__setattr__(self, "hidden_layers", tuple(self.hidden_layers))
        object.__setattr__(self, "activation", self.activation.strip().lower())
        self.validate()

    @property
    def n_inputs(self) -> int:
        """Number of input features."""
        return len(self.input_columns)

    @property
    def n_outputs(self) -> int:
        """Number of predicted target values."""
        return len(self.target_columns)

    def validate(self) -> None:
        """Raise ValueError when the selected model framing is inconsistent."""
        if not self.input_columns:
            raise ValueError("Select at least one input feature.")
        if not self.target_columns:
            raise ValueError("Select at least one target column.")
        if len(self.uncertainty_columns) != len(self.input_columns):
            raise ValueError(
                "Choose exactly one uncertainty column for each input feature."
            )
        if not self.hidden_layers:
            raise ValueError("Choose at least one hidden layer.")
        if any(not name or not name.strip() for name in (
            *self.input_columns, *self.uncertainty_columns, *self.target_columns
        )):
            raise ValueError("Column names cannot be blank.")
        if any(size < 1 for size in self.hidden_layers):
            raise ValueError("Each hidden layer must contain at least one neuron.")
        if self.activation not in SUPPORTED_ACTIVATIONS:
            choices = ", ".join(sorted(SUPPORTED_ACTIVATIONS))
            raise ValueError(f"Unsupported activation '{self.activation}'. Choose: {choices}.")
        selected = [
            *self.input_columns,
            *self.uncertainty_columns,
            *self.target_columns,
        ]
        if len(set(selected)) != len(selected):
            raise ValueError(
                "Input, uncertainty, and target columns must be distinct selections."
            )

    def validate_dataframe(self, dataframe: pd.DataFrame) -> None:
        """Check required columns and ensure their values are finite numbers."""
        self.validate()
        required = [
            *self.input_columns,
            *self.uncertainty_columns,
            *self.target_columns,
        ]
        missing = [name for name in required if name not in dataframe.columns]
        if missing:
            raise ValueError(f"Dataset is missing selected columns: {', '.join(missing)}")
        if dataframe.empty:
            raise ValueError("Dataset has no data rows.")

        invalid: list[str] = []
        for name in required:
            numeric = pd.to_numeric(dataframe[name], errors="coerce")
            if numeric.isna().any() or not numeric.map(math.isfinite).all():
                invalid.append(name)
        if invalid:
            raise ValueError(
                "Selected columns must contain only finite numeric values. "
                f"Check: {', '.join(invalid)}"
            )

    def frame_training_data(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Return columns in legacy order: inputs, input errors, then targets."""
        self.validate_dataframe(dataframe)
        ordered = [
            *self.input_columns,
            *self.uncertainty_columns,
            *self.target_columns,
        ]
        framed = dataframe.loc[:, ordered].apply(pd.to_numeric, errors="raise")
        framed.columns = (
            [f"input_{i}" for i in range(self.n_inputs)]
            + [f"input_error_{i}" for i in range(self.n_inputs)]
            + [f"target_{i}" for i in range(self.n_outputs)]
        )
        return framed.reset_index(drop=True)

    def to_legacy_architecture_text(self) -> str:
        """Serialize the network shape using the legacy annz_net file syntax."""
        lines = [
            "# Generated by the ANNzPlus Python rebuild",
            "# Network shape: inputs, outputs, hidden-layer count and layer sizes",
            f"N_INPUTS {self.n_inputs}",
            f"N_OUTPUTS {self.n_outputs}",
            f"N_LAYERS {len(self.hidden_layers)} {' '.join(map(str, self.hidden_layers))}",
        ]
        return "\n".join(lines) + "\n"

    def write_legacy_architecture(self, destination: str | Path) -> Path:
        """Write a legacy-compatible architecture file and return its path."""
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_legacy_architecture_text(), encoding="utf-8")
        return path

    def to_config_dict(self) -> dict[str, Any]:
        """Serialize model shape and column mapping for the web app to reload."""
        return {
            "input_columns": list(self.input_columns),
            "uncertainty_columns": list(self.uncertainty_columns),
            "target_columns": list(self.target_columns),
            "hidden_layers": list(self.hidden_layers),
            "activation": self.activation,
        }

    def write_config(self, destination: str | Path) -> Path:
        """Save app-specific column mapping next to the legacy architecture."""
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_config_dict(), indent=2) + "\n",
            encoding="utf-8",
        )
        return path


def architecture_from_selections(
    input_columns: Sequence[str],
    uncertainty_columns: Sequence[str],
    target_columns: Sequence[str],
    hidden_layers: Sequence[int],
    activation: str = "leaky_relu",
) -> Architecture:
    """Build a validated Architecture from selections made in a UI."""
    return Architecture(
        input_columns=tuple(input_columns),
        uncertainty_columns=tuple(uncertainty_columns),
        target_columns=tuple(target_columns),
        hidden_layers=tuple(int(size) for size in hidden_layers),
        activation=activation,
    )
