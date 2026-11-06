"""Architecture and data-framing helpers for the ANNzPlus Python rebuild.

The legacy ANNzPlus programs read architecture and catalogue data by position.
This module keeps the user's selected column mapping alongside the network
shape, then exports legacy-compatible architecture and catalogue files.
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
    """Network shape and the dataset columns assigned to each role.

    The legacy catalogue format is positional:
    input values, one uncertainty per input, then optional target values.
    The legacy .net file stores the network shape but not these column names
    or the activation choice, so the app should save its JSON config too.
    """

    input_columns: tuple[str, ...]
    uncertainty_columns: tuple[str, ...]
    target_columns: tuple[str, ...]
    hidden_layers: tuple[int, ...]
    activation: str = "leaky_relu"

    def __post_init__(self) -> None:
        object.__setattr__(self, "input_columns", tuple(self.input_columns))
        object.__setattr__(self, "uncertainty_columns", tuple(self.uncertainty_columns))
        object.__setattr__(self, "target_columns", tuple(self.target_columns))
        object.__setattr__(self, "hidden_layers", tuple(self.hidden_layers))
        object.__setattr__(self, "activation", self.activation.strip().lower())
        self.validate()

    @property
    def n_inputs(self) -> int:
        """Number of selected input features."""
        return len(self.input_columns)

    @property
    def n_outputs(self) -> int:
        """Number of selected prediction targets."""
        return len(self.target_columns)

    def validate(self) -> None:
        """Check that the model shape and selected columns are consistent."""
        if not self.input_columns:
            raise ValueError("Select at least one input feature.")
        if not self.target_columns:
            raise ValueError("Select at least one target column.")
        if len(self.uncertainty_columns) != len(self.input_columns):
            raise ValueError("Choose one uncertainty column for each input feature.")
        if not self.hidden_layers:
            raise ValueError("Choose at least one hidden layer.")
        names = (*self.input_columns, *self.uncertainty_columns, *self.target_columns)
        if any(not name or not name.strip() for name in names):
            raise ValueError("Column names cannot be blank.")
        if len(set(names)) != len(names):
            raise ValueError("Input, uncertainty, and target columns must be distinct.")
        if any(size < 1 for size in self.hidden_layers):
            raise ValueError("Each hidden layer must contain at least one neuron.")
        if self.activation not in SUPPORTED_ACTIVATIONS:
            choices = ", ".join(sorted(SUPPORTED_ACTIVATIONS))
            raise ValueError(f"Unsupported activation '{self.activation}'. Choose: {choices}.")

    def _validate_dataframe(
        self, dataframe: pd.DataFrame, *, include_targets: bool
    ) -> list[str]:
        self.validate()
        required = [*self.input_columns, *self.uncertainty_columns]
        if include_targets:
            required.extend(self.target_columns)

        missing = [name for name in required if name not in dataframe.columns]
        if missing:
            raise ValueError(f"Dataset is missing selected columns: {', '.join(missing)}")
        if dataframe.empty:
            raise ValueError("Dataset has no data rows.")

        invalid = []
        for name in required:
            numeric = pd.to_numeric(dataframe[name], errors="coerce")
            if numeric.isna().any() or not numeric.map(math.isfinite).all():
                invalid.append(name)
        if invalid:
            raise ValueError(
                "Selected columns must contain only finite numeric values. "
                f"Check: {', '.join(invalid)}"
            )
        return required

    def validate_training_dataframe(self, dataframe: pd.DataFrame) -> None:
        """Validate inputs, their uncertainties, and target values for training."""
        self._validate_dataframe(dataframe, include_targets=True)

    def validate_prediction_dataframe(self, dataframe: pd.DataFrame) -> None:
        """Validate inputs and uncertainties for prediction; targets are optional."""
        self._validate_dataframe(dataframe, include_targets=False)

    def frame_training_data(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Select and order training columns as inputs, uncertainties, targets."""
        self.validate_training_dataframe(dataframe)
        ordered = [
            *self.input_columns,
            *self.uncertainty_columns,
            *self.target_columns,
        ]
        return self._frame(dataframe, ordered)

    def frame_prediction_data(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Select and order prediction columns as inputs then uncertainties."""
        self.validate_prediction_dataframe(dataframe)
        ordered = [*self.input_columns, *self.uncertainty_columns]
        return self._frame(dataframe, ordered)

    def _frame(self, dataframe: pd.DataFrame, ordered: list[str]) -> pd.DataFrame:
        framed = dataframe.loc[:, ordered].apply(pd.to_numeric, errors="raise")
        framed.columns = (
            [f"input_{i}" for i in range(self.n_inputs)]
            + [f"input_error_{i}" for i in range(self.n_inputs)]
            + (
                [f"target_{i}" for i in range(self.n_outputs)]
                if len(ordered) == 2 * self.n_inputs + self.n_outputs
                else []
            )
        )
        return framed.reset_index(drop=True)

    @staticmethod
    def legacy_catalogue_text(framed_data: pd.DataFrame) -> str:
        """Serialize a framed catalogue without headers or an index.

        The C++ reader expects whitespace-separated numbers and gets column
        roles from their positions, so names must be kept in the app config.
        """
        return framed_data.to_csv(
            sep=" ",
            header=False,
            index=False,
            float_format="%.10g",
            lineterminator="\n",
        )

    def write_legacy_catalogue(
        self, dataframe: pd.DataFrame, destination: str | Path, *, prediction: bool = False
    ) -> Path:
        """Frame and write a training or prediction catalogue for the legacy app."""
        framed = (
            self.frame_prediction_data(dataframe)
            if prediction
            else self.frame_training_data(dataframe)
        )
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.legacy_catalogue_text(framed), encoding="utf-8")
        return path

    def to_legacy_architecture_text(self) -> str:
        """Serialize network shape using the legacy annz_net file syntax."""
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
        """Return the full model and column mapping for the web app to persist."""
        return {
            "input_columns": list(self.input_columns),
            "uncertainty_columns": list(self.uncertainty_columns),
            "target_columns": list(self.target_columns),
            "hidden_layers": list(self.hidden_layers),
            "activation": self.activation,
        }

    def write_config(self, destination: str | Path) -> Path:
        """Save app-specific column mapping beside the legacy architecture file."""
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_config_dict(), indent=2) + "\n", encoding="utf-8")
        return path


def architecture_from_selections(
    input_columns: Sequence[str],
    uncertainty_columns: Sequence[str],
    target_columns: Sequence[str],
    hidden_layers: Sequence[int],
    activation: str = "leaky_relu",
) -> Architecture:
    """Create a validated Architecture from selections made in a web form."""
    return Architecture(
        input_columns=tuple(input_columns),
        uncertainty_columns=tuple(uncertainty_columns),
        target_columns=tuple(target_columns),
        hidden_layers=tuple(int(size) for size in hidden_layers),
        activation=activation,
    )
