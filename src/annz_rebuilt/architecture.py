"""Model architecture and input/target data framing for the Python app."""
from __future__ import annotations
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Sequence
import pandas as pd

SUPPORTED_ACTIVATIONS = {"leaky_relu", "relu", "tanh", "sigmoid", "mish", "softplus", "silu"}

@dataclass(frozen=True)
class Architecture:
    """Network shape and the catalogue variables assigned to each role."""
    input_columns: tuple[str, ...]
    target_columns: tuple[str, ...]
    hidden_layers: tuple[int, ...]
    activation: str = "leaky_relu"

    def __post_init__(self) -> None:
        object.__setattr__(self, "input_columns", tuple(self.input_columns))
        object.__setattr__(self, "target_columns", tuple(self.target_columns))
        object.__setattr__(self, "hidden_layers", tuple(self.hidden_layers))
        object.__setattr__(self, "activation", self.activation.strip().lower())
        self.validate()

    @property
    def n_inputs(self) -> int:
        return len(self.input_columns)

    @property
    def n_outputs(self) -> int:
        return len(self.target_columns)

    def validate(self) -> None:
        if not self.input_columns:
            raise ValueError("Select at least one input variable.")
        if not self.target_columns:
            raise ValueError("Select at least one target variable.")
        if not self.hidden_layers or any(size < 1 for size in self.hidden_layers):
            raise ValueError("Provide at least one hidden layer with one or more neurons.")
        if self.activation not in SUPPORTED_ACTIVATIONS:
            raise ValueError(f"Unsupported activation: {self.activation}")
        selected = (*self.input_columns, *self.target_columns)
        if any(not name.strip() for name in selected):
            raise ValueError("Variable names cannot be blank.")
        if len(set(selected)) != len(selected):
            raise ValueError("Input and target variables must be distinct.")

    @staticmethod
    def _validate_data(data: pd.DataFrame, columns: Sequence[str]) -> None:
        missing = [name for name in columns if name not in data.columns]
        if missing:
            raise ValueError("Missing selected variables: " + ", ".join(missing))
        if data.empty:
            raise ValueError("The dataset has no data rows.")
        invalid = []
        for name in columns:
            values = pd.to_numeric(data[name], errors="coerce")
            if values.isna().any() or not values.map(math.isfinite).all():
                invalid.append(name)
        if invalid:
            raise ValueError("Selected variables must contain finite numbers. Check: " + ", ".join(invalid))

    def frame_training_data(self, data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return named input and target tables for model training."""
        self._validate_data(data, (*self.input_columns, *self.target_columns))
        inputs = data.loc[:, list(self.input_columns)].apply(pd.to_numeric, errors="raise")
        targets = data.loc[:, list(self.target_columns)].apply(pd.to_numeric, errors="raise")
        return inputs.reset_index(drop=True), targets.reset_index(drop=True)

    def frame_prediction_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return only the input variables required for prediction."""
        self._validate_data(data, self.input_columns)
        return data.loc[:, list(self.input_columns)].apply(
            pd.to_numeric, errors="raise"
        ).reset_index(drop=True)

    def to_config_dict(self) -> dict[str, Any]:
        return {
            "input_columns": list(self.input_columns),
            "target_columns": list(self.target_columns),
            "hidden_layers": list(self.hidden_layers),
            "activation": self.activation,
        }

    def write_config(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_config_dict(), indent=2) + "\n", encoding="utf-8")
        return destination

def architecture_from_selections(
    input_columns: Sequence[str],
    target_columns: Sequence[str],
    hidden_layers: Sequence[int],
    activation: str = "leaky_relu",
) -> Architecture:
    """Create a validated architecture from UI selections."""
    return Architecture(tuple(input_columns), tuple(target_columns), tuple(int(size) for size in hidden_layers), activation)
