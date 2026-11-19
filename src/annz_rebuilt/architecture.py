"""Model architecture and input/target data framing for the Python app."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
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
class NetworkStructure:
    """Describe network topology without choosing activation functions."""

    input_columns: tuple[str, ...]
    target_columns: tuple[str, ...]
    hidden_layers: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "input_columns", tuple(self.input_columns))
        object.__setattr__(self, "target_columns", tuple(self.target_columns))
        object.__setattr__(
            self, "hidden_layers", tuple(int(size) for size in self.hidden_layers)
        )
        if not self.input_columns or not self.target_columns:
            raise ValueError("Select at least one input and one target variable.")
        if not self.hidden_layers or any(size < 1 for size in self.hidden_layers):
            raise ValueError(
                "Provide at least one hidden layer with one or more neurons."
            )
        selected = (*self.input_columns, *self.target_columns)
        if any(not name.strip() for name in selected):
            raise ValueError("Variable names cannot be blank.")
        if len(set(selected)) != len(selected):
            raise ValueError("Input and target variables must be distinct.")

    def to_config_dict(self) -> dict[str, Any]:
        """Return the activation-independent topology for project records."""
        return {
            "input_columns": list(self.input_columns),
            "target_columns": list(self.target_columns),
            "hidden_layers": list(self.hidden_layers),
        }

    def with_activations(
        self, activations: Sequence[str], *, legacy_activation: str | None = None
    ) -> "Architecture":
        """Create a trainable architecture after activation choices are known."""
        names = tuple(str(name).strip().lower() for name in activations)
        if not names:
            raise ValueError("Choose an activation function for every hidden layer.")
        return Architecture(
            self.input_columns,
            self.target_columns,
            self.hidden_layers,
            legacy_activation or names[0],
            names,
        )


@dataclass(frozen=True)
class Architecture:
    """A trainable network configuration, including its chosen activations."""

    input_columns: tuple[str, ...]
    target_columns: tuple[str, ...]
    hidden_layers: tuple[int, ...]
    activation: str = "leaky_relu"
    hidden_activations: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "input_columns", tuple(self.input_columns))
        object.__setattr__(self, "target_columns", tuple(self.target_columns))
        object.__setattr__(self, "hidden_layers", tuple(self.hidden_layers))
        object.__setattr__(self, "activation", self.activation.strip().lower())
        activations = self.hidden_activations
        if activations is None:
            activations = (self.activation,) * len(self.hidden_layers)
        else:
            activations = tuple(str(name).strip().lower() for name in activations)
        object.__setattr__(self, "hidden_activations", activations)
        self.validate()

    @property
    def effective_activations(self) -> tuple[str, ...]:
        """Return the activation function assigned to each hidden layer."""
        return self.hidden_activations or ()

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
            raise ValueError(
                "Provide at least one hidden layer with one or more neurons."
            )
        if self.activation not in SUPPORTED_ACTIVATIONS:
            raise ValueError(f"Unsupported activation: {self.activation}")
        if len(self.effective_activations) != len(self.hidden_layers):
            raise ValueError(
                "Provide exactly one activation function per hidden layer."
            )
        unsupported = sorted(
            set(self.effective_activations).difference(SUPPORTED_ACTIVATIONS)
        )
        if unsupported:
            raise ValueError(
                "Unsupported hidden-layer activations: " + ", ".join(unsupported)
            )
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
            # pd.to_numeric checks that catalogue values can be interpreted as numbers.
            values = pd.to_numeric(data[name], errors="coerce")
            if values.isna().any() or not values.map(math.isfinite).all():
                invalid.append(name)
        if invalid:
            raise ValueError(
                "Selected variables must contain finite numbers. Check: "
                + ", ".join(invalid)
            )

    def frame_training_data(
        self, data: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return named input and target tables for model training."""
        self._validate_data(data, (*self.input_columns, *self.target_columns))
        inputs = data.loc[:, list(self.input_columns)].apply(
            pd.to_numeric, errors="raise"
        )
        targets = data.loc[:, list(self.target_columns)].apply(
            pd.to_numeric, errors="raise"
        )
        # reset_index(drop=True) gives both tables a clean 0..N-1 row index.
        return inputs.reset_index(drop=True), targets.reset_index(drop=True)

    def frame_prediction_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return only the input variables required for prediction."""
        self._validate_data(data, self.input_columns)
        return (
            data.loc[:, list(self.input_columns)]
            .apply(pd.to_numeric, errors="raise")
            .reset_index(drop=True)
        )

    def to_config_dict(self) -> dict[str, Any]:
        config = {
            "input_columns": list(self.input_columns),
            "target_columns": list(self.target_columns),
            "hidden_layers": list(self.hidden_layers),
            "activation": self.activation,
        }
        # Keep old single-activation configs and checkpoint signatures unchanged.
        if len(set(self.effective_activations)) > 1 or (
            self.effective_activations
            and self.effective_activations[0] != self.activation
        ):
            config["hidden_activations"] = list(self.effective_activations)
        return config

    def to_export_config_dict(self) -> dict[str, Any]:
        """Return a self-describing export with one activation per hidden layer.

        The internal config keeps legacy single-activation files compact. Exports
        spell out every layer assignment so users can inspect and recreate the
        optimised network without inferring repeated activations.
        """
        config = self.to_config_dict()
        config["hidden_activations"] = list(self.effective_activations)
        return config

    def write_config(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_export_config_dict(), indent=2) + "\n", encoding="utf-8"
        )
        return destination


def architecture_from_selections(
    input_columns: Sequence[str],
    target_columns: Sequence[str],
    hidden_layers: Sequence[int],
    activation: str = "leaky_relu",
    hidden_activations: Sequence[str] | None = None,
) -> Architecture:
    """Create a validated architecture from UI selections."""
    return Architecture(
        tuple(input_columns),
        tuple(target_columns),
        tuple(int(size) for size in hidden_layers),
        activation,
        tuple(hidden_activations) if hidden_activations is not None else None,
    )
