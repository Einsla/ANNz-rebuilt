"""Portable trained-model bundles for inference and future training continuation."""

from __future__ import annotations

import io
import json
import zipfile
from dataclasses import dataclass
from typing import Any

import pandas as pd

MODEL_FORMAT_VERSION = 1
MANIFEST_NAME = "model.json"


@dataclass(frozen=True)
class ModelBundle:
    architecture: dict[str, Any]
    data_preparation: dict[str, Any]
    checkpoint_bytes: bytes
    training_config: dict[str, Any]
    training_history: pd.DataFrame | None
    optimisation_results: pd.DataFrame | None
    source_data_sha256: str | None = None


def create_model_bundle(
    architecture: dict[str, Any],
    data_preparation: dict[str, Any],
    checkpoint_bytes: bytes,
    *,
    training_config: dict[str, Any] | None = None,
    training_history: pd.DataFrame | None = None,
    optimisation_results: pd.DataFrame | None = None,
    source_data_sha256: str | None = None,
) -> bytes:
    """Create a self-contained model artifact without embedding the catalogue."""
    if not isinstance(checkpoint_bytes, bytes) or not checkpoint_bytes:
        raise ValueError("A trained model checkpoint is required.")
    required_architecture = {"input_columns", "target_columns", "hidden_layers", "activation"}
    if not required_architecture.issubset(architecture):
        raise ValueError("The model architecture configuration is incomplete.")
    required_preparation = {
        "input_columns", "target_columns", "input_mean", "input_scale",
        "target_mean", "target_scale", "rows_per_batch", "batch_size",
        "train_fraction", "validation_fraction", "test_fraction", "random_seed",
    }
    if not required_preparation.issubset(data_preparation):
        raise ValueError("The model's fitted data-preparation settings are incomplete.")

    manifest = {
        "format_version": MODEL_FORMAT_VERSION,
        "architecture": architecture,
        "data_preparation": data_preparation,
        "training_config": training_config or {},
        "source_data_sha256": source_data_sha256,
        "files": {
            "checkpoint": "model/checkpoint.pt",
            "history": "training/history.csv" if training_history is not None else None,
            "optimisation_results": "training/optimisation_results.csv" if optimisation_results is not None else None,
        },
    }
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(MANIFEST_NAME, json.dumps(manifest, indent=2) + "\n")
        archive.writestr("model/checkpoint.pt", checkpoint_bytes)
        if training_history is not None:
            archive.writestr("training/history.csv", training_history.to_csv(index=False, lineterminator="\n"))
        if optimisation_results is not None:
            archive.writestr("training/optimisation_results.csv", optimisation_results.to_csv(index=False, lineterminator="\n"))
        archive.writestr(
            "README.txt",
            "ANNzPlus trained model bundle\n"
            "Contains network architecture, fitted input/target scaling, model weights,\n"
            "optimiser state when available, training settings and loss history.\n"
            "The original training catalogue is intentionally not included.\n"
            "Load this .annzmodel file in the app, then upload a catalogue containing\n"
            "the saved input columns to generate predictions without retraining.\n",
        )
    return output.getvalue()


def open_model_bundle(blob: bytes) -> ModelBundle:
    """Read and validate a portable model bundle."""
    try:
        with zipfile.ZipFile(io.BytesIO(blob), "r") as archive:
            names = set(archive.namelist())
            if MANIFEST_NAME not in names:
                raise ValueError("This file is not an ANNzPlus model bundle.")
            manifest = json.loads(archive.read(MANIFEST_NAME).decode("utf-8"))
            if manifest.get("format_version") != MODEL_FORMAT_VERSION:
                raise ValueError("This model bundle format is unsupported.")
            architecture = manifest.get("architecture")
            preparation = manifest.get("data_preparation")
            training_config = manifest.get("training_config", {})
            files = manifest.get("files", {})
            if not isinstance(architecture, dict) or not isinstance(preparation, dict):
                raise ValueError("The model bundle is missing architecture or scaling settings.")
            if not isinstance(training_config, dict) or not isinstance(files, dict):
                raise ValueError("The model bundle manifest is invalid.")
            checkpoint_path = files.get("checkpoint")
            if checkpoint_path != "model/checkpoint.pt" or checkpoint_path not in names:
                raise ValueError("The model bundle checkpoint is missing or invalid.")
            checkpoint = archive.read(checkpoint_path)
            if not checkpoint:
                raise ValueError("The model bundle checkpoint is empty.")
            history = _read_csv(archive, names, files.get("history"))
            results = _read_csv(archive, names, files.get("optimisation_results"))
            return ModelBundle(
                architecture=architecture,
                data_preparation=preparation,
                checkpoint_bytes=checkpoint,
                training_config=training_config,
                training_history=history,
                optimisation_results=results,
                source_data_sha256=manifest.get("source_data_sha256"),
            )
    except zipfile.BadZipFile as exc:
        raise ValueError("The selected file is not a valid model bundle ZIP.") from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("The model bundle manifest could not be read.") from exc


def _read_csv(archive: zipfile.ZipFile, names: set[str], member: str | None) -> pd.DataFrame | None:
    if member is None:
        return None
    if not isinstance(member, str) or member not in names or member.startswith("/") or ".." in member.split("/"):
        raise ValueError("The model bundle contains an unsafe or missing file reference.")
    return pd.read_csv(io.BytesIO(archive.read(member)))
