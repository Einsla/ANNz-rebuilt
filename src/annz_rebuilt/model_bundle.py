"""Portable trained-model bundles for inference and future training continuation."""

from __future__ import annotations

import io
import json
import zipfile
from dataclasses import dataclass
from typing import Any

import pandas as pd

MODEL_FORMAT_VERSION = 2
SUPPORTED_FORMAT_VERSIONS = {1, MODEL_FORMAT_VERSION}
MANIFEST_NAME = "model.json"


@dataclass(frozen=True)
class ModelBundle:
    architecture: dict[str, Any]
    data_preparation: dict[str, Any]
    checkpoint_bytes: bytes
    training_config: dict[str, Any]
    training_history: pd.DataFrame | None
    optimisation_results: pd.DataFrame | None
    evaluation_report: pd.DataFrame | None = None
    source_data_sha256: str | None = None
    bundle_metadata: dict[str, Any] | None = None
    committee_metadata: dict[str, Any] | None = None


def create_model_bundle(
    architecture: dict[str, Any],
    data_preparation: dict[str, Any],
    checkpoint_bytes: bytes,
    *,
    training_config: dict[str, Any] | None = None,
    training_history: pd.DataFrame | None = None,
    optimisation_results: pd.DataFrame | None = None,
    evaluation_report: pd.DataFrame | None = None,
    source_data_sha256: str | None = None,
    bundle_metadata: dict[str, Any] | None = None,
) -> bytes:
    """Create a self-contained model artifact without embedding the catalogue."""
    if not isinstance(checkpoint_bytes, bytes) or not checkpoint_bytes:
        raise ValueError("A trained model checkpoint is required.")
    required_architecture = {
        "input_columns",
        "target_columns",
        "hidden_layers",
        "activation",
    }
    if not required_architecture.issubset(architecture):
        raise ValueError("The model architecture configuration is incomplete.")
    required_preparation = {
        "input_columns",
        "target_columns",
        "input_mean",
        "input_scale",
        "target_mean",
        "target_scale",
        "rows_per_batch",
        "batch_size",
        "train_fraction",
        "validation_fraction",
        "test_fraction",
        "random_seed",
    }
    if not required_preparation.issubset(data_preparation):
        raise ValueError("The model's fitted data-preparation settings are incomplete.")

    saved_training = dict(training_config or {})
    error_mapping = saved_training.get("prediction_error_columns", {})
    if not isinstance(error_mapping, dict):
        raise ValueError("Prediction error-column mappings must be a JSON object.")
    unknown_error_inputs = set(error_mapping).difference(architecture["input_columns"])
    if unknown_error_inputs:
        raise ValueError(
            "Prediction error mappings refer to unknown model inputs: "
            + ", ".join(sorted(unknown_error_inputs))
        )

    committee_metadata = None
    if saved_training.get("committee_enabled"):
        try:
            from .committee import unpack_committee

            members, committee_metadata = unpack_committee(checkpoint_bytes)
            if len(members) != int(saved_training.get("committee_size", len(members))):
                raise ValueError("Committee size does not match the saved checkpoint.")
            saved_training["committee_size"] = len(members)
            saved_training["committee_members"] = committee_metadata.get(
                "members", saved_training.get("committee_members", [])
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Could not validate committee checkpoint: {exc}") from exc

    metadata = {
        "application": "ANNzPlus Python rebuild",
        "purpose": "trained-model inference and training continuation",
        "includes_training_catalogue": False,
        **(bundle_metadata or {}),
    }
    manifest = {
        "format_version": MODEL_FORMAT_VERSION,
        "architecture": architecture,
        "data_preparation": data_preparation,
        "training_config": saved_training,
        "bundle_metadata": metadata,
        "committee_metadata": committee_metadata,
        "source_data_sha256": source_data_sha256,
        "files": {
            "checkpoint": "model/checkpoint.pt",
            "history": "training/history.csv" if training_history is not None else None,
            "optimisation_results": (
                "training/optimisation_results.csv"
                if optimisation_results is not None
                else None
            ),
            "evaluation_report": (
                "evaluation/metrics.csv" if evaluation_report is not None else None
            ),
        },
    }
    output = io.BytesIO()
    # ZipFile builds a compressed archive in memory; writestr adds files without temporary files.
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(MANIFEST_NAME, json.dumps(manifest, indent=2) + "\n")
        archive.writestr("model/checkpoint.pt", checkpoint_bytes)
        if training_history is not None:
            archive.writestr(
                "training/history.csv",
                training_history.to_csv(index=False, lineterminator="\n"),
            )
        if optimisation_results is not None:
            archive.writestr(
                "training/optimisation_results.csv",
                optimisation_results.to_csv(index=False, lineterminator="\n"),
            )
        if evaluation_report is not None:
            archive.writestr(
                "evaluation/metrics.csv",
                evaluation_report.to_csv(index=False, lineterminator="\n"),
            )
        archive.writestr(
            "README.txt",
            "ANNzPlus trained model bundle (format v2)\n"
            "Contains architecture (including per-layer activations), fitted scaling,\n"
            "network checkpoint(s), optimiser/training settings, split and batch settings,\n"
            "committee member metadata, prediction error-column mappings, loss history,\n"
            "and optimisation results when available.\n"
            "The original training catalogue is intentionally not included.\n"
            "Load this .annzmodel file in the app and upload a catalogue containing\n"
            "the saved input columns. Saved error-column mappings are restored for both single-network and committee prediction.\n",
        )
    return output.getvalue()


def open_model_bundle(blob: bytes) -> ModelBundle:
    """Read and validate a portable model bundle, including legacy v1 bundles."""
    try:
        # BytesIO makes uploaded bytes file-like, allowing ZipFile to read its entries.
        with zipfile.ZipFile(io.BytesIO(blob), "r") as archive:
            names = set(archive.namelist())
            if MANIFEST_NAME not in names:
                raise ValueError("This file is not an ANNzPlus model bundle.")
            manifest = json.loads(archive.read(MANIFEST_NAME).decode("utf-8"))
            if manifest.get("format_version") not in SUPPORTED_FORMAT_VERSIONS:
                raise ValueError("This model bundle format is unsupported.")
            architecture = manifest.get("architecture")
            preparation = manifest.get("data_preparation")
            training_config = manifest.get("training_config", {})
            files = manifest.get("files", {})
            if not isinstance(architecture, dict) or not isinstance(preparation, dict):
                raise ValueError(
                    "The model bundle is missing architecture or scaling settings."
                )
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
            evaluation = _read_csv(archive, names, files.get("evaluation_report"))
            committee_metadata = manifest.get("committee_metadata")
            if committee_metadata is not None and not isinstance(
                committee_metadata, dict
            ):
                raise ValueError("The saved committee metadata is invalid.")
            bundle_metadata = manifest.get("bundle_metadata", {})
            if not isinstance(bundle_metadata, dict):
                raise ValueError("The saved bundle metadata is invalid.")
            return ModelBundle(
                architecture=architecture,
                data_preparation=preparation,
                checkpoint_bytes=checkpoint,
                training_config=training_config,
                training_history=history,
                optimisation_results=results,
                evaluation_report=evaluation,
                source_data_sha256=manifest.get("source_data_sha256"),
                bundle_metadata=bundle_metadata,
                committee_metadata=committee_metadata,
            )
    except zipfile.BadZipFile as exc:
        raise ValueError("The selected file is not a valid model bundle ZIP.") from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("The model bundle manifest could not be read.") from exc


def _read_csv(
    archive: zipfile.ZipFile, names: set[str], member: str | None
) -> pd.DataFrame | None:
    if member is None:
        return None
    if (
        not isinstance(member, str)
        or member not in names
        or member.startswith("/")
        or ".." in member.split("/")
    ):
        raise ValueError(
            "The model bundle contains an unsafe or missing file reference."
        )
    return pd.read_csv(io.BytesIO(archive.read(member)))
