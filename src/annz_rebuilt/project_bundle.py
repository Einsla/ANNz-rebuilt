"""Save and reopen portable project setup ZIP files."""

from __future__ import annotations

import io
import json
from pathlib import PurePath
from typing import Any
import zipfile


PROJECT_FORMAT_VERSION = 1
PROJECT_MANIFEST = "project.json"


def create_project_bundle(
    config: dict[str, Any],
    *,
    catalogue_name: str | None = None,
    catalogue_bytes: bytes | None = None,
) -> bytes:
    """Build a ZIP containing the app configuration and optionally the source data."""
    if (catalogue_name is None) != (catalogue_bytes is None):
        raise ValueError("Provide both the catalogue filename and its contents, or neither.")

    manifest = {
        "format_version": PROJECT_FORMAT_VERSION,
        "architecture": config,
        "catalogue_included": catalogue_bytes is not None,
        "catalogue_path": f"catalogue/{PurePath(catalogue_name).name}"
        if catalogue_name is not None
        else None,
        "training_checkpoint_included": False,
    }
    output = io.BytesIO()
    with zipfile.ZipFile(output, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(PROJECT_MANIFEST, json.dumps(manifest, indent=2) + "\n")
        if catalogue_bytes is not None and catalogue_name is not None:
            archive.writestr(manifest["catalogue_path"], catalogue_bytes)
        archive.writestr(
            "README.txt",
            "Galaxy Neural Network Analysis project bundle\n"
            "Open this ZIP in the local app to restore its architecture settings.\n"
            "Training checkpoints are not included in this prototype.\n",
        )
    return output.getvalue()


def read_project_bundle(bundle_bytes: bytes) -> tuple[dict[str, Any], str | None, bytes | None]:
    """Read project settings and an optional catalogue from a project ZIP."""
    try:
        with zipfile.ZipFile(io.BytesIO(bundle_bytes), mode="r") as archive:
            names = set(archive.namelist())
            if PROJECT_MANIFEST not in names:
                raise ValueError("This ZIP does not contain a project.json manifest.")
            manifest = json.loads(archive.read(PROJECT_MANIFEST).decode("utf-8"))
            if manifest.get("format_version") != PROJECT_FORMAT_VERSION:
                raise ValueError("This project uses an unsupported format version.")
            config = manifest.get("architecture")
            if not isinstance(config, dict):
                raise ValueError("The project manifest has no valid architecture settings.")
            for key in ("input_variables", "target_variables"):
                values = config.get(key)
                if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
                    raise ValueError(f"The project has invalid {key.replace('_', ' ')}.")
            layers = config.get("hidden_layers")
            if (
                not isinstance(layers, list)
                or not 1 <= len(layers) <= 10
                or not all(isinstance(size, int) and 1 <= size <= 4096 for size in layers)
            ):
                raise ValueError("The project has invalid hidden-layer settings.")
            activation = config.get("activation")
            if activation not in {"leaky_relu", "relu", "tanh", "sigmoid", "mish", "softplus", "silu"}:
                raise ValueError("The project has an unsupported activation function.")

            catalogue_path = manifest.get("catalogue_path")
            if catalogue_path is None:
                return config, None, None
            if not isinstance(catalogue_path, str):
                raise ValueError("The project catalogue path is invalid.")
            normalized = catalogue_path.replace("\\", "/")
            if normalized.startswith("/") or ".." in normalized.split("/"):
                raise ValueError("The project catalogue path is unsafe.")
            if normalized not in names:
                raise ValueError("The project refers to a catalogue that is missing from the ZIP.")
            return config, PurePath(normalized).name, archive.read(normalized)
    except zipfile.BadZipFile as exc:
        raise ValueError("The selected file is not a valid project ZIP.") from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("The project manifest could not be read.") from exc
