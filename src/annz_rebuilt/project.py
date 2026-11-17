"""Create and reopen portable project ZIP files."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import PurePosixPath
from typing import Any
import zipfile

import pandas as pd


PROJECT_FORMAT_VERSION = 5
MANIFEST_NAME = "project.json"


@dataclass(frozen=True)
class ProjectArchive:
    """Project configuration, optional source data, reports, and model progress."""

    architecture: dict[str, Any]
    data_preparation: dict[str, Any] | None
    data_source_name: str | None
    data_source_bytes: bytes | None
    data_source_sha256: str | None
    quality_summary: pd.DataFrame | None
    sample_correlations: pd.DataFrame | None
    checkpoint_bytes: bytes | None
    training_history: pd.DataFrame | None
    training_config: dict[str, Any] | None = None
    optimisation_results: pd.DataFrame | None = None


def _safe_archive_path(value: str) -> str:
    normalized = value.replace("\\", "/")
    path = PurePosixPath(normalized)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("The project contains an unsafe file path.")
    return normalized


def create_project_zip(
    architecture_config: dict[str, Any],
    *,
    data_preparation_config: dict[str, Any] | None = None,
    data_source_name: str | None = None,
    data_source_bytes: bytes | None = None,
    data_source_sha256: str | None = None,
    quality_summary: pd.DataFrame | None = None,
    sample_correlations: pd.DataFrame | None = None,
    checkpoint_bytes: bytes | None = None,
    training_history: pd.DataFrame | None = None,
    training_config: dict[str, Any] | None = None,
    optimisation_results: pd.DataFrame | None = None,
) -> bytes:
    """Package settings, optional data, reports, search results, and checkpoint."""
    if data_source_bytes is not None and data_source_name is None:
        raise ValueError("A data source filename is required when including its contents.")
    safe_name = PurePosixPath((data_source_name or "").replace("\\", "/")).name
    if data_source_name is not None and (not safe_name or safe_name in {".", ".."}):
        raise ValueError("The data source filename is invalid.")
    if data_source_bytes is not None:
        data_source_sha256 = hashlib.sha256(data_source_bytes).hexdigest()

    files = {
        "data_source": f"data/{safe_name}" if data_source_bytes is not None else None,
        "quality_summary": "analysis/data_quality_summary.csv" if quality_summary is not None else None,
        "sample_correlations": "analysis/sample_correlations.csv" if sample_correlations is not None else None,
        "training_checkpoint": "training/checkpoint.pt" if checkpoint_bytes is not None else None,
        "training_history": "training/history.csv" if training_history is not None else None,
        "optimisation_results": "training/optimisation_results.csv" if optimisation_results is not None else None,
    }
    manifest = {
        "format_version": PROJECT_FORMAT_VERSION,
        "architecture": architecture_config,
        "data_preparation": data_preparation_config,
        "training_config": training_config,
        "data_source_name": safe_name or None,
        "data_source_sha256": data_source_sha256,
        "files": files,
        "training_checkpoint_included": checkpoint_bytes is not None,
    }

    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(MANIFEST_NAME, json.dumps(manifest, indent=2) + "\n")
        if data_source_bytes is not None:
            archive.writestr(files["data_source"], data_source_bytes)
        if quality_summary is not None:
            archive.writestr(
                files["quality_summary"],
                quality_summary.to_csv(index=False, lineterminator="\n"),
            )
        if sample_correlations is not None:
            archive.writestr(
                files["sample_correlations"],
                sample_correlations.to_csv(index=True, lineterminator="\n"),
            )
        if checkpoint_bytes is not None:
            archive.writestr(files["training_checkpoint"], checkpoint_bytes)
        if training_history is not None:
            archive.writestr(
                files["training_history"],
                training_history.to_csv(index=False, lineterminator="\n"),
            )
        if optimisation_results is not None:
            archive.writestr(
                files["optimisation_results"],
                optimisation_results.to_csv(index=False, lineterminator="\n"),
            )
        archive.writestr(
            "README.txt",
            "Galaxy neural-network analysis project\n"
            "project.json stores architecture, data preparation, and training settings.\n"
            "data/ contains the original source when included.\n"
            "analysis/ contains data-quality and exploratory reports.\n"
            "training/ contains model weights, loss history, and optimisation results when available.\n"
            "Reopen this ZIP in the app to restore the saved configuration.\n",
        )
    return output.getvalue()


def open_project_zip(zip_bytes: bytes) -> ProjectArchive:
    """Read a project ZIP created by this app or an earlier supported version."""
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as archive:
            names = set(archive.namelist())
            if MANIFEST_NAME not in names:
                raise ValueError("This ZIP has no project.json manifest.")
            manifest = json.loads(archive.read(MANIFEST_NAME).decode("utf-8"))
            if manifest.get("format_version") not in {1, 2, 3, 4, PROJECT_FORMAT_VERSION}:
                raise ValueError("This project uses an unsupported format version.")
            architecture = manifest.get("architecture")
            if not isinstance(architecture, dict):
                raise ValueError("The project has no valid architecture configuration.")
            preparation = manifest.get("data_preparation")
            if preparation is not None and not isinstance(preparation, dict):
                raise ValueError("The project data-preparation settings are invalid.")
            training_config = manifest.get("training_config")
            if training_config is not None and not isinstance(training_config, dict):
                raise ValueError("The project training settings are invalid.")
            files = manifest.get("files", {})
            if not isinstance(files, dict):
                files = {}

            data_path = files.get("data_source") or manifest.get("catalogue_path")
            data_name = manifest.get("data_source_name")
            data_bytes = None
            if data_path is not None:
                if not isinstance(data_path, str):
                    raise ValueError("The project data path is invalid.")
                data_path = _safe_archive_path(data_path)
                if data_path not in names:
                    raise ValueError("The source data listed in the manifest is missing.")
                data_name = PurePosixPath(data_path).name
                data_bytes = archive.read(data_path)
                expected = manifest.get("data_source_sha256")
                if expected and hashlib.sha256(data_bytes).hexdigest() != expected:
                    raise ValueError("The source data does not match its recorded checksum.")
            elif data_name is not None and not isinstance(data_name, str):
                raise ValueError("The source data filename in the manifest is invalid.")

            return ProjectArchive(
                architecture=architecture,
                data_preparation=preparation,
                data_source_name=data_name,
                data_source_bytes=data_bytes,
                data_source_sha256=manifest.get("data_source_sha256"),
                quality_summary=_read_csv_member(archive, names, files.get("quality_summary")),
                sample_correlations=_read_csv_member(
                    archive, names, files.get("sample_correlations"), index_col=0
                ),
                checkpoint_bytes=_read_bytes_member(
                    archive, names, files.get("training_checkpoint")
                ),
                training_history=_read_csv_member(
                    archive, names, files.get("training_history")
                ),
                training_config=training_config,
                optimisation_results=_read_csv_member(
                    archive, names, files.get("optimisation_results")
                ),
            )
    except zipfile.BadZipFile as exc:
        raise ValueError("The selected file is not a valid project ZIP.") from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("The project manifest could not be read.") from exc


def _read_bytes_member(archive, names: set[str], archive_path: Any) -> bytes | None:
    if archive_path is None:
        return None
    if not isinstance(archive_path, str):
        raise ValueError("A checkpoint path in the project is invalid.")
    safe_path = _safe_archive_path(archive_path)
    if safe_path not in names:
        raise ValueError(f"The project member {safe_path} is missing.")
    return archive.read(safe_path)


def _read_csv_member(
    archive,
    names: set[str],
    archive_path: Any,
    *,
    index_col: int | None = None,
) -> pd.DataFrame | None:
    if archive_path is None:
        return None
    if not isinstance(archive_path, str):
        raise ValueError("An analysis report path in the project is invalid.")
    safe_path = _safe_archive_path(archive_path)
    if safe_path not in names:
        raise ValueError(f"The project member {safe_path} is missing.")
    return pd.read_csv(io.BytesIO(archive.read(safe_path)), index_col=index_col)
