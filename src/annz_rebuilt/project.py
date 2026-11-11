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


PROJECT_FORMAT_VERSION = 3
MANIFEST_NAME = "project.json"


@dataclass(frozen=True)
class ProjectArchive:
    """Settings, optional source data, preparation, and analysis reports."""

    architecture: dict[str, Any]
    data_preparation: dict[str, Any] | None
    data_source_name: str | None
    data_source_bytes: bytes | None
    data_source_sha256: str | None
    quality_summary: pd.DataFrame | None
    sample_correlations: pd.DataFrame | None


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
) -> bytes:
    """Package settings, optional source data, preparation, and analysis reports."""
    if data_source_bytes is not None and data_source_name is None:
        raise ValueError("A data source filename is required when including its contents.")

    safe_name = PurePosixPath((data_source_name or "").replace("\\", "/")).name
    if data_source_name is not None and (not safe_name or safe_name in {".", ".."}):
        raise ValueError("The data source filename is invalid.")
    if data_source_bytes is not None:
        data_source_sha256 = hashlib.sha256(data_source_bytes).hexdigest()

    files = {
        "data_source": f"data/{safe_name}" if data_source_bytes is not None else None,
        "quality_summary": (
            "analysis/data_quality_summary.csv" if quality_summary is not None else None
        ),
        "sample_correlations": (
            "analysis/sample_correlations.csv"
            if sample_correlations is not None
            else None
        ),
    }
    manifest = {
        "format_version": PROJECT_FORMAT_VERSION,
        "architecture": architecture_config,
        "data_preparation": data_preparation_config,
        "data_source_name": safe_name or None,
        "data_source_sha256": data_source_sha256,
        "files": files,
        "training_checkpoint_included": False,
    }

    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(MANIFEST_NAME, json.dumps(manifest, indent=2) + "\n")
        if data_source_bytes is not None and files["data_source"]:
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
        archive.writestr(
            "README.txt",
            "Galaxy neural-network analysis project\n"
            "project.json stores selected variables, network settings, and preparation.\n"
            "data/ contains the original data source when included.\n"
            "analysis/ contains data-quality statistics and sample correlations when available.\n"
            "Trained weights and training checkpoints are not included in this prototype.\n",
        )
    return output.getvalue()


def open_project_zip(zip_bytes: bytes) -> ProjectArchive:
    """Read a project ZIP, including optional data, preparation, and analysis."""
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as archive:
            names = set(archive.namelist())
            if MANIFEST_NAME not in names:
                raise ValueError("This ZIP has no project.json manifest.")
            manifest = json.loads(archive.read(MANIFEST_NAME).decode("utf-8"))
            version = manifest.get("format_version")
            if version not in {1, 2, PROJECT_FORMAT_VERSION}:
                raise ValueError("This project uses an unsupported format version.")
            architecture = manifest.get("architecture")
            if not isinstance(architecture, dict):
                raise ValueError("The project has no valid architecture configuration.")
            preparation = manifest.get("data_preparation")
            if preparation is not None and not isinstance(preparation, dict):
                raise ValueError("The project data-preparation settings are invalid.")

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
                expected_hash = manifest.get("data_source_sha256")
                if expected_hash and hashlib.sha256(data_bytes).hexdigest() != expected_hash:
                    raise ValueError("The source data in the ZIP does not match its recorded checksum.")
            elif data_name is not None and not isinstance(data_name, str):
                raise ValueError("The source data filename in the manifest is invalid.")

            return ProjectArchive(
                architecture=architecture,
                data_preparation=preparation,
                data_source_name=data_name,
                data_source_bytes=data_bytes,
                data_source_sha256=manifest.get("data_source_sha256"),
                quality_summary=_read_csv_member(
                    archive, names, files.get("quality_summary")
                ),
                sample_correlations=_read_csv_member(
                    archive, names, files.get("sample_correlations"), index_col=0
                ),
            )
    except zipfile.BadZipFile as exc:
        raise ValueError("The selected file is not a valid project ZIP.") from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("The project manifest could not be read.") from exc


def _read_csv_member(
    archive: zipfile.ZipFile,
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
        raise ValueError(f"The analysis report {safe_path} is missing from the ZIP.")
    return pd.read_csv(io.BytesIO(archive.read(safe_path)), index_col=index_col)
