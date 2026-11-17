from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path
from unittest.mock import patch

INSTALLER_PATH = (
    Path(__file__).resolve().parents[1]
    / "installation"
    / "windows"
    / "install.py"
)
SPEC = importlib.util.spec_from_file_location("annz_windows_installer", INSTALLER_PATH)
assert SPEC is not None and SPEC.loader is not None
INSTALLER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(INSTALLER)


def test_installer_selects_nvidia_when_nvidia_smi_finds_gpu():
    result = subprocess.CompletedProcess(
        args=["nvidia-smi", "-L"], returncode=0, stdout="GPU 0: NVIDIA RTX", stderr=""
    )
    with patch.object(INSTALLER.subprocess, "run", return_value=result):
        assert INSTALLER._has_nvidia_gpu() is True


def test_installer_falls_back_when_nvidia_smi_fails():
    result = subprocess.CompletedProcess(
        args=["nvidia-smi", "-L"], returncode=1, stdout="", stderr="not found"
    )
    with patch.object(INSTALLER.subprocess, "run", return_value=result):
        assert INSTALLER._has_nvidia_gpu() is False


def test_installer_falls_back_when_nvidia_smi_is_missing():
    with patch.object(INSTALLER.subprocess, "run", side_effect=OSError):
        assert INSTALLER._has_nvidia_gpu() is False
