"""Install ANNz-rebuilt and select a suitable PyTorch build.

Run from the repository root with:
    python install.py

On Windows/Linux with an NVIDIA GPU, install a CUDA-enabled PyTorch wheel.
On systems without NVIDIA, install the normal PyPI build (CPU fallback).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _run(command: list[str]) -> None:
    subprocess.check_call(command)


def _has_nvidia_gpu() -> bool:
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return result.returncode == 0 and "GPU" in result.stdout
    except OSError:
        return False


def main() -> int:
    project_root = Path(__file__).resolve().parent
    if sys.version_info < (3, 9):
        print("ANNz-rebuilt requires Python 3.9 or newer.")
        return 1

    python = sys.executable
    try:
        if _has_nvidia_gpu():
            # PyTorch 2.8 is the newest CUDA-enabled line supporting Python 3.9.
            # CUDA 12.8 wheels run with a sufficiently recent NVIDIA driver.
            if sys.version_info < (3, 10):
                torch_command = [
                    python, "-m", "pip", "install", "--upgrade", "torch==2.8.0",
                    "--index-url", "https://download.pytorch.org/whl/cu128",
                ]
            else:
                torch_command = [
                    python, "-m", "pip", "install", "--upgrade", "torch",
                    "--index-url", "https://download.pytorch.org/whl/cu132",
                ]
            print("NVIDIA GPU detected: installing CUDA-enabled PyTorch...")
            _run(torch_command)
        else:
            print("No NVIDIA GPU detected: installing PyTorch with CPU fallback.")

        print("Installing ANNz-rebuilt and any remaining dependencies...")
        _run([python, "-m", "pip", "install", "--upgrade", str(project_root)])
    except subprocess.CalledProcessError as exc:
        print("\nInstallation failed. Check the Python version and network connection.")
        return int(exc.returncode or 1)

    print("\nInstallation completed.")
    print("Start the local web interface with:")
    print(f'  "{python}" -m streamlit run "{project_root / "app.py"}"')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
