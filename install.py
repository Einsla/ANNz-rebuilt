"""Install ANNz-rebuilt with a hardware-appropriate PyTorch wheel.

Run install.py from the repository root, preferably inside a virtual environment.
NVIDIA systems use the CUDA 12.8 PyTorch index; other systems use CPU wheels.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PYTORCH_CUDA_INDEX = "https://download.pytorch.org/whl/cu128"
PYTORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"


def _run(command: list[str]) -> None:
    subprocess.check_call(command)


def _has_nvidia_gpu() -> bool:
    """Return True only when the NVIDIA management utility detects a GPU."""
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
    has_nvidia = _has_nvidia_gpu()
    torch_index = PYTORCH_CUDA_INDEX if has_nvidia else PYTORCH_CPU_INDEX

    try:
        print(
            "NVIDIA GPU detected: installing CUDA-enabled PyTorch."
            if has_nvidia
            else "No NVIDIA GPU detected: installing CPU-only PyTorch."
        )
        _run(
            [
                python,
                "-m",
                "pip",
                "install",
                "--upgrade",
                "torch",
                "--index-url",
                torch_index,
            ]
        )

        print("Installing ANNz-rebuilt and its declared runtime dependencies...")
        _run([python, "-m", "pip", "install", "--upgrade", str(project_root)])
    except subprocess.CalledProcessError as exc:
        print("\nInstallation failed. Check Python compatibility and network access.")
        return int(exc.returncode or 1)

    print("\nInstallation completed.")
    print("Launch the application with:")
    print(f'  "{python}" -m streamlit run "{project_root / "app.py"}"')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
