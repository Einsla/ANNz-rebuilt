# Windows installation guide

## Requirements

- Windows 10 or 11, 64-bit.
- Python 3.9 or newer (Python Launcher recommended).
- Internet access for the initial dependency installation.
- NVIDIA GPU optional; CPU execution is supported.

## One-click installation

1. Download the repository ZIP from GitHub and extract it.
2. Keep the extracted directory structure intact.
3. Double-click `Install_ANNzPlus.bat` in the repository root.
4. The installer creates `.venv`, upgrades pip, installs PyTorch and the
   application dependencies, then creates Desktop and Start Menu shortcuts.
5. Start **ANNzPlus Python** from the Desktop shortcut, or double-click
   `Run_ANNzPlus.bat`.

The installer detects NVIDIA hardware using `nvidia-smi`. It chooses the
CUDA 12.8 PyTorch wheel when an NVIDIA GPU is detected and the CPU wheel
otherwise. Intel/AMD CPUs and non-NVIDIA GPUs therefore run through the
reliable CPU backend on Windows; an AMD GPU is not automatically used for
training. PyTorch is downloaded during installation rather than stored in
GitHub because the wheels are large and hardware-specific.

## Manual installation

Run these commands in PowerShell from the repository root:

```powershell
py -3 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe installation\windows\install.py
.venv\Scripts\python.exe -m streamlit run app.py
```

For development and tests:

```powershell
.venv\Scripts\python.exe -m pip install -e ".[dev]"
.venv\Scripts\python.exe -m pytest
.venv\Scripts\python.exe -m ruff check .
```

## Troubleshooting

- If Python is not found, install Python 3.9+ and enable the Python Launcher.
- If shortcut creation is blocked by Windows policy, use
  `Run_ANNzPlus.bat` directly.
- If CUDA is unavailable or GPU memory is insufficient, use the app's CPU
  device option or use mini-batch Adam.
- The first installation requires internet access.

## Updating

Pull or download the latest repository version. Re-run the installer when
dependency declarations change. Existing project data and exported ZIP files
should be backed up separately.
