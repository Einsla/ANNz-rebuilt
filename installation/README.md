# Installation resources

This directory contains installation-specific resources, kept separate from the
application source code and user documentation.

## Windows

See [Windows installation guide](windows/README.md).

- `windows/install.py`: selects the CUDA 12.8 or CPU PyTorch wheel and installs
  the project.
- `windows/Create_ANNzPlus_Shortcut.ps1`: creates Desktop and Start Menu
  shortcuts.
- Repository-root `Install_ANNzPlus.bat`: user-facing one-click installer.
- Repository-root `Run_ANNzPlus.bat`: user-facing application launcher.

## Dependencies

The project dependency declarations remain in the repository-root
`pyproject.toml` so Python build tools can discover them reliably. The
`dependencies/` directory is reserved for future platform-specific
requirements or offline installation manifests.

Large PyTorch wheels and virtual environments are deliberately not committed.
The installer downloads the correct wheel for the target machine.
