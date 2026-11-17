# Dependency management

The canonical runtime dependency list is maintained in the repository-root
`pyproject.toml` under `[project].dependencies`. Development-only tools are
listed in `[project.optional-dependencies].dev`.

The root `requirements.txt` delegates to the local project metadata with
`-e .`, avoiding a second dependency list that could drift out of sync.

PyTorch is installed separately by `installation/windows/install.py` because
the correct wheel depends on the target computer:

- NVIDIA GPU detected: CUDA 12.8 wheel index.
- No NVIDIA GPU detected: CPU wheel index.

Do not commit virtual environments, downloaded wheels, CUDA runtimes, or
machine-specific binaries. They are large and are selected at installation
time. For a reproducible research environment, record the Python version,
PyTorch version, CUDA availability, and the output of `python -m pip freeze`
alongside each experiment rather than treating a machine-specific freeze as
the universal installation specification.
