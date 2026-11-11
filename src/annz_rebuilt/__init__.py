"""Core components for local galaxy neural-network analysis."""

from .architecture import Architecture, architecture_from_selections
from .data_analysis import analyze_catalogue_csv
from .data_preparation import DataPreparation, fit_data_preparation, iter_prepared_batches
from .project import ProjectArchive, create_project_zip, open_project_zip

__all__ = [
    "Architecture",
    "DataPreparation",
    "ProjectArchive",
    "analyze_catalogue_csv",
    "architecture_from_selections",
    "create_project_zip",
    "fit_data_preparation",
    "iter_prepared_batches",
    "open_project_zip",
]
