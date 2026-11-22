"""Core components for local galaxy neural-network analysis."""

from .architecture import Architecture, NetworkStructure, architecture_from_selections
from .data_analysis import analyze_catalogue_csv
from .data_preparation import (
    DataPreparation,
    fit_data_preparation,
    iter_prepared_batches,
)
from .neural_network import (
    build_network,
    evaluate_model,
    train_model,
    write_predictions_csv,
)
from .project import ProjectArchive, create_project_zip, open_project_zip

__all__ = [
    "Architecture",
    "NetworkStructure",
    "DataPreparation",
    "ProjectArchive",
    "analyze_catalogue_csv",
    "architecture_from_selections",
    "build_network",
    "create_project_zip",
    "evaluate_model",
    "fit_data_preparation",
    "iter_prepared_batches",
    "open_project_zip",
    "train_model",
    "write_predictions_csv",
]
