"""Training-only scaling and deterministic train/validation/test splitting."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import BinaryIO, Iterator, Sequence

import numpy as np
import pandas as pd


SPLITS = ("train", "validation", "test")


@dataclass(frozen=True)
class DataPreparation:
    """Split assignments and scaling values ready for later neural-network training."""

    input_columns: tuple[str, ...]
    target_columns: tuple[str, ...]
    train_fraction: float
    validation_fraction: float
    test_fraction: float
    random_seed: int
    rows_per_batch: int
    batch_size: int
    total_rows: int
    usable_rows: int
    dropped_rows: int
    rows_by_split: dict[str, int]
    input_mean: tuple[float, ...]
    input_scale: tuple[float, ...]
    target_mean: tuple[float, ...]
    target_scale: tuple[float, ...]

    def to_config_dict(self) -> dict[str, object]:
        """Convert the settings and fitted scaling values to JSON-compatible data."""
        return asdict(self)

    @classmethod
    def from_config_dict(cls, config: dict[str, object]) -> "DataPreparation":
        """Rebuild preparation settings saved in a project ZIP."""
        values = dict(config)
        values["input_columns"] = tuple(values["input_columns"])
        values["target_columns"] = tuple(values["target_columns"])
        for key in ("input_mean", "input_scale", "target_mean", "target_scale"):
            values[key] = tuple(float(value) for value in values[key])
        values["rows_by_split"] = {
            str(name): int(count) for name, count in values["rows_by_split"].items()
        }
        for key in (
            "random_seed",
            "rows_per_batch",
            "batch_size",
            "total_rows",
            "usable_rows",
            "dropped_rows",
        ):
            values[key] = int(values[key])
        for key in ("train_fraction", "validation_fraction", "test_fraction"):
            values[key] = float(values[key])
        return cls(**values)


def _read_chunks(
    source: str | Path | BinaryIO,
    *,
    separator: str | None,
    rows_per_batch: int,
) -> Iterator[pd.DataFrame]:
    """Read a reusable file source in chunks and normalize its column names."""
    if hasattr(source, "seek"):
        source.seek(0)
    reader = pd.read_csv(source, sep=separator, chunksize=rows_per_batch)
    try:
        for chunk in reader:
            chunk.columns = [str(name).lstrip("#").strip() for name in chunk.columns]
            if len(set(chunk.columns)) != len(chunk.columns):
                raise ValueError("Column names must be unique after removing a leading #.")
            yield chunk
    finally:
        reader.close()


def _split_masks(
    row_count: int,
    *,
    rng: np.random.Generator,
    train_fraction: float,
    validation_fraction: float,
) -> dict[str, np.ndarray]:
    """Assign each row to one split using a repeatable random sequence."""
    draws = rng.random(row_count)
    validation_end = train_fraction + validation_fraction
    return {
        "train": draws < train_fraction,
        "validation": (draws >= train_fraction) & (draws < validation_end),
        "test": draws >= validation_end,
    }


def _numeric_matrix(
    chunk: pd.DataFrame,
    columns: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Convert selected columns to numbers and identify rows where all are finite."""
    missing = [name for name in columns if name not in chunk.columns]
    if missing:
        raise ValueError("Missing selected variables: " + ", ".join(missing))
    numeric = chunk.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce")
    matrix = numeric.to_numpy(dtype="float64", na_value=np.nan)
    return matrix, np.isfinite(matrix).all(axis=1)


def _update_moments(
    values: np.ndarray,
    *,
    count: int,
    total: np.ndarray,
    sum_squares: np.ndarray,
) -> int:
    """Accumulate sums and squared sums for a batch of rows."""
    if len(values) == 0:
        return count
    total += values.sum(axis=0, dtype="float64")
    sum_squares += np.square(values).sum(axis=0, dtype="float64")
    return count + len(values)


def _mean_and_scale(
    count: int,
    total: np.ndarray,
    sum_squares: np.ndarray,
    *,
    label: str,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Calculate means and standard deviations; replace zero scales with 1."""
    if count == 0:
        raise ValueError(f"No usable training rows remain for {label}.")
    mean = total / count
    variance = np.maximum(sum_squares / count - np.square(mean), 0.0)
    scale = np.sqrt(variance)
    scale[scale == 0.0] = 1.0
    return tuple(mean.tolist()), tuple(scale.tolist())


def fit_data_preparation(
    source: str | Path | BinaryIO,
    input_columns: Sequence[str],
    target_columns: Sequence[str],
    *,
    separator: str | None = ",",
    rows_per_batch: int = 100_000,
    batch_size: int = 4_096,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    random_seed: int = 42,
) -> DataPreparation:
    """Fit scaling on training rows and record reproducible data splits.

    A row is dropped if a selected input or target is missing, non-numeric, or
    infinite. Validation and test rows are never used to calculate scaling.
    """
    inputs = tuple(input_columns)
    targets = tuple(target_columns)
    if not inputs or not targets:
        raise ValueError("Choose at least one input variable and one target variable.")
    if set(inputs) & set(targets):
        raise ValueError("Input and target variables must be distinct.")
    if rows_per_batch < 1 or batch_size < 1:
        raise ValueError("Batch sizes must be at least 1.")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("The training share must be between 0 and 1.")
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("The validation share must be between 0 and 1.")
    test_fraction = 1.0 - train_fraction - validation_fraction
    if test_fraction < 0.0:
        raise ValueError("Training and validation shares cannot add to more than 1.")
    if not math.isclose(train_fraction + validation_fraction + test_fraction, 1.0):
        raise ValueError("Train, validation, and test shares must add to 1.")

    rng = np.random.default_rng(random_seed)
    total_rows = 0
    usable_rows = 0
    rows_by_split = {split: 0 for split in SPLITS}
    input_count = 0
    target_count = 0
    input_total = np.zeros(len(inputs), dtype="float64")
    target_total = np.zeros(len(targets), dtype="float64")
    input_squares = np.zeros(len(inputs), dtype="float64")
    target_squares = np.zeros(len(targets), dtype="float64")

    for chunk in _read_chunks(
        source,
        separator=separator,
        rows_per_batch=rows_per_batch,
    ):
        total_rows += len(chunk)
        masks = _split_masks(
            len(chunk),
            rng=rng,
            train_fraction=train_fraction,
            validation_fraction=validation_fraction,
        )
        input_values, valid_inputs = _numeric_matrix(chunk, inputs)
        target_values, valid_targets = _numeric_matrix(chunk, targets)
        valid_rows = valid_inputs & valid_targets
        usable_rows += int(valid_rows.sum())
        for split in SPLITS:
            rows_by_split[split] += int((masks[split] & valid_rows).sum())

        train_rows = masks["train"] & valid_rows
        input_count = _update_moments(
            input_values[train_rows],
            count=input_count,
            total=input_total,
            sum_squares=input_squares,
        )
        target_count = _update_moments(
            target_values[train_rows],
            count=target_count,
            total=target_total,
            sum_squares=target_squares,
        )

    if total_rows == 0:
        raise ValueError("The catalogue has no data rows.")
    input_mean, input_scale = _mean_and_scale(
        input_count, input_total, input_squares, label="inputs"
    )
    target_mean, target_scale = _mean_and_scale(
        target_count, target_total, target_squares, label="targets"
    )
    return DataPreparation(
        input_columns=inputs,
        target_columns=targets,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
        random_seed=random_seed,
        rows_per_batch=rows_per_batch,
        batch_size=batch_size,
        total_rows=total_rows,
        usable_rows=usable_rows,
        dropped_rows=total_rows - usable_rows,
        rows_by_split=rows_by_split,
        input_mean=input_mean,
        input_scale=input_scale,
        target_mean=target_mean,
        target_scale=target_scale,
    )


def iter_prepared_batches(
    source: str | Path | BinaryIO,
    preparation: DataPreparation,
    *,
    split: str = "train",
    batch_size: int | None = None,
    separator: str | None = ",",
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield normalized input and target arrays for one split in bounded batches."""
    if batch_size is None:
        batch_size = preparation.batch_size
    if split not in SPLITS:
        raise ValueError("Split must be train, validation, or test.")
    if batch_size < 1:
        raise ValueError("Batch size must be at least 1.")

    split_rng = np.random.default_rng(preparation.random_seed)
    shuffle_rng = np.random.default_rng(preparation.random_seed + 1)
    input_mean = np.asarray(preparation.input_mean)
    input_scale = np.asarray(preparation.input_scale)
    target_mean = np.asarray(preparation.target_mean)
    target_scale = np.asarray(preparation.target_scale)

    for chunk in _read_chunks(
        source,
        separator=separator,
        rows_per_batch=preparation.rows_per_batch,
    ):
        masks = _split_masks(
            len(chunk),
            rng=split_rng,
            train_fraction=preparation.train_fraction,
            validation_fraction=preparation.validation_fraction,
        )
        input_values, valid_inputs = _numeric_matrix(chunk, preparation.input_columns)
        target_values, valid_targets = _numeric_matrix(chunk, preparation.target_columns)
        chosen = masks[split] & valid_inputs & valid_targets
        inputs = (input_values[chosen] - input_mean) / input_scale
        targets = (target_values[chosen] - target_mean) / target_scale

        if len(inputs):
            order = shuffle_rng.permutation(len(inputs))
            inputs = inputs[order]
            targets = targets[order]
            for start in range(0, len(inputs), batch_size):
                yield inputs[start : start + batch_size], targets[start : start + batch_size]
