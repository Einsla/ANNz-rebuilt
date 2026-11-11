"""Streaming quality checks and summary statistics for numeric catalogues."""
from __future__ import annotations
from pathlib import Path
from typing import BinaryIO, Sequence
import numpy as np
import pandas as pd

def analyze_catalogue_csv(
    source: str | Path | BinaryIO,
    *,
    separator: str | None = ",",
    rows_per_batch: int = 100_000,
    sample_columns: Sequence[str] = (),
    max_sample_rows: int = 20_000,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Profile the CSV in chunks; return summary, exploratory sample, and row count."""
    if rows_per_batch < 1 or max_sample_rows < 1:
        raise ValueError("Batch and sample sizes must be positive.")

    accumulators: dict[str, dict[str, float | int]] = {}
    samples: list[pd.DataFrame] = []
    row_count = 0
    sample_names = list(dict.fromkeys(sample_columns))
    reader = pd.read_csv(source, sep=separator, chunksize=rows_per_batch)
    try:
        for batch in reader:
            batch.columns = [str(name).lstrip("#").strip() for name in batch.columns]
            if len(set(batch.columns)) != len(batch.columns):
                raise ValueError("Column names must be unique after removing a leading #.")
            row_count += len(batch)
            for name in batch.columns:
                values = batch[name]
                numeric = pd.to_numeric(values, errors="coerce")
                numeric_array = numeric.to_numpy(dtype="float64", na_value=np.nan)
                finite_mask = np.isfinite(numeric_array)
                finite_values = numeric_array[finite_mask]
                summary = accumulators.setdefault(name, {
                    "rows": 0, "missing": 0, "non_numeric": 0, "non_finite": 0,
                    "finite": 0, "sum": 0.0, "sum_squares": 0.0,
                    "min": float("inf"), "max": float("-inf"),
                })
                summary["rows"] += len(batch)
                summary["missing"] += int(values.isna().sum())
                summary["non_numeric"] += int((values.notna() & numeric.isna()).sum())
                summary["non_finite"] += int((numeric.notna().to_numpy() & ~finite_mask).sum())
                summary["finite"] += len(finite_values)
                if len(finite_values):
                    summary["sum"] += float(finite_values.sum(dtype="float64"))
                    summary["sum_squares"] += float(np.square(finite_values).sum(dtype="float64"))
                    summary["min"] = min(float(summary["min"]), float(finite_values.min()))
                    summary["max"] = max(float(summary["max"]), float(finite_values.max()))

            chosen = [name for name in sample_names if name in batch.columns]
            if chosen and len(batch):
                samples.append(batch.loc[:, chosen].sample(n=min(100, len(batch)), random_state=row_count % (2**32)))
    finally:
        reader.close()

    if row_count == 0:
        raise ValueError("The catalogue has no data rows.")
    output = []
    for name, stats in accumulators.items():
        count = int(stats["finite"])
        total = float(stats["sum"])
        sum_squares = float(stats["sum_squares"])
        mean = total / count if count else np.nan
        variance = max((sum_squares - total * total / count) / (count - 1), 0.0) if count > 1 else np.nan
        output.append({
            "variable": name, "rows": int(stats["rows"]),
            "missing": int(stats["missing"]), "non_numeric": int(stats["non_numeric"]),
            "non_finite": int(stats["non_finite"]), "finite": count,
            "minimum": float(stats["min"]) if count else np.nan,
            "maximum": float(stats["max"]) if count else np.nan,
            "mean": mean, "std": float(np.sqrt(variance)) if count > 1 else np.nan,
        })

    sample = pd.concat(samples, ignore_index=True) if samples else pd.DataFrame(columns=sample_names)
    if len(sample) > max_sample_rows:
        sample = sample.sample(n=max_sample_rows, random_state=42)
    return pd.DataFrame(output), sample.reset_index(drop=True), row_count

