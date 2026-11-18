"""Committee training and ANNzPlus-compatible uncertainty propagation."""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from typing import BinaryIO, Callable

import numpy as np
import pandas as pd

from .architecture import Architecture
from .data_preparation import DataPreparation, iter_prepared_batches
from .neural_network import _device, _restore_model, _torch, train_model

COMMITTEE_FORMAT_VERSION = 1


def _pack_committee(member_checkpoints: list[bytes], metadata: dict) -> bytes:
    if len(member_checkpoints) < 1 or any(not item for item in member_checkpoints):
        raise ValueError("A committee must contain at least one trained member.")
    output = io.BytesIO()
    manifest = {
        "format_version": COMMITTEE_FORMAT_VERSION,
        "member_count": len(member_checkpoints),
        "metadata": metadata,
    }
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("committee.json", json.dumps(manifest, indent=2))
        for index, checkpoint in enumerate(member_checkpoints):
            archive.writestr(f"members/{index:03d}.pt", checkpoint)
    return output.getvalue()


def unpack_committee(blob: bytes) -> tuple[list[bytes], dict]:
    try:
        with zipfile.ZipFile(io.BytesIO(blob), "r") as archive:
            manifest = json.loads(archive.read("committee.json"))
            if manifest.get("format_version") != COMMITTEE_FORMAT_VERSION:
                raise ValueError("Unsupported committee checkpoint format.")
            count = int(manifest["member_count"])
            if count < 1 or count > 100:
                raise ValueError("Invalid committee member count.")
            members = [archive.read(f"members/{i:03d}.pt") for i in range(count)]
            if any(not member for member in members):
                raise ValueError("A committee member checkpoint is empty.")
            return members, dict(manifest.get("metadata") or {})
    except (zipfile.BadZipFile, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(
            "The selected checkpoint is not a valid committee model."
        ) from exc


def train_committee(
    source: str | Path | BinaryIO,
    architecture: Architecture,
    preparation: DataPreparation,
    *,
    committee_size: int = 5,
    epochs: int = 100,
    learning_rate: float = 0.001,
    optimizer_name: str = "adam",
    separator: str = ",",
    device: str = "auto",
    early_stopping_patience: int | None = None,
    base_seed: int | None = None,
    on_member: Callable[[dict], None] | None = None,
    on_epoch: Callable[[dict], None] | None = None,
) -> dict:
    """Train independent members against the same fixed data splits."""
    if not 1 <= committee_size <= 100:
        raise ValueError("Committee size must be between 1 and 100.")
    if optimizer_name.lower().replace("-", "") in {"vmmin", "legacy_vmmin"} and any(
        name != "tanh" for name in architecture.effective_activations
    ):
        raise ValueError("Legacy vmmin committees require tanh in every hidden layer.")
    seed = preparation.random_seed if base_seed is None else int(base_seed)
    checkpoints: list[bytes] = []
    histories: list[pd.DataFrame] = []
    member_summaries = []
    for index in range(committee_size):
        member_seed = seed + index * 104729

        def member_epoch(event, member=index + 1):
            if on_epoch:
                on_epoch({**event, "member": member, "committee_size": committee_size})

        result = train_model(
            source,
            architecture,
            preparation,
            epochs=epochs,
            learning_rate=learning_rate,
            optimizer_name=optimizer_name,
            separator=separator,
            device=device,
            early_stopping_patience=early_stopping_patience,
            initialization_seed=member_seed,
            on_epoch=member_epoch,
        )
        checkpoints.append(result["checkpoint_bytes"])
        history = result["history"].copy()
        history.insert(0, "member", index + 1)
        histories.append(history)
        member_summaries.append(
            {
                "member": index + 1,
                "seed": member_seed,
                "completed_epoch": result["completed_epoch"],
                "best_epoch": result["best_epoch"],
                "best_validation_loss": result["best_validation_loss"],
                "device": result["device"],
            }
        )
        if on_member:
            on_member(member_summaries[-1])
    checkpoint = _pack_committee(
        checkpoints,
        {
            "committee_size": committee_size,
            "base_seed": seed,
            "optimizer": optimizer_name,
            "members": member_summaries,
        },
    )
    return {
        "checkpoint_bytes": checkpoint,
        "history": pd.concat(histories, ignore_index=True)
        if histories
        else pd.DataFrame(),
        "members": pd.DataFrame(member_summaries),
        "committee_size": committee_size,
        "completed_epoch": max(
            (row["completed_epoch"] for row in member_summaries), default=0
        ),
        "device": member_summaries[0]["device"] if member_summaries else str(device),
    }


def predict_committee(
    inputs: np.ndarray,
    architecture: Architecture,
    preparation: DataPreparation,
    checkpoint_bytes: bytes,
    *,
    input_errors: np.ndarray | None = None,
    device: str = "auto",
) -> dict[str, np.ndarray]:
    """Return mean prediction and C++-style committee/input/combined errors.

    Input errors are 1-sigma uncertainties in the original input-column units.
    Missing input errors are represented by zero. Input errors are assumed
    independent, matching the diagonal covariance assumption in ANNzPlus.
    """
    values = np.asarray(inputs, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != architecture.n_inputs:
        raise ValueError("Input matrix shape does not match the saved architecture.")
    if not np.isfinite(values).all():
        raise ValueError("Prediction inputs must be finite.")
    if input_errors is None:
        errors = np.zeros_like(values)
    else:
        errors = np.asarray(input_errors, dtype=np.float64)
        if errors.shape != values.shape:
            raise ValueError("Input-error matrix must have the same shape as inputs.")
        if not np.isfinite(errors).all() or np.any(errors < 0):
            raise ValueError(
                "Input measurement errors must be finite and non-negative."
            )

    members, _ = unpack_committee(checkpoint_bytes)
    torch, _ = _torch()
    target_device = _device(torch, device)
    x_mean = np.asarray(preparation.input_mean, dtype=np.float64)
    x_scale = np.asarray(preparation.input_scale, dtype=np.float64)
    y_mean = np.asarray(preparation.target_mean, dtype=np.float64)
    y_scale = np.asarray(preparation.target_scale, dtype=np.float64)
    normalized = (values - x_mean) / x_scale

    predictions = []
    jacobians = []
    for member_checkpoint in members:
        model, _ = _restore_model(
            member_checkpoint, architecture, preparation, torch, target_device
        )
        batch_predictions = []
        batch_jacobians = []
        for start in range(0, len(normalized), preparation.batch_size):
            stop = min(start + preparation.batch_size, len(normalized))
            x = (
                torch.as_tensor(
                    normalized[start:stop],
                    dtype=torch.float32,
                    device=target_device,
                )
                .detach()
                .requires_grad_(True)
            )
            output = model(x)
            batch_predictions.append(output.detach().cpu().numpy().astype(np.float64))
            target_jacobian = []
            for target_index in range(architecture.n_outputs):
                gradient = torch.autograd.grad(
                    output[:, target_index].sum(),
                    x,
                    retain_graph=target_index < architecture.n_outputs - 1,
                    create_graph=False,
                )[0]
                target_jacobian.append(gradient.detach().cpu().numpy())
            # (rows, outputs, inputs), converted to original physical units.
            jac = np.stack(target_jacobian, axis=1)
            jac *= y_scale[None, :, None] / x_scale[None, None, :]
            batch_jacobians.append(jac)
        predictions.append(np.concatenate(batch_predictions, axis=0) * y_scale + y_mean)
        jacobians.append(np.concatenate(batch_jacobians, axis=0))

    member_values = np.stack(predictions, axis=0)
    mean_prediction = member_values.mean(axis=0)
    member_count = len(members)
    if member_count > 2:
        committee_variance = (
            member_count
            / (member_count - 1.0)
            * (np.square(member_values).mean(axis=0) - np.square(mean_prediction))
        )
        committee_variance = np.maximum(committee_variance, 0.0)
    else:
        # The original C++ deliberately suppresses committee error for M <= 2.
        committee_variance = np.zeros_like(mean_prediction)

    # Legacy C++ test.cpp averages member Jacobians before propagating errors.
    mean_jacobian = np.mean(np.stack(jacobians, axis=0), axis=0)
    propagated_variance = np.sum(np.square(mean_jacobian * errors[:, None, :]), axis=2)
    total_variance = committee_variance + propagated_variance
    return {
        "prediction": mean_prediction,
        "committee_std": np.sqrt(committee_variance),
        "input_error_std": np.sqrt(propagated_variance),
        "total_std": np.sqrt(total_variance),
        "mean_jacobian": mean_jacobian,
    }


def write_committee_predictions_csv(
    source: str | Path | BinaryIO,
    output_path: str | Path,
    architecture: Architecture,
    preparation: DataPreparation,
    checkpoint_bytes: bytes,
    *,
    error_columns: dict[str, str] | None = None,
    separator: str = ",",
    device: str = "auto",
) -> dict[str, int | str]:
    """Write mean predictions and separate/combined uncertainty columns."""
    if hasattr(source, "seek"):
        source.seek(0)
    reader = pd.read_csv(source, sep=separator, chunksize=preparation.rows_per_batch)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    written = skipped = offset = 0
    header = True
    error_columns = error_columns or {}
    unknown_inputs = set(error_columns).difference(architecture.input_columns)
    if unknown_inputs:
        raise ValueError(
            "Error mappings refer to non-input columns: "
            + ", ".join(sorted(unknown_inputs))
        )
    try:
        with output.open("w", encoding="utf-8", newline="") as handle:
            for chunk in reader:
                chunk.columns = [str(c).lstrip("#").strip() for c in chunk.columns]
                if len(set(chunk.columns)) != len(chunk.columns):
                    raise ValueError(
                        "Prediction column names must be unique after removing a leading #."
                    )
                needed = list(architecture.input_columns) + list(error_columns.values())
                missing = [name for name in needed if name not in chunk.columns]
                if missing:
                    raise ValueError(
                        "Missing input/error columns: " + ", ".join(missing)
                    )
                x = (
                    chunk.loc[:, list(architecture.input_columns)]
                    .apply(pd.to_numeric, errors="coerce")
                    .to_numpy(dtype=np.float64, na_value=np.nan)
                )
                valid = np.isfinite(x).all(axis=1)
                errors = np.zeros_like(x)
                for index, name in enumerate(architecture.input_columns):
                    error_name = error_columns.get(name)
                    if error_name:
                        column = pd.to_numeric(
                            chunk[error_name], errors="coerce"
                        ).to_numpy(dtype=np.float64)
                        valid &= np.isfinite(column) & (column >= 0)
                        errors[:, index] = np.where(np.isfinite(column), column, 0.0)
                row_numbers = np.arange(offset + 1, offset + len(chunk) + 1)[valid]
                offset += len(chunk)
                if not len(row_numbers):
                    skipped += len(chunk)
                    continue
                result = predict_committee(
                    x[valid],
                    architecture,
                    preparation,
                    checkpoint_bytes,
                    input_errors=errors[valid],
                    device=device,
                )
                frame = pd.DataFrame(
                    result["prediction"], columns=architecture.target_columns
                )
                for target_index, target in enumerate(architecture.target_columns):
                    frame[f"{target}_committee_std"] = result["committee_std"][
                        :, target_index
                    ]
                    frame[f"{target}_input_error_std"] = result["input_error_std"][
                        :, target_index
                    ]
                    frame[f"{target}_total_std"] = result["total_std"][:, target_index]
                frame.insert(0, "source_row", row_numbers)
                frame.to_csv(handle, index=False, header=header, lineterminator="\n")
                header = False
                written += len(frame)
                skipped += len(chunk) - len(frame)
    finally:
        reader.close()
    if not written:
        raise ValueError(
            "The prediction catalogue has no rows with valid inputs and errors."
        )
    return {
        "path": str(output.resolve()),
        "predicted_rows": written,
        "skipped_rows": skipped,
    }


def evaluate_committee(
    source: str | Path | BinaryIO,
    architecture: Architecture,
    preparation: DataPreparation,
    checkpoint_bytes: bytes,
    *,
    split: str = "test",
    separator: str = ",",
    device: str = "auto",
) -> pd.DataFrame:
    """Evaluate committee-mean predictions on a named prepared split."""
    if split not in {"train", "validation", "test"}:
        raise ValueError("Choose train, validation, or test.")
    count = 0
    sum_y = np.zeros(architecture.n_outputs, dtype=np.float64)
    sum_y2 = np.zeros_like(sum_y)
    sum_abs = np.zeros_like(sum_y)
    sum_sqerr = np.zeros_like(sum_y)
    sum_committee_std = np.zeros_like(sum_y)
    x_mean = np.asarray(preparation.input_mean)
    x_scale = np.asarray(preparation.input_scale)
    y_mean = np.asarray(preparation.target_mean)
    y_scale = np.asarray(preparation.target_scale)
    for x_scaled, y_scaled in iter_prepared_batches(
        source, preparation, split=split, separator=separator, shuffle_seed=None
    ):
        x_raw = x_scaled * x_scale + x_mean
        result = predict_committee(
            x_raw, architecture, preparation, checkpoint_bytes, device=device
        )
        actual = y_scaled * y_scale + y_mean
        error = result["prediction"] - actual
        count += len(actual)
        sum_y += actual.sum(axis=0)
        sum_y2 += np.square(actual).sum(axis=0)
        sum_abs += np.abs(error).sum(axis=0)
        sum_sqerr += np.square(error).sum(axis=0)
        sum_committee_std += result["committee_std"].sum(axis=0)
    if not count:
        raise ValueError(f"The {split} split has no rows to evaluate.")
    total = np.maximum(sum_y2 - np.square(sum_y) / count, 0.0)
    q2_total = np.maximum(
        sum_y2 - 2.0 * y_mean * sum_y + count * np.square(y_mean), 0.0
    )
    return pd.DataFrame(
        [
            {
                "target": name,
                "rows": count,
                "MAE": sum_abs[i] / count,
                "RMSE": np.sqrt(sum_sqerr[i] / count),
                "R2": 1.0 - sum_sqerr[i] / total[i] if total[i] > 0 else float("nan"),
                "Q2": 1.0 - sum_sqerr[i] / q2_total[i]
                if q2_total[i] > 0
                else float("nan"),
                "mean_committee_std": sum_committee_std[i] / count,
            }
            for i, name in enumerate(architecture.target_columns)
        ]
    )
