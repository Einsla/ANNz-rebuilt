"""Batched neural-network training, evaluation, checkpointing, and prediction."""
from __future__ import annotations

import io
import json
from pathlib import Path
from typing import BinaryIO, Callable

import numpy as np
import pandas as pd

from .architecture import Architecture
from .data_preparation import DataPreparation, iter_prepared_batches


def _torch():
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise RuntimeError("Install the dependencies in requirements.txt to train a network.") from exc
    return torch, nn


def _device(torch, requested: str):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was selected but no CUDA GPU is available.")
    return torch.device(requested)


def build_network(architecture: Architecture):
    """Make a fully connected regression network from the saved layer settings."""
    torch, nn = _torch()
    activations = {
        "leaky_relu": lambda: nn.LeakyReLU(0.01),
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
        "mish": nn.Mish,
        "softplus": nn.Softplus,
        "silu": nn.SiLU,
    }
    if architecture.activation not in activations:
        raise ValueError(f"Unsupported activation: {architecture.activation}")
    layers = []
    width = architecture.n_inputs
    for next_width in architecture.hidden_layers:
        layers.extend((nn.Linear(width, next_width), activations[architecture.activation]()))
        width = next_width
    layers.append(nn.Linear(width, architecture.n_outputs))
    return nn.Sequential(*layers)


def _signature(architecture: Architecture, preparation: DataPreparation) -> str:
    config = {
        "architecture": architecture.to_config_dict(),
        "preparation": preparation.to_config_dict(),
    }
    return json.dumps(config, sort_keys=True, separators=(",", ":"))


def _read_checkpoint(blob: bytes, torch, device, signature: str):
    try:
        state = torch.load(io.BytesIO(blob), map_location=device, weights_only=True)
    except Exception as exc:
        raise ValueError("Could not read this model checkpoint.") from exc
    if not isinstance(state, dict):
        raise ValueError("The checkpoint has an invalid structure.")
    if state.get("signature") != signature:
        raise ValueError("Checkpoint settings do not match this network and prepared dataset.")
    return state


def _pack_checkpoint(torch, model, optimizer, epoch: int, signature: str, history: list[dict]) -> bytes:
    buffer = io.BytesIO()
    torch.save(
        {
            "format_version": 1,
            "epoch": epoch,
            "signature": signature,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "history": history,
        },
        buffer,
    )
    return buffer.getvalue()


def _restore_model(blob: bytes, architecture, preparation, torch, device):
    signature = _signature(architecture, preparation)
    state = _read_checkpoint(blob, torch, device, signature)
    model = build_network(architecture).to(device)
    model.load_state_dict(state["model_state"])
    model.eval()
    return model, state


def _validation_loss(model, source, preparation, separator, torch, device) -> float | None:
    total = 0.0
    rows = 0
    model.eval()
    with torch.no_grad():
        for x_np, y_np in iter_prepared_batches(
            source, preparation, split="validation", separator=separator
        ):
            if not len(x_np):
                continue
            x = torch.as_tensor(x_np, dtype=torch.float32, device=device)
            y = torch.as_tensor(y_np, dtype=torch.float32, device=device)
            total += torch.square(model(x) - y).sum().item()
            rows += len(x_np)
    return total / (rows * len(preparation.target_columns)) if rows else None


def train_model(
    source: str | Path | BinaryIO,
    architecture: Architecture,
    preparation: DataPreparation,
    *,
    epochs: int = 10,
    learning_rate: float = 0.001,
    checkpoint_bytes: bytes | None = None,
    separator: str = ",",
    device: str = "auto",
    on_epoch: Callable[[dict], None] | None = None,
) -> dict:
    """Train additional epochs in mini-batches; return a resumable checkpoint."""
    if epochs < 1 or learning_rate <= 0:
        raise ValueError("Epochs and learning rate must be positive.")
    if tuple(architecture.input_columns) != preparation.input_columns:
        raise ValueError("Network inputs do not match the prepared data.")
    if tuple(architecture.target_columns) != preparation.target_columns:
        raise ValueError("Network targets do not match the prepared data.")
    torch, nn = _torch()
    target_device = _device(torch, device)
    torch.manual_seed(preparation.random_seed)
    model = build_network(architecture).to(target_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    signature = _signature(architecture, preparation)
    history: list[dict] = []
    first_epoch = 0
    if checkpoint_bytes:
        state = _read_checkpoint(checkpoint_bytes, torch, target_device, signature)
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = learning_rate
        history = list(state.get("history", []))
        first_epoch = int(state["epoch"])

    latest = checkpoint_bytes
    for epoch in range(first_epoch + 1, first_epoch + epochs + 1):
        model.train()
        loss_sum = 0.0
        row_count = 0
        for x_np, y_np in iter_prepared_batches(
            source, preparation, split="train", separator=separator
        ):
            if not len(x_np):
                continue
            x = torch.as_tensor(x_np, dtype=torch.float32, device=target_device)
            y = torch.as_tensor(y_np, dtype=torch.float32, device=target_device)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.mse_loss(model(x), y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.detach().item() * len(x_np)
            row_count += len(x_np)
        if row_count == 0:
            raise ValueError("The training split contains no usable rows.")
        record = {
            "epoch": epoch,
            "training_loss": loss_sum / (row_count * len(preparation.target_columns)),
            "validation_loss": _validation_loss(
                model, source, preparation, separator, torch, target_device
            ),
        }
        history.append(record)
        latest = _pack_checkpoint(torch, model, optimizer, epoch, signature, history)
        if on_epoch:
            on_epoch({**record, "checkpoint_bytes": latest, "device": str(target_device)})
    return {
        "checkpoint_bytes": latest,
        "history": pd.DataFrame(history),
        "completed_epoch": first_epoch + epochs,
        "device": str(target_device),
    }


def evaluate_model(
    source: str | Path | BinaryIO,
    architecture: Architecture,
    preparation: DataPreparation,
    checkpoint_bytes: bytes,
    *,
    split: str = "test",
    separator: str = ",",
    device: str = "auto",
    redshift_targets: tuple[str, ...] | list[str] | None = None,
) -> pd.DataFrame:
    """Calculate predictive Q2, R2, MAE/RMSE and ANNz+ photo-z metrics.

    Q2 uses the training-target mean as its reference (external predictive
    coefficient of determination). R2 uses the evaluated split's own mean.
    Photo-z metrics are calculated only for explicitly selected redshift
    targets, or automatically for conventional redshift column names.
    """
    if split not in {"train", "validation", "test"}:
        raise ValueError("Choose train, validation, or test.")
    aliases = {"z", "redshift", "z_spec", "zspec", "specz", "spec_z"}
    if redshift_targets is None:
        photoz_names = {
            name for name in architecture.target_columns
            if name.strip().lower() in aliases
        }
    else:
        photoz_names = set(redshift_targets)
        unknown = photoz_names.difference(architecture.target_columns)
        if unknown:
            raise ValueError("Redshift metrics requested for non-target columns: " + ", ".join(sorted(unknown)))
    torch, _ = _torch()
    target_device = _device(torch, device)
    model, _ = _restore_model(
        checkpoint_bytes, architecture, preparation, torch, target_device
    )
    count = 0
    mean = np.asarray(preparation.target_mean, dtype=np.float64)
    scale = np.asarray(preparation.target_scale, dtype=np.float64)
    sums = np.zeros(architecture.n_outputs)
    squares = np.zeros(architecture.n_outputs)
    abs_error = np.zeros(architecture.n_outputs)
    sq_error = np.zeros(architecture.n_outputs)
    photoz_residuals = {name: [] for name in photoz_names}
    outlier_counts = {name: 0 for name in photoz_names}
    with torch.no_grad():
        for x_np, y_scaled in iter_prepared_batches(
            source, preparation, split=split, separator=separator
        ):
            if not len(x_np):
                continue
            x = torch.as_tensor(x_np, dtype=torch.float32, device=target_device)
            predicted = model(x).cpu().numpy().astype(np.float64) * scale + mean
            actual = np.asarray(y_scaled, dtype=np.float64) * scale + mean
            error = predicted - actual
            count += len(actual)
            sums += actual.sum(axis=0)
            squares += np.square(actual).sum(axis=0)
            abs_error += np.abs(error).sum(axis=0)
            sq_error += np.square(error).sum(axis=0)
            for i, name in enumerate(architecture.target_columns):
                if name in photoz_names:
                    denominator = 1.0 + actual[:, i]
                    if np.any(denominator <= 0.0):
                        raise ValueError(
                            f"Cannot calculate normalized photo-z metrics for {name!r}: "
                            "target redshift must be greater than -1."
                        )
                    delta = error[:, i] / denominator
                    photoz_residuals[name].append(delta)
                    outlier_counts[name] += int(np.count_nonzero(np.abs(delta) >= 0.15))
    if count == 0:
        raise ValueError(f"The {split} split has no rows to evaluate.")
    split_total = np.maximum(squares - np.square(sums) / count, 0.0)
    # External predictive Q2 is referenced to the training-set target mean.
    q2_total = np.maximum(squares - 2.0 * mean * sums + count * np.square(mean), 0.0)
    result = []
    for i, name in enumerate(architecture.target_columns):
        row = {
            "target": name,
            "rows": count,
            "MAE": abs_error[i] / count,
            "RMSE": np.sqrt(sq_error[i] / count),
            "R2": 1 - sq_error[i] / split_total[i] if split_total[i] > 0 else float("nan"),
            "Q2": 1 - sq_error[i] / q2_total[i] if q2_total[i] > 0 else float("nan"),
        }
        if name in photoz_names:
            delta = np.concatenate(photoz_residuals[name])
            row.update({
                "sigma_RMS": float(np.sqrt(np.mean(np.square(delta)))),
                "sigma_68": float((np.quantile(delta, 0.841) - np.quantile(delta, 0.159)) / 2.0),
                "outlier_fraction_percent": 100.0 * outlier_counts[name] / count,
            })
        result.append(row)
    return pd.DataFrame(result)


def write_predictions_csv(
    source: str | Path | BinaryIO,
    output_path: str | Path,
    architecture: Architecture,
    preparation: DataPreparation,
    checkpoint_bytes: bytes,
    *,
    separator: str = ",",
    device: str = "auto",
) -> dict[str, int | str]:
    """Predict a catalogue in chunks and write results directly to a CSV file."""
    torch, _ = _torch()
    target_device = _device(torch, device)
    model, _ = _restore_model(
        checkpoint_bytes, architecture, preparation, torch, target_device
    )
    if hasattr(source, "seek"):
        source.seek(0)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    reader = pd.read_csv(source, sep=separator, chunksize=preparation.rows_per_batch)
    offset = written = 0
    header = True
    try:
        with output.open("w", encoding="utf-8", newline="") as handle:
            for chunk in reader:
                chunk.columns = [str(c).lstrip("#").strip() for c in chunk.columns]
                if len(set(chunk.columns)) != len(chunk.columns):
                    raise ValueError("Prediction column names must be unique after removing a leading #.")
                missing = [c for c in architecture.input_columns if c not in chunk.columns]
                if missing:
                    raise ValueError("Missing input variables: " + ", ".join(missing))
                values = chunk.loc[:, list(architecture.input_columns)].apply(
                    pd.to_numeric, errors="coerce"
                ).to_numpy(dtype="float64", na_value=np.nan)
                valid = np.isfinite(values).all(axis=1)
                source_rows = np.arange(offset + 1, offset + len(chunk) + 1)[valid]
                offset += len(chunk)
                if not len(source_rows):
                    continue
                valid_values = values[valid]
                y = np.empty(
                    (len(valid_values), architecture.n_outputs), dtype=np.float64
                )
                input_mean = np.asarray(preparation.input_mean)
                input_scale = np.asarray(preparation.input_scale)
                # Prediction must obey the same GPU mini-batch limit as training.
                for start in range(0, len(valid_values), preparation.batch_size):
                    end = min(start + preparation.batch_size, len(valid_values))
                    x_np = (
                        valid_values[start:end] - input_mean
                    ) / input_scale
                    x = torch.as_tensor(
                        x_np, dtype=torch.float32, device=target_device
                    )
                    with torch.no_grad():
                        y[start:end] = model(x).cpu().numpy().astype(np.float64)
                y = y * np.asarray(preparation.target_scale) + np.asarray(
                    preparation.target_mean
                )
                frame = pd.DataFrame(y, columns=architecture.target_columns)
                frame.insert(0, "source_row", source_rows)
                frame.to_csv(handle, index=False, header=header, lineterminator="\n")
                header = False
                written += len(frame)
    finally:
        reader.close()
    if not written:
        raise ValueError("The prediction catalogue has no rows with valid inputs.")
    return {
        "path": str(output.resolve()),
        "predicted_rows": written,
        "skipped_rows": offset - written,
    }
