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
        raise RuntimeError(
            "Install the dependencies in requirements.txt to train a network."
        ) from exc
    return torch, nn


def _device(torch, requested: str):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was selected but no CUDA GPU is available.")
    return torch.device(requested)


def build_network(architecture: Architecture):
    """Make a fully connected regression network from the saved layer settings."""
    _, nn = _torch()
    activations = {
        "leaky_relu": lambda: nn.LeakyReLU(0.01),
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
        "mish": nn.Mish,
        "softplus": nn.Softplus,
        "silu": nn.SiLU,
    }
    layers = []
    width = architecture.n_inputs
    for next_width, activation_name in zip(
        architecture.hidden_layers, architecture.effective_activations
    ):
        if activation_name not in activations:
            raise ValueError(f"Unsupported activation: {activation_name}")
        layers.extend((nn.Linear(width, next_width), activations[activation_name]()))
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
        raise ValueError(
            "Checkpoint settings do not match this network and prepared dataset."
        )
    return state


def _pack_checkpoint(
    torch,
    model,
    optimizer,
    epoch: int,
    signature: str,
    history: list[dict],
    optimizer_name: str = "adam",
    extra_state: dict | None = None,
) -> bytes:
    buffer = io.BytesIO()
    state = {
        "format_version": 1,
        "epoch": epoch,
        "signature": signature,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "optimizer_name": optimizer_name,
        "history": history,
    }
    if extra_state:
        state.update(extra_state)
    torch.save(state, buffer)
    return buffer.getvalue()


def _restore_model(blob: bytes, architecture, preparation, torch, device):
    signature = _signature(architecture, preparation)
    state = _read_checkpoint(blob, torch, device, signature)
    model = build_network(architecture).to(device)
    model.load_state_dict(state["model_state"])
    model.eval()
    return model, state


def _validation_loss(
    model, source, preparation, separator, torch, device
) -> float | None:
    total = 0.0
    rows = 0
    model.eval()
    with torch.no_grad():
        for x_np, y_np in iter_prepared_batches(
            source, preparation, split="validation", separator=separator
        ):
            if not len(x_np):
                continue
            parameter_dtype = next(model.parameters()).dtype
            x = torch.as_tensor(x_np, dtype=parameter_dtype, device=device)
            y = torch.as_tensor(y_np, dtype=parameter_dtype, device=device)
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
    optimizer_name: str = "adam",
    checkpoint_bytes: bytes | None = None,
    separator: str = ",",
    device: str = "auto",
    early_stopping_patience: int | None = None,
    initialization_seed: int | None = None,
    on_epoch: Callable[[dict], None] | None = None,
) -> dict:
    """Train with mini-batch Adam or the legacy ANNzPlus vmmin optimiser."""
    optimizer_name = optimizer_name.strip().lower().replace("-", "")
    if optimizer_name in {"quasinewton", "lbfgs", "legacy_vmmin"}:
        optimizer_name = "vmmin"
    if optimizer_name not in {"adam", "vmmin"}:
        raise ValueError("Choose Adam or legacy vmmin (quasi-Newton).")
    if early_stopping_patience is not None and early_stopping_patience < 1:
        raise ValueError("Early-stopping patience must be at least one.")
    if optimizer_name == "vmmin":
        if epochs < 1:
            raise ValueError("Maximum vmmin iterations must be positive.")
        if tuple(architecture.input_columns) != preparation.input_columns:
            raise ValueError("Network inputs do not match the prepared data.")
        if tuple(architecture.target_columns) != preparation.target_columns:
            raise ValueError("Network targets do not match the prepared data.")
        return _train_vmmin(
            source,
            architecture,
            preparation,
            epochs=epochs,
            checkpoint_bytes=checkpoint_bytes,
            separator=separator,
            device=device,
            early_stopping_patience=early_stopping_patience,
            initialization_seed=initialization_seed,
            on_epoch=on_epoch,
        )
    if epochs < 1 or not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("Epochs and learning rate must be positive and finite.")
    if tuple(architecture.input_columns) != preparation.input_columns:
        raise ValueError("Network inputs do not match the prepared data.")
    if tuple(architecture.target_columns) != preparation.target_columns:
        raise ValueError("Network targets do not match the prepared data.")
    if early_stopping_patience is not None and preparation.validation_fraction <= 0:
        raise ValueError("Early stopping requires a non-empty validation split.")
    torch, nn = _torch()
    target_device = _device(torch, device)
    torch.manual_seed(
        preparation.random_seed
        if initialization_seed is None
        else int(initialization_seed)
    )
    model = build_network(architecture).to(target_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    signature = _signature(architecture, preparation)
    history: list[dict] = []
    first_epoch = 0
    if checkpoint_bytes:
        state = _read_checkpoint(checkpoint_bytes, torch, target_device, signature)
        model.load_state_dict(state["model_state"])
        if state.get("optimizer_name", "adam") == "adam":
            optimizer.load_state_dict(state["optimizer_state"])
            history = list(state.get("history", []))
            first_epoch = int(state["epoch"])
        else:
            # Keep trained weights, but initialise fresh Adam state after a switch.
            history = []
            first_epoch = 0
            checkpoint_bytes = _pack_checkpoint(
                torch, model, optimizer, 0, signature, history, "adam"
            )
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = learning_rate
    else:
        checkpoint_bytes = _pack_checkpoint(
            torch, model, optimizer, 0, signature, history, "adam"
        )

    latest = checkpoint_bytes
    best_checkpoint = checkpoint_bytes
    initial_validation_loss = _validation_loss(
        model, source, preparation, separator, torch, target_device
    )
    best_validation_loss = (
        initial_validation_loss if initial_validation_loss is not None else float("inf")
    )
    best_epoch = first_epoch
    epochs_without_improvement = 0
    stopped_early = False
    completed_epoch = first_epoch
    for epoch in range(first_epoch + 1, first_epoch + epochs + 1):
        model.train()
        loss_sum = 0.0
        row_count = 0
        for x_np, y_np in iter_prepared_batches(
            source,
            preparation,
            split="train",
            separator=separator,
            shuffle_seed=preparation.random_seed + epoch,
        ):
            if not len(x_np):
                continue
            x = torch.as_tensor(x_np, dtype=torch.float32, device=target_device)
            y = torch.as_tensor(y_np, dtype=torch.float32, device=target_device)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.mse_loss(model(x), y)
            loss.backward()
            optimizer.step()
            # PyTorch's mean MSE is already averaged over rows and targets.
            # Weight by batch rows, then divide by total rows (not rows ? targets).
            loss_sum += loss.detach().item() * len(x_np)
            row_count += len(x_np)
        if row_count == 0:
            raise ValueError("The training split contains no usable rows.")
        validation_loss = _validation_loss(
            model, source, preparation, separator, torch, target_device
        )
        record = {
            "epoch": epoch,
            "training_loss": loss_sum / row_count,
            "validation_loss": validation_loss,
        }
        history.append(record)
        latest = _pack_checkpoint(
            torch, model, optimizer, epoch, signature, history, "adam"
        )
        completed_epoch = epoch
        if validation_loss is not None and validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch
            best_checkpoint = latest
            epochs_without_improvement = 0
        elif early_stopping_patience is not None and validation_loss is not None:
            epochs_without_improvement += 1
        if on_epoch:
            on_epoch(
                {**record, "checkpoint_bytes": latest, "device": str(target_device)}
            )
        if (
            early_stopping_patience is not None
            and validation_loss is not None
            and epochs_without_improvement >= early_stopping_patience
        ):
            stopped_early = True
            break

    # Always retain the lowest-validation-loss checkpoint when validation exists,
    # even when patience-based early stopping is disabled.
    selected_checkpoint = (
        best_checkpoint if best_validation_loss < float("inf") else latest
    )
    if selected_checkpoint is None:
        selected_checkpoint = latest
    return {
        "checkpoint_bytes": selected_checkpoint,
        "history": pd.DataFrame(history),
        "completed_epoch": completed_epoch,
        "best_epoch": (
            best_epoch if best_validation_loss < float("inf") else completed_epoch
        ),
        "best_validation_loss": (
            best_validation_loss if best_validation_loss < float("inf") else None
        ),
        "stopped_early": stopped_early,
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
            name
            for name in architecture.target_columns
            if name.strip().lower() in aliases
        }
    else:
        photoz_names = set(redshift_targets)
        unknown = photoz_names.difference(architecture.target_columns)
        if unknown:
            raise ValueError(
                "Redshift metrics requested for non-target columns: "
                + ", ".join(sorted(unknown))
            )
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
            "R2": 1 - sq_error[i] / split_total[i]
            if split_total[i] > 0
            else float("nan"),
            "Q2": 1 - sq_error[i] / q2_total[i] if q2_total[i] > 0 else float("nan"),
        }
        if name in photoz_names:
            delta = np.concatenate(photoz_residuals[name])
            row.update(
                {
                    "sigma_RMS": float(np.sqrt(np.mean(np.square(delta)))),
                    "sigma_68": float(
                        (np.quantile(delta, 0.841) - np.quantile(delta, 0.159)) / 2.0
                    ),
                    "outlier_fraction_percent": 100.0 * outlier_counts[name] / count,
                }
            )
        result.append(row)
    return pd.DataFrame(result)


def predict_single(
    inputs: np.ndarray,
    architecture: Architecture,
    preparation: DataPreparation,
    checkpoint_bytes: bytes,
    *,
    input_errors: np.ndarray | None = None,
    device: str = "auto",
) -> dict[str, np.ndarray]:
    """Predict with one network and propagate independent input errors via its Jacobian."""
    values = np.asarray(inputs, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != architecture.n_inputs:
        raise ValueError("Input matrix shape does not match the saved architecture.")
    if len(values) == 0:
        raise ValueError("At least one prediction input row is required.")
    if not np.isfinite(values).all():
        raise ValueError("Prediction inputs must be finite.")
    has_errors = input_errors is not None
    if has_errors:
        errors = np.asarray(input_errors, dtype=np.float64)
        if errors.shape != values.shape:
            raise ValueError("Input-error matrix must have the same shape as inputs.")
        if not np.isfinite(errors).all() or np.any(errors < 0):
            raise ValueError(
                "Input measurement errors must be finite and non-negative."
            )
    else:
        errors = np.zeros_like(values)

    torch, _ = _torch()
    target_device = _device(torch, device)
    model, _ = _restore_model(
        checkpoint_bytes, architecture, preparation, torch, target_device
    )
    x_mean = np.asarray(preparation.input_mean, dtype=np.float64)
    x_scale = np.asarray(preparation.input_scale, dtype=np.float64)
    y_mean = np.asarray(preparation.target_mean, dtype=np.float64)
    y_scale = np.asarray(preparation.target_scale, dtype=np.float64)
    predictions, propagated = [], []
    for start in range(0, len(values), preparation.batch_size):
        stop = min(start + preparation.batch_size, len(values))
        normalized = (values[start:stop] - x_mean) / x_scale
        if has_errors:
            x = (
                torch.as_tensor(normalized, dtype=torch.float32, device=target_device)
                .detach()
                .requires_grad_(True)
            )
            output = model(x)
            gradients = []
            for target_index in range(architecture.n_outputs):
                gradient = torch.autograd.grad(
                    output[:, target_index].sum(),
                    x,
                    retain_graph=target_index < architecture.n_outputs - 1,
                    create_graph=False,
                )[0]
                gradients.append(gradient.detach().cpu().numpy())
            jacobian = np.stack(gradients, axis=1)
            jacobian *= y_scale[None, :, None] / x_scale[None, None, :]
            propagated.append(
                np.sum(np.square(jacobian * errors[start:stop, None, :]), axis=2)
            )
        else:
            with torch.no_grad():
                output = model(
                    torch.as_tensor(
                        normalized, dtype=torch.float32, device=target_device
                    )
                )
            propagated.append(np.zeros((stop - start, architecture.n_outputs)))
        predictions.append(output.detach().cpu().numpy().astype(np.float64))
    prediction = np.concatenate(predictions, axis=0) * y_scale + y_mean
    input_error_std = np.sqrt(np.concatenate(propagated, axis=0))
    return {"prediction": prediction, "input_error_std": input_error_std}


def write_predictions_csv(
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
    """Predict a catalogue in chunks and optionally export Jacobian uncertainties."""
    if hasattr(source, "seek"):
        source.seek(0)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    reader = pd.read_csv(source, sep=separator, chunksize=preparation.rows_per_batch)
    offset = written = skipped = 0
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
                values = (
                    chunk.loc[:, list(architecture.input_columns)]
                    .apply(pd.to_numeric, errors="coerce")
                    .to_numpy(dtype="float64", na_value=np.nan)
                )
                errors = np.zeros_like(values)
                valid = np.isfinite(values).all(axis=1)
                for index, input_name in enumerate(architecture.input_columns):
                    error_name = error_columns.get(input_name)
                    if error_name:
                        column = pd.to_numeric(
                            chunk[error_name], errors="coerce"
                        ).to_numpy(dtype=np.float64)
                        valid &= np.isfinite(column) & (column >= 0)
                        errors[:, index] = np.where(np.isfinite(column), column, 0.0)
                source_rows = np.arange(offset + 1, offset + len(chunk) + 1)[valid]
                offset += len(chunk)
                if not len(source_rows):
                    skipped += len(chunk)
                    continue
                result = predict_single(
                    values[valid],
                    architecture,
                    preparation,
                    checkpoint_bytes,
                    input_errors=errors[valid] if error_columns else None,
                    device=device,
                )
                frame = pd.DataFrame(
                    result["prediction"], columns=architecture.target_columns
                )
                if error_columns:
                    for target_index, target in enumerate(architecture.target_columns):
                        frame[f"{target}_input_error_std"] = result["input_error_std"][
                            :, target_index
                        ]
                frame.insert(0, "source_row", source_rows)
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


def _train_vmmin(
    source,
    architecture,
    preparation,
    *,
    epochs,
    checkpoint_bytes,
    separator,
    device,
    early_stopping_patience,
    initialization_seed,
    on_epoch,
):
    """Train using the original ANNzPlus variable-metric algorithm.

    The legacy optimiser is full-objective, not mini-batch SGD. Source data is
    reread in bounded chunks for each objective and gradient evaluation.
    """
    from .vmmin_training import train_with_vmmin

    torch, _ = _torch()
    target_device = _device(torch, device)
    torch.manual_seed(
        preparation.random_seed
        if initialization_seed is None
        else int(initialization_seed)
    )
    model = build_network(architecture).to(target_device).double()
    signature = _signature(architecture, preparation)
    history = []
    previous_iterations = 0
    initial_alpha = 1.0
    initial_beta = 1.0
    if checkpoint_bytes:
        state = _read_checkpoint(checkpoint_bytes, torch, target_device, signature)
        model.load_state_dict(state["model_state"])
        history = list(state.get("history", []))
        previous_iterations = int(state.get("epoch", 0))
        if state.get("optimizer_name") == "vmmin":
            initial_alpha = float(state.get("vmmin_alpha", 1.0))
            initial_beta = float(state.get("vmmin_beta", 1.0))
    initial_weights = np.concatenate(
        [p.detach().cpu().numpy().reshape(-1) for p in model.parameters()]
    ).astype(np.float64)

    # vmmin is a full-objective optimiser. Patience-based early stopping is not
    # part of the original C++ algorithm; validation RMS is tracked for model
    # selection, but it does not alter the optimiser's trajectory.
    if early_stopping_patience is not None:
        raise ValueError(
            "Legacy vmmin follows its own objective/convergence rules and does "
            "not use validation-loss patience. Disable early stopping for vmmin."
        )

    latest = checkpoint_bytes
    initial_validation_loss = _validation_loss(
        model, source, preparation, separator, torch, target_device
    )
    best_validation_loss = (
        initial_validation_loss if initial_validation_loss is not None else float("inf")
    )
    best_iteration = previous_iterations
    best_checkpoint = _pack_checkpoint(
        torch,
        model,
        None,
        previous_iterations,
        signature,
        history,
        "vmmin",
        {"vmmin_alpha": initial_alpha, "vmmin_beta": initial_beta},
    )
    latest = best_checkpoint

    def callback(index, weights, objective_value):
        nonlocal latest, best_checkpoint, best_validation_loss, best_iteration
        iteration = previous_iterations + index
        validation_loss = _validation_loss(
            model, source, preparation, separator, torch, target_device
        )
        record = {
            "epoch": iteration,
            "training_loss": float(objective_value),
            "validation_loss": validation_loss,
        }
        history.append(record)
        latest = _pack_checkpoint(
            torch,
            model,
            None,
            iteration,
            signature,
            history,
            "vmmin",
            {"vmmin_alpha": initial_alpha, "vmmin_beta": initial_beta},
        )
        if validation_loss is not None and validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_iteration = iteration
            best_checkpoint = latest
        if on_epoch:
            on_epoch(
                {
                    **record,
                    "checkpoint_bytes": latest,
                    "device": str(target_device),
                }
            )

    outcome = train_with_vmmin(
        torch=torch,
        model=model,
        source=source,
        preparation=preparation,
        separator=separator,
        device=target_device,
        initial_weights=initial_weights,
        max_iterations=epochs,
        initial_alpha=initial_alpha,
        initial_beta=initial_beta,
        on_iteration=callback,
    )
    completed = previous_iterations + outcome["iterations"]
    latest = _pack_checkpoint(
        torch,
        model,
        None,
        completed,
        signature,
        history,
        "vmmin",
        {"vmmin_alpha": outcome["alpha"], "vmmin_beta": outcome["beta"]},
    )
    if best_iteration == completed and np.isfinite(best_validation_loss):
        best_checkpoint = latest
    selected = best_checkpoint if np.isfinite(best_validation_loss) else latest
    return {
        "checkpoint_bytes": selected or latest,
        "history": pd.DataFrame(history),
        "completed_epoch": completed,
        "best_epoch": best_iteration
        if best_iteration > previous_iterations
        else completed,
        "best_validation_loss": (
            best_validation_loss if np.isfinite(best_validation_loss) else None
        ),
        "stopped_early": False,
        "device": str(target_device),
        "converged": outcome["converged"],
        "weight_decay_alpha": outcome["alpha"],
        "weight_decay_beta": outcome["beta"],
    }
