"""Neural-network objective and streamed data adapter for legacy vmmin."""

from __future__ import annotations

from typing import Callable

import numpy as np

from .data_preparation import DataPreparation, iter_prepared_batches
from .legacy_vmmin import vmmin


def train_with_vmmin(
    *,
    torch,
    model,
    source,
    preparation: DataPreparation,
    separator: str,
    device,
    initial_weights: np.ndarray,
    max_iterations: int,
    initial_alpha: float = 1.0,
    initial_beta: float = 1.0,
    on_iteration: Callable[[int, np.ndarray, float], None] | None = None,
    max_dense_parameters: int = 5000,
):
    """Train with the legacy sum-of-squares + weight-decay objective.

    Catalogue chunks are streamed from disk for each objective/gradient pass;
    the optimiser still uses the complete training-set objective and gradient.
    """
    parameters = [p for p in model.parameters()]
    parameter_count = sum(p.numel() for p in parameters)
    if parameter_count > max_dense_parameters:
        estimated_mb = parameter_count * parameter_count * 8 / (1024**2)
        raise ValueError(
            f"Legacy vmmin needs a dense {parameter_count} x {parameter_count} "
            f"inverse-Hessian matrix (~{estimated_mb:.0f} MiB). This exceeds the "
            f"configured safety limit of {max_dense_parameters} parameters. "
            "Use a smaller network or Adam for this catalogue."
        )

    model.to(device=device, dtype=torch.float64)
    alpha = float(initial_alpha)
    beta = float(initial_beta)
    train_rows = preparation.rows_by_split.get("train", 0)
    if train_rows < 1:
        raise ValueError("The training split contains no usable rows.")

    def set_weights(vector: np.ndarray) -> None:
        cursor = 0
        with torch.no_grad():
            for parameter in parameters:
                size = parameter.numel()
                values = torch.as_tensor(
                    vector[cursor:cursor + size].reshape(tuple(parameter.shape)),
                    dtype=parameter.dtype,
                    device=device,
                )
                parameter.copy_(values)
                cursor += size

    def objective(vector: np.ndarray) -> float:
        set_weights(vector)
        sum_squares = 0.0
        model.eval()
        with torch.no_grad():
            for x_np, y_np in iter_prepared_batches(
                source, preparation, split="train", separator=separator
            ):
                if not len(x_np):
                    continue
                x = torch.as_tensor(x_np, dtype=torch.float64, device=device)
                y = torch.as_tensor(y_np, dtype=torch.float64, device=device)
                sum_squares += float(torch.square(model(x) - y).sum().item())
        return 0.5 * beta * sum_squares + 0.5 * alpha * float(np.dot(vector, vector))

    def gradient(vector: np.ndarray) -> np.ndarray:
        set_weights(vector)
        accumulated = [torch.zeros_like(p) for p in parameters]
        model.train()
        for x_np, y_np in iter_prepared_batches(
            source, preparation, split="train", separator=separator
        ):
            if not len(x_np):
                continue
            x = torch.as_tensor(x_np, dtype=torch.float64, device=device)
            y = torch.as_tensor(y_np, dtype=torch.float64, device=device)
            residual = model(x) - y
            sum_squares = torch.square(residual).sum()
            grads = torch.autograd.grad(sum_squares, parameters)
            for total, grad in zip(accumulated, grads):
                total.add_(grad.detach())
        flat = np.concatenate(
            [
                (0.5 * beta * total + alpha * parameter).detach()
                .cpu().numpy().reshape(-1)
                for total, parameter in zip(accumulated, parameters)
            ]
        )
        return flat.astype(np.float64, copy=False)

    def update_hyperparameters(vector: np.ndarray) -> None:
        nonlocal alpha, beta
        weight_decay_sum = float(np.dot(vector, vector))
        data_sum_squares = 0.0
        set_weights(vector)
        model.eval()
        with torch.no_grad():
            for x_np, y_np in iter_prepared_batches(
                source, preparation, split="train", separator=separator
            ):
                if not len(x_np):
                    continue
                x = torch.as_tensor(x_np, dtype=torch.float64, device=device)
                y = torch.as_tensor(y_np, dtype=torch.float64, device=device)
                data_sum_squares += float(torch.square(model(x) - y).sum().item())
        if weight_decay_sum > 0 and data_sum_squares > 0:
            # Exact "quick and dirty" update in legacy Tanh/train.cpp.
            alpha = parameter_count / weight_decay_sum
            beta = train_rows / data_sum_squares

    def iteration_callback(index: int, vector: np.ndarray, value: float) -> None:
        if on_iteration is not None:
            on_iteration(index, vector, value)

    result = vmmin(
        initial_weights,
        objective,
        gradient,
        max_iterations=max_iterations,
        on_iteration=iteration_callback,
        on_periodic_update=update_hyperparameters,
    )
    set_weights(result.weights)
    return {
        "weights": result.weights,
        "objective": result.objective,
        "iterations": result.iterations,
        "converged": result.converged,
        "alpha": alpha,
        "beta": beta,
    }
