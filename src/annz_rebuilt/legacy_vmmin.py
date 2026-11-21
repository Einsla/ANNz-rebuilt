"""Faithful Python translation of ANNzPlus Tanh/train.cpp::Training::vmmin."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class VMMinResult:
    weights: np.ndarray
    objective: float
    iterations: int
    converged: bool


def vmmin(
    initial_weights: np.ndarray,
    objective: Callable[[np.ndarray], float],
    gradient: Callable[[np.ndarray], np.ndarray],
    *,
    max_iterations: int,
    on_iteration: Callable[[int, np.ndarray, float], None] | None = None,
    on_periodic_update: Callable[[np.ndarray], None] | None = None,
) -> VMMinResult:
    """Run the dense variable-metric routine as implemented in legacy ANNzPlus.

    The inverse-Hessian approximation is stored as a dense symmetric matrix,
    matching the lower-triangle BFGS update in the C++ source. Memory is O(P²)
    for P trainable parameters.
    """
    if max_iterations < 1:
        raise ValueError("vmmin maximum iterations must be positive.")
    weights = np.asarray(initial_weights, dtype=np.float64).copy()
    n_weights = weights.size
    if n_weights == 0:
        raise ValueError("vmmin requires at least one trainable weight.")

    # np.zeros allocates the dense inverse-Hessian matrix used by the legacy quasi-Newton optimiser.
    # Its memory requirement grows with the square of the number of trainable weights.
    inverse_hessian = np.zeros((n_weights, n_weights), dtype=np.float64)
    current = float(objective(weights))
    minimum = current
    grad = np.asarray(gradient(weights), dtype=np.float64)
    if grad.shape != weights.shape:
        raise ValueError("vmmin gradient shape does not match the weight vector.")

    last_reset = 0
    last_hyper_update = 0
    grad_count = 0
    converged = False
    reset = True

    # Constants copied from Tanh/train.cpp.
    acceptance_tolerance = 1.0e-4
    step_reduction = 0.2
    relative_weight_test = 10.0
    absolute_objective_tolerance = 1.0e-2
    relative_objective_tolerance = 1.0e-8

    while grad_count < max_iterations:
        if reset:
            reset = False
            last_reset = grad_count
            inverse_hessian.fill(0.0)
            np.fill_diagonal(inverse_hessian, 1.0)

        previous_weights = weights.copy()
        previous_gradient = grad.copy()

        # t = -Gg; use symmetric inverse-Hessian approximation.
        direction = -(inverse_hessian @ grad)
        # np.dot calculates a vector dot product, used to test whether the search direction descends.
        grad_projection = float(np.dot(direction, grad))

        if grad_projection >= 0.0:
            if last_reset == grad_count:
                converged = True
                break
            reset = True
            continue

        step_length = 1.0
        accepted = False
        while not (reset or accepted):
            trial = previous_weights + step_length * direction
            changed = np.count_nonzero(
                (relative_weight_test + trial)
                != (relative_weight_test + previous_weights)
            )
            if changed:
                trial_value = float(objective(trial))
                accepted = (
                    trial_value
                    <= minimum + grad_projection * step_length * acceptance_tolerance
                )
                if accepted:
                    weights = trial
                    current = trial_value
                else:
                    step_length *= step_reduction
            else:
                reset = True

        enough = (
            current > absolute_objective_tolerance
            and current < (1.0 - relative_objective_tolerance) * minimum
        )
        if not enough:
            reset = True

        if reset and last_reset == grad_count:
            converged = True
            break

        # Match the source ordering: even a reset request reaches the BFGS
        # update below; the inverse Hessian is reset at the next outer pass.
        minimum = current
        grad_count += 1
        grad = np.asarray(gradient(weights), dtype=np.float64)
        displacement = step_length * direction
        gradient_change = grad - previous_gradient
        curvature = float(np.dot(displacement, gradient_change))

        if curvature > 0.0:
            hessian_gradient = inverse_hessian @ gradient_change
            scale = 1.0 + float(np.dot(gradient_change, hessian_gradient)) / curvature
            inverse_hessian += (
                scale * np.outer(displacement, displacement)
                - np.outer(hessian_gradient, displacement)
                - np.outer(displacement, hessian_gradient)
            ) / curvature
            inverse_hessian = 0.5 * (inverse_hessian + inverse_hessian.T)
        else:
            reset = True

        if grad_count - last_reset > 2 * n_weights:
            reset = True

        if grad_count - last_hyper_update > 29:
            last_hyper_update = grad_count
            if on_periodic_update is not None:
                on_periodic_update(weights.copy())
            # The legacy training code changes alpha/beta and reevaluates E.
            current = float(objective(weights))
            minimum = current

        if on_iteration is not None:
            on_iteration(grad_count, weights.copy(), minimum)

    return VMMinResult(
        weights=weights,
        objective=float(current),
        iterations=grad_count,
        converged=converged,
    )
