#!/usr/bin/env python3
"""Published DEM-error estimators mapped to MintPy interferogram arrays.

The functions in this module operate on NumPy arrays and intentionally keep
MintPy HDF5 I/O outside the numerical methods.  See PUBLISHED_METHODS.md for
the reproduction status and input-domain limitations of each method.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from itertools import combinations
from typing import Any

import numpy as np
from scipy import ndimage, sparse, stats
from scipy.optimize import minimize
from scipy.sparse.linalg import spsolve
from scipy.spatial import Delaunay, QhullError, cKDTree
from sklearn.decomposition import FastICA


@dataclass
class PublishedResult:
    """Common result returned by a published-method reproduction."""

    dem_error: np.ndarray
    valid: np.ndarray
    diagnostics: dict[str, Any] = field(default_factory=dict)


def _dates_and_indices(
    date_pairs: list[tuple[str, str]],
) -> tuple[list[str], dict[str, int]]:
    dates = sorted({date for pair in date_pairs for date in pair})
    return dates, {date: index for index, date in enumerate(dates)}


def _date_years(dates: list[str]) -> np.ndarray:
    parsed = [datetime.strptime(date[:8], "%Y%m%d") for date in dates]
    return np.asarray(
        [(date - parsed[0]).total_seconds() / (365.25 * 86400.0) for date in parsed],
        dtype=np.float64,
    )


def interferogram_design(
    date_pairs: list[tuple[str, str]],
) -> tuple[np.ndarray, list[str]]:
    """Return the interferogram-to-acquisition design with acquisition 0 fixed."""

    dates, indices = _dates_and_indices(date_pairs)
    design = np.zeros((len(date_pairs), len(dates) - 1), dtype=np.float64)
    for row, (master, slave) in enumerate(date_pairs):
        master_index = indices[master]
        slave_index = indices[slave]
        if master_index:
            design[row, master_index - 1] = -1.0
        if slave_index:
            design[row, slave_index - 1] = 1.0
    return design, dates


def invert_to_acquisition_series(
    values: np.ndarray,
    date_pairs: list[tuple[str, str]],
    weights: np.ndarray | None = None,
    rcond: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """Invert interferograms to acquisition values with the first date fixed to zero."""

    observations = np.asarray(values, dtype=np.float64)
    if observations.ndim == 1:
        observations = observations[:, None]
    design, dates = interferogram_design(date_pairs)
    if observations.shape[0] != design.shape[0]:
        raise ValueError("values and date_pairs have inconsistent interferogram counts")
    if weights is None:
        weight_array = np.ones_like(observations)
    else:
        weight_array = np.broadcast_to(np.asarray(weights, dtype=np.float64), observations.shape)

    acquisitions = np.full((len(dates), observations.shape[1]), np.nan, dtype=np.float64)
    acquisitions[0] = 0.0
    valid = np.zeros(observations.shape[1], dtype=bool)
    for pixel in range(observations.shape[1]):
        keep = (
            np.isfinite(observations[:, pixel])
            & np.isfinite(weight_array[:, pixel])
            & (weight_array[:, pixel] > 0)
        )
        if np.sum(keep) < design.shape[1]:
            continue
        weighted_design = design[keep] * np.sqrt(weight_array[keep, pixel])[:, None]
        weighted_values = observations[keep, pixel] * np.sqrt(weight_array[keep, pixel])
        solution, _, rank, _ = np.linalg.lstsq(weighted_design, weighted_values, rcond=rcond)
        if rank != design.shape[1]:
            continue
        acquisitions[1:, pixel] = solution
        valid[pixel] = True
    return acquisitions, valid


def _f_statistic(mixing: np.ndarray, baseline: np.ndarray) -> tuple[float, float]:
    keep = np.isfinite(mixing) & np.isfinite(baseline)
    a = mixing[keep]
    b = baseline[keep]
    if a.size < 3 or np.dot(b, b) <= 0:
        return 0.0, 0.0
    factor = float(np.dot(b, a) / np.dot(b, b))
    residual = a - factor * b
    denominator = float(np.dot(residual, residual) / max(a.size - 1, 1))
    numerator = float(np.dot(factor * b, factor * b))
    return numerator / max(denominator, 1e-15), factor


def nonparametric_ica_2019(
    phase: np.ndarray,
    dem_coefficient: np.ndarray,
    date_pairs: list[tuple[str, str]],
    weights: np.ndarray | None = None,
    alpha: float = 0.01,
    random_state: int = 2019,
) -> PublishedResult:
    """Reproduce Liang et al. (2019) FastICA/SVHT DEM-error estimation."""

    phase = np.asarray(phase, dtype=np.float64)
    coefficient = np.asarray(dem_coefficient, dtype=np.float64)
    if phase.ndim != 2 or coefficient.shape != phase.shape:
        raise ValueError("phase and dem_coefficient must both have shape (ifg, pixel)")

    acquisition_phase, phase_valid = invert_to_acquisition_series(
        phase, date_pairs, weights
    )
    acquisition_coefficient, coefficient_valid = invert_to_acquisition_series(
        coefficient, date_pairs, weights
    )
    sequential_phase = np.diff(acquisition_phase, axis=0)
    sequential_coefficient = np.diff(acquisition_coefficient, axis=0)
    valid = phase_valid & coefficient_valid & np.all(np.isfinite(sequential_phase), axis=0)
    output = np.full(phase.shape[1], np.nan, dtype=np.float64)
    if np.sum(valid) < 8:
        return PublishedResult(output, valid, {"reason": "fewer than 8 valid spatial samples"})

    samples = sequential_phase[:, valid].T
    baseline = np.nanmedian(sequential_coefficient[:, valid], axis=1)
    covariance = np.cov(samples, rowvar=False)
    eigenvalues = np.linalg.eigvalsh(covariance)
    svht = 2.858 * float(np.median(eigenvalues))
    initial_components = max(1, int(np.sum(eigenvalues > svht)))
    maximum_components = min(samples.shape[1], samples.shape[0] - 1)
    selected_source = None
    selected_factor = np.nan
    selected_correlation = np.nan
    selected_f = np.nan
    used_components = initial_components
    critical_f = stats.f.ppf(1.0 - alpha, 1, max(samples.shape[1] - 1, 1))

    for component_count in range(initial_components, maximum_components + 1):
        ica = FastICA(
            n_components=component_count,
            whiten="unit-variance",
            algorithm="parallel",
            fun="logcosh",
            max_iter=2000,
            tol=1e-5,
            random_state=random_state,
        )
        sources = ica.fit_transform(samples)
        mixing = ica.mixing_
        correlations = np.asarray(
            [
                np.corrcoef(mixing[:, index], baseline)[0, 1]
                if np.std(mixing[:, index]) > 0 and np.std(baseline) > 0
                else 0.0
                for index in range(component_count)
            ]
        )
        source_index = int(np.nanargmax(np.abs(correlations)))
        f_value, factor = _f_statistic(mixing[:, source_index], baseline)
        selected_source = sources[:, source_index]
        selected_factor = factor
        selected_correlation = float(correlations[source_index])
        selected_f = f_value
        used_components = component_count
        if f_value > critical_f:
            break

    output[valid] = selected_factor * selected_source
    return PublishedResult(
        output,
        valid & np.isfinite(output),
        {
            "svht": svht,
            "initial_components": initial_components,
            "components": used_components,
            "baseline_correlation": selected_correlation,
            "f_statistic": selected_f,
            "f_critical": float(critical_f),
            "significant": bool(selected_f > critical_f),
            "note": "ICA estimates the spatially centered DEM-error component.",
        },
    )


def _time_groups(times: np.ndarray, span_years: float, overlap: float) -> list[np.ndarray]:
    if times.size < 2:
        return [np.arange(times.size)]
    groups: list[np.ndarray] = []
    start = 0
    while start < times.size - 1:
        end = int(np.searchsorted(times, times[start] + span_years, side="right"))
        end = min(times.size, max(end, start + 3))
        group = np.arange(start, end)
        groups.append(group)
        if end == times.size:
            break
        overlap_count = max(2, int(np.ceil(group.size * overlap)))
        next_start = max(start + 1, end - overlap_count)
        start = next_start
    return groups


def _full_deformation_terms(times: np.ndarray) -> np.ndarray:
    local = times - times[0]
    return np.column_stack(
        (
            local,
            local**2,
            local**3,
            np.sin(2.0 * np.pi * times),
            np.cos(2.0 * np.pi * times),
        )
    )


def _select_hypothesis_terms(
    observations: np.ndarray,
    times: np.ndarray,
    alpha: float,
) -> np.ndarray:
    terms = _full_deformation_terms(times)
    design = np.column_stack((np.ones(times.size), terms))
    keep = np.isfinite(observations)
    if np.sum(keep) <= design.shape[1]:
        return np.zeros(terms.shape[1], dtype=bool)
    x = design[keep]
    y = observations[keep]
    parameters, _, rank, _ = np.linalg.lstsq(x, y, rcond=None)
    if rank != design.shape[1]:
        return np.zeros(terms.shape[1], dtype=bool)
    fitted = x @ parameters
    sse = float(np.sum((y - fitted) ** 2))
    ssr = float(np.sum((fitted - np.mean(y)) ** 2))
    p = terms.shape[1]
    dof = y.size - p - 1
    if dof <= 0:
        return np.zeros(terms.shape[1], dtype=bool)
    f_value = (ssr / p) / max(sse / dof, 1e-15)
    if f_value <= stats.f.ppf(1.0 - alpha, p, dof):
        return np.zeros(terms.shape[1], dtype=bool)
    covariance = (sse / dof) * np.linalg.pinv(x.T @ x)
    standard_error = np.sqrt(np.maximum(np.diag(covariance)[1:], 1e-15))
    t_value = np.abs(parameters[1:] / standard_error)
    return t_value > stats.t.ppf(1.0 - alpha / 2.0, dof)


def adaptive_hypothesis_2021(
    phase: np.ndarray,
    dem_coefficient: np.ndarray,
    date_pairs: list[tuple[str, str]],
    weights: np.ndarray | None = None,
    alpha: float = 0.01,
    group_span_years: float = 1.0,
    overlap: float = 0.2,
) -> PublishedResult:
    """Reproduce the grouped F/t-test adaptive deformation model of Du et al. (2021)."""

    phase = np.asarray(phase, dtype=np.float64)
    coefficient = np.asarray(dem_coefficient, dtype=np.float64)
    acquisition_phase, phase_valid = invert_to_acquisition_series(phase, date_pairs, weights)
    acquisition_coefficient, coefficient_valid = invert_to_acquisition_series(
        coefficient, date_pairs, weights
    )
    _, dates = interferogram_design(date_pairs)
    times = _date_years(dates)
    groups = _time_groups(times, group_span_years, overlap)
    output = np.full(phase.shape[1], np.nan, dtype=np.float64)
    selected_term_counts = np.zeros(phase.shape[1], dtype=np.int16)

    for pixel in range(phase.shape[1]):
        if not (phase_valid[pixel] and coefficient_valid[pixel]):
            continue
        selections = [
            _select_hypothesis_terms(acquisition_phase[group, pixel], times[group], alpha)
            for group in groups
        ]
        offsets = np.cumsum([0] + [int(np.sum(selection)) for selection in selections])
        rows: list[np.ndarray] = []
        observations: list[float] = []
        parameter_count = int(offsets[-1]) + 1
        for group_index, group in enumerate(groups):
            terms = _full_deformation_terms(times[group])[:, selections[group_index]]
            for local_index in range(group.size - 1):
                row = np.zeros(parameter_count, dtype=np.float64)
                if terms.shape[1]:
                    row[offsets[group_index] : offsets[group_index + 1]] = (
                        terms[local_index + 1] - terms[local_index]
                    )
                row[-1] = (
                    acquisition_coefficient[group[local_index + 1], pixel]
                    - acquisition_coefficient[group[local_index], pixel]
                )
                rows.append(row)
                observations.append(
                    acquisition_phase[group[local_index + 1], pixel]
                    - acquisition_phase[group[local_index], pixel]
                )

        for group_index in range(len(groups) - 1):
            overlap_dates = np.intersect1d(groups[group_index], groups[group_index + 1])
            if overlap_dates.size < 2:
                continue
            first_terms = _full_deformation_terms(times[groups[group_index]])[
                :, selections[group_index]
            ]
            second_terms = _full_deformation_terms(times[groups[group_index + 1]])[
                :, selections[group_index + 1]
            ]
            first_lookup = {date: index for index, date in enumerate(groups[group_index])}
            second_lookup = {date: index for index, date in enumerate(groups[group_index + 1])}
            for left, right in zip(overlap_dates[:-1], overlap_dates[1:]):
                row = np.zeros(parameter_count, dtype=np.float64)
                if first_terms.shape[1]:
                    row[offsets[group_index] : offsets[group_index + 1]] = (
                        first_terms[first_lookup[right]] - first_terms[first_lookup[left]]
                    )
                if second_terms.shape[1]:
                    row[offsets[group_index + 1] : offsets[group_index + 2]] -= (
                        second_terms[second_lookup[right]] - second_terms[second_lookup[left]]
                    )
                rows.append(row)
                observations.append(0.0)

        design = np.asarray(rows)
        values = np.asarray(observations)
        if design.shape[0] < parameter_count:
            continue
        solution, _, rank, _ = np.linalg.lstsq(design, values, rcond=None)
        if rank == parameter_count:
            output[pixel] = solution[-1]
            selected_term_counts[pixel] = int(offsets[-1])

    valid = np.isfinite(output)
    return PublishedResult(
        output,
        valid,
        {
            "groups": [group.tolist() for group in groups],
            "alpha": alpha,
            "mean_selected_terms": float(np.mean(selected_term_counts[valid])) if np.any(valid) else np.nan,
            "selected_term_counts": selected_term_counts,
        },
    )


def _ri_l1(
    observed: np.ndarray,
    predicted: np.ndarray,
    weights: np.ndarray,
) -> float:
    residual = (
        np.abs(np.sin(observed) - np.sin(predicted))
        + np.abs(np.cos(observed) - np.cos(predicted))
    )
    return float(np.sum(weights * residual) / max(np.sum(weights), 1e-15))


def igs_cmaes_2021_equivalent(
    wrapped_phase: np.ndarray,
    dem_coefficient: np.ndarray,
    temporal_design: np.ndarray,
    coherence: np.ndarray | None = None,
    velocity_bounds: tuple[float, float] = (-4.0, 4.0),
    dem_bounds: tuple[float, float] = (-200.0, 200.0),
    scales: tuple[int, ...] = (8, 7, 6, 5, 4, 3, 2),
    num_initial: int = 5,
    loss_threshold: float = 0.6,
    maxiter: int = 80,
) -> PublishedResult:
    """Map the IGS-CMAES RI-L1 search to SciPy's bounded Powell solver.

    The author implementation's iterative grid selection and RI-L1 objective
    are retained.  Powell replaces the unavailable ``cma`` dependency.
    """

    observed = np.asarray(wrapped_phase, dtype=np.float64)
    coefficient = np.asarray(dem_coefficient, dtype=np.float64)
    temporal = np.asarray(temporal_design, dtype=np.float64).reshape(-1)
    if observed.ndim != 2 or coefficient.shape != observed.shape:
        raise ValueError("wrapped_phase and dem_coefficient must have shape (ifg, pixel)")
    if temporal.size != observed.shape[0]:
        raise ValueError("temporal_design length must equal the interferogram count")
    if coherence is None:
        base_weights = np.ones_like(observed)
    else:
        base_weights = np.asarray(coherence, dtype=np.float64)
        if base_weights.shape != observed.shape:
            base_weights = np.broadcast_to(base_weights, observed.shape)
    output = np.full(observed.shape[1], np.nan, dtype=np.float64)
    velocity = np.full(observed.shape[1], np.nan, dtype=np.float64)
    evaluations = np.zeros(observed.shape[1], dtype=np.int32)
    losses = np.full(observed.shape[1], np.nan, dtype=np.float64)
    velocity_half_range = 0.5 * (velocity_bounds[1] - velocity_bounds[0])

    for pixel in range(observed.shape[1]):
        weights = np.where(
            np.isfinite(observed[:, pixel])
            & np.isfinite(coefficient[:, pixel])
            & np.isfinite(base_weights[:, pixel])
            & (base_weights[:, pixel] >= 0.5),
            base_weights[:, pixel],
            0.0,
        )
        if np.sum(weights > 0) < 3:
            continue

        def objective(parameters: np.ndarray) -> float:
            predicted = temporal * parameters[0] + coefficient[:, pixel] * parameters[1]
            return _ri_l1(observed[:, pixel], predicted, weights)

        candidates: list[tuple[float, float, float]] = []
        grid_evaluations = 0
        for scale in scales:
            velocity_step = max(scale * velocity_half_range / 50.0, 1e-6)
            dem_step = max(scale * 2.0, 0.5)
            velocity_grid = np.arange(
                velocity_bounds[0], velocity_bounds[1] + velocity_step / 2.0, velocity_step
            )
            dem_grid = np.arange(dem_bounds[0], dem_bounds[1] + dem_step / 2.0, dem_step)
            velocity_grid = np.clip(velocity_grid, *velocity_bounds)
            dem_grid = np.clip(dem_grid, *dem_bounds)
            velocity_grid = np.unique(velocity_grid)
            dem_grid = np.unique(dem_grid)
            current: list[tuple[float, float, float]] = []
            for velocity_value in velocity_grid:
                predicted_velocity = temporal * velocity_value
                predicted = predicted_velocity[:, None] + coefficient[:, pixel, None] * dem_grid[None, :]
                residual = (
                    np.abs(np.sin(observed[:, pixel, None]) - np.sin(predicted))
                    + np.abs(np.cos(observed[:, pixel, None]) - np.cos(predicted))
                )
                grid_losses = np.sum(weights[:, None] * residual, axis=0) / max(np.sum(weights), 1e-15)
                grid_evaluations += dem_grid.size
                best_indices = np.argsort(grid_losses)[:num_initial]
                current.extend(
                    (float(grid_losses[index]), float(velocity_value), float(dem_grid[index]))
                    for index in best_indices
                )
            current.sort(key=lambda item: item[0])
            candidates = []
            for candidate in current:
                normalized = np.asarray(
                    [
                        20.0 * (candidate[1] - velocity_bounds[0]) / max(2.0 * velocity_half_range, 1e-15) - 10.0,
                        20.0 * (candidate[2] - dem_bounds[0]) / max(dem_bounds[1] - dem_bounds[0], 1e-15) - 10.0,
                    ]
                )
                if all(
                    np.linalg.norm(
                        normalized
                        - np.asarray(
                            [
                                20.0 * (existing[1] - velocity_bounds[0]) / max(2.0 * velocity_half_range, 1e-15) - 10.0,
                                20.0 * (existing[2] - dem_bounds[0]) / max(dem_bounds[1] - dem_bounds[0], 1e-15) - 10.0,
                            ]
                        )
                    )
                    > np.pi
                    for existing in candidates
                ):
                    candidates.append(candidate)
                if len(candidates) == num_initial:
                    break
            if len(candidates) == num_initial and candidates[0][0] <= loss_threshold:
                break

        best_loss = np.inf
        best_parameters = None
        local_evaluations = 0
        for _, velocity_start, dem_start in candidates:
            start_parameters = np.asarray(
                [
                    np.clip(velocity_start, *velocity_bounds),
                    np.clip(dem_start, *dem_bounds),
                ]
            )
            candidate_parameters = start_parameters.copy()
            candidate_loss = objective(candidate_parameters)
            local_evaluations += 1
            result = minimize(
                objective,
                start_parameters,
                method="Powell",
                bounds=(velocity_bounds, dem_bounds),
                options={"maxiter": maxiter, "xtol": 1e-4, "ftol": 1e-5},
            )
            local_evaluations += int(result.nfev)
            if result.fun < candidate_loss:
                candidate_parameters = result.x
                candidate_loss = float(result.fun)

            velocity_step = (velocity_bounds[1] - velocity_bounds[0]) / 20.0
            dem_step = (dem_bounds[1] - dem_bounds[0]) / 20.0
            for _ in range(6):
                local_velocity = np.clip(
                    candidate_parameters[0] + velocity_step * np.arange(-2, 3),
                    *velocity_bounds,
                )
                local_dem = np.clip(
                    candidate_parameters[1] + dem_step * np.arange(-2, 3),
                    *dem_bounds,
                )
                for velocity_value in np.unique(local_velocity):
                    predicted_velocity = temporal * velocity_value
                    predicted = (
                        predicted_velocity[:, None]
                        + coefficient[:, pixel, None] * np.unique(local_dem)[None, :]
                    )
                    residual = (
                        np.abs(np.sin(observed[:, pixel, None]) - np.sin(predicted))
                        + np.abs(np.cos(observed[:, pixel, None]) - np.cos(predicted))
                    )
                    local_losses = np.sum(weights[:, None] * residual, axis=0) / max(
                        np.sum(weights), 1e-15
                    )
                    local_evaluations += local_losses.size
                    local_index = int(np.argmin(local_losses))
                    local_loss = float(local_losses[local_index])
                    if local_loss < candidate_loss:
                        candidate_loss = local_loss
                        candidate_parameters = np.asarray(
                            [velocity_value, np.unique(local_dem)[local_index]]
                        )
                velocity_step *= 0.5
                dem_step *= 0.5

            rounded_candidate = round(candidate_loss, 4)
            rounded_best = round(best_loss, 4)
            candidate_norm = np.linalg.norm(
                [
                    candidate_parameters[0] / max(velocity_half_range, 1e-15),
                    candidate_parameters[1]
                    / max(0.5 * (dem_bounds[1] - dem_bounds[0]), 1e-15),
                ]
            )
            best_norm = (
                np.linalg.norm(
                    [
                        best_parameters[0] / max(velocity_half_range, 1e-15),
                        best_parameters[1]
                        / max(0.5 * (dem_bounds[1] - dem_bounds[0]), 1e-15),
                    ]
                )
                if best_parameters is not None
                else np.inf
            )
            if rounded_candidate < rounded_best or (
                rounded_candidate == rounded_best and candidate_norm < best_norm
            ):
                best_loss = candidate_loss
                best_parameters = candidate_parameters
        if best_parameters is not None:
            velocity[pixel] = best_parameters[0]
            output[pixel] = best_parameters[1]
            losses[pixel] = best_loss
            evaluations[pixel] = grid_evaluations + local_evaluations

    valid = np.isfinite(output)
    return PublishedResult(
        output,
        valid,
        {
            "velocity": velocity,
            "ri_l1": losses,
            "objective_evaluations": evaluations,
            "optimizer": "IGS grid + bounded Powell (CMA-ES dependency unavailable)",
        },
    )


def _wrapped_sobel_direction(phase: np.ndarray) -> np.ndarray:
    wrapped = lambda value: np.angle(np.exp(1j * value))
    padded = np.pad(phase, 1, mode="edge")
    i1, i2, i3 = padded[:-2, :-2], padded[:-2, 1:-1], padded[:-2, 2:]
    i4, i6 = padded[1:-1, :-2], padded[1:-1, 2:]
    i7, i8, i9 = padded[2:, :-2], padded[2:, 1:-1], padded[2:, 2:]
    gx = wrapped(i3 - i1) + 2.0 * wrapped(i6 - i4) + wrapped(i9 - i7)
    gy = wrapped(i7 - i1) + 2.0 * wrapped(i8 - i2) + wrapped(i9 - i3)
    return np.arctan2(gy, gx)


def pgdc_detect_2025(
    wrapped_phase: np.ndarray,
    perpendicular_baseline: np.ndarray,
    temporal_days: np.ndarray,
    coherence: np.ndarray | None = None,
    threshold: float = 0.5,
    max_temporal_days: float = 60.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Implement the modified-Sobel phase gradient direction consistency detector."""

    phase = np.asarray(wrapped_phase, dtype=np.float64)
    if phase.ndim != 3:
        raise ValueError("wrapped_phase must have shape (ifg, rows, columns)")
    baselines = np.asarray(perpendicular_baseline, dtype=np.float64).reshape(-1)
    days = np.asarray(temporal_days, dtype=np.float64).reshape(-1)
    selected = np.isfinite(days) & (np.abs(days) <= max_temporal_days)
    if np.sum(selected) < min(3, phase.shape[0]):
        shortest = np.argsort(np.abs(days))[: min(3, phase.shape[0])]
        selected[shortest] = True
    indices = np.flatnonzero(selected)
    directions = np.empty((indices.size, *phase.shape[1:]), dtype=np.float64)
    if coherence is None:
        direction_weights = np.ones_like(directions)
    else:
        coherence_array = np.asarray(coherence, dtype=np.float64)
        direction_weights = np.asarray(coherence_array[indices], dtype=np.float64)
    for output_index, interferogram_index in enumerate(indices):
        normalized_phase = phase[interferogram_index]
        if baselines[interferogram_index] < 0:
            normalized_phase = -normalized_phase
        directions[output_index] = _wrapped_sobel_direction(normalized_phase)
    direction_weights = np.where(np.isfinite(directions), direction_weights, 0.0)
    resultant = np.sum(direction_weights * np.exp(1j * directions), axis=0)
    weight_sum = np.sum(direction_weights, axis=0)
    gdc = np.divide(
        np.abs(resultant),
        weight_sum,
        out=np.zeros(phase.shape[1:], dtype=np.float64),
        where=weight_sum > 0,
    )
    centers = gdc > threshold
    detected = ndimage.binary_dilation(centers, structure=np.ones((3, 3), dtype=bool))
    return gdc, detected, indices


def _network_edges(points: np.ndarray, maximum_length: float, neighbors: int = 3) -> list[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    if points.shape[0] >= 3:
        try:
            triangulation = Delaunay(points[:, ::-1])
            for simplex in triangulation.simplices:
                for left, right in combinations(simplex, 2):
                    edges.add(tuple(sorted((int(left), int(right)))))
        except QhullError:
            pass
    if points.shape[0] >= 2:
        tree = cKDTree(points)
        _, indices = tree.query(points, k=min(neighbors + 1, points.shape[0]))
        if indices.ndim == 1:
            indices = indices[:, None]
        for left, row in enumerate(indices):
            for right in row[1:]:
                edges.add(tuple(sorted((left, int(right)))))
    return [
        edge
        for edge in sorted(edges)
        if np.linalg.norm(points[edge[0]] - points[edge[1]]) < maximum_length
    ]


def _combined_interferogram_pairs(
    date_pairs: list[tuple[str, str]],
) -> list[tuple[int, int, int]]:
    records = []
    for index, (master, slave) in enumerate(date_pairs):
        master_date = datetime.strptime(master[:8], "%Y%m%d")
        slave_date = datetime.strptime(slave[:8], "%Y%m%d")
        duration = abs((slave_date - master_date).days)
        midpoint = master_date + (slave_date - master_date) / 2
        records.append((index, duration, midpoint))
    records.sort(key=lambda item: item[2])
    pairs: list[tuple[int, int, int]] = []
    for first, second in zip(records[:-1], records[1:]):
        if first[1] <= 0 or second[1] <= 0:
            continue
        factor = max(1, int(round(second[1] / first[1])))
        if factor <= 3 and abs(second[1] - factor * first[1]) <= max(3, int(0.2 * second[1])):
            pairs.append((first[0], second[0], factor))
    return pairs


def pgdc_estimate_2025(
    wrapped_phase: np.ndarray,
    dem_coefficient: np.ndarray,
    date_pairs: list[tuple[str, str]],
    detected: np.ndarray,
    coherence: np.ndarray | None = None,
    dem_bounds: tuple[float, float] = (-200.0, 200.0),
    dem_step: float = 1.0,
    maximum_edge_length: float = 30.0,
) -> PublishedResult:
    """Estimate PGDC relative DEM errors using TPC arc search and network adjustment."""

    phase = np.asarray(wrapped_phase, dtype=np.float64)
    coefficient = np.asarray(dem_coefficient, dtype=np.float64)
    if phase.ndim != 3 or coefficient.shape != phase.shape:
        raise ValueError("wrapped_phase and dem_coefficient must have shape (ifg, y, x)")
    points = np.argwhere(np.asarray(detected, dtype=bool))
    output = np.zeros(phase.shape[1:], dtype=np.float64)
    valid_map = np.zeros(phase.shape[1:], dtype=bool)
    if points.shape[0] < 2:
        return PublishedResult(output, valid_map, {"reason": "fewer than two detected pixels"})
    edges = _network_edges(points, maximum_edge_length)
    combinations_ = _combined_interferogram_pairs(date_pairs)
    if not edges or not combinations_:
        return PublishedResult(output, valid_map, {"reason": "no valid arcs or combined interferograms"})

    candidates = np.arange(dem_bounds[0], dem_bounds[1] + dem_step / 2.0, dem_step)
    relative = np.full(len(edges), np.nan, dtype=np.float64)
    tpc = np.zeros(len(edges), dtype=np.float64)
    for edge_index, (left, right) in enumerate(edges):
        ly, lx = points[left]
        ry, rx = points[right]
        arc_phase = np.angle(
            np.exp(1j * (phase[:, ry, rx] - phase[:, ly, lx]))
        )
        # For an arc, phi_j - phi_i = c_bar * (h_j - h_i).  The geometry
        # coefficient varies slowly in space, so use the endpoint mean.
        arc_coefficient = 0.5 * (
            coefficient[:, ry, rx] + coefficient[:, ly, lx]
        )
        combined_phase = []
        combined_coefficient = []
        for first, second, factor in combinations_:
            combined_phase.append(
                np.angle(np.exp(1j * (arc_phase[second] - factor * arc_phase[first])))
            )
            combined_coefficient.append(arc_coefficient[second] - factor * arc_coefficient[first])
        combined_phase_array = np.asarray(combined_phase)
        combined_coefficient_array = np.asarray(combined_coefficient)
        keep = np.isfinite(combined_phase_array) & np.isfinite(combined_coefficient_array)
        if np.sum(keep) < 2:
            continue
        residual = combined_phase_array[keep, None] - combined_coefficient_array[keep, None] * candidates[None, :]
        coherence_values = np.abs(np.mean(np.exp(1j * residual), axis=0))
        best = int(np.argmax(coherence_values))
        relative[edge_index] = candidates[best]
        tpc[edge_index] = coherence_values[best]

    usable_edges = [edge for edge, value in zip(edges, relative) if np.isfinite(value)]
    usable_relative = relative[np.isfinite(relative)]
    usable_tpc = tpc[np.isfinite(relative)]
    adjacency: list[list[int]] = [[] for _ in range(points.shape[0])]
    for left, right in usable_edges:
        adjacency[left].append(right)
        adjacency[right].append(left)
    visited = np.zeros(points.shape[0], dtype=bool)
    component_count = 0
    for start in range(points.shape[0]):
        if visited[start] or not adjacency[start]:
            continue
        stack = [start]
        component = []
        visited[start] = True
        while stack:
            node = stack.pop()
            component.append(node)
            for neighbor in adjacency[node]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    stack.append(neighbor)
        component_set = set(component)
        component_edges = [
            index
            for index, edge in enumerate(usable_edges)
            if edge[0] in component_set and edge[1] in component_set
        ]
        if not component_edges:
            continue
        component_count += 1
        component_points = points[component]
        boundary_score = (
            (component_points[:, 0] == np.min(component_points[:, 0]))
            | (component_points[:, 0] == np.max(component_points[:, 0]))
            | (component_points[:, 1] == np.min(component_points[:, 1]))
            | (component_points[:, 1] == np.max(component_points[:, 1]))
        )
        boundary_nodes = np.asarray(component)[boundary_score]
        if coherence is None:
            reference = int(boundary_nodes[0])
        else:
            mean_coherence = np.nanmean(np.asarray(coherence), axis=0)
            reference = int(
                boundary_nodes[
                    np.nanargmax([mean_coherence[tuple(points[node])] for node in boundary_nodes])
                ]
            )
        unknown_nodes = [node for node in component if node != reference]
        lookup = {node: index for index, node in enumerate(unknown_nodes)}
        design = np.zeros((len(component_edges), len(unknown_nodes)), dtype=np.float64)
        values = np.zeros(len(component_edges), dtype=np.float64)
        weights = np.zeros(len(component_edges), dtype=np.float64)
        for row, edge_index in enumerate(component_edges):
            left, right = usable_edges[edge_index]
            if left != reference:
                design[row, lookup[left]] = -1.0
            if right != reference:
                design[row, lookup[right]] = 1.0
            values[row] = usable_relative[edge_index]
            weights[row] = max(usable_tpc[edge_index], 1e-3)
        weighted_design = design * np.sqrt(weights)[:, None]
        weighted_values = values * np.sqrt(weights)
        solution = np.linalg.lstsq(weighted_design, weighted_values, rcond=None)[0]
        for node, value in zip(unknown_nodes, solution):
            output[tuple(points[node])] = value
            valid_map[tuple(points[node])] = True
        valid_map[tuple(points[reference])] = True

    return PublishedResult(
        output,
        valid_map,
        {
            "detected_pixels": int(points.shape[0]),
            "arcs": len(usable_edges),
            "combined_interferograms": len(combinations_),
            "subnetworks": component_count,
            "mean_tpc": float(np.mean(usable_tpc)) if usable_tpc.size else np.nan,
        },
    )


def dynamic_height_sbas_2025(
    phase: np.ndarray,
    dem_coefficient: np.ndarray,
    date_pairs: list[tuple[str, str]],
    weights: np.ndarray | None = None,
    minimum_side_acquisitions: int = 3,
) -> PublishedResult:
    """Search the surface-height change date using the improved SBAS model."""

    observations = np.asarray(phase, dtype=np.float64)
    coefficient = np.asarray(dem_coefficient, dtype=np.float64)
    if observations.ndim != 2 or coefficient.shape != observations.shape:
        raise ValueError("phase and dem_coefficient must have shape (ifg, pixel)")
    if weights is None:
        weight_array = np.ones_like(observations)
    else:
        weight_array = np.broadcast_to(np.asarray(weights, dtype=np.float64), observations.shape)
    dates, indices = _dates_and_indices(date_pairs)
    years = _date_years(dates)
    temporal = np.asarray(
        [years[indices[slave]] - years[indices[master]] for master, slave in date_pairs]
    )
    before_best = np.full(observations.shape[1], np.nan)
    after_best = np.full(observations.shape[1], np.nan)
    velocity_before = np.full(observations.shape[1], np.nan)
    velocity_after = np.full(observations.shape[1], np.nan)
    best_rss = np.full(observations.shape[1], np.inf)
    best_index = np.full(observations.shape[1], -1, dtype=np.int16)

    for change_index in range(minimum_side_acquisitions, len(dates) - minimum_side_acquisitions + 1):
        before = np.asarray(
            [indices[master] < change_index and indices[slave] < change_index for master, slave in date_pairs]
        )
        after = np.asarray(
            [indices[master] >= change_index and indices[slave] >= change_index for master, slave in date_pairs]
        )
        if np.sum(before) < 2 or np.sum(after) < 2:
            continue
        for pixel in range(observations.shape[1]):
            parameters = []
            rss = 0.0
            valid_fit = True
            for subset in (before, after):
                keep = (
                    subset
                    & np.isfinite(observations[:, pixel])
                    & np.isfinite(coefficient[:, pixel])
                    & np.isfinite(weight_array[:, pixel])
                    & (weight_array[:, pixel] > 0)
                )
                if np.sum(keep) < 2:
                    valid_fit = False
                    break
                design = np.column_stack((coefficient[keep, pixel], temporal[keep]))
                sqrt_weight = np.sqrt(weight_array[keep, pixel])
                solution, _, rank, _ = np.linalg.lstsq(
                    design * sqrt_weight[:, None],
                    observations[keep, pixel] * sqrt_weight,
                    rcond=None,
                )
                if rank != 2:
                    valid_fit = False
                    break
                residual = observations[keep, pixel] - design @ solution
                rss += float(np.sum(weight_array[keep, pixel] * residual**2))
                parameters.append(solution)
            if valid_fit and rss < best_rss[pixel]:
                best_rss[pixel] = rss
                best_index[pixel] = change_index
                before_best[pixel], velocity_before[pixel] = parameters[0]
                after_best[pixel], velocity_after[pixel] = parameters[1]

    valid = best_index >= 0
    return PublishedResult(
        after_best,
        valid,
        {
            "dem_error_before": before_best,
            "dem_error_after": after_best,
            "height_change": after_best - before_best,
            "change_index": best_index,
            "change_date": np.asarray(
                [dates[index] if index >= 0 and index < len(dates) else "" for index in best_index]
            ),
            "velocity_before": velocity_before,
            "velocity_after": velocity_after,
            "rss": best_rss,
        },
    )


def fractal_surface_regularize_2015_adapted(
    initial_dem_error: np.ndarray,
    magnitude_proxy: np.ndarray,
    valid_mask: np.ndarray | None = None,
    fidelity_weight: float = 1.0,
    range_weight: float = 2.0,
    azimuth_weight: float = 0.25,
) -> PublishedResult:
    """Linearized fractal-scattering regularizer for the MintPy data domain.

    The paper requires interferometric magnitude and jointly updates phase and
    magnitude.  MintPy's ifgramStack does not contain that magnitude, so this
    adapted form uses a supplied proxy to construct the fourth-root range-
    gradient prior from m = beta(c + grad(phi))**4.
    """

    initial = np.asarray(initial_dem_error, dtype=np.float64)
    proxy = np.asarray(magnitude_proxy, dtype=np.float64)
    if initial.ndim != 2 or proxy.shape != initial.shape:
        raise ValueError("initial_dem_error and magnitude_proxy must be 2-D with equal shape")
    if valid_mask is None:
        valid = np.isfinite(initial) & np.isfinite(proxy)
    else:
        valid = np.asarray(valid_mask, dtype=bool) & np.isfinite(initial) & np.isfinite(proxy)
    output = np.full_like(initial, np.nan)
    coordinates = np.argwhere(valid)
    if coordinates.size == 0:
        return PublishedResult(output, valid, {"reason": "no valid pixels"})
    index_map = np.full(initial.shape, -1, dtype=np.int64)
    index_map[valid] = np.arange(coordinates.shape[0])

    horizontal_edges = []
    vertical_edges = []
    for row, column in coordinates:
        if column + 1 < initial.shape[1] and valid[row, column + 1]:
            horizontal_edges.append(((row, column), (row, column + 1)))
        if row + 1 < initial.shape[0] and valid[row + 1, column]:
            vertical_edges.append(((row, column), (row + 1, column)))

    def difference_matrix(edges: list[tuple[tuple[int, int], tuple[int, int]]]) -> sparse.csr_matrix:
        if not edges:
            return sparse.csr_matrix((0, coordinates.shape[0]))
        rows = np.repeat(np.arange(len(edges)), 2)
        columns = []
        values = []
        for left, right in edges:
            columns.extend((index_map[left], index_map[right]))
            values.extend((-1.0, 1.0))
        return sparse.coo_matrix(
            (values, (rows, columns)), shape=(len(edges), coordinates.shape[0])
        ).tocsr()

    dx = difference_matrix(horizontal_edges)
    dy = difference_matrix(vertical_edges)
    initial_vector = initial[valid]
    if horizontal_edges:
        initial_gradient = dx @ initial_vector
        edge_proxy = np.asarray(
            [0.5 * (proxy[left] + proxy[right]) for left, right in horizontal_edges]
        )
        proxy_root = np.maximum(edge_proxy, 1e-8) ** 0.25
        proxy_root /= max(float(np.median(proxy_root)), 1e-8)
        gradient_scale = max(float(np.median(np.abs(initial_gradient))), 1e-6)
        target_gradient = np.sign(initial_gradient) * gradient_scale * proxy_root
        edge_weight = 1.0 / np.sqrt(initial_gradient**2 + gradient_scale**2)
        wx = sparse.diags(edge_weight)
    else:
        target_gradient = np.empty(0)
        wx = sparse.csr_matrix((0, 0))
    if vertical_edges:
        initial_vertical = dy @ initial_vector
        vertical_scale = max(float(np.median(np.abs(initial_vertical))), 1e-6)
        wy = sparse.diags(1.0 / np.sqrt(initial_vertical**2 + vertical_scale**2))
    else:
        wy = sparse.csr_matrix((0, 0))

    normal = fidelity_weight * sparse.eye(coordinates.shape[0], format="csr")
    rhs = fidelity_weight * initial_vector
    if horizontal_edges:
        normal = normal + range_weight * (dx.T @ wx @ dx)
        rhs = rhs + range_weight * (dx.T @ wx @ target_gradient)
    if vertical_edges:
        normal = normal + azimuth_weight * (dy.T @ wy @ dy)
    solution = spsolve(normal.tocsc(), rhs)
    output[valid] = solution
    smoothed = ndimage.convolve1d(
        np.where(valid, output, 0.0),
        weights=np.asarray([0.1, 0.8, 0.1]),
        axis=0,
        mode="nearest",
    )
    support = ndimage.convolve1d(valid.astype(np.float64), [0.1, 0.8, 0.1], axis=0, mode="nearest")
    output[valid] = smoothed[valid] / np.maximum(support[valid], 1e-8)
    return PublishedResult(
        output,
        valid,
        {
            "status": "adapted",
            "range_edges": len(horizontal_edges),
            "azimuth_edges": len(vertical_edges),
            "azimuth_kernel": [0.1, 0.8, 0.1],
            "limitation": "magnitude_proxy is not the interferometric magnitude required by the paper",
        },
    )
