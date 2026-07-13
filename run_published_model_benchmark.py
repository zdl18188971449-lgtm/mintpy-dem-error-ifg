#!/usr/bin/env python3
"""Benchmark published DEM-error methods against the current project model."""

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

import mintpy_dem_error_ifg as current
import published_dem_error_models as published


METHOD_LABELS = {
    "hybrid_optimal_2026": "Hybrid optimal (2026)",
    "linear_huber": "Linear Huber",
    "adaptive_vce_huber_graph": "Current adaptive graph",
    "ica_2019": "Nonparametric ICA (2019)",
    "adaptive_ht_2021": "Adaptive HT (2021)",
    "igs_cmaes_2021_equivalent": "IGS-CMAES equivalent (2021)",
    "pgdc_2025": "PGDC (2025)",
    "dynamic_height_2025": "Dynamic height SBAS (2025)",
    "fractal_2015_adapted": "Fractal adapted (2015)",
    "initial_spike_map": "Unregularized estimate",
}


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run scenario-matched published DEM-error method comparisons.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-o", "--output-dir", default="published_model_benchmark")
    parser.add_argument("--size", type=int, default=18)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return (values - np.mean(values)) / max(np.std(values), 1e-8)


def _dates(count: int, spacing_days: int = 36) -> list[str]:
    reference = datetime(2020, 1, 1)
    return [
        (reference + timedelta(days=spacing_days * index)).strftime("%Y%m%d")
        for index in range(count)
    ]


def _pairs(dates: list[str], neighbors: int = 3) -> list[tuple[str, str]]:
    return [
        (dates[left], dates[right])
        for left in range(len(dates))
        for right in range(left + 1, min(len(dates), left + neighbors + 1))
    ]


def _pair_values(
    acquisition_values: np.ndarray,
    date_pairs: list[tuple[str, str]],
    dates: list[str],
) -> np.ndarray:
    lookup = {date: index for index, date in enumerate(dates)}
    return np.asarray(
        [
            acquisition_values[lookup[slave]] - acquisition_values[lookup[master]]
            for master, slave in date_pairs
        ]
    )


def _terrain_and_truth(
    rng: np.random.Generator, size: int
) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.mgrid[-1:1:complex(size), -1:1:complex(size)]
    terrain = 520.0 * np.exp(-((xx + 0.25) ** 2 / 0.16 + (yy - 0.05) ** 2 / 0.28))
    terrain += 180.0 * np.sin(2.6 * xx) * np.cos(2.0 * yy)
    terrain += 55.0 * gaussian_filter(rng.normal(size=(size, size)), size / 12.0)
    dem_error = 9.0 * _normalize(gaussian_filter(rng.laplace(size=(size, size)), size / 14.0))
    dem_error += 4.0 * np.tanh(_normalize(terrain))
    dem_error -= np.mean(dem_error)
    return terrain, dem_error


def _static_stack(
    rng: np.random.Generator, size: int
) -> dict[str, np.ndarray | list[tuple[str, str]] | list[str]]:
    dates = _dates(12)
    date_pairs = _pairs(dates, 3)
    terrain, dem_error = _terrain_and_truth(rng, size)
    acquisition_baseline = rng.normal(0.0, 230.0, len(dates))
    acquisition_baseline[0] = 0.0
    baseline = _pair_values(acquisition_baseline, date_pairs, dates)
    coefficient = np.repeat(
        (0.0018 * baseline)[:, None], size * size, axis=1
    )
    years = np.asarray(
        [
            (datetime.strptime(date, "%Y%m%d") - datetime.strptime(dates[0], "%Y%m%d")).days
            / 365.25
            for date in dates
        ]
    )
    velocity = 0.9 * _normalize(gaussian_filter(rng.normal(size=(size, size)), size / 9.0))
    acceleration = 0.35 * _normalize(gaussian_filter(rng.normal(size=(size, size)), size / 8.0))
    seasonal = 0.25 * _normalize(gaussian_filter(rng.normal(size=(size, size)), size / 7.0))
    acquisition_phase = years[:, None, None] * velocity
    acquisition_phase += 0.5 * years[:, None, None] ** 2 * acceleration
    acquisition_phase += np.sin(2.0 * np.pi * years)[:, None, None] * seasonal
    deformation = _pair_values(acquisition_phase, date_pairs, dates).reshape(len(date_pairs), -1)
    atmosphere = np.asarray(
        [
            0.10 * gaussian_filter(rng.normal(size=(size, size)), size / 10.0)
            for _ in date_pairs
        ]
    ).reshape(len(date_pairs), -1)
    noise = rng.normal(0.0, 0.035, (len(date_pairs), size * size))
    phase = coefficient * dem_error.reshape(1, -1) + deformation + atmosphere + noise
    coherence = np.clip(0.9 - 1.8 * np.abs(noise), 0.55, 0.97)
    return {
        "dates": dates,
        "date_pairs": date_pairs,
        "terrain": terrain,
        "truth": dem_error,
        "phase": phase,
        "coefficient": coefficient,
        "coherence": coherence,
        "baseline": baseline,
        "years": years,
    }


def _linear_huber(
    phase: np.ndarray,
    coefficient: np.ndarray,
    date_pairs: list[tuple[str, str]],
    weights: np.ndarray,
) -> np.ndarray:
    design, _ = current.build_deformation_design(date_pairs, 1)
    solution, _ = current.solve_model_block(
        phase,
        coefficient,
        design,
        np.ones(len(date_pairs), dtype=bool),
        weights,
        robust=True,
        huber_iterations=8,
    )
    return solution[:, 0]


def _current_adaptive_graph(
    phase: np.ndarray,
    coefficient: np.ndarray,
    date_pairs: list[tuple[str, str]],
    weights: np.ndarray,
    terrain: np.ndarray,
) -> np.ndarray:
    solution, valid, information, _, _, _ = current.solve_adaptive_model_block(
        phase,
        coefficient,
        date_pairs,
        np.ones(len(date_pairs), dtype=bool),
        weights,
        huber_iterations=8,
    )
    initial = solution[:, 0].reshape(terrain.shape)
    regularized, _ = current.terrain_graph_regularize(
        initial,
        information.reshape(terrain.shape),
        terrain,
        valid.reshape(terrain.shape),
        regularization=8.0,
        iterations=40,
    )
    return regularized.reshape(-1)


def _metrics(truth: np.ndarray, estimate: np.ndarray, mask: np.ndarray | None = None) -> dict[str, float]:
    truth = np.asarray(truth, dtype=np.float64).reshape(-1)
    estimate = np.asarray(estimate, dtype=np.float64).reshape(-1)
    valid = np.isfinite(truth) & np.isfinite(estimate)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool).reshape(-1)
    if not np.any(valid):
        return {"rmse_m": np.nan, "mae_m": np.nan, "correlation": np.nan}
    residual = estimate[valid] - truth[valid]
    correlation = (
        float(np.corrcoef(truth[valid], estimate[valid])[0, 1])
        if np.std(truth[valid]) > 0 and np.std(estimate[valid]) > 0
        else np.nan
    )
    return {
        "rmse_m": float(np.sqrt(np.mean(residual**2))),
        "mae_m": float(np.mean(np.abs(residual))),
        "correlation": correlation,
    }


def _append_result(
    rows: list[dict],
    scenario: str,
    method: str,
    status: str,
    truth: np.ndarray,
    estimate: np.ndarray,
    runtime: float,
    mask: np.ndarray | None = None,
    extra: dict | None = None,
) -> None:
    row = {
        "scenario": scenario,
        "method": method,
        "label": METHOD_LABELS[method],
        "reproduction_status": status,
        **_metrics(truth, estimate, mask),
        "runtime_s": runtime,
        "extra": json.dumps(extra or {}, ensure_ascii=True),
    }
    rows.append(row)


def run_benchmark(size: int, seed: int) -> tuple[list[dict], dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    maps: dict[str, np.ndarray] = {}
    static = _static_stack(rng, size)
    phase = static["phase"]
    coefficient = static["coefficient"]
    coherence = static["coherence"]
    date_pairs = static["date_pairs"]
    terrain = static["terrain"]
    truth = static["truth"]
    weights = coherence**2
    maps["static_truth"] = truth

    start = time.perf_counter()
    linear = _linear_huber(phase, coefficient, date_pairs, weights)
    runtime = time.perf_counter() - start
    maps["static_linear"] = linear.reshape(size, size)
    _append_result(rows, "static_nonlinear", "linear_huber", "project_baseline", truth, linear, runtime)

    start = time.perf_counter()
    graph = _current_adaptive_graph(phase, coefficient, date_pairs, weights, terrain)
    runtime = time.perf_counter() - start
    maps["static_graph"] = graph.reshape(size, size)
    _append_result(
        rows,
        "static_nonlinear",
        "adaptive_vce_huber_graph",
        "current",
        truth,
        graph,
        runtime,
    )

    start = time.perf_counter()
    ica = published.nonparametric_ica_2019(phase, coefficient, date_pairs, weights)
    runtime = time.perf_counter() - start
    maps["static_ica"] = ica.dem_error.reshape(size, size)
    _append_result(
        rows,
        "static_nonlinear",
        "ica_2019",
        "strict",
        truth,
        ica.dem_error,
        runtime,
        extra={key: value for key, value in ica.diagnostics.items() if np.isscalar(value)},
    )

    start = time.perf_counter()
    adaptive = published.adaptive_hypothesis_2021(phase, coefficient, date_pairs, weights)
    runtime = time.perf_counter() - start
    maps["static_adaptive"] = adaptive.dem_error.reshape(size, size)
    _append_result(
        rows,
        "static_nonlinear",
        "adaptive_ht_2021",
        "strict",
        truth,
        adaptive.dem_error,
        runtime,
        extra={"mean_selected_terms": adaptive.diagnostics["mean_selected_terms"]},
    )

    start = time.perf_counter()
    hybrid_static = published.hybrid_optimal_2026(
        phase.reshape(len(date_pairs), size, size),
        coefficient.reshape(len(date_pairs), size, size),
        date_pairs,
        coherence.reshape(len(date_pairs), size, size),
        terrain=terrain,
        dem_bounds=(-80.0, 80.0),
    )
    runtime = time.perf_counter() - start
    maps["static_hybrid"] = hybrid_static.dem_error
    _append_result(
        rows,
        "static_nonlinear",
        "hybrid_optimal_2026",
        "new_hybrid",
        truth,
        hybrid_static.dem_error,
        runtime,
        extra={
            "dynamic_pixels": int(np.sum(hybrid_static.diagnostics["dynamic_mask"])),
            "sparse_mode": bool(hybrid_static.diagnostics["sparse_mode"]),
        },
    )

    # Wrapped linear scenario for the IGS objective.
    temporal_design, _ = current.build_deformation_design(date_pairs, 1)
    velocity = 1.1 * _normalize(gaussian_filter(rng.normal(size=(size, size)), size / 8.0)).reshape(-1)
    wrapped_truth = 0.75 * truth.reshape(-1)
    maps["wrapped_truth"] = wrapped_truth.reshape(size, size)
    clean_wrapped_phase = coefficient * wrapped_truth[None, :]
    clean_wrapped_phase += temporal_design[:, :1] @ velocity[None, :]
    noisy_unwrapped = clean_wrapped_phase + rng.normal(0.0, 0.02, clean_wrapped_phase.shape)
    yy, xx = np.mgrid[:size, :size]
    unwrap_region = (
        ((xx - 0.56 * size) / (0.22 * size)) ** 2
        + ((yy - 0.44 * size) / (0.17 * size)) ** 2
        <= 1.0
    )
    for interferogram_index in (3, 11, 17):
        if interferogram_index < noisy_unwrapped.shape[0]:
            noisy_unwrapped[interferogram_index, unwrap_region.reshape(-1)] += 2.0 * np.pi
    wrapped = np.angle(np.exp(1j * noisy_unwrapped))
    start = time.perf_counter()
    igs = published.igs_cmaes_2021_equivalent(
        wrapped,
        coefficient,
        temporal_design[:, 0],
        coherence,
        velocity_bounds=(-4.0, 4.0),
        dem_bounds=(-80.0, 80.0),
        loss_threshold=0.3,
        maxiter=50,
    )
    runtime = time.perf_counter() - start
    maps["wrapped_igs"] = igs.dem_error.reshape(size, size)
    _append_result(
        rows,
        "wrapped_linear",
        "igs_cmaes_2021_equivalent",
        "equivalent",
        wrapped_truth,
        igs.dem_error,
        runtime,
        extra={
            "mean_ri_l1": float(np.nanmean(igs.diagnostics["ri_l1"])),
            "mean_evaluations": float(np.nanmean(igs.diagnostics["objective_evaluations"])),
        },
    )

    start = time.perf_counter()
    wrapped_linear_huber = _linear_huber(noisy_unwrapped, coefficient, date_pairs, weights)
    runtime = time.perf_counter() - start
    maps["wrapped_linear"] = wrapped_linear_huber.reshape(size, size)
    _append_result(
        rows,
        "wrapped_linear",
        "linear_huber",
        "project_baseline_unwrapped",
        wrapped_truth,
        wrapped_linear_huber,
        runtime,
    )

    start = time.perf_counter()
    hybrid_wrapped = published.hybrid_optimal_2026(
        noisy_unwrapped.reshape(len(date_pairs), size, size),
        coefficient.reshape(len(date_pairs), size, size),
        date_pairs,
        coherence.reshape(len(date_pairs), size, size),
        wrapped_phase=wrapped.reshape(len(date_pairs), size, size),
        terrain=terrain,
        velocity_bounds=(-4.0, 4.0),
        dem_bounds=(-80.0, 80.0),
    )
    runtime = time.perf_counter() - start
    maps["wrapped_hybrid"] = hybrid_wrapped.dem_error
    _append_result(
        rows,
        "wrapped_linear",
        "hybrid_optimal_2026",
        "new_hybrid",
        wrapped_truth,
        hybrid_wrapped.dem_error,
        runtime,
        extra=hybrid_wrapped.diagnostics["igs"],
    )

    # Sparse DEM-error scenario for PGDC detection and network estimation.
    sparse_dates = _dates(11, 24)
    sparse_pairs = list(zip(sparse_dates[:-1], sparse_dates[1:]))
    sparse_truth = np.zeros((size, size), dtype=np.float64)
    yy, xx = np.mgrid[:size, :size]
    sparse_truth[((xx - 0.55 * size) / (0.16 * size)) ** 2 + ((yy - 0.5 * size) / (0.2 * size)) ** 2 <= 1] = 24.0
    sparse_baseline = np.asarray([-300, -220, -150, -80, 70, 130, 200, 280, 340, 410.0])
    sparse_coefficient = np.broadcast_to(
        0.0025 * sparse_baseline[:, None, None], (len(sparse_pairs), size, size)
    )
    sparse_phase = sparse_coefficient * sparse_truth
    sparse_phase += rng.normal(0.0, 0.02, sparse_phase.shape)
    sparse_wrapped = np.angle(np.exp(1j * sparse_phase))
    start = time.perf_counter()
    gdc, detected, selected = published.pgdc_detect_2025(
        sparse_wrapped,
        sparse_baseline,
        np.full(len(sparse_pairs), 24.0),
        threshold=0.4,
    )
    pgdc = published.pgdc_estimate_2025(
        sparse_wrapped,
        sparse_coefficient,
        sparse_pairs,
        detected,
        dem_bounds=(-60.0, 60.0),
        dem_step=0.5,
        maximum_edge_length=max(6.0, size / 2.0),
    )
    runtime = time.perf_counter() - start
    maps["sparse_truth"] = sparse_truth
    maps["sparse_gdc"] = gdc
    maps["sparse_pgdc"] = pgdc.dem_error
    _append_result(
        rows,
        "sparse_dem",
        "pgdc_2025",
        "equivalent",
        sparse_truth,
        pgdc.dem_error,
        runtime,
        extra={
            "precision": float(np.sum(detected & (sparse_truth != 0)) / max(np.sum(detected), 1)),
            "recall": float(np.sum(detected & (sparse_truth != 0)) / max(np.sum(sparse_truth != 0), 1)),
            "selected_ifgrams": selected.tolist(),
            **pgdc.diagnostics,
        },
    )
    sparse_phase_flat = sparse_phase.reshape(len(sparse_pairs), -1)
    sparse_coefficient_flat = sparse_coefficient.reshape(len(sparse_pairs), -1)
    sparse_weights = np.ones_like(sparse_phase_flat)
    start = time.perf_counter()
    sparse_graph = _current_adaptive_graph(
        sparse_phase_flat,
        sparse_coefficient_flat,
        sparse_pairs,
        sparse_weights,
        np.zeros((size, size)),
    )
    runtime = time.perf_counter() - start
    maps["sparse_graph"] = sparse_graph.reshape(size, size)
    _append_result(
        rows,
        "sparse_dem",
        "adaptive_vce_huber_graph",
        "current",
        sparse_truth,
        sparse_graph,
        runtime,
    )

    start = time.perf_counter()
    hybrid_sparse = published.hybrid_optimal_2026(
        sparse_phase,
        sparse_coefficient,
        sparse_pairs,
        np.full_like(sparse_phase, 0.9),
        wrapped_phase=sparse_wrapped,
        terrain=np.zeros((size, size)),
        pgdc_threshold=0.4,
        velocity_bounds=(-4.0, 4.0),
        dem_bounds=(-60.0, 60.0),
    )
    runtime = time.perf_counter() - start
    maps["sparse_hybrid"] = hybrid_sparse.dem_error
    _append_result(
        rows,
        "sparse_dem",
        "hybrid_optimal_2026",
        "new_hybrid",
        sparse_truth,
        hybrid_sparse.dem_error,
        runtime,
        extra={
            "sparse_mode": bool(hybrid_sparse.diagnostics["sparse_mode"]),
            "active_pixels": int(np.sum(hybrid_sparse.diagnostics["active_mask"])),
        },
    )

    # Dynamic surface-height scenario.
    dynamic_dates = _dates(12, 40)
    dynamic_pairs = _pairs(dynamic_dates, 3)
    dynamic_lookup = {date: index for index, date in enumerate(dynamic_dates)}
    dynamic_baseline_acquisition = rng.normal(0.0, 250.0, len(dynamic_dates))
    dynamic_baseline_acquisition[0] = 0.0
    dynamic_coefficient_acquisition = 0.0018 * dynamic_baseline_acquisition
    dynamic_coefficient = _pair_values(
        dynamic_coefficient_acquisition, dynamic_pairs, dynamic_dates
    )
    dynamic_coefficient = np.repeat(dynamic_coefficient[:, None], size * size, axis=1)
    before_truth = 0.55 * truth
    height_change = np.zeros_like(before_truth)
    height_change[((xx - 0.48 * size) / (0.22 * size)) ** 2 + ((yy - 0.52 * size) / (0.18 * size)) ** 2 <= 1] = 16.0
    after_truth = before_truth + height_change
    change_index = 6
    dynamic_phase = np.empty((len(dynamic_pairs), size * size), dtype=np.float64)
    dynamic_years = np.arange(len(dynamic_dates)) * 40.0 / 365.25
    dynamic_velocity = 0.35 * _normalize(gaussian_filter(rng.normal(size=(size, size)), size / 8.0)).reshape(-1)
    for pair_index, (master, slave) in enumerate(dynamic_pairs):
        master_index = dynamic_lookup[master]
        slave_index = dynamic_lookup[slave]
        if slave_index < change_index:
            topographic = dynamic_coefficient[pair_index] * before_truth.reshape(-1)
        elif master_index >= change_index:
            topographic = dynamic_coefficient[pair_index] * after_truth.reshape(-1)
        else:
            topographic = (
                dynamic_coefficient_acquisition[slave_index] * after_truth.reshape(-1)
                - dynamic_coefficient_acquisition[master_index] * before_truth.reshape(-1)
            )
        dynamic_phase[pair_index] = topographic
        dynamic_phase[pair_index] += (
            dynamic_years[slave_index] - dynamic_years[master_index]
        ) * dynamic_velocity
    dynamic_phase += rng.normal(0.0, 0.02, dynamic_phase.shape)
    dynamic_weights = np.ones_like(dynamic_phase)
    start = time.perf_counter()
    dynamic = published.dynamic_height_sbas_2025(
        dynamic_phase, dynamic_coefficient, dynamic_pairs, dynamic_weights
    )
    runtime = time.perf_counter() - start
    maps["dynamic_truth"] = height_change
    maps["dynamic_estimate"] = dynamic.diagnostics["height_change"].reshape(size, size)
    _append_result(
        rows,
        "dynamic_height",
        "dynamic_height_2025",
        "strict",
        height_change,
        dynamic.diagnostics["height_change"],
        runtime,
        extra={
            "mean_change_index_error": float(
                np.mean(np.abs(dynamic.diagnostics["change_index"][dynamic.valid] - change_index))
            )
        },
    )
    start = time.perf_counter()
    dynamic_static = _current_adaptive_graph(
        dynamic_phase,
        dynamic_coefficient,
        dynamic_pairs,
        dynamic_weights,
        terrain,
    )
    runtime = time.perf_counter() - start
    maps["dynamic_static"] = np.zeros((size, size))
    _append_result(
        rows,
        "dynamic_height",
        "adaptive_vce_huber_graph",
        "current_static_height_assumption",
        height_change,
        np.zeros_like(dynamic_static),
        runtime,
        extra={"estimated_static_dem_rmse_after_m": _metrics(after_truth, dynamic_static)["rmse_m"]},
    )

    start = time.perf_counter()
    hybrid_dynamic = published.hybrid_optimal_2026(
        dynamic_phase.reshape(len(dynamic_pairs), size, size),
        dynamic_coefficient.reshape(len(dynamic_pairs), size, size),
        dynamic_pairs,
        np.full((len(dynamic_pairs), size, size), 0.9),
        terrain=terrain,
        velocity_bounds=(-4.0, 4.0),
        dem_bounds=(-80.0, 80.0),
    )
    runtime = time.perf_counter() - start
    maps["dynamic_hybrid"] = hybrid_dynamic.diagnostics["height_change"]
    _append_result(
        rows,
        "dynamic_height",
        "hybrid_optimal_2026",
        "new_hybrid",
        height_change,
        hybrid_dynamic.diagnostics["height_change"],
        runtime,
        extra={
            "dynamic_pixels": int(np.sum(hybrid_dynamic.diagnostics["dynamic_mask"])),
            "mean_change_index_error": float(
                np.mean(
                    np.abs(
                        hybrid_dynamic.diagnostics["change_index"][
                            hybrid_dynamic.diagnostics["dynamic_mask"]
                        ]
                        - change_index
                    )
                )
            ),
        },
    )

    # Fractal paper's single-image/magnitude domain is represented as a post-regularizer.
    spike_initial = graph.reshape(size, size) + rng.normal(0.0, 1.2, (size, size))
    spike_mask = rng.random((size, size)) < 0.035
    spike_initial[spike_mask] += rng.choice([-1.0, 1.0], np.sum(spike_mask)) * rng.uniform(8.0, 16.0, np.sum(spike_mask))
    magnitude_proxy = (0.2 + np.abs(np.gradient(truth, axis=1))) ** 4
    start = time.perf_counter()
    fractal = published.fractal_surface_regularize_2015_adapted(
        spike_initial, magnitude_proxy
    )
    runtime = time.perf_counter() - start
    maps["fractal_initial"] = spike_initial
    maps["fractal_estimate"] = fractal.dem_error
    _append_result(
        rows,
        "fractal_spikes",
        "initial_spike_map",
        "input",
        truth,
        spike_initial,
        0.0,
    )
    _append_result(
        rows,
        "fractal_spikes",
        "fractal_2015_adapted",
        "adapted",
        truth,
        fractal.dem_error,
        runtime,
        extra=fractal.diagnostics,
    )
    return rows, maps


def _plot_maps(maps: dict[str, np.ndarray], output_dir: Path) -> None:
    figure, axes = plt.subplots(4, 4, figsize=(13.2, 12.0), constrained_layout=True)
    limits = {
        "static": max(float(np.nanpercentile(np.abs(maps["static_truth"]), 98)), 1e-6),
        "sparse": max(float(np.nanmax(np.abs(maps["sparse_truth"]))), 1e-6),
        "dynamic": max(float(np.nanmax(np.abs(maps["dynamic_truth"]))), 1e-6),
        "wrapped": max(float(0.75 * np.nanpercentile(np.abs(maps["static_truth"]), 98)), 1e-6),
        "fractal": max(
            float(
                np.nanpercentile(
                    np.abs(
                        np.concatenate(
                            (
                                maps["static_truth"].reshape(-1),
                                maps["fractal_initial"].reshape(-1),
                                maps["fractal_estimate"].reshape(-1),
                            )
                        )
                    ),
                    98,
                )
            ),
            1e-6,
        ),
    }
    panels = [
        ("static_truth", "Static truth", "RdBu_r", "static"),
        ("static_hybrid", "Hybrid optimal 2026", "RdBu_r", "static"),
        ("static_adaptive", "Adaptive HT 2021", "RdBu_r", "static"),
        ("static_ica", "ICA 2019", "RdBu_r", "static"),
        ("sparse_truth", "Sparse truth", "RdBu_r", "sparse"),
        ("sparse_gdc", "GDC detection score", "viridis", "gdc"),
        ("sparse_pgdc", "PGDC 2025", "RdBu_r", "sparse"),
        ("sparse_hybrid", "Hybrid optimal 2026", "RdBu_r", "sparse"),
        ("dynamic_truth", "Height-change truth", "RdBu_r", "dynamic"),
        ("dynamic_estimate", "Dynamic SBAS 2025", "RdBu_r", "dynamic"),
        ("dynamic_hybrid", "Hybrid optimal 2026", "RdBu_r", "dynamic"),
        ("wrapped_igs", "IGS equivalent DEM", "RdBu_r", "wrapped"),
        ("static_truth", "Fractal target", "RdBu_r", "fractal"),
        ("fractal_initial", "Unregularized spikes", "RdBu_r", "fractal"),
        ("fractal_estimate", "Fractal adapted 2015", "RdBu_r", "fractal"),
        ("static_graph", "Current estimate before spikes", "RdBu_r", "fractal"),
    ]
    for axis, (key, title, cmap, group) in zip(axes.flat, panels):
        values = maps[key]
        if cmap == "RdBu_r":
            limit = limits[group]
            image = axis.imshow(values, cmap=cmap, vmin=-limit, vmax=limit)
        else:
            image = axis.imshow(values, cmap=cmap, vmin=0.0, vmax=1.0)
        axis.set_title(title, fontsize=10)
        axis.set_xticks([])
        axis.set_yticks([])
        figure.colorbar(image, ax=axis, shrink=0.72, pad=0.02)
    figure.suptitle("Scenario-matched DEM-error method comparison", fontsize=15)
    figure.savefig(output_dir / "published_method_comparison.png", dpi=220)
    figure.savefig(output_dir / "published_method_comparison.pdf")
    plt.close(figure)


def _plot_rmse(rows: list[dict], output_dir: Path) -> None:
    figure, axis = plt.subplots(figsize=(11.5, 6.5), constrained_layout=True)
    ordered = sorted(rows, key=lambda row: (row["scenario"], row["rmse_m"]))
    labels = [f"{row['scenario']} | {row['label']}" for row in ordered]
    values = [row["rmse_m"] for row in ordered]
    colors = [
        "#167D8D" if row["reproduction_status"] in {"strict", "current"} else "#D97841"
        for row in ordered
    ]
    positions = np.arange(len(ordered))
    axis.barh(positions, values, color=colors)
    axis.set_yticks(positions, labels)
    axis.invert_yaxis()
    axis.set_xlabel("DEM / height-change RMSE (m)")
    axis.grid(axis="x", color="#D9D9D9", linewidth=0.7)
    axis.set_axisbelow(True)
    axis.set_title("RMSE is comparable only within the same scenario")
    figure.savefig(output_dir / "published_method_rmse.png", dpi=220)
    figure.savefig(output_dir / "published_method_rmse.pdf")
    plt.close(figure)


def main(argv: list[str] | None = None) -> int:
    args = create_parser().parse_args(argv)
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output directory is not empty, use --overwrite: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, maps = run_benchmark(args.size, args.seed)
    with (output_dir / "published_method_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(output_dir / "published_benchmark_maps.npz", **maps)
    (output_dir / "benchmark_config.json").write_text(
        json.dumps({"size": args.size, "seed": args.seed}, indent=2), encoding="utf-8"
    )
    _plot_maps(maps, output_dir)
    _plot_rmse(rows, output_dir)
    print(f"wrote {len(rows)} benchmark rows to {output_dir}")
    for row in rows:
        print(
            f"  {row['scenario']:18s} {row['method']:30s} "
            f"RMSE={row['rmse_m']:.3f} m runtime={row['runtime_s']:.3f} s"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
