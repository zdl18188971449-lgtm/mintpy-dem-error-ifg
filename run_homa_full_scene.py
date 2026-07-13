#!/usr/bin/env python3
"""Apply a validation-driven HOMA-DEM profile to a full MintPy scene."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import mintpy_dem_error_ifg as current
import published_dem_error_models as published


_WORKER: dict = {}


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run tiled HOMA-DEM correction on a full MintPy stack.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("ifgram_stack")
    parser.add_argument("-g", "--geometry", required=True)
    parser.add_argument("--mask")
    parser.add_argument("--mask-dataset")
    parser.add_argument("--mintpy-dem", help="Existing MintPy demErr.h5")
    parser.add_argument("--timeseries-before", help="Existing timeseries before DEM correction")
    parser.add_argument(
        "--timeseries-mintpy-corrected",
        help="Existing MintPy DEM-corrected timeseries for comparison",
    )
    parser.add_argument("-o", "--output-dir", required=True)
    parser.add_argument("--block-rows", type=int, default=32)
    parser.add_argument("--halo-rows", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--min-coherence", type=float, default=0.5)
    parser.add_argument("--dem-bound", type=float, default=200.0)
    parser.add_argument("--velocity-bound", type=float, default=20.0)
    parser.add_argument("--graph-lambda", type=float, default=1.0)
    parser.add_argument("--graph-iterations", type=int, default=30)
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _init_worker(
    stack_path: str,
    geometry_path: str,
    mask_path: str | None,
    mask_dataset: str | None,
    config: dict,
) -> None:
    stack = h5py.File(stack_path, "r")
    geometry = h5py.File(geometry_path, "r")
    mask = h5py.File(mask_path, "r") if mask_path else None
    pairs = current.read_date_pairs(stack["date"])
    selected = (
        np.asarray(stack["dropIfgram"], dtype=bool)
        if "dropIfgram" in stack
        else np.ones(len(pairs), dtype=bool)
    )
    selected_pairs = [pair for pair, keep in zip(pairs, selected) if keep]
    baseline_mode, baseline_source = current.inspect_baseline_mode(
        stack, geometry, pairs
    )
    selected_mask_dataset = (
        current.choose_mask_dataset(mask, mask_dataset) if mask else None
    )
    _WORKER.update(
        {
            "stack": stack,
            "geometry": geometry,
            "mask": mask,
            "mask_dataset": selected_mask_dataset,
            "pairs": pairs,
            "selected": selected,
            "selected_pairs": selected_pairs,
            "baseline_mode": baseline_mode,
            "baseline_source": baseline_source,
            "config": config,
        }
    )


def _read_coefficient(
    stack: h5py.File,
    geometry: h5py.File,
    pairs: list[tuple[str, str]],
    baseline_mode: str,
    baseline_source,
    row_slice: slice,
    width: int,
) -> tuple[np.ndarray, np.ndarray]:
    bperp = current.read_bperp_block(
        baseline_mode,
        baseline_source,
        geometry,
        pairs,
        row_slice,
        width,
    )
    incidence = np.asarray(
        geometry["incidenceAngle"][row_slice, :], dtype=np.float64
    ).reshape(-1)
    slant_range = np.asarray(
        geometry["slantRangeDistance"][row_slice, :], dtype=np.float64
    ).reshape(-1)
    sine = np.sin(np.deg2rad(incidence))
    inverse_geometry = np.divide(
        1.0,
        slant_range * sine,
        out=np.full_like(slant_range, np.nan),
        where=(slant_range > 0) & np.isfinite(sine) & (sine != 0),
    )
    coefficient = (
        -4.0 * np.pi / float(stack.attrs["WAVELENGTH"])
    ) * bperp * inverse_geometry[None, :]
    return coefficient, inverse_geometry


def _process_tile(task: tuple[int, int, int, int]) -> dict:
    core_start, core_stop, read_start, read_stop = task
    stack = _WORKER["stack"]
    geometry = _WORKER["geometry"]
    config = _WORKER["config"]
    selected = _WORKER["selected"]
    width = int(stack["unwrapPhase"].shape[2])
    row_slice = slice(read_start, read_stop)
    phase = np.asarray(
        stack["unwrapPhase"][selected, row_slice, :], dtype=np.float64
    )
    coherence = (
        np.asarray(stack["coherence"][selected, row_slice, :], dtype=np.float64)
        if "coherence" in stack
        else np.ones_like(phase)
    )
    coefficient, inverse_geometry = _read_coefficient(
        stack,
        geometry,
        _WORKER["pairs"],
        _WORKER["baseline_mode"],
        _WORKER["baseline_source"],
        row_slice,
        width,
    )
    coefficient = coefficient[selected].reshape(phase.shape)
    terrain = np.asarray(geometry["height"][row_slice, :], dtype=np.float64)
    valid = np.isfinite(inverse_geometry).reshape(terrain.shape)
    if _WORKER["mask"] is not None:
        valid &= np.asarray(
            _WORKER["mask"][_WORKER["mask_dataset"]][row_slice, :],
            dtype=bool,
        )
    coherence[:, ~valid] = 0.0
    started = time.perf_counter()
    result = published.hybrid_optimal_2026(
        phase,
        coefficient,
        _WORKER["selected_pairs"],
        coherence,
        terrain=terrain,
        min_coherence=config["min_coherence"],
        graph_lambda=config["graph_lambda"],
        graph_iterations=config["graph_iterations"],
        velocity_bounds=(-config["velocity_bound"], config["velocity_bound"]),
        dem_bounds=(-config["dem_bound"], config["dem_bound"]),
        enable_ica=False,
        enable_dynamic=False,
        enable_igs=False,
    )
    local_start = core_start - read_start
    local_stop = local_start + (core_stop - core_start)
    crop = np.s_[local_start:local_stop, :]
    diagnostics = result.diagnostics
    return {
        "core_start": core_start,
        "core_stop": core_stop,
        "runtime_s": time.perf_counter() - started,
        "dem_error": np.asarray(result.dem_error[crop], dtype=np.float32),
        "valid": np.asarray(result.valid[crop], dtype=np.uint8),
        "dem_error_std": np.asarray(
            diagnostics["dem_error_std"][crop], dtype=np.float32
        ),
        "observability": np.asarray(
            diagnostics["dem_observability"][crop], dtype=np.float32
        ),
        "source_model": np.asarray(
            diagnostics["source_model"][crop], dtype=np.uint8
        ),
        "active_mask": np.asarray(
            diagnostics["active_mask"][crop], dtype=np.uint8
        ),
        "graph_blend": np.asarray(
            diagnostics["graph_blend"][crop], dtype=np.float32
        ),
        "gdc": np.asarray(diagnostics["gdc"][crop], dtype=np.float32),
        "pgdc_detected": np.asarray(
            diagnostics["pgdc_detected"][crop], dtype=np.uint8
        ),
        "model_order": np.asarray(
            diagnostics["deformation_model_order"][crop], dtype=np.uint8
        ),
        "source_names": diagnostics["source_names"],
    }


def _create_component_file(
    path: Path,
    stack_path: Path,
    shape: tuple[int, int],
    block_rows: int,
    config: dict,
) -> h5py.File:
    length, width = shape
    chunks = (min(block_rows, length), min(512, width))
    with h5py.File(stack_path, "r") as stack:
        attributes = dict(stack.attrs)
    file = h5py.File(path, "w")
    for key, value in attributes.items():
        file.attrs[key] = value
    file.attrs["FILE_TYPE"] = "homaDemErrorComponent"
    file.attrs["UNIT"] = "m"
    file.attrs["HOMA_PROFILE"] = "full_scene_static_no_ica_no_igs"
    file.attrs["HOMA_CONFIG"] = json.dumps(config)
    specifications = {
        "demError": (np.float32, np.nan),
        "demErrorStd": (np.float32, np.nan),
        "demObservability": (np.float32, 0.0),
        "sourceModel": (np.uint8, 0),
        "validMask": (np.uint8, 0),
        "activeMask": (np.uint8, 0),
        "graphBlend": (np.float32, 0.0),
        "gdc": (np.float32, 0.0),
        "pgdcDetected": (np.uint8, 0),
        "deformationModelOrder": (np.uint8, 0),
    }
    for name, (dtype, fill) in specifications.items():
        file.create_dataset(
            name,
            shape=(length, width),
            dtype=dtype,
            chunks=chunks,
            compression="gzip",
            fillvalue=fill,
        )
    return file


def _estimate_and_apply_dem_scale(
    stack_path: Path,
    geometry_path: Path,
    mask_path: Path | None,
    mask_dataset: str | None,
    component_path: Path,
    min_coherence: float,
    block_rows: int,
) -> dict:
    mask_handle = h5py.File(mask_path, "r") if mask_path else None
    try:
        with h5py.File(stack_path, "r") as stack, h5py.File(
            geometry_path, "r"
        ) as geometry, h5py.File(component_path, "r+") as component:
            pairs = current.read_date_pairs(stack["date"])
            mode, source = current.inspect_baseline_mode(stack, geometry, pairs)
            num_ifgram, length, width = stack["unwrapPhase"].shape
            numerator = np.zeros(num_ifgram, dtype=np.float64)
            denominator = np.zeros(num_ifgram, dtype=np.float64)
            before_sum = np.zeros(num_ifgram, dtype=np.float64)
            weight_sum = np.zeros(num_ifgram, dtype=np.float64)
            selected_mask_dataset = (
                current.choose_mask_dataset(mask_handle, mask_dataset)
                if mask_handle
                else None
            )
            source_dataset = (
                component["demErrorRaw"]
                if "demErrorRaw" in component
                else component["demError"]
            )
            for start in range(0, length, block_rows):
                stop = min(start + block_rows, length)
                row_slice = slice(start, stop)
                coefficient, inverse_geometry = _read_coefficient(
                    stack, geometry, pairs, mode, source, row_slice, width
                )
                phase = np.asarray(
                    stack["unwrapPhase"][:, row_slice, :], dtype=np.float64
                ).reshape(num_ifgram, -1)
                coherence = (
                    np.asarray(
                        stack["coherence"][:, row_slice, :], dtype=np.float64
                    ).reshape(num_ifgram, -1)
                    if "coherence" in stack
                    else np.ones_like(phase)
                )
                dem_error = np.asarray(
                    source_dataset[row_slice, :], dtype=np.float64
                ).reshape(-1)
                spatial_mask = np.isfinite(inverse_geometry) & np.isfinite(dem_error)
                if mask_handle is not None:
                    spatial_mask &= np.asarray(
                        mask_handle[selected_mask_dataset][row_slice, :], dtype=bool
                    ).reshape(-1)
                dem_phase = coefficient * np.nan_to_num(dem_error, nan=0.0)[None, :]
                valid = (
                    np.isfinite(phase)
                    & np.isfinite(dem_phase)
                    & spatial_mask[None, :]
                    & (coherence >= min_coherence)
                )
                weights = np.where(valid, coherence**2, 0.0)
                numerator += np.sum(weights * phase * dem_phase, axis=1)
                denominator += np.sum(weights * dem_phase**2, axis=1)
                before_sum += np.sum(weights * phase**2, axis=1)
                weight_sum += np.sum(weights, axis=1)
            total_denominator = float(np.sum(denominator))
            raw_scale = (
                float(np.sum(numerator) / total_denominator)
                if total_denominator > 0
                else 0.0
            )
            scale = float(np.clip(raw_scale, 0.0, 1.0))
            raw_dataset = (
                component["demErrorRaw"]
                if "demErrorRaw" in component
                else component.create_dataset(
                    "demErrorRaw",
                    shape=(length, width),
                    dtype=np.float32,
                    chunks=component["demError"].chunks,
                    compression="gzip",
                    fillvalue=np.nan,
                )
            )
            for start in range(0, length, block_rows):
                stop = min(start + block_rows, length)
                raw = np.asarray(
                    source_dataset[start:stop, :], dtype=np.float32
                )
                if source_dataset.name != raw_dataset.name:
                    raw_dataset[start:stop, :] = raw
                component["demError"][start:stop, :] = (scale * raw).astype(
                    np.float32
                )
            component.attrs["DEM_ERROR_GLOBAL_SCALE"] = scale
            component.attrs["DEM_ERROR_RAW_SCALE"] = raw_scale
            before_global = float(
                np.sqrt(np.sum(before_sum) / max(np.sum(weight_sum), 1.0))
            )
            after_global = float(
                np.sqrt(
                    np.sum(
                        before_sum
                        - 2.0 * scale * numerator
                        + scale**2 * denominator
                    )
                    / max(np.sum(weight_sum), 1.0)
                )
            )
            per_ifgram_scale = np.divide(
                numerator,
                denominator,
                out=np.zeros_like(numerator),
                where=denominator > 0,
            )
            return {
                "raw_global_scale": raw_scale,
                "applied_global_scale": scale,
                "weighted_rms_before_rad": before_global,
                "weighted_rms_after_rad": after_global,
                "weighted_rms_ratio": after_global / max(before_global, 1e-15),
                "unclipped_per_ifgram_scale": per_ifgram_scale.tolist(),
            }
    finally:
        if mask_handle is not None:
            mask_handle.close()


def _write_corrected_stack(
    input_path: Path,
    geometry_path: Path,
    component_path: Path,
    output_path: Path,
    block_rows: int,
    min_coherence: float,
) -> list[dict]:
    shutil.copy2(input_path, output_path)
    metrics = []
    with h5py.File(input_path, "r") as stack, h5py.File(
        geometry_path, "r"
    ) as geometry, h5py.File(component_path, "r") as component, h5py.File(
        output_path, "r+"
    ) as corrected:
        pairs = current.read_date_pairs(stack["date"])
        mode, source = current.inspect_baseline_mode(stack, geometry, pairs)
        num_ifgram, length, width = stack["unwrapPhase"].shape
        original_sum = np.zeros(num_ifgram, dtype=np.float64)
        corrected_sum = np.zeros(num_ifgram, dtype=np.float64)
        count = np.zeros(num_ifgram, dtype=np.int64)
        weighted_original_sum = np.zeros(num_ifgram, dtype=np.float64)
        weighted_corrected_sum = np.zeros(num_ifgram, dtype=np.float64)
        weight_sum = np.zeros(num_ifgram, dtype=np.float64)
        for start in range(0, length, block_rows):
            stop = min(start + block_rows, length)
            row_slice = slice(start, stop)
            coefficient, _ = _read_coefficient(
                stack, geometry, pairs, mode, source, row_slice, width
            )
            dem_error = np.asarray(component["demError"][row_slice, :], dtype=np.float64).reshape(-1)
            phase = np.asarray(stack["unwrapPhase"][:, row_slice, :], dtype=np.float64)
            dem_phase = coefficient * np.nan_to_num(dem_error, nan=0.0)[None, :]
            corrected_phase = phase.reshape(num_ifgram, -1) - dem_phase
            valid = np.isfinite(phase.reshape(num_ifgram, -1)) & np.isfinite(dem_error)[None, :]
            coherence = (
                np.asarray(stack["coherence"][:, row_slice, :], dtype=np.float64).reshape(num_ifgram, -1)
                if "coherence" in stack
                else np.ones_like(corrected_phase)
            )
            weights = np.where(valid & (coherence >= min_coherence), coherence**2, 0.0)
            original_sum += np.sum(np.where(valid, phase.reshape(num_ifgram, -1) ** 2, 0.0), axis=1)
            corrected_sum += np.sum(np.where(valid, corrected_phase**2, 0.0), axis=1)
            count += np.sum(valid, axis=1)
            weighted_original_sum += np.sum(
                weights * phase.reshape(num_ifgram, -1) ** 2, axis=1
            )
            weighted_corrected_sum += np.sum(weights * corrected_phase**2, axis=1)
            weight_sum += np.sum(weights, axis=1)
            corrected["unwrapPhase"][:, row_slice, :] = corrected_phase.reshape(
                num_ifgram, stop - start, width
            ).astype(corrected["unwrapPhase"].dtype)
        corrected.attrs["DEM_ERROR_CORRECTED"] = "yes"
        corrected.attrs["DEM_ERROR_MODEL"] = "HOMA_full_scene_static"
        dates = current.read_date_pairs(stack["date"])
        for index, pair in enumerate(dates):
            metrics.append(
                {
                    "ifgram_index": index,
                    "date12": f"{pair[0]}_{pair[1]}",
                    "original_rms_rad": float(np.sqrt(original_sum[index] / max(count[index], 1))),
                    "homa_corrected_rms_rad": float(np.sqrt(corrected_sum[index] / max(count[index], 1))),
                    "weighted_original_rms_rad": float(
                        np.sqrt(weighted_original_sum[index] / max(weight_sum[index], 1e-15))
                    ),
                    "weighted_homa_corrected_rms_rad": float(
                        np.sqrt(weighted_corrected_sum[index] / max(weight_sum[index], 1e-15))
                    ),
                    "valid_pixels": int(count[index]),
                }
            )
    return metrics


def _velocity_from_timeseries(values: np.ndarray, dates: np.ndarray) -> np.ndarray:
    decoded = [
        value.decode() if isinstance(value, bytes) else str(value) for value in dates
    ]
    date_values = np.asarray(
        [np.datetime64(f"{date[:4]}-{date[4:6]}-{date[6:8]}") for date in decoded]
    )
    years = (date_values - date_values[0]).astype("timedelta64[D]").astype(float) / 365.25
    centered = years - np.mean(years)
    return np.tensordot(centered, values, axes=(0, 0)) / np.sum(centered**2)


def _write_timeseries_products(
    before_path: Path,
    mintpy_corrected_path: Path | None,
    geometry_path: Path,
    component_path: Path,
    output_path: Path,
    comparison_path: Path,
    mintpy_dem_path: Path | None,
    block_rows: int,
) -> dict:
    shutil.copy2(before_path, output_path)
    with h5py.File(before_path, "r") as before, h5py.File(
        output_path, "r+"
    ) as homa, h5py.File(geometry_path, "r") as geometry, h5py.File(
        component_path, "r"
    ) as component:
        length, width = component["demError"].shape
        bperp = np.asarray(before["bperp"], dtype=np.float64)
        dates = np.asarray(before["date"])
        comparison = h5py.File(comparison_path, "w")
        for key, value in before.attrs.items():
            comparison.attrs[key] = value
        comparison.attrs["FILE_TYPE"] = "homaFullSceneComparison"
        chunks = (min(block_rows, length), min(512, width))
        datasets = {
            name: comparison.create_dataset(
                name,
                shape=(length, width),
                dtype=np.float32,
                chunks=chunks,
                compression="gzip",
            )
            for name in (
                "velocityBefore",
                "velocityHoma",
                "velocityMintPy",
                "velocityHomaMinusMintPy",
                "demHoma",
                "demMintPy",
                "demHomaMinusMintPy",
            )
        }
        mintpy_handle = h5py.File(mintpy_corrected_path, "r") if mintpy_corrected_path else None
        mintpy_dem_handle = h5py.File(mintpy_dem_path, "r") if mintpy_dem_path else None
        velocity_difference_sum = 0.0
        velocity_difference_count = 0
        dem_difference_values = []
        try:
            for start in range(0, length, block_rows):
                stop = min(start + block_rows, length)
                row_slice = slice(start, stop)
                dem = np.asarray(component["demError"][row_slice, :], dtype=np.float64)
                incidence = np.asarray(geometry["incidenceAngle"][row_slice, :], dtype=np.float64)
                slant_range = np.asarray(geometry["slantRangeDistance"][row_slice, :], dtype=np.float64)
                inverse = 1.0 / (slant_range * np.sin(np.deg2rad(incidence)))
                input_values = np.asarray(before["timeseries"][:, row_slice, :], dtype=np.float64)
                displacement = bperp[:, None, None] * inverse[None, :, :] * np.nan_to_num(dem, nan=0.0)[None, :, :]
                homa_values = input_values - displacement
                homa["timeseries"][:, row_slice, :] = homa_values.astype(homa["timeseries"].dtype)
                velocity_before = _velocity_from_timeseries(input_values, dates)
                velocity_homa = _velocity_from_timeseries(homa_values, dates)
                if mintpy_handle is not None:
                    mintpy_values = np.asarray(
                        mintpy_handle["timeseries"][:, row_slice, :], dtype=np.float64
                    )
                    velocity_mintpy = _velocity_from_timeseries(mintpy_values, dates)
                else:
                    velocity_mintpy = np.full_like(velocity_homa, np.nan)
                if mintpy_dem_handle is not None:
                    mintpy_dem = np.asarray(
                        mintpy_dem_handle[next(iter(mintpy_dem_handle.keys()))][row_slice, :],
                        dtype=np.float64,
                    )
                    valid_dem = np.isfinite(dem) & np.isfinite(mintpy_dem)
                    if np.any(valid_dem):
                        dem_difference_values.append((dem[valid_dem] - mintpy_dem[valid_dem]).astype(np.float32))
                else:
                    mintpy_dem = np.full_like(dem, np.nan)
                datasets["velocityBefore"][row_slice, :] = velocity_before.astype(np.float32)
                datasets["velocityHoma"][row_slice, :] = velocity_homa.astype(np.float32)
                datasets["velocityMintPy"][row_slice, :] = velocity_mintpy.astype(np.float32)
                datasets["velocityHomaMinusMintPy"][row_slice, :] = (velocity_homa - velocity_mintpy).astype(np.float32)
                datasets["demHoma"][row_slice, :] = dem.astype(np.float32)
                datasets["demMintPy"][row_slice, :] = mintpy_dem.astype(np.float32)
                datasets["demHomaMinusMintPy"][row_slice, :] = (dem - mintpy_dem).astype(np.float32)
                valid_velocity = np.isfinite(velocity_homa) & np.isfinite(velocity_mintpy)
                velocity_difference_sum += float(
                    np.sum((velocity_homa[valid_velocity] - velocity_mintpy[valid_velocity]) ** 2)
                )
                velocity_difference_count += int(np.sum(valid_velocity))
            homa.attrs["DEM_ERROR_CORRECTED"] = "yes"
            homa.attrs["DEM_ERROR_MODEL"] = "HOMA_full_scene_static"
        finally:
            if mintpy_handle is not None:
                mintpy_handle.close()
            if mintpy_dem_handle is not None:
                mintpy_dem_handle.close()
            comparison.close()
    if dem_difference_values:
        differences = np.concatenate(dem_difference_values)
        dem_offset = float(np.median(differences))
        centered = differences - dem_offset
        dem_rmse = float(np.sqrt(np.mean(centered**2)))
        dem_mae = float(np.mean(np.abs(centered)))
    else:
        dem_offset = dem_rmse = dem_mae = np.nan
    return {
        "dem_homa_minus_mintpy_median_m": dem_offset,
        "dem_homa_mintpy_centered_rmse_m": dem_rmse,
        "dem_homa_mintpy_centered_mae_m": dem_mae,
        "velocity_homa_mintpy_rmse_m_per_year": float(
            np.sqrt(velocity_difference_sum / max(velocity_difference_count, 1))
        ),
        "velocity_comparison_pixels": velocity_difference_count,
    }


def _configure_plot_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 7.0,
            "axes.titlesize": 7.2,
            "axes.labelsize": 7.0,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def _save_figure(figure: plt.Figure, base: Path, dpi: int) -> None:
    figure.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    figure.savefig(base.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    figure.savefig(
        base.with_suffix(".tiff"),
        dpi=dpi,
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(figure)


def _robust_limit(values: np.ndarray, percentile: float = 98.0) -> float:
    finite = np.abs(values[np.isfinite(values)])
    return float(np.percentile(finite, percentile)) if finite.size else 1.0


def _plot_comparisons(
    stack_path: Path,
    geometry_path: Path,
    component_path: Path,
    comparison_path: Path | None,
    mintpy_dem_path: Path | None,
    corrected_stack_path: Path,
    output_dir: Path,
    dpi: int,
) -> None:
    _configure_plot_style()
    with h5py.File(stack_path, "r") as stack:
        _, length, width = stack["unwrapPhase"].shape
        selected = (
            np.asarray(stack["dropIfgram"], dtype=bool)
            if "dropIfgram" in stack
            else np.ones(stack["unwrapPhase"].shape[0], dtype=bool)
        )
        coherence_means = np.asarray(
            [np.nanmean(stack["coherence"][index]) if selected[index] else -np.inf for index in range(len(selected))]
        )
        best_index = int(np.argmax(coherence_means))
        pairs = current.read_date_pairs(stack["date"])
    stride = max(1, int(np.ceil(max(length, width) / 1000)))
    sample = np.s_[::stride, ::stride]
    with h5py.File(component_path, "r") as component:
        homa_dem = np.asarray(component["demError"][sample], dtype=np.float64)
        observability = np.asarray(component["demObservability"][sample], dtype=np.float64)
    if mintpy_dem_path:
        with h5py.File(mintpy_dem_path, "r") as mintpy_dem:
            mintpy_name = next(iter(mintpy_dem.keys()))
            mintpy_values = np.asarray(mintpy_dem[mintpy_name][sample], dtype=np.float64)
    else:
        mintpy_values = np.full_like(homa_dem, np.nan)
    valid = np.isfinite(homa_dem) & np.isfinite(mintpy_values)
    offset = float(np.median(homa_dem[valid] - mintpy_values[valid])) if np.any(valid) else 0.0
    dem_difference = homa_dem - mintpy_values - offset
    if comparison_path:
        with h5py.File(comparison_path, "r") as comparison:
            velocity_before = np.asarray(comparison["velocityBefore"][sample], dtype=np.float64) * 1000.0
            velocity_mintpy = np.asarray(comparison["velocityMintPy"][sample], dtype=np.float64) * 1000.0
            velocity_homa = np.asarray(comparison["velocityHoma"][sample], dtype=np.float64) * 1000.0
            velocity_difference = velocity_homa - velocity_mintpy
    else:
        velocity_before = velocity_mintpy = velocity_homa = velocity_difference = np.full_like(homa_dem, np.nan)
    dem_limit = max(_robust_limit(mintpy_values), _robust_limit(homa_dem))
    difference_limit = _robust_limit(dem_difference)
    velocity_limit = max(_robust_limit(velocity_before), _robust_limit(velocity_homa))
    velocity_difference_limit = _robust_limit(velocity_difference)
    figure, axes = plt.subplots(2, 4, figsize=(7.16, 4.05))
    panels = [
        (mintpy_values, "MintPy DEM error", "RdBu_r", -dem_limit, dem_limit, "m"),
        (homa_dem, "Applied HOMA-DEM error", "RdBu_r", -dem_limit, dem_limit, "m"),
        (dem_difference, "HOMA - MintPy (offset removed)", "RdBu_r", -difference_limit, difference_limit, "m"),
        (observability, "HOMA observability", "viridis", 0, _robust_limit(observability), "normalized"),
        (velocity_before, "Velocity before DEM correction", "RdBu_r", -velocity_limit, velocity_limit, "mm/yr"),
        (velocity_mintpy, "MintPy-corrected velocity", "RdBu_r", -velocity_limit, velocity_limit, "mm/yr"),
        (velocity_homa, "HOMA-corrected velocity", "RdBu_r", -velocity_limit, velocity_limit, "mm/yr"),
        (velocity_difference, "HOMA - MintPy velocity", "RdBu_r", -velocity_difference_limit, velocity_difference_limit, "mm/yr"),
    ]
    for index, (axis, panel) in enumerate(zip(axes.flat, panels)):
        values, title, cmap, vmin, vmax, unit = panel
        image = axis.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest", rasterized=True)
        axis.set_title(title)
        axis.set_xticks([])
        axis.set_yticks([])
        axis.text(0.02, 0.98, f"({chr(ord('a') + index)})", transform=axis.transAxes, va="top", fontweight="bold", bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 0.7})
        colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.025)
        colorbar.ax.tick_params(labelsize=5.5)
        colorbar.set_label(unit, fontsize=6.0)
    figure.tight_layout()
    _save_figure(figure, output_dir / "homa_full_scene_dem_velocity_comparison", dpi)

    with h5py.File(stack_path, "r") as stack, h5py.File(corrected_stack_path, "r") as corrected, h5py.File(geometry_path, "r") as geometry:
        original_phase = np.asarray(stack["unwrapPhase"][best_index, sample[0], sample[1]], dtype=np.float64)
        homa_corrected = np.asarray(corrected["unwrapPhase"][best_index, sample[0], sample[1]], dtype=np.float64)
        incidence = np.asarray(geometry["incidenceAngle"][sample], dtype=np.float64)
        slant_range = np.asarray(geometry["slantRangeDistance"][sample], dtype=np.float64)
        bperp = float(stack["bperp"][best_index])
        coefficient = (-4.0 * np.pi / float(stack.attrs["WAVELENGTH"])) * bperp / (slant_range * np.sin(np.deg2rad(incidence)))
        mintpy_corrected = original_phase - coefficient * np.nan_to_num(mintpy_values, nan=0.0)
    phase_limit = max(_robust_limit(original_phase), _robust_limit(homa_corrected))
    correction_limit = _robust_limit(original_phase - homa_corrected)
    phase_panels = [
        (original_phase, "Before DEM correction", -phase_limit, phase_limit),
        (mintpy_corrected, "Existing MintPy correction", -phase_limit, phase_limit),
        (homa_corrected, "HOMA-DEM correction", -phase_limit, phase_limit),
        (homa_corrected - mintpy_corrected, "HOMA - MintPy corrected phase", -correction_limit, correction_limit),
    ]
    figure, axes = plt.subplots(1, 4, figsize=(7.16, 2.05))
    for index, (axis, (values, title, vmin, vmax)) in enumerate(zip(axes, phase_panels)):
        image = axis.imshow(values, cmap="RdBu_r", vmin=vmin, vmax=vmax, interpolation="nearest", rasterized=True)
        axis.set_title(title)
        axis.set_xticks([])
        axis.set_yticks([])
        axis.text(0.02, 0.98, f"({chr(ord('a') + index)})", transform=axis.transAxes, va="top", fontweight="bold", bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 0.7})
        colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.025)
        colorbar.set_label("rad", fontsize=6.0)
        colorbar.ax.tick_params(labelsize=5.5)
    figure.suptitle(
        f"Highest-coherence interferogram: {pairs[best_index][0]}-{pairs[best_index][1]}",
        fontsize=7.5,
        fontweight="bold",
    )
    figure.tight_layout()
    _save_figure(figure, output_dir / "homa_best_ifgram_before_after", dpi)


def _prepare_output_directory(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"output exists, use --overwrite: {path}")
        resolved = path.resolve()
        if resolved == Path(resolved.anchor) or len(resolved.parts) < 4:
            raise ValueError(f"refusing to remove unsafe output directory: {resolved}")
        shutil.rmtree(resolved)
    path.mkdir(parents=True)


def main(argv: list[str] | None = None) -> int:
    args = create_parser().parse_args(argv)
    if args.block_rows < 1 or args.halo_rows < 0 or args.workers < 1:
        raise ValueError("block rows/workers must be positive and halo rows nonnegative")
    stack_path = Path(args.ifgram_stack).resolve()
    geometry_path = Path(args.geometry).resolve()
    mask_path = Path(args.mask).resolve() if args.mask else None
    output_dir = Path(args.output_dir).resolve()
    _prepare_output_directory(output_dir, args.overwrite)
    config = {
        "min_coherence": args.min_coherence,
        "dem_bound": args.dem_bound,
        "velocity_bound": args.velocity_bound,
        "graph_lambda": args.graph_lambda,
        "graph_iterations": args.graph_iterations,
        "block_rows": args.block_rows,
        "halo_rows": args.halo_rows,
        "workers": args.workers,
        "enable_ica": False,
        "enable_dynamic": False,
        "enable_igs": False,
        "enable_pgdc": True,
        "enable_graph": True,
    }
    with h5py.File(stack_path, "r") as stack:
        _, length, width = stack["unwrapPhase"].shape
    component_path = output_dir / "demComponent_homa_full_scene.h5"
    corrected_stack_path = output_dir / "ifgramStack_homaDem.h5"
    component = _create_component_file(
        component_path, stack_path, (length, width), args.block_rows, config
    )
    tasks = []
    for start in range(0, length, args.block_rows):
        stop = min(start + args.block_rows, length)
        tasks.append(
            (
                start,
                stop,
                max(0, start - args.halo_rows),
                min(length, stop + args.halo_rows),
            )
        )
    tile_rows = []
    source_names = None
    started = time.perf_counter()
    try:
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=_init_worker,
            initargs=(
                str(stack_path),
                str(geometry_path),
                str(mask_path) if mask_path else None,
                args.mask_dataset,
                config,
            ),
        ) as executor:
            futures = {executor.submit(_process_tile, task): task for task in tasks}
            completed = 0
            for future in as_completed(futures):
                result = future.result()
                row_slice = slice(result["core_start"], result["core_stop"])
                component["demError"][row_slice, :] = result["dem_error"]
                component["demErrorStd"][row_slice, :] = result["dem_error_std"]
                component["demObservability"][row_slice, :] = result["observability"]
                component["sourceModel"][row_slice, :] = result["source_model"]
                component["validMask"][row_slice, :] = result["valid"]
                component["activeMask"][row_slice, :] = result["active_mask"]
                component["graphBlend"][row_slice, :] = result["graph_blend"]
                component["gdc"][row_slice, :] = result["gdc"]
                component["pgdcDetected"][row_slice, :] = result["pgdc_detected"]
                component["deformationModelOrder"][row_slice, :] = result["model_order"]
                source_names = result["source_names"]
                tile_rows.append(
                    {
                        "row_start": result["core_start"],
                        "row_stop": result["core_stop"],
                        "runtime_s": result["runtime_s"],
                        "valid_pixels": int(np.sum(result["valid"])),
                    }
                )
                completed += 1
                print(
                    f"tiles {completed}/{len(tasks)} rows "
                    f"{result['core_start']}:{result['core_stop']} "
                    f"{result['runtime_s']:.1f} s",
                    flush=True,
                )
        component.attrs["SOURCE_NAMES"] = json.dumps(source_names or [])
    finally:
        component.close()
    print("estimating global DEM correction scale", flush=True)
    scale_metrics = _estimate_and_apply_dem_scale(
        stack_path,
        geometry_path,
        mask_path,
        args.mask_dataset,
        component_path,
        args.min_coherence,
        args.block_rows,
    )
    with (output_dir / "tile_metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(tile_rows[0]))
        writer.writeheader()
        writer.writerows(sorted(tile_rows, key=lambda row: row["row_start"]))
    print("writing corrected interferogram stack", flush=True)
    ifgram_metrics = _write_corrected_stack(
        stack_path,
        geometry_path,
        component_path,
        corrected_stack_path,
        args.block_rows,
        args.min_coherence,
    )
    with (output_dir / "ifgram_before_after_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(ifgram_metrics[0]))
        writer.writeheader()
        writer.writerows(ifgram_metrics)
    comparison_path = None
    timeseries_metrics = {}
    if args.timeseries_before:
        print("writing HOMA-corrected timeseries", flush=True)
        comparison_path = output_dir / "homa_full_scene_comparison.h5"
        timeseries_metrics = _write_timeseries_products(
            Path(args.timeseries_before).resolve(),
            Path(args.timeseries_mintpy_corrected).resolve()
            if args.timeseries_mintpy_corrected
            else None,
            geometry_path,
            component_path,
            output_dir / "timeseries_tropHgt_homaDem.h5",
            comparison_path,
            Path(args.mintpy_dem).resolve() if args.mintpy_dem else None,
            args.block_rows,
        )
    summary = {
        "stack": str(stack_path),
        "geometry": str(geometry_path),
        "mask": str(mask_path) if mask_path else None,
        "shape": [length, width],
        "tiles": len(tasks),
        "processing_runtime_s": time.perf_counter() - started,
        "profile": config,
        "scale_metrics": scale_metrics,
        "timeseries_metrics": timeseries_metrics,
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("drawing comparison figures", flush=True)
    _plot_comparisons(
        stack_path,
        geometry_path,
        component_path,
        comparison_path,
        Path(args.mintpy_dem).resolve() if args.mintpy_dem else None,
        corrected_stack_path,
        output_dir,
        args.dpi,
    )
    print(f"full-scene outputs written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
