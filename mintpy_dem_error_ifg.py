#!/usr/bin/env python3
"""Estimate and remove residual DEM phase from a MintPy interferogram stack.

The implementation is inspired by vceSAR's direct MintPy HDF5 workflow, but
the inversion and file writing code here is independent.  It estimates one
DEM-error value per pixel from all selected interferograms, reconstructs the
DEM phase for every interferogram, and writes model-specific corrected stacks.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np


@dataclass(frozen=True)
class ModelSpec:
    name: str
    poly_order: int
    solver: str
    phase_velocity: bool = False


MODEL_SPECS = {
    "linear_ols": ModelSpec("linear_ols", 1, "ols"),
    "quadratic_ols": ModelSpec("quadratic_ols", 2, "ols"),
    "linear_velocity": ModelSpec("linear_velocity", 1, "ols", True),
    "linear_wls": ModelSpec("linear_wls", 1, "coherence"),
    "quadratic_wls": ModelSpec("quadratic_wls", 2, "coherence"),
    "linear_huber": ModelSpec("linear_huber", 1, "huber"),
    "quadratic_huber": ModelSpec("quadratic_huber", 2, "huber"),
    "linear_vce": ModelSpec("linear_vce", 1, "vce"),
    "quadratic_vce": ModelSpec("quadratic_vce", 2, "vce"),
}


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Correct residual DEM phase in each MintPy interferogram.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("ifgram_stack", help="MintPy inputs/ifgramStack.h5")
    parser.add_argument(
        "-g",
        "--geometry",
        required=True,
        help="MintPy geometryRadar.h5 with incidenceAngle and slantRangeDistance",
    )
    parser.add_argument(
        "-o", "--output-dir", default="dem_error_ifg", help="Output directory"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=sorted(MODEL_SPECS),
        default=["linear_ols", "quadratic_ols", "linear_wls", "linear_huber"],
        help="Calibration models to run",
    )
    parser.add_argument(
        "--periodic",
        nargs="*",
        type=float,
        default=[],
        metavar="YEARS",
        help="Optional periodic deformation terms appended to every model",
    )
    parser.add_argument(
        "--step-date",
        nargs="*",
        default=[],
        metavar="YYYYMMDD",
        help="Optional step functions appended to every model",
    )
    parser.add_argument("--mask", help="Optional MintPy mask HDF5 file")
    parser.add_argument("--mask-dataset", help="Dataset name in --mask")
    parser.add_argument(
        "--min-coherence",
        type=float,
        default=0.0,
        help="Ignore individual observations below this coherence",
    )
    parser.add_argument(
        "--variance-file",
        help="vceSAR-compatible variance HDF5 required by *_vce models",
    )
    parser.add_argument(
        "--variance-dataset",
        default="model_parameters",
        help="Variance dataset: (ifg,), (ifg,y,x), or model_parameters (ifg,>=3)",
    )
    parser.add_argument(
        "--use-all-ifgrams",
        action="store_true",
        help="Fit all interferograms instead of only dropIfgram=True entries",
    )
    parser.add_argument("--block-rows", type=int, default=8)
    parser.add_argument("--huber-delta", type=float, default=1.345)
    parser.add_argument("--huber-iterations", type=int, default=5)
    parser.add_argument("--rcond", type=float, default=1e-8)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print network/model diagnostics only",
    )
    return parser


def decode_text(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def read_date_pairs(dataset: h5py.Dataset) -> list[tuple[str, str]]:
    data = dataset[:]
    if data.ndim != 2 or data.shape[1] != 2:
        raise ValueError(f"date dataset must have shape (ifg, 2), got {data.shape}")
    return [(decode_text(row[0]), decode_text(row[1])) for row in data]


def parse_datetime(text: str) -> datetime:
    for fmt in ("%Y%m%d", "%Y%m%dT%H%M", "%Y%m%dT%H%M%S"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            pass
    raise ValueError(f"unsupported acquisition date: {text}")


def date_years(date_pairs: Iterable[tuple[str, str]]) -> dict[str, float]:
    dates = sorted({date for pair in date_pairs for date in pair})
    parsed = {date: parse_datetime(date) for date in dates}
    ref = parsed[dates[0]]
    return {date: (value - ref).total_seconds() / (365.25 * 86400.0) for date, value in parsed.items()}


def build_deformation_design(
    date_pairs: list[tuple[str, str]],
    poly_order: int,
    periods: list[float] | None = None,
    step_dates: list[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Return interferogram-domain deformation design matrix.

    The constant polynomial column cancels in an interferogram, so polynomial
    columns start at velocity (order 1).
    """
    periods = periods or []
    step_dates = step_dates or []
    if poly_order < 1:
        raise ValueError("poly_order must be at least 1")
    if any(period <= 0 for period in periods):
        raise ValueError("periods must be positive")

    years = date_years(date_pairs)
    columns: list[np.ndarray] = []
    labels: list[str] = []

    for order in range(1, poly_order + 1):
        values = [
            (years[slave] ** order - years[master] ** order) / math.factorial(order)
            for master, slave in date_pairs
        ]
        columns.append(np.asarray(values, dtype=np.float64))
        labels.append({1: "velocity", 2: "acceleration"}.get(order, f"poly_{order}"))

    for period in periods:
        omega = 2.0 * np.pi / period
        columns.append(
            np.asarray(
                [
                    np.cos(omega * years[slave]) - np.cos(omega * years[master])
                    for master, slave in date_pairs
                ],
                dtype=np.float64,
            )
        )
        columns.append(
            np.asarray(
                [
                    np.sin(omega * years[slave]) - np.sin(omega * years[master])
                    for master, slave in date_pairs
                ],
                dtype=np.float64,
            )
        )
        labels.extend([f"cos_{period:g}yr", f"sin_{period:g}yr"])

    for step_date in step_dates:
        step_time = parse_datetime(step_date)
        values = []
        for master, slave in date_pairs:
            master_flag = float(parse_datetime(master) > step_time)
            slave_flag = float(parse_datetime(slave) > step_time)
            values.append(slave_flag - master_flag)
        columns.append(np.asarray(values, dtype=np.float64))
        labels.append(f"step_{step_date}")

    return np.column_stack(columns), labels


def network_components(
    date_pairs: list[tuple[str, str]], selected: np.ndarray
) -> list[list[str]]:
    adjacency: dict[str, set[str]] = {}
    for keep, (master, slave) in zip(selected, date_pairs):
        if not keep:
            continue
        adjacency.setdefault(master, set()).add(slave)
        adjacency.setdefault(slave, set()).add(master)

    components: list[list[str]] = []
    unseen = set(adjacency)
    while unseen:
        start = unseen.pop()
        todo = [start]
        component = [start]
        while todo:
            node = todo.pop()
            for neighbor in adjacency[node]:
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    todo.append(neighbor)
                    component.append(neighbor)
        components.append(sorted(component))
    return sorted(components, key=lambda item: (-len(item), item))


def choose_mask_dataset(handle: h5py.File, requested: str | None) -> str:
    if requested:
        if requested not in handle:
            raise KeyError(f"mask dataset not found: {requested}")
        return requested
    if "mask" in handle:
        return "mask"
    for name, obj in handle.items():
        if isinstance(obj, h5py.Dataset) and obj.ndim == 2:
            return name
    raise ValueError("mask file contains no 2D dataset")


def dataset_create_kwargs(dataset: h5py.Dataset) -> dict:
    kwargs = {}
    if dataset.chunks is not None:
        kwargs["chunks"] = dataset.chunks
    if dataset.compression is not None:
        kwargs["compression"] = dataset.compression
        kwargs["compression_opts"] = dataset.compression_opts
    if dataset.shuffle:
        kwargs["shuffle"] = True
    if dataset.fletcher32:
        kwargs["fletcher32"] = True
    return kwargs


def prepare_output_files(
    input_file: h5py.File,
    output_dir: Path,
    spec: ModelSpec,
    phase_dataset: str,
    overwrite: bool,
) -> tuple[h5py.File, h5py.Dataset, h5py.File, h5py.Dataset, h5py.Dataset, Path, Path]:
    corrected_path = output_dir / f"ifgramStack_demErr_{spec.name}.h5"
    component_path = output_dir / f"demComponent_{spec.name}.h5"
    for path in (corrected_path, component_path):
        if path.exists():
            if not overwrite:
                raise FileExistsError(f"output exists, use --overwrite: {path}")
            path.unlink()

    corrected_file = h5py.File(corrected_path, "w")
    for key, value in input_file.attrs.items():
        corrected_file.attrs[key] = value
    corrected_file.attrs["DEM_ERROR_MODEL"] = spec.name
    corrected_file.attrs["DEM_ERROR_CORRECTED"] = "yes"
    for name in input_file:
        if name != phase_dataset:
            input_file.copy(name, corrected_file, name=name)
    phase_source = input_file[phase_dataset]
    corrected_phase = corrected_file.create_dataset(
        phase_dataset,
        shape=phase_source.shape,
        dtype=np.float32,
        **dataset_create_kwargs(phase_source),
    )
    for key, value in phase_source.attrs.items():
        corrected_phase.attrs[key] = value

    component_file = h5py.File(component_path, "w")
    for key, value in input_file.attrs.items():
        component_file.attrs[key] = value
    component_file.attrs["FILE_TYPE"] = "demErrorIfgramStack"
    component_file.attrs["UNIT"] = "radian"
    component_file.attrs["DEM_ERROR_MODEL"] = spec.name
    for name in ("date", "bperp", "dropIfgram"):
        if name in input_file:
            input_file.copy(name, component_file, name=name)
    dem_phase = component_file.create_dataset(
        "demPhase",
        shape=phase_source.shape,
        dtype=np.float32,
        **dataset_create_kwargs(phase_source),
    )
    dem_error = component_file.create_dataset(
        "demError",
        shape=phase_source.shape[1:],
        dtype=np.float32,
        chunks=True,
        compression="gzip",
        compression_opts=4,
        fillvalue=np.nan,
    )
    dem_error.attrs["UNIT"] = "m"
    return (
        corrected_file,
        corrected_phase,
        component_file,
        dem_phase,
        dem_error,
        corrected_path,
        component_path,
    )


def normalize_weights(weights: np.ndarray) -> np.ndarray:
    totals = np.sum(weights, axis=1, keepdims=True)
    counts = np.sum(weights > 0, axis=1, keepdims=True)
    means = np.divide(totals, counts, out=np.ones_like(totals), where=counts > 0)
    return np.divide(weights, means, out=np.zeros_like(weights), where=means > 0)


def solve_weighted(
    design: np.ndarray,
    observations: np.ndarray,
    weights: np.ndarray,
    rcond: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve independent weighted least-squares systems for many pixels."""
    num_pixel, _, num_param = design.shape
    solution = np.full((num_pixel, num_param), np.nan, dtype=np.float64)
    obs_count = np.sum(weights > 0, axis=1)
    candidates = np.flatnonzero(obs_count >= num_param)
    if candidates.size == 0:
        return solution, np.zeros(num_pixel, dtype=bool)

    design_i = design[candidates]
    obs_i = observations[candidates]
    weight_i = normalize_weights(weights[candidates])
    normal = np.einsum("pmq,pm,pmr->pqr", design_i, weight_i, design_i, optimize=True)
    rhs = np.einsum("pmq,pm,pm->pq", design_i, weight_i, obs_i, optimize=True)

    singular = np.linalg.svd(normal, compute_uv=False)
    threshold = rcond * singular[:, :1]
    full_rank = np.sum(singular > threshold, axis=1) == num_param
    if np.any(full_rank):
        inverse = np.linalg.pinv(normal[full_rank], rcond=rcond)
        solved = np.einsum("pqr,pr->pq", inverse, rhs[full_rank], optimize=True)
        solution[candidates[full_rank]] = solved

    valid = np.all(np.isfinite(solution), axis=1)
    return solution, valid


def solve_model_block(
    phase: np.ndarray,
    dem_coefficient: np.ndarray,
    deformation_design: np.ndarray,
    fit_ifgrams: np.ndarray,
    base_weights: np.ndarray,
    phase_velocity: bool = False,
    temporal_baseline: np.ndarray | None = None,
    robust: bool = False,
    huber_delta: float = 1.345,
    huber_iterations: int = 5,
    rcond: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate [DEM error, deformation parameters] for a phase block.

    Parameters use interferogram-major arrays with shape (num_ifg, num_pixel).
    """
    phase_fit = np.asarray(phase[fit_ifgrams].T, dtype=np.float64)
    dem_fit = np.asarray(dem_coefficient[fit_ifgrams].T, dtype=np.float64)
    defo_fit = np.asarray(deformation_design[fit_ifgrams], dtype=np.float64)
    weights = np.asarray(base_weights[fit_ifgrams].T, dtype=np.float64)

    num_pixel, num_ifg = phase_fit.shape
    num_param = 1 + defo_fit.shape[1]
    design = np.empty((num_pixel, num_ifg, num_param), dtype=np.float64)
    design[:, :, 0] = dem_fit
    design[:, :, 1:] = defo_fit[None, :, :]

    finite = np.isfinite(phase_fit) & np.isfinite(dem_fit) & np.isfinite(weights)
    weights = np.where(finite & (weights > 0), weights, 0.0)
    observations = np.where(finite, phase_fit, 0.0)
    design = np.where(finite[:, :, None], design, 0.0)

    if phase_velocity:
        if temporal_baseline is None:
            raise ValueError("temporal_baseline is required for phase-velocity inversion")
        dt = np.abs(np.asarray(temporal_baseline[fit_ifgrams], dtype=np.float64))
        if np.any(dt <= 0):
            raise ValueError("phase-velocity inversion requires non-zero temporal baselines")
        scale = 1.0 / dt
        observations *= scale[None, :]
        design *= scale[None, :, None]

    solution, valid = solve_weighted(design, observations, weights, rcond)
    if robust and np.any(valid):
        robust_pixels = np.flatnonzero(valid)
        robust_design = design[robust_pixels]
        robust_observations = observations[robust_pixels]
        robust_weights = weights[robust_pixels]
        robust_solution = solution[robust_pixels]
        for _ in range(max(1, huber_iterations)):
            prediction = np.einsum(
                "pmq,pq->pm", robust_design, robust_solution, optimize=True
            )
            residual = robust_observations - prediction
            residual[robust_weights <= 0] = np.nan
            median = np.nanmedian(residual, axis=1, keepdims=True)
            mad = np.nanmedian(np.abs(residual - median), axis=1, keepdims=True)
            scale = np.maximum(1.4826 * mad, 1e-6)
            ratio = np.abs(residual) / (huber_delta * scale)
            huber_weight = np.ones_like(ratio)
            large = ratio > 1.0
            huber_weight[large] = 1.0 / ratio[large]
            huber_weight[~np.isfinite(huber_weight)] = 0.0
            current_weights = robust_weights * huber_weight
            robust_solution, robust_valid = solve_weighted(
                robust_design,
                robust_observations,
                current_weights,
                rcond,
            )
            if not np.all(robust_valid):
                robust_pixels = robust_pixels[robust_valid]
                robust_design = robust_design[robust_valid]
                robust_observations = robust_observations[robust_valid]
                robust_weights = robust_weights[robust_valid]
                robust_solution = robust_solution[robust_valid]
                if robust_pixels.size == 0:
                    break
        solution[:] = np.nan
        solution[robust_pixels] = robust_solution
        valid = np.all(np.isfinite(solution), axis=1)

    return solution, valid


def inspect_baseline_mode(
    stack: h5py.File, geometry: h5py.File, date_pairs: list[tuple[str, str]]
) -> tuple[str, object]:
    bperp = stack["bperp"]
    if bperp.ndim == 3:
        return "stack_3d", bperp
    if bperp.ndim != 1 or bperp.shape[0] != len(date_pairs):
        raise ValueError(f"unsupported ifgramStack bperp shape: {bperp.shape}")

    if "bperp" in geometry and isinstance(geometry["bperp"], h5py.Dataset):
        dataset = geometry["bperp"]
        if dataset.ndim == 3 and "date" in geometry:
            dates = [decode_text(value) for value in geometry["date"][:]]
            return "geometry_3d", (dataset, {date: idx for idx, date in enumerate(dates)})

    named = {name[6:]: name for name in geometry if name.startswith("bperp-")}
    if named and all(date in named for pair in date_pairs for date in pair):
        return "geometry_named", named
    return "stack_1d", np.asarray(bperp[:], dtype=np.float64)


def read_bperp_block(
    mode: str,
    source,
    geometry: h5py.File,
    date_pairs: list[tuple[str, str]],
    row_slice: slice,
    width: int,
) -> np.ndarray:
    if mode == "stack_1d":
        return np.broadcast_to(source[:, None], (len(date_pairs), (row_slice.stop - row_slice.start) * width))
    if mode == "stack_3d":
        return np.asarray(source[:, row_slice, :], dtype=np.float64).reshape(len(date_pairs), -1)
    if mode == "geometry_3d":
        dataset, date_index = source
        output = np.empty((len(date_pairs), (row_slice.stop - row_slice.start) * width), dtype=np.float64)
        for idx, (master, slave) in enumerate(date_pairs):
            output[idx] = (
                np.asarray(dataset[date_index[slave], row_slice, :], dtype=np.float64)
                - np.asarray(dataset[date_index[master], row_slice, :], dtype=np.float64)
            ).reshape(-1)
        return output
    if mode == "geometry_named":
        output = np.empty((len(date_pairs), (row_slice.stop - row_slice.start) * width), dtype=np.float64)
        for idx, (master, slave) in enumerate(date_pairs):
            output[idx] = (
                np.asarray(geometry[source[slave]][row_slice, :], dtype=np.float64)
                - np.asarray(geometry[source[master]][row_slice, :], dtype=np.float64)
            ).reshape(-1)
        return output
    raise ValueError(f"unsupported baseline mode: {mode}")


def read_mask_block(
    mask_file: h5py.File | None,
    mask_dataset: str | None,
    row_slice: slice,
    width: int,
) -> np.ndarray:
    if mask_file is None:
        return np.ones((row_slice.stop - row_slice.start) * width, dtype=bool)
    return np.asarray(mask_file[mask_dataset][row_slice, :]).reshape(-1) != 0


def prepare_vce_source(
    variance_file: h5py.File | None,
    dataset_name: str,
    num_ifgram: int,
):
    if variance_file is None:
        return None
    if dataset_name not in variance_file:
        raise KeyError(f"variance dataset not found: {dataset_name}")
    dataset = variance_file[dataset_name]
    if dataset.ndim == 1 and dataset.shape[0] == num_ifgram:
        return ("vector", np.asarray(dataset[:], dtype=np.float64))
    if dataset.ndim == 2 and dataset.shape[0] == num_ifgram and dataset.shape[1] >= 3:
        values = np.asarray(dataset[:, 0], dtype=np.float64) + np.asarray(dataset[:, 2], dtype=np.float64)
        return ("vector", values)
    if dataset.ndim == 3 and dataset.shape[0] == num_ifgram:
        return ("stack", dataset)
    raise ValueError(
        f"unsupported variance dataset shape {dataset.shape}; expected (ifg,), (ifg,>=3), or (ifg,y,x)"
    )


def get_base_weights(
    spec: ModelSpec,
    coherence: np.ndarray | None,
    vce_source,
    row_slice: slice,
    phase_shape: tuple[int, int, int],
    min_coherence: float,
) -> np.ndarray:
    num_ifgram, _, width = phase_shape
    num_pixel = (row_slice.stop - row_slice.start) * width
    weights = np.ones((num_ifgram, num_pixel), dtype=np.float64)

    if coherence is not None:
        coh = np.asarray(coherence[:, row_slice, :], dtype=np.float64).reshape(num_ifgram, -1)
        if min_coherence > 0:
            weights[coh < min_coherence] = 0.0
    else:
        coh = None

    if spec.solver == "coherence":
        if coh is None:
            raise ValueError(f"{spec.name} requires a coherence dataset")
        coh_clip = np.clip(coh, 1e-3, 0.999)
        phase_weight = coh_clip**2 / np.maximum(1.0 - coh_clip**2, 1e-6)
        weights *= np.clip(phase_weight, 0.0, 1e3)
    elif spec.solver == "vce":
        if vce_source is None:
            raise ValueError(f"{spec.name} requires --variance-file")
        mode, source = vce_source
        if mode == "vector":
            variance = np.broadcast_to(source[:, None], weights.shape)
        else:
            variance = np.asarray(source[:, row_slice, :], dtype=np.float64).reshape(num_ifgram, -1)
        weights *= np.divide(1.0, variance, out=np.zeros_like(variance), where=variance > 0)

    return weights


def summarize_input(args, stack: h5py.File, geometry: h5py.File) -> dict:
    required_stack = ["unwrapPhase", "date", "bperp"]
    required_geometry = ["incidenceAngle", "slantRangeDistance"]
    missing = [name for name in required_stack if name not in stack]
    missing += [name for name in required_geometry if name not in geometry]
    if missing:
        raise KeyError(f"missing required datasets: {missing}")

    phase = stack["unwrapPhase"]
    if phase.ndim != 3:
        raise ValueError(f"unwrapPhase must be 3D, got {phase.shape}")
    date_pairs = read_date_pairs(stack["date"])
    if len(date_pairs) != phase.shape[0]:
        raise ValueError("date and unwrapPhase interferogram counts differ")
    if geometry["incidenceAngle"].shape != phase.shape[1:]:
        raise ValueError("incidenceAngle shape differs from unwrapPhase spatial shape")
    if geometry["slantRangeDistance"].shape != phase.shape[1:]:
        raise ValueError("slantRangeDistance shape differs from unwrapPhase spatial shape")
    if "WAVELENGTH" not in stack.attrs:
        raise KeyError("ifgramStack metadata is missing WAVELENGTH")

    selected = np.ones(phase.shape[0], dtype=bool)
    if not args.use_all_ifgrams and "dropIfgram" in stack:
        selected = np.asarray(stack["dropIfgram"][:], dtype=bool)
    components = network_components(date_pairs, selected)
    baseline_mode, baseline_source = inspect_baseline_mode(stack, geometry, date_pairs)

    diagnostics = {
        "shape": tuple(int(value) for value in phase.shape),
        "selected_ifgrams": int(np.sum(selected)),
        "total_ifgrams": int(phase.shape[0]),
        "acquisitions": len({date for pair in date_pairs for date in pair}),
        "network_components": components,
        "baseline_mode": baseline_mode,
        "models": {},
    }
    scalar_bperp = np.asarray(stack["bperp"][:], dtype=np.float64) if stack["bperp"].ndim == 1 else None
    for model_name in args.models:
        spec = MODEL_SPECS[model_name]
        deformation, labels = build_deformation_design(
            date_pairs, spec.poly_order, args.periodic, args.step_date
        )
        if scalar_bperp is not None:
            rank = int(np.linalg.matrix_rank(np.column_stack((scalar_bperp, deformation))[selected]))
        else:
            rank = None
        diagnostics["models"][model_name] = {
            "parameters": 1 + deformation.shape[1],
            "deformation_terms": labels,
            "representative_rank": rank,
        }
    return diagnostics | {
        "date_pairs": date_pairs,
        "selected": selected,
        "baseline_source": baseline_source,
    }


def print_diagnostics(diagnostics: dict) -> None:
    print("MintPy DEM-error interferogram correction")
    print(f"  stack shape: {diagnostics['shape']}")
    print(
        f"  fit interferograms: {diagnostics['selected_ifgrams']} / {diagnostics['total_ifgrams']}"
    )
    print(f"  acquisitions: {diagnostics['acquisitions']}")
    print(f"  baseline mode: {diagnostics['baseline_mode']}")
    sizes = [len(component) for component in diagnostics["network_components"]]
    print(f"  network components: {len(sizes)} {sizes}")
    if len(sizes) > 1:
        print("  WARNING: the selected interferogram network is disconnected")
    for name, model in diagnostics["models"].items():
        rank = model["representative_rank"]
        rank_text = "pixel-dependent" if rank is None else str(rank)
        print(
            f"  model {name}: parameters={model['parameters']}, rank={rank_text}, "
            f"terms={model['deformation_terms']}"
        )
        if rank is not None and rank < model["parameters"]:
            print(f"  WARNING: model {name} is rank deficient")


def run(args: argparse.Namespace) -> list[Path]:
    input_path = Path(args.ifgram_stack).resolve()
    geometry_path = Path(args.geometry).resolve()
    output_dir = Path(args.output_dir).resolve()
    if args.block_rows < 1:
        raise ValueError("--block-rows must be positive")
    if not 0 <= args.min_coherence < 1:
        raise ValueError("--min-coherence must be in [0, 1)")

    mask_handle = h5py.File(args.mask, "r") if args.mask else None
    variance_handle = h5py.File(args.variance_file, "r") if args.variance_file else None
    output_paths: list[Path] = []
    try:
        with h5py.File(input_path, "r") as stack, h5py.File(geometry_path, "r") as geometry:
            diagnostics = summarize_input(args, stack, geometry)
            print_diagnostics(diagnostics)
            if args.dry_run:
                return output_paths

            output_dir.mkdir(parents=True, exist_ok=True)
            phase_source = stack["unwrapPhase"]
            coherence_source = stack["coherence"] if "coherence" in stack else None
            num_ifgram, length, width = phase_source.shape
            date_pairs = diagnostics["date_pairs"]
            selected = diagnostics["selected"]
            baseline_mode = diagnostics["baseline_mode"]
            baseline_source = diagnostics["baseline_source"]
            wavelength = float(stack.attrs["WAVELENGTH"])
            range_to_phase = -4.0 * np.pi / wavelength
            years = date_years(date_pairs)
            temporal_baseline = np.asarray(
                [years[slave] - years[master] for master, slave in date_pairs],
                dtype=np.float64,
            )
            mask_dataset = choose_mask_dataset(mask_handle, args.mask_dataset) if mask_handle else None
            vce_source = prepare_vce_source(
                variance_handle, args.variance_dataset, num_ifgram
            )
            if variance_handle is not None and "date" in variance_handle:
                variance_dates = read_date_pairs(variance_handle["date"])
                if variance_dates != date_pairs:
                    raise ValueError(
                        "variance-file date pairs do not match ifgramStack order"
                    )

            summary_rows = []
            for model_name in args.models:
                spec = MODEL_SPECS[model_name]
                deformation_design, deformation_labels = build_deformation_design(
                    date_pairs, spec.poly_order, args.periodic, args.step_date
                )
                (
                    corrected_file,
                    corrected_phase,
                    component_file,
                    dem_phase_output,
                    dem_error_output,
                    corrected_path,
                    component_path,
                ) = prepare_output_files(
                    stack, output_dir, spec, "unwrapPhase", args.overwrite
                )
                for output_file in (corrected_file, component_file):
                    output_file.attrs["DEM_ERROR_POLY_ORDER"] = spec.poly_order
                    output_file.attrs["DEM_ERROR_SOLVER"] = spec.solver
                    output_file.attrs["DEM_ERROR_PHASE_VELOCITY"] = str(
                        spec.phase_velocity
                    )
                    output_file.attrs["DEM_ERROR_PERIODIC_YEARS"] = str(args.periodic)
                    output_file.attrs["DEM_ERROR_STEP_DATES"] = str(args.step_date)
                    output_file.attrs["DEM_ERROR_FIT_IFGRAMS"] = int(np.sum(selected))
                output_paths.extend([corrected_path, component_path])

                sum_original2 = np.zeros(num_ifgram, dtype=np.float64)
                sum_corrected2 = np.zeros(num_ifgram, dtype=np.float64)
                sum_component2 = np.zeros(num_ifgram, dtype=np.float64)
                sum_residual2 = np.zeros(num_ifgram, dtype=np.float64)
                metric_count = np.zeros(num_ifgram, dtype=np.int64)
                dem_sum = 0.0
                dem_sum2 = 0.0
                dem_count = 0

                print(f"\nmodel: {spec.name}")
                try:
                    for y0 in range(0, length, args.block_rows):
                        y1 = min(length, y0 + args.block_rows)
                        row_slice = slice(y0, y1)
                        phase = np.asarray(phase_source[:, row_slice, :], dtype=np.float64).reshape(num_ifgram, -1)
                        incidence = np.asarray(
                            geometry["incidenceAngle"][row_slice, :], dtype=np.float64
                        ).reshape(-1)
                        slant_range = np.asarray(
                            geometry["slantRangeDistance"][row_slice, :], dtype=np.float64
                        ).reshape(-1)
                        sine_incidence = np.sin(np.deg2rad(incidence))
                        inverse_geometry = np.divide(
                            1.0,
                            slant_range * sine_incidence,
                            out=np.full_like(slant_range, np.nan),
                            where=(slant_range > 0) & (sine_incidence != 0),
                        )
                        bperp = read_bperp_block(
                            baseline_mode,
                            baseline_source,
                            geometry,
                            date_pairs,
                            row_slice,
                            width,
                        )
                        dem_coefficient = range_to_phase * bperp * inverse_geometry[None, :]
                        weights = get_base_weights(
                            spec,
                            coherence_source,
                            vce_source,
                            row_slice,
                            phase_source.shape,
                            args.min_coherence,
                        )
                        spatial_mask = read_mask_block(
                            mask_handle, mask_dataset, row_slice, width
                        )
                        spatial_mask &= np.isfinite(inverse_geometry)
                        spatial_mask &= ~np.all((phase == 0) | ~np.isfinite(phase), axis=0)
                        weights[:, ~spatial_mask] = 0.0

                        solution, valid = solve_model_block(
                            phase,
                            dem_coefficient,
                            deformation_design,
                            selected,
                            weights,
                            phase_velocity=spec.phase_velocity,
                            temporal_baseline=temporal_baseline,
                            robust=spec.solver == "huber",
                            huber_delta=args.huber_delta,
                            huber_iterations=args.huber_iterations,
                            rcond=args.rcond,
                        )
                        dem_error = solution[:, 0]
                        component = np.zeros_like(phase)
                        component[:, valid] = dem_coefficient[:, valid] * dem_error[valid][None, :]
                        corrected = phase - component

                        prediction = component.copy()
                        if deformation_design.shape[1] > 0 and np.any(valid):
                            prediction[:, valid] += (
                                deformation_design @ solution[valid, 1:].T
                            )
                        residual = phase - prediction
                        metric_mask = np.isfinite(phase) & valid[None, :]
                        sum_original2 += np.sum(np.where(metric_mask, phase**2, 0.0), axis=1)
                        sum_corrected2 += np.sum(np.where(metric_mask, corrected**2, 0.0), axis=1)
                        sum_component2 += np.sum(np.where(metric_mask, component**2, 0.0), axis=1)
                        sum_residual2 += np.sum(np.where(metric_mask, residual**2, 0.0), axis=1)
                        metric_count += np.sum(metric_mask, axis=1)
                        if np.any(valid):
                            values = dem_error[valid]
                            dem_sum += float(np.sum(values))
                            dem_sum2 += float(np.sum(values**2))
                            dem_count += int(values.size)

                        corrected_phase[:, row_slice, :] = corrected.reshape(num_ifgram, y1 - y0, width).astype(np.float32)
                        dem_phase_output[:, row_slice, :] = component.reshape(num_ifgram, y1 - y0, width).astype(np.float32)
                        dem_error_output[row_slice, :] = dem_error.reshape(y1 - y0, width).astype(np.float32)
                        print(f"  rows {y0}:{y1}, valid pixels {int(np.sum(valid))}")
                finally:
                    corrected_file.close()
                    component_file.close()

                original_rms = np.full(num_ifgram, np.nan, dtype=np.float64)
                corrected_rms = np.full(num_ifgram, np.nan, dtype=np.float64)
                component_rms = np.full(num_ifgram, np.nan, dtype=np.float64)
                residual_rms = np.full(num_ifgram, np.nan, dtype=np.float64)
                has_metrics = metric_count > 0
                original_rms[has_metrics] = np.sqrt(
                    sum_original2[has_metrics] / metric_count[has_metrics]
                )
                corrected_rms[has_metrics] = np.sqrt(
                    sum_corrected2[has_metrics] / metric_count[has_metrics]
                )
                component_rms[has_metrics] = np.sqrt(
                    sum_component2[has_metrics] / metric_count[has_metrics]
                )
                residual_rms[has_metrics] = np.sqrt(
                    sum_residual2[has_metrics] / metric_count[has_metrics]
                )
                metrics_path = output_dir / f"ifgram_metrics_{spec.name}.csv"
                with metrics_path.open("w", newline="", encoding="utf-8") as file:
                    writer = csv.writer(file)
                    writer.writerow(
                        [
                            "master",
                            "slave",
                            "fit_used",
                            "valid_pixels",
                            "original_rms_rad",
                            "corrected_rms_rad",
                            "dem_component_rms_rad",
                            "model_residual_rms_rad",
                        ]
                    )
                    for idx, (master, slave) in enumerate(date_pairs):
                        writer.writerow(
                            [
                                master,
                                slave,
                                int(selected[idx]),
                                int(metric_count[idx]),
                                original_rms[idx],
                                corrected_rms[idx],
                                component_rms[idx],
                                residual_rms[idx],
                            ]
                        )
                output_paths.append(metrics_path)

                dem_mean = dem_sum / dem_count if dem_count else np.nan
                dem_variance = dem_sum2 / dem_count - dem_mean**2 if dem_count else np.nan
                summary_rows.append(
                    {
                        "model": spec.name,
                        "poly_order": spec.poly_order,
                        "solver": spec.solver,
                        "phase_velocity": int(spec.phase_velocity),
                        "num_parameters": 1 + len(deformation_labels),
                        "valid_dem_pixels": dem_count,
                        "dem_error_mean_m": dem_mean,
                        "dem_error_std_m": np.sqrt(max(dem_variance, 0.0)) if dem_count else np.nan,
                        "mean_model_residual_rms_rad": float(np.nanmean(residual_rms)),
                        "corrected_stack": corrected_path.name,
                        "component_file": component_path.name,
                    }
                )

            summary_path = output_dir / "model_comparison.csv"
            with summary_path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=list(summary_rows[0]))
                writer.writeheader()
                writer.writerows(summary_rows)
            output_paths.append(summary_path)
    finally:
        if mask_handle is not None:
            mask_handle.close()
        if variance_handle is not None:
            variance_handle.close()
    return output_paths


def main(argv: list[str] | None = None) -> int:
    args = create_parser().parse_args(argv)
    try:
        outputs = run(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=os.sys.stderr)
        return 1
    if outputs:
        print("\noutputs:")
        for path in outputs:
            print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
