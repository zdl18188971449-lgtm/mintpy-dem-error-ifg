#!/usr/bin/env python3
"""Generate a MintPy-like synthetic stack and benchmark DEM-error models.

The component-based simulation follows the design described by
InterferogramSimulator-main: terrain, deformation, atmosphere, water, and
noise are generated separately. Unlike its independent image samples, this
script builds an acquisition-consistent interferogram network with known DEM
error and writes the HDF5 datasets required by mintpy_dem_error_ifg.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timedelta
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

import mintpy_dem_error_ifg as correction


DEFAULT_MODELS = [
    "linear_ols",
    "quadratic_ols",
    "linear_wls",
    "quadratic_wls",
    "linear_huber",
    "quadratic_huber",
    "linear_vce",
    "quadratic_vce",
    "adaptive_vce_huber_graph",
]

TERRAIN_TYPES = ["plain", "hills", "mountain", "valley", "urban"]


def display_model_name(name: str) -> str:
    return "adaptive_graph" if name == "adaptive_vce_huber_graph" else name


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Simulate, correct, score, and plot a MintPy DEM-error benchmark"
    )
    parser.add_argument("-o", "--output", default="synthetic_benchmark")
    parser.add_argument("--size", type=int, default=96)
    parser.add_argument("--num-acquisitions", type=int, default=12)
    parser.add_argument("--max-neighbor", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--terrain-type", choices=TERRAIN_TYPES, default="hills")
    parser.add_argument("--wavelength", type=float, default=0.236)
    parser.add_argument("--baseline-time-correlation", type=float, default=0.0)
    parser.add_argument("--atmosphere-scale", type=float, default=1.0)
    parser.add_argument("--noise-scale", type=float, default=1.0)
    parser.add_argument("--acceleration-scale", type=float, default=1.0)
    parser.add_argument("--outlier-cycles", type=float, default=1.0)
    parser.add_argument("--adaptive-bic-margin", type=float, default=2.0)
    parser.add_argument("--graph-lambda", type=float, default=8.0)
    parser.add_argument("--graph-iterations", type=int, default=40)
    parser.add_argument("--models", nargs="+", choices=sorted(correction.MODEL_SPECS), default=DEFAULT_MODELS)
    parser.add_argument("--block-rows", type=int, default=24)
    parser.add_argument("--min-coherence", type=float, default=0.25)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    scale = np.std(values)
    return (values - np.mean(values)) / scale if scale > 0 else np.zeros_like(values)


def smooth_field(
    rng: np.random.Generator, shape: tuple[int, int], correlation_pixels: float
) -> np.ndarray:
    noise = rng.normal(size=shape)
    fy = np.fft.fftfreq(shape[0])[:, None]
    fx = np.fft.fftfreq(shape[1])[None, :]
    spectrum = np.exp(-2.0 * math.pi**2 * correlation_pixels**2 * (fx**2 + fy**2))
    return normalize(np.fft.ifft2(np.fft.fft2(noise) * spectrum).real)


def gaussian_surface(
    xx: np.ndarray,
    yy: np.ndarray,
    center_x: float,
    center_y: float,
    sigma_x: float,
    sigma_y: float,
) -> np.ndarray:
    return np.exp(
        -0.5
        * (
            ((xx - center_x) / sigma_x) ** 2
            + ((yy - center_y) / sigma_y) ** 2
        )
    )


def build_terrain_and_dem_error(
    rng: np.random.Generator,
    shape: tuple[int, int],
    xx: np.ndarray,
    yy: np.ndarray,
    terrain_type: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    size = shape[0]
    broad = smooth_field(rng, shape, size / 8.0)
    detail = smooth_field(rng, shape, size / 16.0)

    if terrain_type == "plain":
        terrain = 120.0 + 22.0 * broad + 18.0 * xx + 8.0 * yy
        river_axis = 0.15 * np.sin(2.5 * np.pi * xx) - 0.10
        water_mask = np.abs(yy - river_axis) < 0.045
        dem_error = 2.8 * broad + 1.2 * normalize(terrain)
        dem_error[(xx > -0.65) & (xx < -0.15) & (yy > 0.15) & (yy < 0.55)] += 3.5
        dem_error[(xx > 0.25) & (yy < -0.25)] -= 2.5
    elif terrain_type == "hills":
        hill = gaussian_surface(xx, yy, -0.25, 0.15, 0.32, 0.42)
        terrain = 450.0 + 180.0 * broad + 520.0 * hill + 90.0 * xx
        water_field = smooth_field(rng, shape, size / 7.0)
        water_mask = water_field > np.quantile(water_field, 0.91)
        local_error = gaussian_surface(xx, yy, 0.35, -0.15, 0.20, 0.28)
        dem_error = 3.5 * broad + 2.0 * normalize(terrain) + 10.0 * local_error
        dem_error[(yy > 0.35) & (xx < -0.35)] += 4.0
    elif terrain_type == "mountain":
        ridge_axis = 0.22 * np.sin(1.7 * np.pi * xx) + 0.10
        ridge = np.exp(-0.5 * ((yy - ridge_axis) / 0.20) ** 2)
        peak = gaussian_surface(xx, yy, -0.45, -0.20, 0.24, 0.30)
        terrain = 950.0 + 720.0 * ridge + 620.0 * peak + 280.0 * broad + 110.0 * xx
        water_mask = terrain < np.quantile(terrain, 0.035)
        slope = np.hypot(*np.gradient(normalize(terrain)))
        dem_error = 4.5 * broad + 3.5 * detail + 3.0 * normalize(terrain)
        dem_error += 7.0 * normalize(slope) + 5.0 * peak
    elif terrain_type == "valley":
        valley_axis = 0.24 * np.sin(1.8 * np.pi * xx) - 0.05
        valley = np.exp(-0.5 * ((yy - valley_axis) / 0.18) ** 2)
        terrain = 780.0 + 480.0 * (1.0 - valley) + 190.0 * broad + 80.0 * xx
        water_mask = np.abs(yy - valley_axis) < 0.035
        side_bias = np.tanh((yy - valley_axis) / 0.16)
        dem_error = 3.2 * broad + 7.0 * (valley - np.mean(valley))
        dem_error += 4.0 * side_bias + 1.5 * normalize(terrain)
    elif terrain_type == "urban":
        terrain = 85.0 + 12.0 * broad + 8.0 * xx
        building_error = np.zeros(shape, dtype=np.float64)
        blocks = [
            (-0.75, -0.35, -0.65, -0.25, 55.0, 7.0),
            (-0.20, 0.15, -0.15, 0.30, 85.0, -6.0),
            (0.35, 0.75, -0.55, -0.10, 65.0, 9.0),
            (0.25, 0.60, 0.30, 0.70, 45.0, 5.0),
            (-0.70, -0.30, 0.35, 0.72, 70.0, -8.0),
        ]
        for x0, x1, y0, y1, height, error in blocks:
            footprint = (xx >= x0) & (xx <= x1) & (yy >= y0) & (yy <= y1)
            terrain[footprint] += height
            building_error[footprint] += error
        water_mask = (xx > 0.72) & (yy > 0.45)
        dem_error = 2.0 * broad + 1.5 * detail + building_error
        dem_error += 1.0 * normalize(terrain)
    else:
        raise ValueError(f"unsupported terrain type: {terrain_type}")

    dem_error = np.asarray(dem_error, dtype=np.float64)
    dem_error -= np.mean(dem_error[~water_mask])
    return terrain, dem_error, water_mask


def acquisition_dates(count: int) -> list[str]:
    start = datetime(2020, 1, 1)
    return [(start + timedelta(days=46 * idx)).strftime("%Y%m%d") for idx in range(count)]


def interferogram_pairs(count: int, max_neighbor: int) -> list[tuple[int, int]]:
    return [
        (master, slave)
        for master in range(count - 1)
        for slave in range(master + 1, min(count, master + max_neighbor + 1))
    ]


def acquisition_baselines(
    rng: np.random.Generator, count: int, time_correlation: float
) -> np.ndarray:
    if not -0.95 <= time_correlation <= 0.95:
        raise ValueError("--baseline-time-correlation must be in [-0.95, 0.95]")
    time_axis = normalize(np.arange(count, dtype=np.float64))
    random_axis = rng.normal(size=count)
    random_axis -= np.dot(random_axis, time_axis) / np.dot(time_axis, time_axis) * time_axis
    random_axis = normalize(random_axis)
    mixed = (
        math.sqrt(1.0 - time_correlation**2) * random_axis
        + time_correlation * time_axis
    )
    baselines = 320.0 * normalize(mixed)
    return baselines - baselines[0]


def write_dataset(file: h5py.File, name: str, data: np.ndarray) -> h5py.Dataset:
    return file.create_dataset(
        name,
        data=data,
        compression="gzip" if data.ndim >= 2 else None,
        compression_opts=4 if data.ndim >= 2 else None,
        shuffle=bool(data.ndim >= 2),
    )


def simulate(args: argparse.Namespace, root: Path) -> dict:
    if args.size < 16:
        raise ValueError("--size must be at least 16")
    if args.num_acquisitions < 5:
        raise ValueError("--num-acquisitions must be at least 5")
    if not 1 <= args.max_neighbor < args.num_acquisitions:
        raise ValueError("--max-neighbor must be between 1 and num-acquisitions - 1")

    terrain_seed, process_seed = np.random.SeedSequence(args.seed).spawn(2)
    terrain_rng = np.random.default_rng(terrain_seed)
    rng = np.random.default_rng(process_seed)
    shape = (args.size, args.size)
    y = np.linspace(-1.0, 1.0, args.size)
    x = np.linspace(-1.0, 1.0, args.size)
    xx, yy = np.meshgrid(x, y)

    terrain, dem_error, water_mask = build_terrain_and_dem_error(
        terrain_rng, shape, xx, yy, args.terrain_type
    )
    terrain_norm = normalize(terrain)

    velocity = -1.8 * gaussian_surface(xx, yy, 0.05, 0.05, 0.28, 0.24)
    velocity += 0.25 * xx
    acceleration = args.acceleration_scale * 1.4 * gaussian_surface(
        xx, yy, -0.20, 0.22, 0.24, 0.30
    )

    dates = acquisition_dates(args.num_acquisitions)
    pairs = interferogram_pairs(args.num_acquisitions, args.max_neighbor)
    date_pairs = [(dates[master], dates[slave]) for master, slave in pairs]
    years = np.asarray(
        [
            (datetime.strptime(date, "%Y%m%d") - datetime.strptime(dates[0], "%Y%m%d")).days
            / 365.25
            for date in dates
        ]
    )
    deformation_acquisition = (
        years[:, None, None] * velocity[None, :, :]
        + 0.5 * years[:, None, None] ** 2 * acceleration[None, :, :]
    )

    baseline_center = acquisition_baselines(
        rng, args.num_acquisitions, args.baseline_time_correlation
    )
    baseline_shape = 1.0 + 0.025 * xx - 0.018 * yy
    baseline_acquisition = baseline_center[:, None, None] * baseline_shape[None, :, :]
    bperp = np.asarray(
        [np.mean(baseline_acquisition[slave] - baseline_acquisition[master]) for master, slave in pairs]
    )

    incidence = 34.0 + 4.0 * (xx + 1.0) / 2.0 + 0.4 * yy
    slant_range = 850000.0 + 18000.0 * (xx + 1.0) / 2.0
    range_to_phase = -4.0 * np.pi / args.wavelength
    inverse_geometry = 1.0 / (slant_range * np.sin(np.deg2rad(incidence)))

    atmosphere_acquisition = np.empty((args.num_acquisitions, *shape), dtype=np.float64)
    for idx in range(args.num_acquisitions):
        turbulent = 0.32 * smooth_field(rng, shape, args.size / 10.0)
        stratified = rng.normal(scale=0.10) * terrain_norm
        atmosphere_acquisition[idx] = args.atmosphere_scale * (turbulent + stratified)

    terrain_gradient = np.hypot(*np.gradient(terrain_norm))
    terrain_gradient /= max(np.max(terrain_gradient), 1e-6)

    num_ifg = len(pairs)
    dem_phase = np.empty((num_ifg, *shape), dtype=np.float64)
    deformation_phase = np.empty_like(dem_phase)
    atmosphere_phase = np.empty_like(dem_phase)
    noise_phase = np.empty_like(dem_phase)
    unwrap_error = np.zeros_like(dem_phase)
    coherence = np.empty_like(dem_phase)

    for idx, (master, slave) in enumerate(pairs):
        bperp_map = baseline_acquisition[slave] - baseline_acquisition[master]
        dem_phase[idx] = range_to_phase * bperp_map * inverse_geometry * dem_error
        deformation_phase[idx] = deformation_acquisition[slave] - deformation_acquisition[master]
        independent_turbulence = 0.08 * smooth_field(rng, shape, args.size / 14.0)
        atmosphere_phase[idx] = args.atmosphere_scale * independent_turbulence
        atmosphere_phase[idx] += atmosphere_acquisition[slave] - atmosphere_acquisition[master]

        coherence_i = rng.uniform(0.76, 0.94) - 0.16 * terrain_gradient
        coherence_i -= 0.06 * np.abs(deformation_phase[idx]) / max(
            np.max(np.abs(deformation_phase[idx])), 1e-6
        )
        coherence_i += 0.025 * smooth_field(rng, shape, args.size / 16.0)
        coherence_i = np.clip(coherence_i, 0.20, 0.98)
        coherence_i[water_mask] = rng.uniform(0.04, 0.14, size=np.sum(water_mask))
        coherence[idx] = coherence_i

        sigma = np.sqrt(
            np.maximum(1.0 - coherence_i**2, 1e-6)
            / np.maximum(8.0 * coherence_i**2, 1e-6)
        )
        noise_phase[idx] = args.noise_scale * sigma * rng.normal(size=shape)
        noise_phase[idx, water_mask] = rng.uniform(-np.pi, np.pi, size=np.sum(water_mask))

    outlier_index = int(np.argmax(np.abs(bperp)))
    outlier_region = ((xx + 0.15) / 0.24) ** 2 + ((yy - 0.10) / 0.20) ** 2 <= 1.0
    unwrap_error[outlier_index, outlier_region] = 2.0 * np.pi * args.outlier_cycles

    clean_phase = dem_phase + deformation_phase
    unwrap_phase = clean_phase + atmosphere_phase + noise_phase + unwrap_error
    wrapped_phase = np.angle(np.exp(1j * unwrap_phase))
    valid_mask = (~water_mask) & (np.mean(coherence, axis=0) >= args.min_coherence)

    input_dir = root / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    stack_path = input_dir / "ifgramStack.h5"
    geometry_path = input_dir / "geometryRadar.h5"
    mask_path = root / "maskTempCoh.h5"
    variance_path = root / "simulationVariance.h5"
    truth_path = root / "simulationTruth.h5"

    with h5py.File(stack_path, "w") as file:
        file.attrs.update(
            {
                "FILE_TYPE": "ifgramStack",
                "LENGTH": str(args.size),
                "WIDTH": str(args.size),
                "WAVELENGTH": str(args.wavelength),
                "PROCESSOR": "synthetic",
            }
        )
        write_dataset(file, "unwrapPhase", unwrap_phase.astype(np.float32))
        write_dataset(file, "wrapPhase", wrapped_phase.astype(np.float32))
        write_dataset(file, "coherence", coherence.astype(np.float32))
        file.create_dataset("date", data=np.asarray(date_pairs, dtype="S8"))
        file.create_dataset("bperp", data=bperp.astype(np.float32))
        file.create_dataset("dropIfgram", data=np.ones(num_ifg, dtype=bool))

    with h5py.File(geometry_path, "w") as file:
        write_dataset(file, "height", terrain.astype(np.float32))
        write_dataset(file, "incidenceAngle", incidence.astype(np.float32))
        write_dataset(file, "slantRangeDistance", slant_range.astype(np.float32))
        write_dataset(file, "bperp", baseline_acquisition.astype(np.float32))
        file.create_dataset("date", data=np.asarray(dates, dtype="S8"))

    with h5py.File(mask_path, "w") as file:
        file.attrs["FILE_TYPE"] = "mask"
        write_dataset(file, "mask", valid_mask.astype(np.uint8))

    variance = np.var(atmosphere_phase + noise_phase, axis=(1, 2))
    variance = np.maximum(variance, 1e-6)
    with h5py.File(variance_path, "w") as file:
        file.create_dataset("phaseVariance", data=variance.astype(np.float32))
        file.create_dataset("date", data=np.asarray(date_pairs, dtype="S8"))

    with h5py.File(truth_path, "w") as file:
        file.attrs["OUTLIER_IFGRAM"] = "_".join(date_pairs[outlier_index])
        write_dataset(file, "demError", dem_error.astype(np.float32))
        write_dataset(file, "demPhase", dem_phase.astype(np.float32))
        write_dataset(file, "deformationPhase", deformation_phase.astype(np.float32))
        write_dataset(file, "atmospherePhase", atmosphere_phase.astype(np.float32))
        write_dataset(file, "noisePhase", noise_phase.astype(np.float32))
        write_dataset(file, "unwrapError", unwrap_error.astype(np.float32))
        write_dataset(file, "cleanPhase", clean_phase.astype(np.float32))
        write_dataset(file, "validMask", valid_mask.astype(np.uint8))
        file.create_dataset("date", data=np.asarray(date_pairs, dtype="S8"))
        file.create_dataset("bperp", data=bperp.astype(np.float32))

    metadata = {
        "seed": args.seed,
        "terrain_type": args.terrain_type,
        "shape": list(shape),
        "num_acquisitions": args.num_acquisitions,
        "num_interferograms": num_ifg,
        "max_neighbor": args.max_neighbor,
        "baseline_time_correlation": args.baseline_time_correlation,
        "atmosphere_scale": args.atmosphere_scale,
        "noise_scale": args.noise_scale,
        "acceleration_scale": args.acceleration_scale,
        "outlier_cycles": args.outlier_cycles,
        "adaptive_bic_margin": args.adaptive_bic_margin,
        "graph_lambda": args.graph_lambda,
        "graph_iterations": args.graph_iterations,
        "outlier_ifgram": "_".join(date_pairs[outlier_index]),
    }
    (root / "simulation_config.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return {
        "stack": stack_path,
        "geometry": geometry_path,
        "mask": mask_path,
        "variance": variance_path,
        "truth": truth_path,
        "outlier_index": outlier_index,
    }


def run_correction(args: argparse.Namespace, paths: dict, root: Path) -> Path:
    output = root / "correction"
    command = [
        str(paths["stack"]),
        "-g",
        str(paths["geometry"]),
        "--mask",
        str(paths["mask"]),
        "--variance-file",
        str(paths["variance"]),
        "--variance-dataset",
        "phaseVariance",
        "-o",
        str(output),
        "--models",
        *args.models,
        "--block-rows",
        str(args.block_rows),
        "--min-coherence",
        str(args.min_coherence),
        "--adaptive-bic-margin",
        str(args.adaptive_bic_margin),
        "--graph-lambda",
        str(args.graph_lambda),
        "--graph-iterations",
        str(args.graph_iterations),
        "--overwrite",
    ]
    result = correction.main(command)
    if result != 0:
        raise RuntimeError(f"DEM-error correction failed with status {result}")
    return output


def nmad(values: np.ndarray) -> float:
    center = np.median(values)
    return float(1.4826 * np.median(np.abs(values - center)))


def evaluate(models: list[str], paths: dict, correction_dir: Path, root: Path) -> list[dict]:
    with h5py.File(paths["truth"], "r") as truth_file, h5py.File(paths["stack"], "r") as stack_file:
        truth_dem = np.asarray(truth_file["demError"][:], dtype=np.float64)
        truth_dem_phase = np.asarray(truth_file["demPhase"][:], dtype=np.float64)
        mask = np.asarray(truth_file["validMask"][:], dtype=bool)
        observed = np.asarray(stack_file["unwrapPhase"][:], dtype=np.float64)
        target_corrected = observed - truth_dem_phase

    network_residual = {}
    with (correction_dir / "model_comparison.csv").open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            network_residual[row["model"]] = float(row["mean_model_residual_rms_rad"])

    rows = []
    for model in models:
        with h5py.File(correction_dir / f"demComponent_{model}.h5", "r") as file:
            estimate = np.asarray(file["demError"][:], dtype=np.float64)
            estimated_phase = np.asarray(file["demPhase"][:], dtype=np.float64)
        with h5py.File(correction_dir / f"ifgramStack_demErr_{model}.h5", "r") as file:
            corrected = np.asarray(file["unwrapPhase"][:], dtype=np.float64)

        valid = mask & np.isfinite(estimate)
        error = estimate[valid] - truth_dem[valid]
        truth_std = float(np.std(truth_dem[valid]))
        dem_rmse = float(np.sqrt(np.mean(error**2)))
        phase_valid = np.broadcast_to(valid, truth_dem_phase.shape)
        phase_error = estimated_phase[phase_valid] - truth_dem_phase[phase_valid]
        correction_error = corrected[phase_valid] - target_corrected[phase_valid]
        correlation = np.corrcoef(estimate[valid], truth_dem[valid])[0, 1]
        rows.append(
            {
                "model": model,
                "valid_pixels": int(np.sum(valid)),
                "dem_bias_m": float(np.mean(error)),
                "dem_mae_m": float(np.mean(np.abs(error))),
                "dem_rmse_m": dem_rmse,
                "truth_dem_std_m": truth_std,
                "dem_nrmse": dem_rmse / truth_std if truth_std > 0 else np.nan,
                "dem_nmad_m": nmad(error),
                "dem_correlation": float(correlation),
                "dem_phase_rmse_rad": float(np.sqrt(np.mean(phase_error**2))),
                "corrected_truth_rmse_rad": float(np.sqrt(np.mean(correction_error**2))),
                "network_model_residual_rms_rad": network_residual[model],
            }
        )

    with (root / "benchmark_metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def image_limits(values: np.ndarray, mask: np.ndarray | None = None) -> tuple[float, float]:
    sample = values[mask] if mask is not None else values.reshape(-1)
    limit = float(np.nanpercentile(np.abs(sample), 98.0))
    return -max(limit, 1e-6), max(limit, 1e-6)


def plot_results(rows: list[dict], paths: dict, correction_dir: Path, root: Path) -> None:
    ranked = sorted(rows, key=lambda row: row["dem_rmse_m"])
    shown = ranked[: min(3, len(ranked))]
    with h5py.File(paths["truth"], "r") as truth_file, h5py.File(paths["stack"], "r") as stack_file:
        truth_dem = np.asarray(truth_file["demError"][:], dtype=np.float64)
        truth_phase = np.asarray(truth_file["demPhase"][:], dtype=np.float64)
        mask = np.asarray(truth_file["validMask"][:], dtype=bool)
        observed = np.asarray(stack_file["unwrapPhase"][:], dtype=np.float64)
        date_pairs = correction.read_date_pairs(stack_file["date"])

    estimates = {}
    estimated_phases = {}
    corrected_phases = {}
    for row in shown:
        model = row["model"]
        with h5py.File(correction_dir / f"demComponent_{model}.h5", "r") as file:
            estimates[model] = np.asarray(file["demError"][:], dtype=np.float64)
            estimated_phases[model] = np.asarray(file["demPhase"][:], dtype=np.float64)
        with h5py.File(correction_dir / f"ifgramStack_demErr_{model}.h5", "r") as file:
            corrected_phases[model] = np.asarray(file["unwrapPhase"][:], dtype=np.float64)

    dem_limits = image_limits(truth_dem, mask)
    figure, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
    image = axes[0, 0].imshow(np.where(mask, truth_dem, np.nan), cmap="RdBu_r", vmin=dem_limits[0], vmax=dem_limits[1])
    axes[0, 0].set_title("True DEM error (m)")
    figure.colorbar(image, ax=axes[0, 0], shrink=0.78)
    for column, row in enumerate(shown, start=1):
        model = row["model"]
        image = axes[0, column].imshow(
            np.where(mask, estimates[model], np.nan), cmap="RdBu_r", vmin=dem_limits[0], vmax=dem_limits[1]
        )
        axes[0, column].set_title(
            f"{display_model_name(model)}\nRMSE={row['dem_rmse_m']:.2f} m"
        )
        figure.colorbar(image, ax=axes[0, column], shrink=0.78)

    model_names = [row["model"] for row in rows]
    rmse_values = [row["dem_rmse_m"] for row in rows]
    axes[1, 0].bar(np.arange(len(rows)), rmse_values, color="#287271")
    axes[1, 0].set_xticks(
        np.arange(len(rows)), [display_model_name(name) for name in model_names], rotation=55, ha="right"
    )
    axes[1, 0].set_ylabel("DEM RMSE (m)")
    axes[1, 0].set_title("All-model truth benchmark")
    for column, row in enumerate(shown, start=1):
        model = row["model"]
        error = estimates[model] - truth_dem
        error_limits = image_limits(error, mask)
        image = axes[1, column].imshow(
            np.where(mask, error, np.nan), cmap="RdBu_r", vmin=error_limits[0], vmax=error_limits[1]
        )
        axes[1, column].set_title(f"{display_model_name(model)} estimation error")
        figure.colorbar(image, ax=axes[1, column], shrink=0.78)
    for axis in axes.flat:
        axis.set_xticks([]) if axis is not axes[1, 0] else None
        axis.set_yticks([]) if axis is not axes[1, 0] else None
    figure.savefig(root / "synthetic_dem_comparison.png", dpi=180)
    figure.savefig(root / "synthetic_dem_comparison.pdf")
    plt.close(figure)

    best_model = shown[0]["model"]
    ifg_index = int(np.argmax([np.sqrt(np.mean(phase[mask] ** 2)) for phase in truth_phase]))
    target = observed[ifg_index] - truth_phase[ifg_index]
    correction_error = corrected_phases[best_model][ifg_index] - target
    phase_limits = image_limits(observed[ifg_index], mask)
    error_limits = image_limits(correction_error, mask)
    panels = [
        (observed[ifg_index], "Observed unwrapped phase", phase_limits),
        (truth_phase[ifg_index], "True DEM phase", image_limits(truth_phase[ifg_index], mask)),
        (estimated_phases[best_model][ifg_index], f"Estimated DEM phase\n{display_model_name(best_model)}", image_limits(truth_phase[ifg_index], mask)),
        (corrected_phases[best_model][ifg_index], "Corrected phase", phase_limits),
        (correction_error, "Correction error", error_limits),
    ]
    figure, axes = plt.subplots(1, 5, figsize=(20, 4.2), constrained_layout=True)
    for axis, (data, title, limits) in zip(axes, panels):
        image = axis.imshow(np.where(mask, data, np.nan), cmap="RdBu_r", vmin=limits[0], vmax=limits[1])
        axis.set_title(title)
        axis.set_xticks([])
        axis.set_yticks([])
        figure.colorbar(image, ax=axis, shrink=0.72)
    master, slave = date_pairs[ifg_index]
    figure.suptitle(f"Interferogram {master}_{slave}")
    figure.savefig(root / "synthetic_ifgram_comparison.png", dpi=180)
    figure.savefig(root / "synthetic_ifgram_comparison.pdf")
    plt.close(figure)


def main(argv: list[str] | None = None) -> int:
    args = create_parser().parse_args(argv)
    root = Path(args.output).resolve()
    if root.exists() and any(root.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output directory is not empty, use --overwrite: {root}")
    root.mkdir(parents=True, exist_ok=True)

    paths = simulate(args, root)
    correction_dir = run_correction(args, paths, root)
    rows = evaluate(args.models, paths, correction_dir, root)
    plot_results(rows, paths, correction_dir, root)

    print("\nSynthetic benchmark ranking by DEM RMSE:")
    for rank, row in enumerate(sorted(rows, key=lambda item: item["dem_rmse_m"]), start=1):
        print(
            f"  {rank}. {row['model']}: RMSE={row['dem_rmse_m']:.3f} m, "
            f"MAE={row['dem_mae_m']:.3f} m, r={row['dem_correlation']:.3f}"
        )
    print(f"\nOutputs: {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
