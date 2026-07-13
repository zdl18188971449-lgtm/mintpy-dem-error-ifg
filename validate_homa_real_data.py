#!/usr/bin/env python3
"""Validate HOMA-DEM with held-out interferograms and external elevation."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

import mintpy_dem_error_ifg as current
import published_dem_error_models as published
import run_published_model_benchmark as benchmark


METHOD_LABELS = {
    "uncorrected": "No DEM correction",
    "linear_huber": "Linear Huber",
    "adaptive_ht_2021": "Adaptive HT",
    "current_graph": "Current graph",
    "hybrid_optimal_2026": "HOMA-DEM",
    "mintpy_dem_error": "MintPy demErr",
    "homa_no_ica": "HOMA-DEM, no ICA",
    "homa_no_dynamic": "HOMA-DEM, no dynamic",
    "homa_no_pgdc": "HOMA-DEM, no PGDC",
    "homa_no_igs": "HOMA-DEM, no IGS",
    "homa_no_graph": "HOMA-DEM, no graph",
}

ABLATION_VARIANTS = {
    "homa_no_ica": {"enable_ica": False},
    "homa_no_dynamic": {"enable_dynamic": False},
    "homa_no_pgdc": {"enable_pgdc": False},
    "homa_no_igs": {"enable_igs": False},
    "homa_no_graph": {"enable_graph": False},
}


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Real MintPy holdout and external-elevation validation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("ifgram_stack")
    parser.add_argument("-g", "--geometry", required=True)
    parser.add_argument("--mask")
    parser.add_argument("--mask-dataset")
    parser.add_argument("--mintpy-dem-reference")
    parser.add_argument("--lookup", help="MintPy geometryGeo.h5 with radar lookup coordinates")
    parser.add_argument(
        "--external-dem",
        help="Georeferenced external DEM/DSM raster path or COG URL",
    )
    parser.add_argument("-o", "--output-dir", default="homa_real_validation")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--min-coherence", type=float, default=0.5)
    parser.add_argument("--dem-bound", type=float, default=200.0)
    parser.add_argument("--velocity-bound", type=float, default=20.0)
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 7.0,
            "axes.titlesize": 7.5,
            "axes.labelsize": 7.2,
            "xtick.labelsize": 6.3,
            "ytick.labelsize": 6.3,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
        }
    )


def _mean_ci(values: np.ndarray) -> tuple[float, float, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return np.nan, np.nan, np.nan
    mean = float(np.mean(finite))
    if finite.size < 2:
        return mean, np.nan, np.nan
    half_width = float(
        stats.t.ppf(0.975, finite.size - 1)
        * np.std(finite, ddof=1)
        / np.sqrt(finite.size)
    )
    return mean, mean - half_width, mean + half_width


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _network_connected(pairs: list[tuple[str, str]]) -> bool:
    dates = sorted({date for pair in pairs for date in pair})
    if not dates:
        return False
    neighbors = {date: set() for date in dates}
    for master, slave in pairs:
        neighbors[master].add(slave)
        neighbors[slave].add(master)
    visited = {dates[0]}
    pending = [dates[0]]
    while pending:
        date = pending.pop()
        for neighbor in neighbors[date] - visited:
            visited.add(neighbor)
            pending.append(neighbor)
    return len(visited) == len(dates)


def _load_inputs(args: argparse.Namespace) -> dict:
    stack_path = Path(args.ifgram_stack).resolve()
    geometry_path = Path(args.geometry).resolve()
    mask_handle = h5py.File(args.mask, "r") if args.mask else None
    try:
        with h5py.File(stack_path, "r") as stack, h5py.File(
            geometry_path, "r"
        ) as geometry:
            phase = np.asarray(stack["unwrapPhase"], dtype=np.float64)
            coherence = (
                np.asarray(stack["coherence"], dtype=np.float64)
                if "coherence" in stack
                else np.ones_like(phase)
            )
            wrapped = (
                np.asarray(stack["wrapPhase"], dtype=np.float64)
                if "wrapPhase" in stack
                else np.angle(np.exp(1j * phase))
            )
            pairs = current.read_date_pairs(stack["date"])
            selected = (
                np.asarray(stack["dropIfgram"], dtype=bool)
                if "dropIfgram" in stack
                else np.ones(phase.shape[0], dtype=bool)
            )
            _, length, width = phase.shape
            baseline_mode, baseline_source = current.inspect_baseline_mode(
                stack, geometry, pairs
            )
            bperp = current.read_bperp_block(
                baseline_mode,
                baseline_source,
                geometry,
                pairs,
                slice(0, length),
                width,
            )
            incidence = np.asarray(
                geometry["incidenceAngle"], dtype=np.float64
            ).reshape(-1)
            slant_range = np.asarray(
                geometry["slantRangeDistance"], dtype=np.float64
            ).reshape(-1)
            sine = np.sin(np.deg2rad(incidence))
            inverse_geometry = np.divide(
                1.0,
                slant_range * sine,
                out=np.full_like(slant_range, np.nan),
                where=(slant_range > 0) & (sine != 0),
            )
            coefficient = (
                -4.0 * np.pi / float(stack.attrs["WAVELENGTH"])
            ) * bperp * inverse_geometry[None, :]
            spatial_mask = np.isfinite(inverse_geometry)
            if mask_handle is not None:
                dataset = current.choose_mask_dataset(
                    mask_handle, args.mask_dataset
                )
                spatial_mask &= np.asarray(
                    mask_handle[dataset], dtype=bool
                ).reshape(-1)
            terrain = (
                np.asarray(geometry["height"], dtype=np.float64)
                if "height" in geometry
                else np.zeros((length, width), dtype=np.float64)
            )
            weights = coherence.reshape(phase.shape[0], -1) ** 2
            weights[coherence.reshape(phase.shape[0], -1) < args.min_coherence] = 0.0
            weights[:, ~spatial_mask] = 0.0
            attributes = {key: stack.attrs[key] for key in stack.attrs}
    finally:
        if mask_handle is not None:
            mask_handle.close()
    return {
        "stack_path": stack_path,
        "phase": phase,
        "wrapped": wrapped,
        "coherence": coherence,
        "pairs": pairs,
        "selected": selected,
        "coefficient": coefficient.reshape(phase.shape),
        "weights": weights.reshape(phase.shape),
        "terrain": terrain,
        "spatial_mask": spatial_mask.reshape(length, width),
        "attributes": attributes,
    }


def _fit_dem_methods(data: dict, train: np.ndarray, args: argparse.Namespace) -> dict:
    phase = data["phase"][train]
    coefficient = data["coefficient"][train]
    coherence = data["coherence"][train]
    wrapped = data["wrapped"][train]
    weights = data["weights"][train]
    pairs = [pair for pair, keep in zip(data["pairs"], train) if keep]
    terrain = data["terrain"]
    shape = terrain.shape
    phase_flat = phase.reshape(phase.shape[0], -1)
    coefficient_flat = coefficient.reshape(coefficient.shape[0], -1)
    weights_flat = weights.reshape(weights.shape[0], -1)
    linear = benchmark._linear_huber(
        phase_flat, coefficient_flat, pairs, weights_flat
    ).reshape(shape)
    adaptive = published.adaptive_hypothesis_2021(
        phase_flat, coefficient_flat, pairs, weights_flat
    ).dem_error.reshape(shape)
    graph = benchmark._current_adaptive_graph(
        phase_flat, coefficient_flat, pairs, weights_flat, terrain
    ).reshape(shape)
    homa = published.hybrid_optimal_2026(
        phase,
        coefficient,
        pairs,
        coherence,
        wrapped_phase=wrapped,
        terrain=terrain,
        min_coherence=args.min_coherence,
        velocity_bounds=(-args.velocity_bound, args.velocity_bound),
        dem_bounds=(-args.dem_bound, args.dem_bound),
    )
    estimates = {
        "uncorrected": None,
        "linear_huber": linear,
        "adaptive_ht_2021": adaptive,
        "current_graph": graph,
        "hybrid_optimal_2026": homa,
    }
    for method, switches in ABLATION_VARIANTS.items():
        estimates[method] = published.hybrid_optimal_2026(
            phase,
            coefficient,
            pairs,
            coherence,
            wrapped_phase=wrapped,
            terrain=terrain,
            min_coherence=args.min_coherence,
            velocity_bounds=(-args.velocity_bound, args.velocity_bound),
            dem_bounds=(-args.dem_bound, args.dem_bound),
            **switches,
        )
    return estimates


def _homa_dem_phase(
    result: published.PublishedResult,
    coefficient: np.ndarray,
    pairs: list[tuple[str, str]],
) -> np.ndarray:
    shape = coefficient.shape
    coefficient_flat = coefficient.reshape(shape[0], -1)
    before = np.asarray(result.diagnostics["dem_error_before"]).reshape(-1)
    after = np.asarray(result.diagnostics["dem_error_after"]).reshape(-1)
    dynamic = np.asarray(result.diagnostics["dynamic_mask"]).reshape(-1)
    change_index = np.asarray(result.diagnostics["change_index"]).reshape(-1)
    dates = sorted({date for pair in pairs for date in pair})
    date_index = {date: index for index, date in enumerate(dates)}
    dem_phase = coefficient_flat * after[None, :]
    for index, (master, slave) in enumerate(pairs):
        before_pixels = dynamic & (date_index[slave] < change_index)
        after_pixels = dynamic & (date_index[master] >= change_index)
        spanning = dynamic & ~before_pixels & ~after_pixels
        dem_phase[index, before_pixels] = (
            coefficient_flat[index, before_pixels] * before[before_pixels]
        )
        dem_phase[index, after_pixels] = (
            coefficient_flat[index, after_pixels] * after[after_pixels]
        )
        dem_phase[index, spanning] = 0.0
    return dem_phase.reshape(shape)


def _profile_residual(
    phase: np.ndarray,
    dem_phase: np.ndarray,
    weights: np.ndarray,
    design: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
) -> np.ndarray:
    observations = (phase - dem_phase).reshape(phase.shape[0], -1)
    weight_flat = weights.reshape(weights.shape[0], -1)
    residual = np.full((int(np.sum(test)), observations.shape[1]), np.nan)
    train_design = design[train]
    test_design = design[test]
    for pixel in range(observations.shape[1]):
        valid = (
            np.isfinite(observations[:, pixel])
            & np.isfinite(weight_flat[:, pixel])
            & (weight_flat[:, pixel] > 0)
        )
        fit = train & valid
        evaluate = test & valid
        if np.sum(fit) < train_design.shape[1] or not np.any(evaluate):
            continue
        sqrt_weight = np.sqrt(weight_flat[fit, pixel])
        solution, _, rank, _ = np.linalg.lstsq(
            design[fit] * sqrt_weight[:, None],
            observations[fit, pixel] * sqrt_weight,
            rcond=None,
        )
        if rank != design.shape[1]:
            continue
        test_positions = np.flatnonzero(test)
        for output_index, ifgram_index in enumerate(test_positions):
            if evaluate[ifgram_index]:
                residual[output_index, pixel] = (
                    observations[ifgram_index, pixel]
                    - test_design[output_index] @ solution
                )
    return residual


def _weighted_metrics(residual: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    valid = np.isfinite(residual) & np.isfinite(weights) & (weights > 0)
    if not np.any(valid):
        return {"rmse_rad": np.nan, "mae_rad": np.nan, "count": 0}
    value = residual[valid]
    weight = weights[valid]
    return {
        "rmse_rad": float(np.sqrt(np.sum(weight * value**2) / np.sum(weight))),
        "mae_rad": float(np.sum(weight * np.abs(value)) / np.sum(weight)),
        "count": int(np.sum(valid)),
    }


def _holdout_validation(data: dict, args: argparse.Namespace) -> tuple[list[dict], list[dict]]:
    selected_indices = np.flatnonzero(data["selected"])
    if args.folds < 2 or args.folds > selected_indices.size:
        raise ValueError("--folds must be between 2 and the selected interferogram count")
    fold_ids = np.arange(selected_indices.size) % args.folds
    design, _ = current.build_deformation_design(data["pairs"], 3, periods=[1.0])
    rows = []
    for fold in range(args.folds):
        test = np.zeros(len(data["pairs"]), dtype=bool)
        test[selected_indices[fold_ids == fold]] = True
        train = data["selected"] & ~test
        training_pairs = [pair for pair, keep in zip(data["pairs"], train) if keep]
        if not _network_connected(training_pairs):
            raise ValueError(f"training network is disconnected in fold {fold}")
        estimates = _fit_dem_methods(data, train, args)
        for method, estimate in estimates.items():
            if method == "hybrid_optimal_2026" or method in ABLATION_VARIANTS:
                dem_phase = _homa_dem_phase(
                    estimate, data["coefficient"], data["pairs"]
                )
            elif estimate is None:
                dem_phase = np.zeros_like(data["phase"])
            else:
                dem_phase = data["coefficient"] * estimate[None, :, :]
            residual = _profile_residual(
                data["phase"],
                dem_phase,
                data["weights"],
                design,
                train,
                test,
            )
            metrics = _weighted_metrics(residual, data["weights"][test].reshape(residual.shape))
            rows.append(
                {
                    "fold": fold,
                    "method": method,
                    "label": METHOD_LABELS[method],
                    "train_ifgrams": int(np.sum(train)),
                    "test_ifgrams": int(np.sum(test)),
                    **metrics,
                }
            )
    summary = []
    for method in METHOD_LABELS:
        records = [row for row in rows if row["method"] == method]
        if not records:
            continue
        rmse = np.asarray([row["rmse_rad"] for row in records])
        mean, low, high = _mean_ci(rmse)
        summary.append(
            {
                "method": method,
                "label": METHOD_LABELS[method],
                "folds": len(records),
                "mean_holdout_rmse_rad": mean,
                "std_holdout_rmse_rad": float(np.nanstd(rmse, ddof=1)),
                "mean_ci95_low_rad": max(low, 0.0),
                "mean_ci95_high_rad": high,
                "median_holdout_rmse_rad": float(np.nanmedian(rmse)),
            }
        )
    return rows, summary


def _read_reference_subset(path: Path, data: dict) -> np.ndarray:
    with h5py.File(path, "r") as file:
        datasets = []
        file.visititems(
            lambda name, item: datasets.append(name)
            if isinstance(item, h5py.Dataset) and item.ndim == 2
            else None
        )
        if not datasets:
            raise ValueError(f"no 2D dataset found in {path}")
        reference = np.asarray(file[datasets[0]], dtype=np.float64)
    shape = data["terrain"].shape
    if reference.shape == shape:
        return reference
    attributes = data["attributes"]
    x0 = int(attributes.get("SUBSET_XMIN", 0))
    y0 = int(attributes.get("SUBSET_YMIN", 0))
    subset = reference[y0 : y0 + shape[0], x0 : x0 + shape[1]]
    if subset.shape != shape:
        raise ValueError("MintPy DEM reference does not cover the input subset")
    return subset


def _relative_map_metrics(reference: np.ndarray, estimate: np.ndarray) -> dict[str, float]:
    valid = np.isfinite(reference) & np.isfinite(estimate)
    if not np.any(valid):
        return {"offset_m": np.nan, "rmse_m": np.nan, "mae_m": np.nan, "correlation": np.nan}
    offset = float(np.median(reference[valid] - estimate[valid]))
    residual = estimate[valid] + offset - reference[valid]
    correlation = (
        float(np.corrcoef(reference[valid], estimate[valid])[0, 1])
        if np.std(reference[valid]) > 0 and np.std(estimate[valid]) > 0
        else np.nan
    )
    return {
        "offset_m": offset,
        "rmse_m": float(np.sqrt(np.mean(residual**2))),
        "mae_m": float(np.mean(np.abs(residual))),
        "correlation": correlation,
    }


def _lookup_external_samples(
    lookup_path: Path, external_path: str | Path, data: dict
) -> list[dict]:
    try:
        import rasterio
        from pyproj import Transformer
    except ImportError as error:
        raise RuntimeError("external validation requires rasterio and pyproj") from error
    attributes = data["attributes"]
    radar_x0 = int(attributes.get("SUBSET_XMIN", 0))
    radar_y0 = int(attributes.get("SUBSET_YMIN", 0))
    length, width = data["terrain"].shape
    best: dict[tuple[int, int], tuple[float, float, float]] = {}
    with h5py.File(lookup_path, "r") as lookup:
        azimuth = lookup["azimuthCoord"]
        range_coordinate = lookup["rangeCoord"]
        x_first = float(lookup.attrs["X_FIRST"])
        x_step = float(lookup.attrs["X_STEP"])
        y_first = float(lookup.attrs["Y_FIRST"])
        y_step = float(lookup.attrs["Y_STEP"])
        for start in range(0, azimuth.shape[0], 256):
            stop = min(start + 256, azimuth.shape[0])
            azimuth_block = np.asarray(azimuth[start:stop], dtype=np.float64)
            range_block = np.asarray(range_coordinate[start:stop], dtype=np.float64)
            local_row = np.rint(azimuth_block).astype(np.int64) - radar_y0
            local_column = np.rint(range_block).astype(np.int64) - radar_x0
            valid = (
                np.isfinite(azimuth_block)
                & np.isfinite(range_block)
                & (local_row >= 0)
                & (local_row < length)
                & (local_column >= 0)
                & (local_column < width)
            )
            for block_row, geo_column in zip(*np.where(valid)):
                row = int(local_row[block_row, geo_column])
                column = int(local_column[block_row, geo_column])
                distance = float(
                    (azimuth_block[block_row, geo_column] - (radar_y0 + row)) ** 2
                    + (range_block[block_row, geo_column] - (radar_x0 + column)) ** 2
                )
                geo_row = start + block_row
                longitude = x_first + geo_column * x_step
                latitude = y_first + geo_row * y_step
                key = (row, column)
                if key not in best or distance < best[key][0]:
                    best[key] = (distance, longitude, latitude)
    points = [(value[1], value[2]) for _, value in sorted(best.items())]
    with rasterio.open(external_path) as raster:
        if raster.crs is None:
            raise ValueError("external DEM has no coordinate reference system")
        transformer = Transformer.from_crs("EPSG:4326", raster.crs, always_xy=True)
        raster_points = [transformer.transform(lon, lat) for lon, lat in points]
        external_values = [float(value[0]) for value in raster.sample(raster_points)]
        nodata = raster.nodata
    rows = []
    for ((radar_row, radar_column), (_, longitude, latitude)), external in zip(
        sorted(best.items()), external_values
    ):
        if nodata is not None and external == nodata:
            external = np.nan
        rows.append(
            {
                "radar_row": radar_row,
                "radar_column": radar_column,
                "longitude": longitude,
                "latitude": latitude,
                "external_height_m": external,
                "input_height_m": float(data["terrain"][radar_row, radar_column]),
                "calibration": (radar_row + radar_column) % 2 == 0,
            }
        )
    return rows


def _external_metrics(
    samples: list[dict], corrections: dict[str, np.ndarray]
) -> tuple[list[dict], list[dict]]:
    details = []
    metrics = []
    for method, correction in corrections.items():
        predicted = np.asarray(
            [
                sample["input_height_m"]
                + correction[sample["radar_row"], sample["radar_column"]]
                for sample in samples
            ],
            dtype=np.float64,
        )
        external = np.asarray([sample["external_height_m"] for sample in samples])
        calibration = np.asarray([sample["calibration"] for sample in samples])
        valid = np.isfinite(predicted) & np.isfinite(external)
        calibration_valid = valid & calibration
        evaluation_valid = valid & ~calibration
        offset = float(np.median(external[calibration_valid] - predicted[calibration_valid]))
        residual = predicted[evaluation_valid] + offset - external[evaluation_valid]
        metrics.append(
            {
                "method": method,
                "label": METHOD_LABELS[method],
                "calibration_samples": int(np.sum(calibration_valid)),
                "evaluation_samples": int(np.sum(evaluation_valid)),
                "calibrated_vertical_offset_m": offset,
                "external_rmse_m": float(np.sqrt(np.mean(residual**2))),
                "external_mae_m": float(np.mean(np.abs(residual))),
            }
        )
        for sample_index, sample in enumerate(samples):
            details.append(
                {
                    **sample,
                    "method": method,
                    "predicted_height_m": predicted[sample_index],
                    "calibrated_height_m": predicted[sample_index] + offset,
                    "residual_m": predicted[sample_index] + offset - external[sample_index],
                }
            )
    return metrics, details


def _plot_results(
    holdout: list[dict], external: list[dict], output_dir: Path, dpi: int
) -> None:
    panels = 2 if external else 1
    figure, axes = plt.subplots(1, panels, figsize=(7.16, 3.8))
    axes = np.atleast_1d(axes)
    methods = [row["method"] for row in holdout]
    positions = np.arange(len(methods))[::-1]
    means = np.asarray([row["mean_holdout_rmse_rad"] for row in holdout])
    lows = np.asarray([row["mean_ci95_low_rad"] for row in holdout])
    highs = np.asarray([row["mean_ci95_high_rad"] for row in holdout])
    colors = ["#B23A2B" if method == "hybrid_optimal_2026" else "#777777" for method in methods]
    axes[0].barh(positions, means, color=colors, height=0.58)
    axes[0].errorbar(
        means,
        positions,
        xerr=np.vstack((means - lows, highs - means)),
        fmt="none",
        ecolor="black",
        capsize=2,
        linewidth=0.7,
    )
    axes[0].set_yticks(positions, [METHOD_LABELS[m] for m in methods])
    axes[0].set_xlabel("Held-out phase RMSE (rad), mean and 95% CI")
    axes[0].set_title("(a) MintPy network holdout", fontweight="bold")
    axes[0].spines[["top", "right"]].set_visible(False)
    if external:
        methods = [row["method"] for row in external]
        positions = np.arange(len(methods))[::-1]
        values = [row["external_rmse_m"] for row in external]
        colors = ["#B23A2B" if method == "hybrid_optimal_2026" else "#777777" for method in methods]
        axes[1].barh(positions, values, color=colors, height=0.58)
        axes[1].set_yticks(positions, [METHOD_LABELS[m] for m in methods])
        axes[1].set_xlabel("External elevation RMSE (m)")
        axes[1].set_title("(b) Independent DSM evaluation", fontweight="bold")
        axes[1].spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    for suffix in ("pdf", "svg"):
        figure.savefig(output_dir / f"homa_real_validation.{suffix}", bbox_inches="tight")
    figure.savefig(
        output_dir / "homa_real_validation.tiff",
        dpi=dpi,
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    figure.savefig(
        output_dir / "homa_real_validation.png", dpi=dpi, bbox_inches="tight"
    )
    plt.close(figure)


def main(argv: list[str] | None = None) -> int:
    args = create_parser().parse_args(argv)
    if bool(args.lookup) != bool(args.external_dem):
        raise ValueError("--lookup and --external-dem must be provided together")
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"output exists, use --overwrite: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    data = _load_inputs(args)
    holdout_rows, holdout_summary = _holdout_validation(data, args)
    _write_csv(output_dir / "mintpy_holdout_folds.csv", holdout_rows)
    _write_csv(output_dir / "mintpy_holdout_summary.csv", holdout_summary)

    full_train = data["selected"].copy()
    estimates = _fit_dem_methods(data, full_train, args)
    corrections = {
        "uncorrected": np.zeros_like(data["terrain"]),
        "linear_huber": estimates["linear_huber"],
        "adaptive_ht_2021": estimates["adaptive_ht_2021"],
        "current_graph": estimates["current_graph"],
        "hybrid_optimal_2026": estimates["hybrid_optimal_2026"].dem_error,
    }
    np.savez_compressed(output_dir / "real_dem_error_maps.npz", **corrections)

    reference_metrics = []
    if args.mintpy_dem_reference:
        reference = _read_reference_subset(
            Path(args.mintpy_dem_reference).resolve(), data
        )
        corrections["mintpy_dem_error"] = reference
        for method, estimate in corrections.items():
            if method == "uncorrected":
                continue
            reference_metrics.append(
                {
                    "method": method,
                    "label": METHOD_LABELS[method],
                    **_relative_map_metrics(reference, estimate),
                }
            )
        _write_csv(output_dir / "mintpy_reference_agreement.csv", reference_metrics)

    external_metrics = []
    if args.lookup and args.external_dem:
        samples = _lookup_external_samples(
            Path(args.lookup).resolve(), args.external_dem, data
        )
        external_metrics, details = _external_metrics(samples, corrections)
        _write_csv(output_dir / "external_elevation_metrics.csv", external_metrics)
        _write_csv(output_dir / "external_elevation_samples.csv", details)

    (output_dir / "validation_config.json").write_text(
        json.dumps(
            {
                "ifgram_stack": str(data["stack_path"]),
                "geometry": str(Path(args.geometry).resolve()),
                "mask": str(Path(args.mask).resolve()) if args.mask else None,
                "folds": args.folds,
                "selected_ifgrams": int(np.sum(data["selected"])),
                "mintpy_dem_reference": args.mintpy_dem_reference,
                "lookup": args.lookup,
                "external_dem": args.external_dem,
                "external_offset_calibration": "checkerboard half-sample median; evaluated on disjoint half",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _configure_style()
    _plot_results(holdout_summary, external_metrics, output_dir, args.dpi)
    print(f"real-data validation written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
