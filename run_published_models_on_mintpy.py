#!/usr/bin/env python3
"""Run published DEM-error method reproductions on a MintPy ifgramStack."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np

import mintpy_dem_error_ifg as current
import published_dem_error_models as published


METHODS = (
    "hybrid_optimal_2026",
    "fractal_2015_adapted",
    "ica_2019",
    "adaptive_ht_2021",
    "igs_cmaes_2021_equivalent",
    "pgdc_2025",
    "dynamic_height_2025",
)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Apply published DEM-error estimators to a MintPy interferogram stack.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("ifgram_stack")
    parser.add_argument("-g", "--geometry", required=True)
    parser.add_argument("-o", "--output-dir", default="published_dem_error_ifg")
    parser.add_argument("--models", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--mask")
    parser.add_argument("--mask-dataset")
    parser.add_argument("--min-coherence", type=float, default=0.5)
    parser.add_argument("--pgdc-threshold", type=float, default=0.5)
    parser.add_argument("--pgdc-max-temporal-days", type=float, default=60.0)
    parser.add_argument("--pgdc-dem-step", type=float, default=1.0)
    parser.add_argument("--dem-bound", type=float, default=200.0)
    parser.add_argument("--velocity-bound", type=float, default=20.0)
    parser.add_argument("--hybrid-graph-lambda", type=float, default=1.0)
    parser.add_argument("--hybrid-dynamic-alpha", type=float, default=0.01)
    parser.add_argument("--hybrid-min-height-change", type=float, default=2.0)
    parser.add_argument("--hybrid-max-height-change", type=float, default=100.0)
    parser.add_argument("--hybrid-min-dynamic-component", type=int, default=9)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _json_value(value):
    if isinstance(value, np.ndarray):
        if value.size <= 100:
            return value.tolist()
        finite = value[np.isfinite(value)] if np.issubdtype(value.dtype, np.number) else np.asarray([])
        return {
            "shape": list(value.shape),
            "mean": float(np.mean(finite)) if finite.size else None,
            "min": float(np.min(finite)) if finite.size else None,
            "max": float(np.max(finite)) if finite.size else None,
        }
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_outputs(
    input_stack: Path,
    output_dir: Path,
    model: str,
    original_phase: np.ndarray,
    dem_error: np.ndarray,
    dem_phase: np.ndarray,
    diagnostics: dict,
    overwrite: bool,
) -> list[Path]:
    corrected_path = output_dir / f"ifgramStack_demErr_{model}.h5"
    component_path = output_dir / f"demComponent_{model}.h5"
    diagnostics_path = output_dir / f"diagnostics_{model}.json"
    for path in (corrected_path, component_path, diagnostics_path):
        if path.exists() and not overwrite:
            raise FileExistsError(f"output exists, use --overwrite: {path}")
    shutil.copy2(input_stack, corrected_path)
    corrected = original_phase - dem_phase
    with h5py.File(corrected_path, "r+") as file:
        file["unwrapPhase"][:] = corrected.astype(file["unwrapPhase"].dtype)
        file.attrs["DEM_ERROR_MODEL"] = model
        file.attrs["DEM_ERROR_REPRODUCTION"] = diagnostics.get("reproduction_status", "")
    with h5py.File(component_path, "w") as file:
        file.attrs["FILE_TYPE"] = "demErrorComponent"
        file.attrs["DEM_ERROR_MODEL"] = model
        file.create_dataset("demError", data=dem_error.astype(np.float32), compression="gzip")
        file.create_dataset("demPhase", data=dem_phase.astype(np.float32), compression="gzip")
        diagnostic_datasets = {
            "dem_error_before": "demErrorBefore",
            "height_change": "heightChange",
            "source_model": "sourceModel",
            "dynamic_mask": "dynamicMask",
            "change_index": "changeIndex",
            "deformation_model_order": "deformationModelOrder",
            "dem_error_std": "demErrorStd",
            "dem_information": "demInformation",
            "dem_observability": "demObservability",
            "graph_blend": "graphBlend",
            "gdc": "gdc",
            "pgdc_detected": "pgdcDetected",
            "active_mask": "activeMask",
            "unwrap_cycle_fraction": "unwrapCycleFraction",
            "candidate_spread": "candidateSpread",
        }
        for key, dataset_name in diagnostic_datasets.items():
            value = diagnostics.get(key)
            if isinstance(value, np.ndarray) and value.shape == dem_error.shape:
                file.create_dataset(dataset_name, data=value, compression="gzip")
    diagnostics_path.write_text(
        json.dumps({key: _json_value(value) for key, value in diagnostics.items()}, indent=2),
        encoding="utf-8",
    )
    return [corrected_path, component_path, diagnostics_path]


def run(args: argparse.Namespace) -> list[Path]:
    input_stack = Path(args.ifgram_stack).resolve()
    geometry_path = Path(args.geometry).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.dem_bound <= 0 or args.velocity_bound <= 0:
        raise ValueError("--dem-bound and --velocity-bound must be positive")
    mask_handle = h5py.File(args.mask, "r") if args.mask else None
    outputs: list[Path] = []
    try:
        with h5py.File(input_stack, "r") as stack, h5py.File(geometry_path, "r") as geometry:
            date_pairs = current.read_date_pairs(stack["date"])
            original_phase = np.asarray(stack["unwrapPhase"], dtype=np.float64)
            wrapped_phase = (
                np.asarray(stack["wrapPhase"], dtype=np.float64)
                if "wrapPhase" in stack
                else np.angle(np.exp(1j * original_phase))
            )
            coherence = (
                np.asarray(stack["coherence"], dtype=np.float64)
                if "coherence" in stack
                else np.ones_like(original_phase)
            )
            num_ifgram, length, width = original_phase.shape
            selected = (
                np.asarray(stack["dropIfgram"][:], dtype=bool)
                if "dropIfgram" in stack
                else np.ones(num_ifgram, dtype=bool)
            )
            baseline_mode, baseline_source = current.inspect_baseline_mode(
                stack, geometry, date_pairs
            )
            bperp = current.read_bperp_block(
                baseline_mode,
                baseline_source,
                geometry,
                date_pairs,
                slice(0, length),
                width,
            )
            incidence = np.asarray(geometry["incidenceAngle"], dtype=np.float64).reshape(-1)
            slant_range = np.asarray(
                geometry["slantRangeDistance"], dtype=np.float64
            ).reshape(-1)
            inverse_geometry = np.divide(
                1.0,
                slant_range * np.sin(np.deg2rad(incidence)),
                out=np.full_like(slant_range, np.nan),
                where=(slant_range > 0) & (np.sin(np.deg2rad(incidence)) != 0),
            )
            coefficient = (
                -4.0 * np.pi / float(stack.attrs["WAVELENGTH"])
            ) * bperp * inverse_geometry[None, :]
            spatial_mask = np.isfinite(inverse_geometry)
            if mask_handle is not None:
                dataset_name = current.choose_mask_dataset(mask_handle, args.mask_dataset)
                spatial_mask &= np.asarray(mask_handle[dataset_name][:], dtype=bool).reshape(-1)
            weights = coherence.reshape(num_ifgram, -1) ** 2
            weights[coherence.reshape(num_ifgram, -1) < args.min_coherence] = 0.0
            weights[:, ~spatial_mask] = 0.0
            phase_flat = original_phase.reshape(num_ifgram, -1)
            wrapped_flat = wrapped_phase.reshape(num_ifgram, -1)
            terrain = (
                np.asarray(geometry["height"], dtype=np.float64)
                if "height" in geometry
                else np.zeros((length, width), dtype=np.float64)
            )
            dates = sorted({date for pair in date_pairs for date in pair})
            date_lookup = {date: index for index, date in enumerate(dates)}
            temporal_days = np.asarray(
                [
                    (datetime.strptime(slave[:8], "%Y%m%d") - datetime.strptime(master[:8], "%Y%m%d")).days
                    for master, slave in date_pairs
                ],
                dtype=np.float64,
            )
            temporal_years = temporal_days / 365.25

            fit_phase = phase_flat[selected]
            fit_wrapped = wrapped_flat[selected]
            fit_coefficient = coefficient[selected]
            fit_weights = weights[selected]
            fit_pairs = [pair for pair, keep in zip(date_pairs, selected) if keep]
            fit_coherence = coherence[selected].reshape(np.sum(selected), -1)
            fit_bperp = np.nanmedian(bperp[selected], axis=1)
            fit_days = temporal_days[selected]

            linear_design, _ = current.build_deformation_design(fit_pairs, 1)
            linear_solution, linear_valid = current.solve_model_block(
                fit_phase,
                fit_coefficient,
                linear_design,
                np.ones(len(fit_pairs), dtype=bool),
                fit_weights,
                robust=True,
                huber_iterations=8,
            )
            linear_dem = linear_solution[:, 0]

            for model in args.models:
                print(f"running {model}")
                if model == "hybrid_optimal_2026":
                    result = published.hybrid_optimal_2026(
                        fit_phase.reshape(np.sum(selected), length, width),
                        fit_coefficient.reshape(np.sum(selected), length, width),
                        fit_pairs,
                        fit_coherence.reshape(np.sum(selected), length, width),
                        wrapped_phase=fit_wrapped.reshape(
                            np.sum(selected), length, width
                        ),
                        terrain=terrain,
                        min_coherence=args.min_coherence,
                        pgdc_threshold=args.pgdc_threshold,
                        dynamic_alpha=args.hybrid_dynamic_alpha,
                        minimum_height_change=args.hybrid_min_height_change,
                        maximum_height_change=args.hybrid_max_height_change,
                        minimum_dynamic_component_pixels=args.hybrid_min_dynamic_component,
                        graph_lambda=args.hybrid_graph_lambda,
                        velocity_bounds=(-args.velocity_bound, args.velocity_bound),
                        dem_bounds=(-args.dem_bound, args.dem_bound),
                    )
                    result.dem_error = result.dem_error.reshape(-1)
                    result.valid = result.valid.reshape(-1)
                    status = "new_hybrid"
                elif model == "ica_2019":
                    result = published.nonparametric_ica_2019(
                        fit_phase, fit_coefficient, fit_pairs, fit_weights
                    )
                    status = "strict"
                elif model == "adaptive_ht_2021":
                    result = published.adaptive_hypothesis_2021(
                        fit_phase, fit_coefficient, fit_pairs, fit_weights
                    )
                    status = "strict"
                elif model == "igs_cmaes_2021_equivalent":
                    result = published.igs_cmaes_2021_equivalent(
                        fit_wrapped,
                        fit_coefficient,
                        temporal_years[selected],
                        fit_coherence,
                        velocity_bounds=(-args.velocity_bound, args.velocity_bound),
                        dem_bounds=(-args.dem_bound, args.dem_bound),
                        loss_threshold=0.3,
                    )
                    status = "equivalent"
                elif model == "pgdc_2025":
                    gdc, detected, pgdc_selected = published.pgdc_detect_2025(
                        fit_wrapped.reshape(np.sum(selected), length, width),
                        fit_bperp,
                        fit_days,
                        fit_coherence.reshape(np.sum(selected), length, width),
                        threshold=args.pgdc_threshold,
                        max_temporal_days=args.pgdc_max_temporal_days,
                    )
                    result = published.pgdc_estimate_2025(
                        fit_wrapped.reshape(np.sum(selected), length, width),
                        fit_coefficient.reshape(np.sum(selected), length, width),
                        fit_pairs,
                        detected,
                        fit_coherence.reshape(np.sum(selected), length, width),
                        dem_bounds=(-args.dem_bound, args.dem_bound),
                        dem_step=args.pgdc_dem_step,
                    )
                    result.dem_error = result.dem_error.reshape(-1)
                    result.valid = result.valid.reshape(-1)
                    result.diagnostics.update(
                        {"gdc_mean": float(np.nanmean(gdc)), "gradient_ifgrams": pgdc_selected}
                    )
                    status = "equivalent"
                elif model == "dynamic_height_2025":
                    result = published.dynamic_height_sbas_2025(
                        fit_phase, fit_coefficient, fit_pairs, fit_weights
                    )
                    status = "strict"
                elif model == "fractal_2015_adapted":
                    proxy = np.nanmean(fit_coherence, axis=0).reshape(length, width)
                    regularized = published.fractal_surface_regularize_2015_adapted(
                        linear_dem.reshape(length, width),
                        proxy,
                        linear_valid.reshape(length, width) & spatial_mask.reshape(length, width),
                    )
                    result = published.PublishedResult(
                        regularized.dem_error.reshape(-1),
                        regularized.valid.reshape(-1),
                        regularized.diagnostics,
                    )
                    status = "adapted"
                else:
                    raise ValueError(f"unsupported model: {model}")

                dem_error = np.asarray(result.dem_error, dtype=np.float64).reshape(-1)
                dem_error[~spatial_mask] = np.nan
                if model in {"dynamic_height_2025", "hybrid_optimal_2026"}:
                    before = np.asarray(result.diagnostics["dem_error_before"]).reshape(-1)
                    after = np.asarray(result.diagnostics["dem_error_after"]).reshape(-1)
                    change_index = np.asarray(result.diagnostics["change_index"]).reshape(-1)
                    dynamic_pixels = (
                        np.asarray(result.diagnostics["dynamic_mask"]).reshape(-1)
                        if model == "hybrid_optimal_2026"
                        else result.valid
                    )
                    dem_phase = coefficient * after[None, :]
                    spanning_count = 0
                    for ifgram_index, (master, slave) in enumerate(date_pairs):
                        master_index = date_lookup[master]
                        slave_index = date_lookup[slave]
                        before_pixels = dynamic_pixels & (slave_index < change_index)
                        after_pixels = dynamic_pixels & (master_index >= change_index)
                        spanning_pixels = (
                            dynamic_pixels & ~before_pixels & ~after_pixels
                        )
                        dem_phase[ifgram_index, before_pixels] = (
                            coefficient[ifgram_index, before_pixels] * before[before_pixels]
                        )
                        dem_phase[ifgram_index, after_pixels] = (
                            coefficient[ifgram_index, after_pixels] * after[after_pixels]
                        )
                        dem_phase[ifgram_index, spanning_pixels] = 0.0
                        spanning_count += int(np.sum(spanning_pixels & result.valid))
                    result.diagnostics["uncorrected_spanning_observations"] = spanning_count
                else:
                    dem_phase = coefficient * dem_error[None, :]
                dem_phase[:, ~np.isfinite(dem_error)] = 0.0
                diagnostics = {
                    "model": model,
                    "reproduction_status": status,
                    "fit_interferograms": int(np.sum(selected)),
                    "valid_pixels": int(np.sum(result.valid)),
                    **result.diagnostics,
                }
                outputs.extend(
                    _write_outputs(
                        input_stack,
                        output_dir,
                        model,
                        original_phase,
                        dem_error.reshape(length, width),
                        dem_phase.reshape(num_ifgram, length, width),
                        diagnostics,
                        args.overwrite,
                    )
                )
    finally:
        if mask_handle is not None:
            mask_handle.close()
    return outputs


def main(argv: list[str] | None = None) -> int:
    args = create_parser().parse_args(argv)
    outputs = run(args)
    print("outputs:")
    for path in outputs:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
