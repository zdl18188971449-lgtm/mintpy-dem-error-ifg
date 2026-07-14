#!/usr/bin/env python3
"""Run MintPy DEM-error correction with a replaceable numerical estimator.

All data preparation, geometry handling, temporal models, block splitting,
metadata, correction, residual generation, and HDF5 writing are delegated to
the installed MintPy implementation.  Only estimate_dem_error() is replaced
when ``--estimator homa_huber`` is selected.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

import numpy as np
from scipy import linalg


ESTIMATORS = ("mintpy_ols", "mintpy_ols_batch", "homa_huber")


def _solve_batched_weighted_lstsq(
    design: np.ndarray,
    observations: np.ndarray,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Solve independent, small least-squares systems with column scaling."""
    if weights is None:
        weighted_design = design
        weighted_observations = observations
    else:
        sqrt_weight = np.sqrt(weights)
        weighted_design = design * sqrt_weight[:, :, None]
        weighted_observations = observations * sqrt_weight

    column_scale = np.linalg.norm(weighted_design, axis=1)
    column_scale = np.maximum(column_scale, np.finfo(np.float64).tiny)
    scaled_design = weighted_design / column_scale[:, None, :]
    normal_matrix = np.einsum(
        "bnk,bnl->bkl",
        scaled_design,
        scaled_design,
        optimize=True,
    )
    normal_rhs = np.einsum(
        "bnk,bn->bk",
        scaled_design,
        weighted_observations,
        optimize=True,
    )
    scaled_solution = np.linalg.solve(normal_matrix, normal_rhs[:, :, None])[:, :, 0]
    return scaled_solution / column_scale


def estimate_dem_error_huber_batch(
    ts0: np.ndarray,
    G_geom0: np.ndarray,
    G_defo0: np.ndarray,
    tbase: np.ndarray,
    date_flag: np.ndarray | None = None,
    phase_velocity: bool = False,
    *,
    huber_delta: float = 1.345,
    max_iterations: int = 8,
    tolerance: float = 1e-5,
    pixel_batch_size: int = 32768,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized Huber IRLS for pixel-wise MintPy geometry matrices."""
    ts0 = np.asarray(ts0)
    G_geom0 = np.asarray(G_geom0)
    G_defo0 = np.asarray(G_defo0)
    tbase = np.asarray(tbase)
    if ts0.ndim == 1:
        ts0 = ts0.reshape(-1, 1)
    if G_geom0.ndim == 1:
        G_geom0 = G_geom0.reshape(-1, 1)
    if G_geom0.shape[1] == 1 and ts0.shape[1] > 1:
        G_geom0 = np.repeat(G_geom0, ts0.shape[1], axis=1)
    if date_flag is None:
        date_flag = np.ones(ts0.shape[0], dtype=np.bool_)

    num_date, num_pixel = ts0.shape
    num_param = G_defo0.shape[1] + 1 - int(phase_velocity)
    delta_z = np.empty(num_pixel, dtype=np.float64)
    ts_cor = np.empty((num_date, num_pixel), dtype=np.float64)
    ts_res = np.empty((num_date, num_pixel), dtype=np.float64)

    for start in range(0, num_pixel, pixel_batch_size):
        stop = min(start + pixel_batch_size, num_pixel)
        observations_all = np.asarray(ts0[:, start:stop].T, dtype=np.float64)
        geometry_all = np.asarray(G_geom0[:, start:stop].T, dtype=np.float64)
        batch_size = stop - start
        deformation_all = np.broadcast_to(
            np.asarray(G_defo0, dtype=np.float64),
            (batch_size, *G_defo0.shape),
        )
        design_all = np.concatenate((geometry_all[:, :, None], deformation_all), axis=2)

        observations = observations_all[:, date_flag]
        design = design_all[:, date_flag, :]
        if phase_velocity:
            tbase_diff = np.diff(tbase[date_flag], axis=0).reshape(1, -1)
            observations = np.diff(observations, axis=1) / tbase_diff
            design = np.diff(design, axis=1) / tbase_diff[:, :, None]
            design = np.concatenate((design[:, :, :1], design[:, :, 2:]), axis=2)
            design_all = np.concatenate((design_all[:, :, :1], design_all[:, :, 2:]), axis=2)

        if design.shape[2] != num_param:
            raise ValueError("unexpected MintPy design-matrix size")

        solution = _solve_batched_weighted_lstsq(design, observations)
        active = np.ones(batch_size, dtype=bool)
        for _ in range(max_iterations):
            residual = observations - np.einsum(
                "bnk,bk->bn",
                design,
                solution,
                optimize=True,
            )
            center = np.median(residual, axis=1)
            scale = 1.4826 * np.median(np.abs(residual - center[:, None]), axis=1)
            scale_floor = np.finfo(np.float64).eps * np.maximum(
                1.0,
                np.max(np.abs(observations), axis=1),
            )
            active &= np.isfinite(scale) & (scale > scale_floor)
            if not np.any(active):
                break

            denominator = huber_delta * np.maximum(scale, scale_floor)
            normalized = np.abs(residual) / denominator[:, None]
            weights = np.ones_like(normalized)
            outlier = normalized > 1.0
            weights[outlier] = 1.0 / normalized[outlier]
            next_solution = _solve_batched_weighted_lstsq(
                design[active],
                observations[active],
                weights[active],
            )
            previous = solution[active]
            relative_change = np.linalg.norm(next_solution - previous, axis=1)
            relative_change /= np.maximum(1.0, np.linalg.norm(previous, axis=1))
            solution[active] = next_solution
            active_indices = np.flatnonzero(active)
            active[active_indices[relative_change <= tolerance]] = False

        prediction = np.einsum(
            "bnk,bk->bn",
            design_all,
            solution,
            optimize=True,
        )
        dem_displacement = design_all[:, :, 0] * solution[:, :1]
        delta_z[start:stop] = solution[:, 0]
        ts_cor[:, start:stop] = (observations_all - dem_displacement).T
        ts_res[:, start:stop] = (observations_all - prediction).T

    return delta_z, ts_cor, ts_res


def correct_dem_error_patch_huber(
    G_defo: np.ndarray,
    ts_file: str,
    geom_file: str | None = None,
    box: tuple[int, int, int, int] | None = None,
    date_flag: np.ndarray | None = None,
    phase_velocity: bool = False,
    *,
    huber_delta: float = 1.345,
    max_iterations: int = 8,
    tolerance: float = 1e-5,
    pixel_batch_size: int = 32768,
):
    """MintPy's patch preparation with a batched Huber numerical kernel."""
    from mintpy.dem_error import read_geometry
    from mintpy.objects import timeseries
    from mintpy.utils import readfile

    ts_obj = timeseries(ts_file)
    ts_obj.open(print_msg=False)
    if box:
        num_row = box[3] - box[1]
        num_col = box[2] - box[0]
    else:
        num_row = ts_obj.length
        num_col = ts_obj.width
    num_pixel = num_row * num_col
    num_date = ts_obj.numDate
    tbase = np.asarray(ts_obj.tbase, dtype=np.float32) / 365.25

    ts_data = readfile.read(ts_file, box=box)[0].reshape(num_date, -1)
    sin_inc_angle, range_dist, pbase = read_geometry(ts_file, geom_file, box=box)

    print("skip pixels with ZERO in ALL acquisitions")
    mask = np.nanmean(ts_data, axis=0) != 0.0
    print("skip pixels with NaN  in ANY acquisitions")
    mask *= np.sum(np.isnan(ts_data), axis=0) == 0

    import os

    tcoh_file = os.path.join(os.path.dirname(ts_file), "temporalCoherence.h5")
    if os.path.isfile(tcoh_file):
        print("skip pixels with ZERO temporal coherence")
        tcoh = readfile.read(tcoh_file, box=box)[0].flatten()
        mask *= tcoh != 0.0
    if range_dist.size != 1:
        print("skip pixels with ZERO / NaN value in incidenceAngle / slantRangeDistance")
        for geom_data in (sin_inc_angle, range_dist):
            mask *= geom_data != 0.0
            mask *= ~np.isnan(geom_data)

    num_pixel2inv = int(np.sum(mask))
    percentage = num_pixel2inv / num_pixel * 100
    print(
        f"number of pixels to invert: {num_pixel2inv} out of {num_pixel} "
        f"({percentage:.1f}%)"
    )
    delta_z = np.zeros(num_pixel, dtype=np.float32)
    ts_cor = np.zeros((num_date, num_pixel), dtype=np.float32)
    ts_res = np.zeros((num_date, num_pixel), dtype=np.float32)
    if num_pixel2inv < 1:
        return (
            delta_z.reshape(num_row, num_col),
            ts_cor.reshape(num_date, num_row, num_col),
            ts_res.reshape(num_date, num_row, num_col),
            box,
        )

    estimator_name = "Huber IRLS" if max_iterations else "OLS"
    print(
        f"estimating DEM error with batched {estimator_name} "
        f"({pixel_batch_size} pixels per batch) ..."
    )
    denominator = np.asarray(range_dist).reshape(-1) * np.asarray(sin_inc_angle).reshape(-1)
    if denominator.size == 1:
        geometry = pbase / denominator[0]
    elif pbase.shape[1] == 1:
        geometry = pbase / denominator[mask].reshape(1, -1)
    else:
        geometry = pbase[:, mask] / denominator[mask].reshape(1, -1)

    delta_z_i, ts_cor_i, ts_res_i = estimate_dem_error_huber_batch(
        ts_data[:, mask],
        geometry,
        G_defo,
        tbase,
        date_flag=date_flag,
        phase_velocity=phase_velocity,
        huber_delta=huber_delta,
        max_iterations=max_iterations,
        tolerance=tolerance,
        pixel_batch_size=pixel_batch_size,
    )
    delta_z[mask] = delta_z_i
    ts_cor[:, mask] = ts_cor_i
    ts_res[:, mask] = ts_res_i
    return (
        delta_z.reshape(num_row, num_col),
        ts_cor.reshape(num_date, num_row, num_col),
        ts_res.reshape(num_date, num_row, num_col),
        box,
    )


def estimate_dem_error_huber(
    ts0: np.ndarray,
    G0: np.ndarray,
    tbase: np.ndarray,
    date_flag: np.ndarray | None = None,
    phase_velocity: bool = False,
    cond: float = 1e-8,
    display: bool = False,
    sharey: bool = True,
    *,
    huber_delta: float = 1.345,
    max_iterations: int = 8,
    tolerance: float = 1e-5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate DEM error using Huber IRLS on MintPy's exact linear system."""
    ts0 = np.asarray(ts0)
    G0 = np.asarray(G0)
    tbase = np.asarray(tbase)
    if ts0.ndim == 1:
        ts0 = ts0.reshape(-1, 1)
    if date_flag is None:
        date_flag = np.ones(ts0.shape[0], dtype=np.bool_)

    G = G0[date_flag, :]
    ts = ts0[date_flag, :]
    design_all = G0

    if phase_velocity:
        tbase_diff = np.diff(tbase[date_flag], axis=0).reshape(-1, 1)
        ts = np.diff(ts, axis=0) / np.repeat(tbase_diff, ts.shape[1], axis=1)
        G = np.diff(G, axis=0) / np.repeat(tbase_diff, G.shape[1], axis=1)
        G = np.hstack((G[:, :1], G[:, 2:]))
        design_all = np.hstack((G0[:, :1], G0[:, 2:]))

    solution = linalg.lstsq(G, ts, cond=cond)[0]
    for pixel in range(ts.shape[1]):
        x = solution[:, pixel]
        y = ts[:, pixel]
        for _ in range(max_iterations):
            residual = y - G @ x
            center = np.median(residual)
            scale = 1.4826 * np.median(np.abs(residual - center))
            scale_floor = np.finfo(np.float64).eps * max(1.0, np.max(np.abs(y)))
            if not np.isfinite(scale) or scale <= scale_floor:
                break

            normalized = np.abs(residual) / (huber_delta * scale)
            weights = np.ones_like(normalized)
            outlier = normalized > 1.0
            weights[outlier] = 1.0 / normalized[outlier]
            sqrt_weight = np.sqrt(weights)
            next_x = linalg.lstsq(
                G * sqrt_weight[:, None],
                y * sqrt_weight,
                cond=cond,
            )[0]
            relative_change = np.linalg.norm(next_x - x) / max(1.0, np.linalg.norm(x))
            x = next_x
            if relative_change <= tolerance:
                break
        solution[:, pixel] = x

    delta_z = solution[0, :]
    dem_displacement = np.dot(
        design_all[:, 0].reshape(-1, 1),
        delta_z.reshape(1, -1),
    )
    ts_cor = ts0 - dem_displacement
    ts_res = ts0 - np.dot(design_all, solution)

    if display:
        from matplotlib import pyplot as plt

        _, axes = plt.subplots(
            nrows=4,
            ncols=1,
            figsize=(8, 8),
            sharex=True,
            sharey=sharey,
        )
        titles = ["Original TS", "Corrected TS", "Fitting residual", "Fitted defo model"]
        for axis, data, title in zip(
            axes,
            [ts0, ts_cor, ts_res, ts_cor - ts_res],
            titles,
        ):
            axis.plot(data, ".")
            axis.set_title(title)
        plt.show()

    return delta_z, ts_cor, ts_res


def _parse_wrapper_args(iargs: Sequence[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--estimator", choices=ESTIMATORS, default="homa_huber")
    parser.add_argument("--huber-delta", type=float, default=1.345)
    parser.add_argument("--huber-iterations", type=int, default=8)
    parser.add_argument("--huber-tolerance", type=float, default=1e-5)
    parser.add_argument("--pixel-batch-size", type=int, default=32768)
    options, mintpy_args = parser.parse_known_args(iargs)
    if options.huber_delta <= 0:
        parser.error("--huber-delta must be positive")
    if options.huber_iterations < 1:
        parser.error("--huber-iterations must be at least 1")
    if options.huber_tolerance <= 0:
        parser.error("--huber-tolerance must be positive")
    if options.pixel_batch_size < 1:
        parser.error("--pixel-batch-size must be at least 1")
    return options, mintpy_args


def main(iargs: Sequence[str] | None = None) -> int:
    options, mintpy_args = _parse_wrapper_args(sys.argv[1:] if iargs is None else iargs)

    from mintpy import dem_error
    from mintpy.cli import dem_error as dem_error_cli

    inps = dem_error_cli.cmd_line_parse(mintpy_args)
    if options.estimator == "mintpy_ols":
        estimator = dem_error.estimate_dem_error
    else:
        if inps.update_mode:
            raise ValueError(
                "homa_huber does not use MintPy --update mode because native output "
                "metadata does not identify the replacement estimator; use a separate "
                "output directory and omit --update"
            )
        if inps.cluster:
            raise ValueError(
                "homa_huber requires MintPy clustering to be disabled because the "
                "estimator replacement is local to this Python process; omit "
                "--cluster and set mintpy.compute.cluster = none in the template"
            )

        def estimator(*args, **kwargs):
            return estimate_dem_error_huber(
                *args,
                **kwargs,
                huber_delta=options.huber_delta,
                max_iterations=options.huber_iterations,
                tolerance=options.huber_tolerance,
            )

        def patch_estimator(*args, **kwargs):
            return correct_dem_error_patch_huber(
                *args,
                **kwargs,
                huber_delta=options.huber_delta,
                max_iterations=(
                    options.huber_iterations
                    if options.estimator == "homa_huber"
                    else 0
                ),
                tolerance=options.huber_tolerance,
                pixel_batch_size=options.pixel_batch_size,
            )

    print(f"DEM-error numerical estimator: {options.estimator}")

    original_estimator = dem_error.estimate_dem_error
    original_patch_estimator = dem_error.correct_dem_error_patch
    dem_error.estimate_dem_error = estimator
    if options.estimator != "mintpy_ols":
        dem_error.correct_dem_error_patch = patch_estimator
    try:
        if inps.update_mode and dem_error.run_or_skip(inps) == "skip":
            return 0
        dem_error.correct_dem_error(inps)
    finally:
        dem_error.estimate_dem_error = original_estimator
        dem_error.correct_dem_error_patch = original_patch_estimator
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
