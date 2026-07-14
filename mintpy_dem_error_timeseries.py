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


ESTIMATORS = ("mintpy_ols", "homa_huber")


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
    options, mintpy_args = parser.parse_known_args(iargs)
    if options.huber_delta <= 0:
        parser.error("--huber-delta must be positive")
    if options.huber_iterations < 1:
        parser.error("--huber-iterations must be at least 1")
    if options.huber_tolerance <= 0:
        parser.error("--huber-tolerance must be positive")
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

    print(f"DEM-error numerical estimator: {options.estimator}")

    original_estimator = dem_error.estimate_dem_error
    dem_error.estimate_dem_error = estimator
    try:
        if inps.update_mode and dem_error.run_or_skip(inps) == "skip":
            return 0
        dem_error.correct_dem_error(inps)
    finally:
        dem_error.estimate_dem_error = original_estimator
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
